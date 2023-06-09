//HEAD_DSPH
/*
<DUALSPHYSICS>  Copyright (c) 2019 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/).

EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

This file is part of DualSPHysics.

DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>.
*/

/// \file JSphCpu.cpp \brief Implements the class \ref JSphCpu.

#include "JSphCpu.h"
#include "JCellDivCpu.h"
#include "JPartFloatBi4.h"
#include "Functions.h"
#include "JSphMotion.h"
#include "JArraysCpu.h"
#include "JSphDtFixed.h"
#include "JWaveGen.h"
#include "JMLPistons.h"     //<vs_mlapiston>
#include "JRelaxZones.h"    //<vs_rzone>
#include "JChronoObjects.h" //<vs_chroono>
#include "JDamping.h"
#include "JXml.h"
#include "JSaveDt.h"
#include "JTimeOut.h"
#include "JSphAccInput.h"
#include "JGaugeSystem.h"
#include "JSphBoundCorr.h"  //<vs_innlet>
#include <climits>

using namespace std;


//==============================================================================
/// Prepare variables for interaction functions for non-Newtonian formulation.
/// Prepara variables para interaccion.
//==============================================================================
void JSphCpu::ComputePress_NN(unsigned np, unsigned npb) {
    //-Prepare values of rhop for interaction. | Prepara datos derivados de rhop para interaccion.
    const int n = int(np);
#ifdef OMP_USE
#pragma omp parallel for schedule (static) if(n>OMP_LIMIT_COMPUTELIGHT)
#endif
    for (int p = 0; p < n; p++) {
        float rhozero_ph; float cteb_ph; float gamma_ph;
        const typecode cod = Codec[p];
        if (CODE_IsFluid(cod)) {
            unsigned cp = CODE_GetTypeValue(cod);
            rhozero_ph = PhaseArray[cp].rho;
            cteb_ph = PhaseArray[cp].CteB;
            gamma_ph = PhaseArray[cp].Gamma;
        }
        else {
            rhozero_ph = RhopZero;
            cteb_ph = CteB;
            gamma_ph = Gamma;
        }
        const float rhop = Velrhopc[p].w, rhop_r0 = rhop / rhozero_ph;
        Pressc[p] = cteb_ph*(pow(rhop_r0, gamma_ph) - 1.0f);
    }
}
//==============================================================================
//Full tensors
//==============================================================================
/// These functions return values for the tensors and invariants.
//==============================================================================
//==============================================================================
/// Calculates the velocity gradient (full matrix)
//==============================================================================
void JSphCpu::GetVelocityGradients_FDA(float rr2, float drx, float dry, float drz
    , float dvx, float dvy, float dvz, tmatrix3f &dvelp1, float &div_vel)const
{
    //vel gradients
    dvelp1.a11 = dvx*drx / rr2; dvelp1.a12 = dvx*dry / rr2; dvelp1.a13 = dvx*drz / rr2; //Fan et al., 2010
    dvelp1.a21 = dvy*drx / rr2; dvelp1.a22 = dvy*dry / rr2; dvelp1.a23 = dvy*drz / rr2;
    dvelp1.a31 = dvz*drx / rr2; dvelp1.a32 = dvz*dry / rr2; dvelp1.a33 = dvz*drz / rr2;
    div_vel = (dvelp1.a11 + dvelp1.a22 + dvelp1.a33) / 3.f;
}
//==============================================================================
/// Calculates the Strain Rate Tensor (full matrix)
//==============================================================================
void JSphCpu::GetStrainRateTensor(const tmatrix3f &dvelp1, float div_vel, float &I_D, float &II_D, float &J1_D, float &J2_D, float &div_D_tensor, float &D_tensor_magn, tmatrix3f &D_tensor)const   
{
    //Strain tensor and invariant
	D_tensor.a11 = dvelp1.a11 - div_vel;          D_tensor.a12 = 0.5f*(dvelp1.a12 + dvelp1.a21);      D_tensor.a13 = 0.5f*(dvelp1.a13 + dvelp1.a31);
	D_tensor.a21 = 0.5f*(dvelp1.a21 + dvelp1.a12);      D_tensor.a22 = dvelp1.a22 - div_vel;          D_tensor.a23 = 0.5f*(dvelp1.a23 + dvelp1.a32);
	D_tensor.a31 = 0.5f*(dvelp1.a31 + dvelp1.a13);      D_tensor.a32 = 0.5f*(dvelp1.a32 + dvelp1.a23);      D_tensor.a33 = dvelp1.a33 - div_vel;
    div_D_tensor = (D_tensor.a11 + D_tensor.a22 + D_tensor.a33) / 3.f;

    //I_D - the first invariant -
    I_D = D_tensor.a11 + D_tensor.a22 + D_tensor.a33;
    //II_D - the second invariant - expnaded form witout symetry 
    float II_D_1 = D_tensor.a11*D_tensor.a22 + D_tensor.a22*D_tensor.a33 + D_tensor.a11*D_tensor.a33;
    float II_D_2 = D_tensor.a12*D_tensor.a21 + D_tensor.a23*D_tensor.a32 + D_tensor.a13*D_tensor.a31;
    II_D = -II_D_1 + II_D_2; 
    //deformation tensor magnitude
    D_tensor_magn = sqrt((II_D));
    if (II_D < 0.f) {
        printf("****D_tensor_magn is negative**** \n");     
    }
    //Main Strain rate invariants
    J1_D = I_D; J2_D = I_D*I_D - 2.f*II_D;
}
//==============================================================================
/// Calculates the effective visocity
//==============================================================================
void JSphCpu::GetEta_Effective(const typecode pp1, float tau_yield, float D_tensor_magn, float visco, float m_NN, float n_NN, float reg_NN, float reg_strain, float &visco_etap1)const
{ // Robin: added reg_NN and reg_strain
	//if (D_tensor_magn != D_tensor_magn)printf("at eta D_tensor_magn=%f\n", D_tensor_magn);

  bool tiny_strain = 0 ; // Robin : I added this to have a possibly more reliable condition for 
                         // whether we have passed our minimum strain.
  visco_etap1 = 0;       // R : just resetting this (maybe can delete)
  float visco_etap1_term1 = 0, visco_etap1_term2 = 0; 
  // R : Just initialising these two terms.

  if (D_tensor_magn < reg_strain)
  {
    D_tensor_magn = reg_strain;
    tiny_strain = 1; // R: This lets us set the condition for small-strain conditions more reliably than using a float
  } // R: avoid potentially dividing by zero

  if (tiny_strain == 0)
  {
    visco_etap1_term1 = visco * pow(2.0f * D_tensor_magn, (n_NN - 1.0f)); 
    visco_etap1_term2 = PhaseCte[pp1].tau_max / (2.0f * D_tensor_magn);
    // Robin: Check the factors of 2 here.
    // Term 2 is calculated without the Papast. regularisation, because we already have two
    // other types of regularisation going on. (Relating to strain, and to the max. viscosity.)
    // To include it, just add the following code to term_2:
    // * (1.f - exp( -m_NN * 2.0f * D_tensor_magn));
    // I originally also included the HBP model as an option similar to the original 
    // implementation (i.e. using different variables to pick an option). 
    // I did not do that in this version of the code for simplicity. 
  
    visco_etap1 = visco_etap1_term1 + visco_etap1_term2;
  }


  if (tiny_strain == 1 || visco_etap1 > reg_NN)
  {
    visco_etap1 = reg_NN; // R: this regularises the 
  }

// Robin : original code is below. 
/*
	float miou_yield = ( PhaseCte[pp1].tau_max ? PhaseCte[pp1].tau_max / (2.0f*D_tensor_magn) : (tau_yield) / (2.0f*D_tensor_magn) ); //HPB will adjust eta		
	//if tau_max exists
	if (PhaseCte[pp1].tau_max && D_tensor_magn <= PhaseCte[pp1].tau_max / (2.f*PhaseCte[pp1].Bi_multi*visco)) { //multiplier
		miou_yield = PhaseCte[pp1].Bi_multi*visco;
	}
	//Papanastasiou
	float visco_etap1_term1 =( PhaseCte[pp1].tau_max ? miou_yield : miou_yield *(1.f - exp(-m_NN*D_tensor_magn)) ); // Robin: I think there is a factor of two missing here.
	if (D_tensor_magn <= ALMOSTZERO) visco_etap1_term1 = (PhaseCte[pp1].tau_max ? miou_yield : m_NN*tau_yield);
	//HB
	float visco_etap1_term2 = visco*pow(D_tensor_magn, (n_NN - 1.0f)); // Robin: I think there is a factor of two missing here as well.
	if (D_tensor_magn <= ALMOSTZERO)visco_etap1_term2 = visco;
	visco_etap1 = visco_etap1_term1 + visco_etap1_term2;
*/

	/*
    //use according to YOUR criteria
    float tyield = (D_tensor_magn <= tau_yield / (2.f*visco) ? (PhaseCte[pp1].tau_max? PhaseCte[pp1].tau_max: tau_yield / (2.f*visco)) : tau_yield / (2.0f*D_tensor_magn));     
    //use according to YOUR criteria
    if (tyield != tyield)tyield = (PhaseCte[pp1].tau_max ? PhaseCte[pp1].tau_max:tau_yield / (2.f*visco));      
    float visco_etap1_term1 = tyield *(1.f - exp(-m_NN*D_tensor_magn)); // Robin : 2.0f
    float visco_etap1_term2 = visco*pow(D_tensor_magn, n_NN - 1); // Robin : 2.0f
    visco_etap1 = visco_etap1_term1 + visco_etap1_term2;
	*/
}
//==============================================================================
/// Calculates the stress Tensor (full matrix)
//==============================================================================
void JSphCpu::GetStressTensor(const tmatrix3f &D_tensor, float visco_etap1, float &I_t, float &II_t, float &J1_t, float &J2_t, float &tau_tensor_magn, tmatrix3f &tau_tensor)const
{
    //Stress tensor and invariant   
    tau_tensor.a11 = 2.f*visco_etap1*(D_tensor.a11);    tau_tensor.a12 = 2.f*visco_etap1*D_tensor.a12;      tau_tensor.a13 = 2.f*visco_etap1*D_tensor.a13;
    tau_tensor.a21 = 2.f*visco_etap1*D_tensor.a21;      tau_tensor.a22 = 2.f*visco_etap1*(D_tensor.a22);    tau_tensor.a23 = 2.f*visco_etap1*D_tensor.a23;
    tau_tensor.a31 = 2.f*visco_etap1*D_tensor.a31;      tau_tensor.a32 = 2.f*visco_etap1*D_tensor.a32;      tau_tensor.a33 = 2.f*visco_etap1*(D_tensor.a33);

    //I_t - the first invariant -
    I_t = tau_tensor.a11 + tau_tensor.a22 + tau_tensor.a33;
    //II_t - the second invariant - expnaded form witout symetry 
    float II_t_1 = tau_tensor.a11*tau_tensor.a22 + tau_tensor.a22*tau_tensor.a33 + tau_tensor.a11*tau_tensor.a33;
    float II_t_2 = tau_tensor.a12*tau_tensor.a21 + tau_tensor.a23*tau_tensor.a32 + tau_tensor.a13*tau_tensor.a31;
    II_t = -II_t_1 + II_t_2;
    //stress tensor magnitude
    tau_tensor_magn = sqrt(II_t); // Robin: corrected
    if (II_t < 0.f) {
        printf("****tau_tensor_magn is negative**** \n");
    }
    //Main Strain rate invariants
    J1_t = I_t; J2_t = I_t*I_t - 2.f*II_t;
}

//==============================================================================
//symetric tensors
//==============================================================================
/// Calculates the velocity gradients symetric
//==============================================================================
void JSphCpu::GetVelocityGradients_SPH_tsym(float massp2, const tfloat4 &velrhop2, float dvx, float dvy, float dvz, float frx, float fry, float frz
    , tsymatrix3f &gradvelp1)const
{
    //vel gradients
    const float volp2 = -massp2 / velrhop2.w;
    float dv = dvx*volp2; gradvelp1.xx += dv*frx; gradvelp1.xy += dv*fry; gradvelp1.xz += dv*frz;
          dv = dvy*volp2; gradvelp1.xy += dv*frx; gradvelp1.yy += dv*fry; gradvelp1.yz += dv*frz;
          dv = dvz*volp2; gradvelp1.xz += dv*frx; gradvelp1.yz += dv*fry; gradvelp1.zz += dv*frz;
}
//==============================================================================
/// Calculates the Strain Rate Tensor (symetric)
//==============================================================================
void JSphCpu::GetStrainRateTensor_tsym(const tsymatrix3f &dvelp1, float &I_D, float &II_D, float &J1_D, float &J2_D, float &div_D_tensor, float &D_tensor_magn, tsymatrix3f &D_tensor)const
{
    //Strain tensor and invariant
    float div_vel= (dvelp1.xx + dvelp1.yy + dvelp1.zz) / 3.f;
    D_tensor.xx = dvelp1.xx - div_vel;      D_tensor.xy = 0.5f*(dvelp1.xy);     D_tensor.xz = 0.5f*(dvelp1.xz);
                                            D_tensor.yy = dvelp1.yy - div_vel;  D_tensor.yz = 0.5f*(dvelp1.yz);
                                                                                D_tensor.zz = dvelp1.zz - div_vel;
    //the off-diagonal entries of velocity gradients are i.e. 0.5f*(du/dy+dvdx) with dvelp1.xy=du/dy+dvdx
    div_D_tensor = (D_tensor.xx + D_tensor.yy + D_tensor.zz) / 3.f;

    //I_D - the first invariant -
    I_D = D_tensor.xx + D_tensor.yy + D_tensor.zz;
    //II_D - the second invariant - expnaded form witout symetry 
    float II_D_1 = D_tensor.xx*D_tensor.yy + D_tensor.yy*D_tensor.zz + D_tensor.xx*D_tensor.zz;
    float II_D_2 = D_tensor.xy*D_tensor.xy + D_tensor.yz*D_tensor.yz + D_tensor.xz*D_tensor.xz;
    II_D = -II_D_1 + II_D_2;
    ////deformation tensor magnitude
    D_tensor_magn = sqrt((II_D));
    if (II_D < 0.f) {
        printf("****D_tensor_magn is negative**** \n");
    }
    //Main Strain rate invariants
    J1_D = I_D; J2_D = I_D*I_D - 2.f*II_D;
}
//==============================================================================
/// Calculates the Stress Tensor (symetric)
//==============================================================================
void JSphCpu::GetStressTensor_sym(const tsymatrix3f &D_tensorp1, float visco_etap1, float &I_t, float &II_t, float &J1_t, float &J2_t, float &tau_tensor_magn, tsymatrix3f &tau_tensorp1)const
{
    //Stress tensor and invariant
    tau_tensorp1.xx = 2.f*visco_etap1*(D_tensorp1.xx);  tau_tensorp1.xy = 2.f*visco_etap1*D_tensorp1.xy;    tau_tensorp1.xz = 2.f*visco_etap1*D_tensorp1.xz;
                                                        tau_tensorp1.yy = 2.f*visco_etap1*(D_tensorp1.yy);  tau_tensorp1.yz = 2.f*visco_etap1*D_tensorp1.yz;
                                                                                                            tau_tensorp1.zz = 2.f*visco_etap1*(D_tensorp1.zz);
    //I_t - the first invariant -
    I_t = tau_tensorp1.xx + tau_tensorp1.yy + tau_tensorp1.zz;
    //II_t - the second invariant - expnaded form witout symetry 
    float II_t_1 = tau_tensorp1.xx*tau_tensorp1.yy + tau_tensorp1.yy*tau_tensorp1.zz + tau_tensorp1.xx*tau_tensorp1.zz;
    float II_t_2 = tau_tensorp1.xy*tau_tensorp1.xy + tau_tensorp1.yz*tau_tensorp1.yz + tau_tensorp1.xz*tau_tensorp1.xz;
    II_t = -II_t_1 + II_t_2;
    //stress tensor magnitude
    tau_tensor_magn = sqrt(II_t);
    if (II_t < 0.f) {
        printf("****tau_tensor_magn is negative**** \n");
    }
    //Main Stress rate invariants
    J1_t = I_t; J2_t = I_t*I_t - 2.f*II_t;
}

//end_of_file
