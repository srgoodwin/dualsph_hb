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
/// Perform interaction between particles for NN using the FDA approach. Bound-Fluid/Float
//==============================================================================
template<TpKernel tker, TpFtMode ftmode> void JSphCpu::InteractionForcesBound_NN_FDA
(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial
    , const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
    , const tdouble3 *pos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
    , float &viscdt, float *ar)const
{
    //-Initialize viscth to calculate max viscdt with OpenMP. | Inicializa viscth para calcular visdt maximo con OpenMP.
    float viscth[OMP_MAXTHREADS*OMP_STRIDE];
    for (int th = 0; th<OmpThreads; th++)viscth[th*OMP_STRIDE] = 0;
    //-Starts execution using OpenMP.
    const int pfin = int(pinit + n);
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
    for (int p1 = int(pinit); p1<pfin; p1++) {
        float visc = 0, arp1 = 0;

        //-Load data of particle p1. | Carga datos de particula p1.
        const tdouble3 posp1 = pos[p1];
        const bool rsymp1 = (Symmetry && posp1.y <= Dosh); //<vs_syymmetry>
        const tfloat3 velp1 = TFloat3(velrhop[p1].x, velrhop[p1].y, velrhop[p1].z);

        //-Obtain limits of interaction. | Obtiene limites de interaccion.
        int cxini, cxfin, yini, yfin, zini, zfin;
        GetInteractionCells(dcell[p1], hdiv, nc, cellzero, cxini, cxfin, yini, yfin, zini, zfin);

        //-Search for neighbours in adjacent cells. | Busqueda de vecinos en celdas adyacentes.
        for (int z = zini; z<zfin; z++) {
            const int zmod = (nc.w)*z + cellinitial; //-Sum from start of fluid cells. | Le suma donde empiezan las celdas de fluido.
            for (int y = yini; y<yfin; y++) {
                int ymod = zmod + nc.x*y;
                const unsigned pini = beginendcell[cxini + ymod];
                const unsigned pfin = beginendcell[cxfin + ymod];

                //-Interaction of boundary with type Fluid/Float | Interaccion de Bound con varias Fluid/Float.
                //---------------------------------------------------------------------------------------------
                bool rsym = false; //<vs_syymmetry>
                for (unsigned p2 = pini; p2<pfin; p2++) {
                    const float drx = float(posp1.x - pos[p2].x);
                    float dry = float(posp1.y - pos[p2].y);
                    if (rsym)    dry = float(posp1.y + pos[p2].y); //<vs_syymmetry>
                    const float drz = float(posp1.z - pos[p2].z);
                    const float rr2 = drx*drx + dry*dry + drz*drz;
                    if (rr2 <= Fourh2 && rr2 >= ALMOSTZERO) {
                        //-Wendland, Cubic Spline or Gaussian kernel.
                        float frx, fry, frz;
                        if (tker == KERNEL_Wendland)     GetKernelWendland(rr2, drx, dry, drz, frx, fry, frz);
                        else if (tker == KERNEL_Cubic)   GetKernelCubic(rr2, drx, dry, drz, frx, fry, frz);
                        else if (tker == KERNEL_Gaussian)GetKernelGaussian(rr2, drx, dry, drz, frx, fry, frz);
                        else if (tker == KERNEL_WendlandC6)GetKernelWendlandC6(rr2, drx, dry, drz, frx, fry, frz);  //<vs_praticalsskq>

                                                                                                                    //===== Get mass of particle p2 ===== 
                        float massp2 = MassFluid; //-Contains particle mass of incorrect fluid. | Contiene masa de particula por defecto fluid.
                        bool compute = true;      //-Deactivate when using DEM and/or bound-float. | Se desactiva cuando se usa DEM y es bound-float.
                        if (USE_FLOATING) {
                            bool ftp2 = CODE_IsFloating(code[p2]);
                            if (ftp2)massp2 = FtObjs[CODE_GetTypeValue(code[p2])].massp;
                            compute = !(USE_FTEXTERNAL && ftp2); //-Deactivate when using DEM/Chrono and/or bound-float. | Se desactiva cuando se usa DEM/Chrono y es bound-float.
                        }

                        if (compute) {
                            //-Density derivative.
                            //const float dvx=velp1.x-velrhop[p2].x, dvy=velp1.y-velrhop[p2].y, dvz=velp1.z-velrhop[p2].z;
                            tfloat4 velrhop2 = velrhop[p2];
                            if (rsym)velrhop2.y = -velrhop2.y; //<vs_syymmetry>
                            const float dvx = velp1.x - velrhop2.x, dvy = velp1.y - velrhop2.y, dvz = velp1.z - velrhop2.z;
                            if (compute)arp1 += massp2*(dvx*frx + dvy*fry + dvz*frz);

                            {//-Viscosity.
                                const float dot = drx*dvx + dry*dvy + drz*dvz;
                                const float dot_rr2 = dot / (rr2 + Eta2);
                                visc = max(dot_rr2, visc);
                            }
                        }
                        rsym = (rsymp1 && !rsym && float(posp1.y - dry) <= Dosh); //<vs_syymmetry>
                        if (rsym)p2--;                                       //<vs_syymmetry>
                    }
                    else rsym = false;                                      //<vs_syymmetry>
                }
            }
        }
        //-Sum results together. | Almacena resultados.
        if (arp1 || visc) {
            ar[p1] += arp1;
            const int th = omp_get_thread_num();
            if (visc>viscth[th*OMP_STRIDE])viscth[th*OMP_STRIDE] = visc;
        }
    }
    //-Keep max value in viscdt. | Guarda en viscdt el valor maximo.
    for (int th = 0; th<OmpThreads; th++)if (viscdt<viscth[th*OMP_STRIDE])viscdt = viscth[th*OMP_STRIDE];
}

//==============================================================================
/// Perform interaction between particles for NN using the FDA approach: Fluid/Float-Fluid/Float or Fluid/Float-Bound
//==============================================================================
template<TpKernel tker, TpFtMode ftmode, TpVisco lamsps, TpDensity tdensity, bool shift> void JSphCpu::InteractionForcesFluid_NN_FDA_All
(unsigned n, unsigned pinit, tint4 nc, int hdiv, unsigned cellinitial, float visco, float *visco_eta
    , const unsigned *beginendcell, tint3 cellzero, const unsigned *dcell
    , const tsymatrix3f* tau, tsymatrix3f* gradvel
    , const tdouble3 *pos, const tfloat4 *velrhop, const typecode *code, const unsigned *idp
    , const float *press
    , float &viscdt, float &viscetadt, float *ar, tfloat3 *ace, float *delta
    , TpShifting shiftmode, tfloat4 *shiftposfs)const
{
    const bool boundp2 = (!cellinitial); //-Interaction with type boundary (Bound). | Interaccion con Bound.
    
    //-Initialize viscth to calculate viscdt maximo con OpenMP. | Inicializa viscth para calcular visdt maximo con OpenMP.
    float viscth[OMP_MAXTHREADS*OMP_STRIDE];
    float viscetath[OMP_MAXTHREADS*OMP_STRIDE];
    for (int th = 0; th < OmpThreads; th++) { 
        viscth[th*OMP_STRIDE] = 0; 
        viscetath[th*OMP_STRIDE] = 0;
    }
    
    //-Initialise execution with OpenMP. | Inicia ejecucion con OpenMP.
    const int pfin = int(pinit + n);
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
    for (int p1 = int(pinit); p1<pfin; p1++) {
        float visc = 0, arp1 = 0, deltap1 = 0;
        tfloat3 acep1 = TFloat3(0);
        //-Variables for Shifting.
        tfloat4 shiftposfsp1;
        if (shift)shiftposfsp1 = shiftposfs[p1];

        //-Obtain data of particle p1 in case of floating objects. | Obtiene datos de particula p1 en caso de existir floatings.
        bool ftp1 = false;     //-Indicate if it is floating. | Indica si es floating.      
        if (USE_FLOATING) {
            ftp1 = CODE_IsFloating(code[p1]);
            if (ftp1 && tdensity != DDT_None)deltap1 = FLT_MAX; //-DDT is not applied to floating particles.
            if (ftp1 && shift)shiftposfsp1.x = FLT_MAX;  //-For floating objects do not calculate shifting. | Para floatings no se calcula shifting.
        }

        //-Obtain data of particle p1.
        const tdouble3 posp1 = pos[p1];
        const tfloat3 velp1 = TFloat3(velrhop[p1].x, velrhop[p1].y, velrhop[p1].z);
        const float rhopp1 = velrhop[p1].w;
        const float pressp1 = press[p1];
        //const tsymatrix3f taup1 = (lamsps == VISCO_Artificial ? gradvelp1 : tau[p1]);
        const bool rsymp1 = (Symmetry && posp1.y <= Dosh); //<vs_syymmetry>             
        //<vs_non-Newtonian>
        const typecode pp1 = CODE_GetTypeValue(code[p1]);
        float visco_etap1 = 0;

        //-Obtain interaction limits.
        int cxini, cxfin, yini, yfin, zini, zfin;
        GetInteractionCells(dcell[p1], hdiv, nc, cellzero, cxini, cxfin, yini, yfin, zini, zfin);

        //-Search for neighbours in adjacent cells.
        for (int z = zini; z<zfin; z++) {
            const int zmod = (nc.w)*z + cellinitial; //-Sum from start of fluid or boundary cells. | Le suma donde empiezan las celdas de fluido o bound.
            for (int y = yini; y<yfin; y++) {
                int ymod = zmod + nc.x*y;
                const unsigned pini = beginendcell[cxini + ymod];
                const unsigned pfin = beginendcell[cxfin + ymod];

                //-Interaction of Fluid with type Fluid or Bound. | Interaccion de Fluid con varias Fluid o Bound.
                //------------------------------------------------------------------------------------------------
                bool rsym = false; //<vs_syymmetry>
                for (unsigned p2 = pini; p2<pfin; p2++) {
                    const float drx = float(posp1.x - pos[p2].x);
                    float dry = float(posp1.y - pos[p2].y);
                    if (rsym)    dry = float(posp1.y + pos[p2].y); //<vs_syymmetry>
                    const float drz = float(posp1.z - pos[p2].z);
                    const float rr2 = drx*drx + dry*dry + drz*drz;
                    if (rr2 <= Fourh2 && rr2 >= ALMOSTZERO) {
                        //-Wendland, Cubic Spline or Gaussian kernel.
                        float frx, fry, frz;
                        if (tker == KERNEL_Wendland)     GetKernelWendland(rr2, drx, dry, drz, frx, fry, frz);
                        else if (tker == KERNEL_Cubic)   GetKernelCubic(rr2, drx, dry, drz, frx, fry, frz);
                        else if (tker == KERNEL_Gaussian)GetKernelGaussian(rr2, drx, dry, drz, frx, fry, frz);
                        else if (tker == KERNEL_WendlandC6)GetKernelWendlandC6(rr2, drx, dry, drz, frx, fry, frz);  //<vs_praticalsskq>

                        //===== Get mass of particle p2 =====                       
                        const typecode pp2 = CODE_GetTypeValue(code[p2]); //<vs_non-Newtonian>
                        float massp2 = (boundp2 ? MassBound : PhaseArray[pp2].mass); //-Contiene masa de particula segun sea bound o fluid.
                        //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PhaseArray[pp1].mass : PhaseArray[pp2].mass);
                        //Floating
                        bool ftp2 = false;    //-Indicate if it is floating | Indica si es floating.
                        bool compute = true;  //-Deactivate when using DEM and if it is of type float-float or float-bound | Se desactiva cuando se usa DEM y es float-float o float-bound.
                        if (USE_FLOATING) {
                            ftp2 = CODE_IsFloating(code[p2]);
                            if (ftp2)massp2 = FtObjs[CODE_GetTypeValue(code[p2])].massp;
#ifdef DELTA_HEAVYFLOATING
                            if (ftp2 && tdensity == DDT_DDT && massp2 <= (MassFluid*1.2f))deltap1 = FLT_MAX;
#else
                            if (ftp2 && tdensity == DDT_DDT)deltap1 = FLT_MAX;
#endif
                            if (ftp2 && shift && shiftmode == SHIFT_NoBound)shiftposfsp1.x = FLT_MAX; //-With floating objects do not use shifting. | Con floatings anula shifting.
                            compute = !(USE_FTEXTERNAL && ftp1 && (boundp2 || ftp2)); //-Deactivate when using DEM and if it is of type float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
                        }

                        tfloat4 velrhop2 = velrhop[p2];
                        if (rsym)velrhop2.y = -velrhop2.y; //<vs_syymmetry>
                        //===== Acceleration ===== 
                        if (compute) {
                            const float prs = (pressp1 + press[p2]) / (rhopp1*velrhop2.w) + (tker == KERNEL_Cubic ? GetKernelCubicTensil(rr2, rhopp1, pressp1, velrhop2.w, press[p2]) : 0);
                            const float p_vpm = -prs*massp2;
                            acep1.x += p_vpm*frx; acep1.y += p_vpm*fry; acep1.z += p_vpm*frz;
                        }

                        //-Density derivative.
                        const float rhop1over2 = rhopp1 / velrhop2.w; //<vs_non-Newtonian>
                        float dvx = velp1.x - velrhop2.x, dvy = velp1.y - velrhop2.y, dvz = velp1.z - velrhop2.z;
                        if (compute)arp1 += massp2*(dvx*frx + dvy*fry + dvz*frz)*rhop1over2; //<vs_non-Newtonian>

                        const float cbar = max(PhaseArray[pp2].Cs0, PhaseArray[pp2].Cs0); //<vs_non-Newtonian>
                        //-Density Diffusion Term (DeltaSPH Molteni).
                        if (tdensity == DDT_DDT && deltap1 != FLT_MAX) {
                            const float visc_densi=DDT2h*cbar*(rhop1over2-1.f)/(rr2+Eta2);
                            const float dot3 = (drx*frx + dry*fry + drz*frz);
                            const float delta = (pp1 == pp2 ? visc_densi*dot3*massp2 : 0); //<vs_non-Newtonian>
                            //deltap1 = (boundp2 ? FLT_MAX : deltap1 + delta);
                            deltap1 = (boundp2 && TBoundary == BC_DBC ? FLT_MAX : deltap1 + delta);
                        }
                        //-Density Diffusion Term (Fourtakas et al 2019).  //<vs_dtt2_ini>
                        if ((tdensity == DDT_DDT2 || (tdensity == DDT_DDT2Full && !boundp2)) && deltap1 != FLT_MAX && !ftp2) {
                            const float rh = 1.f + DDTgz*drz;
                            const float drhop = RhopZero*pow(rh, 1.f / Gamma) - RhopZero;
                            const float visc_densi = DDT2h*cbar*((velrhop2.w - rhopp1) - drhop) / (rr2 + Eta2);
                            const float dot3 = (drx*frx + dry*fry + drz*frz);
                            const float delta = (pp1 == pp2 ? visc_densi*dot3*massp2 / velrhop2.w : 0); //<vs_non-Newtonian>
                            deltap1 = (boundp2 ? FLT_MAX : deltap1 - delta); //-blocks it makes it boil - bloody DBC
                        }  //<vs_dtt2_end>

                        //-Shifting correction.
                        if (shift && shiftposfsp1.x != FLT_MAX) {
                            bool heavyphase = (PhaseArray[pp1].mass > PhaseArray[pp2].mass && pp1 != pp2 ? true : false);
                            const float massrhop = massp2 / velrhop2.w;
                            const bool noshift = (boundp2 && (shiftmode == SHIFT_NoBound || (shiftmode == SHIFT_NoFixed && CODE_IsFixed(code[p2]))));
                            shiftposfsp1.x = (noshift ? FLT_MAX : (heavyphase ? 0 : shiftposfsp1.x + massrhop*frx)); //-For boundary do not use shifting. | Con boundary anula shifting.                            
                            shiftposfsp1.y += (heavyphase ? 0 : massrhop*fry); //<vs_non-Newtonian>
                            shiftposfsp1.z += (heavyphase ? 0 : massrhop*frz); //<vs_non-Newtonian>
                            shiftposfsp1.w -= (heavyphase ? 0 : massrhop*(drx*frx + dry*fry + drz*frz)); //<vs_non-Newtonian>
                        }
                        
                        //===== Viscosity ===== 
                        if (compute) {
                            const float dot = drx*dvx + dry*dvy + drz*dvz;
                            const float dot_rr2 = dot / (rr2 + Eta2);
                            visc = max(dot_rr2, visc);
                            //<vs_non-Newtonian>
                            const float visco_NN = PhaseCte[pp2].visco;
                            if (lamsps==VISCO_Artificial) {//-Artificial viscosity.                         
                                if (dot<0) {
                                    const float amubar = H*dot_rr2;  //amubar=CTE.h*dot/(rr2+CTE.eta2);
                                    const float robar = (rhopp1 + velrhop2.w)*0.5f;
                                    const float pi_visc = (-visco_NN*cbar*amubar / robar)*massp2;
                                    acep1.x -= pi_visc*frx; acep1.y -= pi_visc*fry; acep1.z -= pi_visc*frz;
                                }
                            }
                            else if (lamsps==VISCO_LaminarSPS || lamsps == VISCO_ConstEq) {
                                {
                                    //vel gradients
                                    if (boundp2) { //this applies no slip on tensor                                     
                                        dvx = 2.f*velp1.x; dvy = 2.f*velp1.y; dvz = 2.f*velp1.z;  //fomraly I should use the moving BC vel as ug=2ub-uf
                                    }                                   
                                    tmatrix3f dvelp1; float div_vel;
                                    GetVelocityGradients_FDA(rr2, drx, dry, drz, dvx, dvy, dvz, dvelp1, div_vel);

                                    //Strain rate tensor 
                                    tmatrix3f D_tensor; float div_D_tensor; float D_tensor_magn;
                                    float I_D, II_D; float J1_D, J2_D;
                                    GetStrainRateTensor(dvelp1, div_vel, I_D, II_D, J1_D, J2_D, div_D_tensor, D_tensor_magn, D_tensor);
									
                                    //Effective viscosity                                   
                                    float m_NN = PhaseCte[pp2].m_NN; float n_NN = PhaseCte[pp2].n_NN; float reg_NN = PhaseCte[pp2].reg_NN; float reg_strain = PhaseCte[pp2].reg_strain; float tau_yield = PhaseCte[pp2].tau_yield; // Robin: added reg_NN and reg_strain
                                    GetEta_Effective(pp1,tau_yield, D_tensor_magn, visco_NN, m_NN, n_NN, reg_NN, reg_strain, visco_etap1);                                  

                                    if (lamsps == VISCO_LaminarSPS) {//-Laminar contribution.
                                        //Morris Operator
                                        const float temp = 2.0f*(visco_etap1) / ((rr2 + Eta2)*velrhop2.w);
                                        const float vtemp = massp2*temp*(drx*frx + dry*fry + drz*frz);
                                        acep1.x += vtemp*dvx; acep1.y += vtemp*dvy; acep1.z += vtemp*dvz;
                                    }
                                    else if (lamsps == VISCO_ConstEq) {
                                        //stress tensor tau
                                        tmatrix3f tau_tensor; float tau_tensor_magn;
                                        float I_t, II_t; float J1_t, J2_t;
                                        GetStressTensor(D_tensor, visco_etap1, I_t, II_t, J1_t, J2_t, tau_tensor_magn, tau_tensor);

                                        //viscous forces                
                                        float taux = (tau_tensor.a11*frx + tau_tensor.a12*fry + tau_tensor.a13*frz) / (velrhop2.w); 
                                        float tauy = (tau_tensor.a21*frx + tau_tensor.a22*fry + tau_tensor.a23*frz) / (velrhop2.w);
                                        float tauz = (tau_tensor.a31*frx + tau_tensor.a32*fry + tau_tensor.a33*frz) / (velrhop2.w);
                                        acep1.x += taux*massp2; acep1.y += tauy*massp2; acep1.z += tauz*massp2;         
                                    }
                                }
                                //-SPS turbulence model.
                                //-SPS turbulence model is disabled in beta version                             
                            }
                        }
                        rsym = (rsymp1 && !rsym && float(posp1.y - dry) <= Dosh); //<vs_syymmetry>
                        if (rsym)p2--;                                       //<vs_syymmetry>
                    }
                    else rsym = false;                                      //<vs_syymmetry>
                }
            }
        }
        //-Sum results together. | Almacena resultados.
        if (shift || arp1 || acep1.x || acep1.y || acep1.z || visc) {
            if (tdensity != DDT_None) {
                if (delta)delta[p1] = (delta[p1] == FLT_MAX || deltap1 == FLT_MAX ? FLT_MAX : delta[p1] + deltap1);
                else if (deltap1 != FLT_MAX)arp1 += deltap1;
            }
            ar[p1] += arp1;
            ace[p1] = ace[p1] + acep1;
            const int th = omp_get_thread_num();
            if (visc>viscth[th*OMP_STRIDE])viscth[th*OMP_STRIDE] = visc;
			//const float viou = visco_etap1 / rhopp1;
            if (visco_etap1 > viscetath[th*OMP_STRIDE])viscetath[th*OMP_STRIDE] = visco_etap1;
            if (visco_etap1)visco_eta[p1] = visco_etap1;
            if (shift)shiftposfs[p1] = shiftposfsp1;
            //auxnn[p1] = visco_etap1; //to be used if an auxilary is needed for debug or otherwise.
        }
    }
    //-Keep max value in viscdt. | Guarda en viscdt el valor maximo.
    for (int th = 0; th < OmpThreads; th++) { 
        if (viscdt < viscth[th*OMP_STRIDE])viscdt = viscth[th*OMP_STRIDE]; 
        if (viscetadt < viscetath[th*OMP_STRIDE])viscetadt = viscetath[th*OMP_STRIDE];
    }
}

//==============================================================================
/// Interaction of Fluid-Fluid/Bound & Bound-Fluid (forces and DEM) for NN using the FDA approach
//==============================================================================
template<TpKernel tker, TpFtMode ftmode, TpVisco lamsps, TpDensity tdensity, bool shift>
  void JSphCpu::Interaction_ForcesCpuT_NN_FDA(const stinterparmsc &t,StInterResultc &res)const
{
  const tint4 nc = TInt4(int(t.ncells.x), int(t.ncells.y), int(t.ncells.z), int(t.ncells.x*t.ncells.y));
  const tint3 cellzero = TInt3(t.cellmin.x, t.cellmin.y, t.cellmin.z);
  const unsigned cellfluid = nc.w*nc.z + 1;
  const int hdiv = (CellMode == CELLMODE_H ? 2 : 1);
  float viscdt=res.viscdt;
  float viscetadt=res.viscetadt;
  if (t.npf) {
    //-Interaction Fluid-Fluid.
    InteractionForcesFluid_NN_FDA_All<tker, ftmode, lamsps, tdensity, shift>(t.npf, t.npb, nc, hdiv, cellfluid, Visco, t.visco_eta, t.begincell, cellzero, t.dcell, t.spstau, t.spsgradvel, t.pos, t.velrhop, t.code, t.idp, t.press, viscdt, viscetadt, t.ar, t.ace, t.delta, t.shiftmode, t.shiftposfs);
    //-Interaction Fluid-Bound.
    InteractionForcesFluid_NN_FDA_All<tker, ftmode, lamsps, tdensity, shift>(t.npf, t.npb, nc, hdiv, 0, Visco*ViscoBoundFactor, t.visco_eta, t.begincell, cellzero, t.dcell, t.spstau, t.spsgradvel, t.pos, t.velrhop, t.code, t.idp, t.press, viscdt, viscetadt, t.ar, t.ace, t.delta, t.shiftmode, t.shiftposfs);

    //-Interaction of DEM Floating-Bound & Floating-Floating. //(DEM)
    if (UseDEM)InteractionForcesDEM(CaseNfloat, nc, hdiv, cellfluid, t.begincell, cellzero, t.dcell, FtRidp, DemData, t.pos, t.velrhop, t.code, t.idp, viscdt, t.ace);
  }
  if (t.npbok) {
    //-Interaction Bound-Fluid.
    InteractionForcesBound_NN_FDA<tker, ftmode>(t.npbok, 0, nc, hdiv, cellfluid, t.begincell, cellzero, t.dcell, t.pos, t.velrhop, t.code, t.idp, viscdt, t.ar);
  }
  res.viscdt=viscdt;
  res.viscetadt=viscetadt;
}

//end_of_file
