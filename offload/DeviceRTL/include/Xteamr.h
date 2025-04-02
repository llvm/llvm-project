//===---------------- Xteamr.h - OpenMP interface ----------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DeviceRTL Header file: Xteamr.h
//     External __kmpc headers for cross team reduction functions defined
//     in DeviceRTL/src/Xteamr.cpp. Clang generates a call to one of these
//     functions when it encounter a reduction. The specific function depends
//     on datatype and warpsize. The number of waves must be a power of 2.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_XTEAMR_H
#define OMPTARGET_DEVICERTL_XTEAMR_H
#include "DeviceTypes.h"
#include "Synchronization.h"

#define _CD double _Complex
#define _CF float _Complex
#define _US unsigned short
#define _UI unsigned int
#define _UL unsigned long
#define _INLINE_ATTR_ __attribute__((flatten, always_inline))
#define _RF_LDS volatile __gpu_local

extern "C" {
/// External cross team reduction (xteamr) helper functions
///
/// The template for name of xteamr helper function is:
/// __kmpc_xteamr_<dtype>_<max_waves>x<WSZ> where
///    <dtype> is letter(s) representing data type, e.g. d=double.
///    <max_waves> maximum number of waves in thread block.
///    <WSZ>   warp size, 32 or 64.
///    IS_FAST There is an optional template boolean type (defaulting to false)
///    that indicates if an atomic add should be used instead of the last
///    reduction round. This applies to only sum reduction currently.
/// Example: __kmpc_xteamr_d_16x64 is the reduction helper function
///          for all reductions with data type double for warp size 64.
/// All xteamr helper functions are defined in Xteamr.cpp. They each call the
/// internal templated function _xteam_reduction also defined in Xteamr.cpp.
/// Clang/flang code generation for C, C++, and FORTRAN instantiate a call to
/// a helper function for each reduction used in an OpenMP target region.
///
/// \param  Input thread local reduction value
/// \param  Pointer to result value
/// \param  Global array of team values for this reduction instance
/// \param  Pointer to atomic counter of completed teams
/// \param  Function pointer to reduction function (sum,min,max)
/// \param  Function pointer to reduction function on LDS memory
/// \param  Reduction null value
/// \param  Outer loop iteration value, 0 to numteams*numthreads
/// \param  Number of teams

/// External intra-team reduction (iteamr) helper functions
///
/// The name template for intra-team helper functions is
/// __kmpc_iteamr_<dtype>_<max_waves>x<WSZ> where
///    <dtype> is letter(s) representing data type, e.g. d=double.
///    <max_waves> maximum number of waves in thread block.
///    <WSZ>   warp size, 32 or 64.
/// All iteamr helper functions are defined in Xteamr.cpp. They each call the
/// internal templated function _iteam_reduction also defined in Xteamr.cpp.
///
/// \param  Input thread local reduction value
/// \param  Pointer to result value
/// \param  Function pointer to reduction function (sum,min,max)
/// \param  Function pointer to reduction function on LDS memory
/// \param  Reduction null value
/// \param  Outer loop iteration value, 0 to numthreads
///
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_d_16x64(
    double v, double *r_ptr, double *tvs, uint32_t *td,
    void (*_rf)(double *, double),
    void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_d_16x64_fast_sum(
    double v, double *r_ptr, double *tvs, uint32_t *td,
    void (*_rf)(double *, double),
    void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_d_16x64(double v, double *r_ptr,
                                         void (*_rf)(double *, double),
                                         void (*_rf_lds)(_RF_LDS double *,
                                                         _RF_LDS double *),
                                         const double rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_f_16x64(
    float v, float *r_ptr, float *tvs, uint32_t *td,
    void (*_rf)(float *, float),
    void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_f_16x64_fast_sum(
    float v, float *r_ptr, float *tvs, uint32_t *td,
    void (*_rf)(float *, float),
    void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_f_16x64(float v, float *r_ptr,
                                         void (*_rf)(float *, float),
                                         void (*_rf_lds)(_RF_LDS float *,
                                                         _RF_LDS float *),
                                         const float rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_h_16x64(
    _Float16 v, _Float16 *r_ptr, _Float16 *tvs, uint32_t *td,
    void (*_rf)(_Float16 *, _Float16),
    void (*_rf_lds)(_RF_LDS _Float16 *, _RF_LDS _Float16 *), const _Float16 rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_h_16x64_fast_sum(
    _Float16 v, _Float16 *r_ptr, _Float16 *tvs, uint32_t *td,
    void (*_rf)(_Float16 *, _Float16),
    void (*_rf_lds)(_RF_LDS _Float16 *, _RF_LDS _Float16 *), const _Float16 rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_h_16x64(_Float16 v, _Float16 *r_ptr,
                                         void (*_rf)(_Float16 *, _Float16),
                                         void (*_rf_lds)(_RF_LDS _Float16 *,
                                                         _RF_LDS _Float16 *),
                                         const _Float16 rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_bf_16x64(
    __bf16 v, __bf16 *r_ptr, __bf16 *tvs, uint32_t *td,
    void (*_rf)(__bf16 *, __bf16),
    void (*_rf_lds)(_RF_LDS __bf16 *, _RF_LDS __bf16 *), const __bf16 rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_bf_16x64_fast_sum(
    __bf16 v, __bf16 *r_ptr, __bf16 *tvs, uint32_t *td,
    void (*_rf)(__bf16 *, __bf16),
    void (*_rf_lds)(_RF_LDS __bf16 *, _RF_LDS __bf16 *), const __bf16 rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_bf_16x64(__bf16 v, __bf16 *r_ptr,
                                          void (*_rf)(__bf16 *, __bf16),
                                          void (*_rf_lds)(_RF_LDS __bf16 *,
                                                          _RF_LDS __bf16 *),
                                          const __bf16 rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cd_16x64(
    _CD v, _CD *r_ptr, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
    void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_cd_16x64_fast_sum(
    _CD v, _CD *r_ptr, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
    void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_cd_16x64(_CD v, _CD *r_ptr,
                                          void (*_rf)(_CD *, _CD),
                                          void (*_rf_lds)(_RF_LDS _CD *,
                                                          _RF_LDS _CD *),
                                          const _CD rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cf_16x64(
    _CF v, _CF *r_ptr, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
    void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_cf_16x64_fast_sum(
    _CF v, _CF *r_ptr, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
    void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_cf_16x64(_CF v, _CF *r_ptr,
                                          void (*_rf)(_CF *, _CF),
                                          void (*_rf_lds)(_RF_LDS _CF *,
                                                          _RF_LDS _CF *),
                                          const _CF rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_s_16x64(
    short v, short *r_ptr, short *tvs, uint32_t *td,
    void (*_rf)(short *, short),
    void (*_rf_lds)(_RF_LDS short *, _RF_LDS short *), const short rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_s_16x64_fast_sum(
    short v, short *r_ptr, short *tvs, uint32_t *td,
    void (*_rf)(short *, short),
    void (*_rf_lds)(_RF_LDS short *, _RF_LDS short *), const short rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_s_16x64(short v, short *r_ptr,
                                         void (*_rf)(short *, short),
                                         void (*_rf_lds)(_RF_LDS short *,
                                                         _RF_LDS short *),
                                         const short rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_us_16x64(
    _US v, _US *r_ptr, _US *tvs, uint32_t *td, void (*_rf)(_US *, _US),
    void (*_rf_lds)(_RF_LDS _US *, _RF_LDS _US *), const _US rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_us_16x64_fast_sum(
    _US v, _US *r_ptr, _US *tvs, uint32_t *td, void (*_rf)(_US *, _US),
    void (*_rf_lds)(_RF_LDS _US *, _RF_LDS _US *), const _US rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_us_16x64(_US v, _US *r_ptr,
                                          void (*_rf)(_US *, _US),
                                          void (*_rf_lds)(_RF_LDS _US *,
                                                          _RF_LDS _US *),
                                          const _US rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_i_16x64(
    int v, int *r_ptr, int *tvs, uint32_t *td, void (*_rf)(int *, int),
    void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_i_16x64_fast_sum(
    int v, int *r_ptr, int *tvs, uint32_t *td, void (*_rf)(int *, int),
    void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_i_16x64(int v, int *r_ptr,
                                         void (*_rf)(int *, int),
                                         void (*_rf_lds)(_RF_LDS int *,
                                                         _RF_LDS int *),
                                         const int rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ui_16x64(
    _UI v, _UI *r_ptr, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
    void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_ui_16x64_fast_sum(
    _UI v, _UI *r_ptr, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
    void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_ui_16x64(_UI v, _UI *r_ptr,
                                          void (*_rf)(_UI *, _UI),
                                          void (*_rf_lds)(_RF_LDS _UI *,
                                                          _RF_LDS _UI *),
                                          const _UI rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_l_16x64(
    long v, long *r_ptr, long *tvs, uint32_t *td, void (*_rf)(long *, long),
    void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_l_16x64_fast_sum(
    long v, long *r_ptr, long *tvs, uint32_t *td, void (*_rf)(long *, long),
    void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_l_16x64(long v, long *r_ptr,
                                         void (*_rf)(long *, long),
                                         void (*_rf_lds)(_RF_LDS long *,
                                                         _RF_LDS long *),
                                         const long rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ul_16x64(
    _UL v, _UL *r_ptr, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
    void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_ul_16x64_fast_sum(
    _UL v, _UL *r_ptr, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
    void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_ul_16x64(_UL v, _UL *r_ptr,
                                          void (*_rf)(_UL *, _UL),
                                          void (*_rf_lds)(_RF_LDS _UL *,
                                                          _RF_LDS _UL *),
                                          const _UL rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_d_32x32(
    double v, double *r_ptr, double *tvs, uint32_t *td,
    void (*_rf)(double *, double),
    void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_d_32x32_fast_sum(
    double v, double *r_ptr, double *tvs, uint32_t *td,
    void (*_rf)(double *, double),
    void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_d_32x32(double v, double *r_ptr,
                                         void (*_rf)(double *, double),
                                         void (*_rf_lds)(_RF_LDS double *,
                                                         _RF_LDS double *),
                                         const double rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_f_32x32(
    float v, float *r_ptr, float *tvs, uint32_t *td,
    void (*_rf)(float *, float),
    void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_f_32x32_fast_sum(
    float v, float *r_ptr, float *tvs, uint32_t *td,
    void (*_rf)(float *, float),
    void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_f_32x32(float v, float *r_ptr,
                                         void (*_rf)(float *, float),
                                         void (*_rf_lds)(_RF_LDS float *,
                                                         _RF_LDS float *),
                                         const float rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_h_32x32(
    _Float16 v, _Float16 *r_ptr, _Float16 *tvs, uint32_t *td,
    void (*_rf)(_Float16 *, _Float16),
    void (*_rf_lds)(_RF_LDS _Float16 *, _RF_LDS _Float16 *), const _Float16 rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_h_32x32_fast_sum(
    _Float16 v, _Float16 *r_ptr, _Float16 *tvs, uint32_t *td,
    void (*_rf)(_Float16 *, _Float16),
    void (*_rf_lds)(_RF_LDS _Float16 *, _RF_LDS _Float16 *), const _Float16 rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_h_32x32(_Float16 v, _Float16 *r_ptr,
                                         void (*_rf)(_Float16 *, _Float16),
                                         void (*_rf_lds)(_RF_LDS _Float16 *,
                                                         _RF_LDS _Float16 *),
                                         const _Float16 rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_bf_32x32(
    __bf16 v, __bf16 *r_ptr, __bf16 *tvs, uint32_t *td,
    void (*_rf)(__bf16 *, __bf16),
    void (*_rf_lds)(_RF_LDS __bf16 *, _RF_LDS __bf16 *), const __bf16 rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_bf_32x32_fast_sum(
    __bf16 v, __bf16 *r_ptr, __bf16 *tvs, uint32_t *td,
    void (*_rf)(__bf16 *, __bf16),
    void (*_rf_lds)(_RF_LDS __bf16 *, _RF_LDS __bf16 *), const __bf16 rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_bf_32x32(__bf16 v, __bf16 *r_ptr,
                                          void (*_rf)(__bf16 *, __bf16),
                                          void (*_rf_lds)(_RF_LDS __bf16 *,
                                                          _RF_LDS __bf16 *),
                                          const __bf16 rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cd_32x32(
    _CD v, _CD *r_ptr, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
    void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_cd_32x32_fast_sum(
    _CD v, _CD *r_ptr, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
    void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_cd_32x32(_CD v, _CD *r_ptr,
                                          void (*_rf)(_CD *, _CD),
                                          void (*_rf_lds)(_RF_LDS _CD *,
                                                          _RF_LDS _CD *),
                                          const _CD rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cf_32x32(
    _CF v, _CF *r_ptr, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
    void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_cf_32x32_fast_sum(
    _CF v, _CF *r_ptr, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
    void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_cf_32x32(_CF v, _CF *r_ptr,
                                          void (*_rf)(_CF *, _CF),
                                          void (*_rf_lds)(_RF_LDS _CF *,
                                                          _RF_LDS _CF *),
                                          const _CF rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_s_32x32(
    short v, short *r_ptr, short *tvs, uint32_t *td,
    void (*_rf)(short *, short),
    void (*_rf_lds)(_RF_LDS short *, _RF_LDS short *), const short rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_s_32x32_fast_sum(
    short v, short *r_ptr, short *tvs, uint32_t *td,
    void (*_rf)(short *, short),
    void (*_rf_lds)(_RF_LDS short *, _RF_LDS short *), const short rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_s_32x32(short v, short *r_ptr,
                                         void (*_rf)(short *, short),
                                         void (*_rf_lds)(_RF_LDS short *,
                                                         _RF_LDS short *),
                                         const short rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_us_32x32(
    _US v, _US *r_ptr, _US *tvs, uint32_t *td, void (*_rf)(_US *, _US),
    void (*_rf_lds)(_RF_LDS _US *, _RF_LDS _US *), const _US rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_us_32x32_fast_sum(
    _US v, _US *r_ptr, _US *tvs, uint32_t *td, void (*_rf)(_US *, _US),
    void (*_rf_lds)(_RF_LDS _US *, _RF_LDS _US *), const _US rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_us_32x32(_US v, _US *r_ptr,
                                          void (*_rf)(_US *, _US),
                                          void (*_rf_lds)(_RF_LDS _US *,
                                                          _RF_LDS _US *),
                                          const _US rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_i_32x32(
    int v, int *r_ptr, int *tvs, uint32_t *td, void (*_rf)(int *, int),
    void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_i_32x32_fast_sum(
    int v, int *r_ptr, int *tvs, uint32_t *td, void (*_rf)(int *, int),
    void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_i_32x32(int v, int *r_ptr,
                                         void (*_rf)(int *, int),
                                         void (*_rf_lds)(_RF_LDS int *,
                                                         _RF_LDS int *),
                                         const int rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ui_32x32(
    _UI v, _UI *r_ptr, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
    void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_ui_32x32_fast_sum(
    _UI v, _UI *r_ptr, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
    void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_ui_32x32(_UI v, _UI *r_ptr,
                                          void (*_rf)(_UI *, _UI),
                                          void (*_rf_lds)(_RF_LDS _UI *,
                                                          _RF_LDS _UI *),
                                          const _UI rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_l_32x32(
    long v, long *r_ptr, long *tvs, uint32_t *td, void (*_rf)(long *, long),
    void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_l_32x32_fast_sum(
    long v, long *r_ptr, long *tvs, uint32_t *td, void (*_rf)(long *, long),
    void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_l_32x32(long v, long *r_ptr,
                                         void (*_rf)(long *, long),
                                         void (*_rf_lds)(_RF_LDS long *,
                                                         _RF_LDS long *),
                                         const long rnv, const uint64_t k);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ul_32x32(
    _UL v, _UL *r_ptr, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
    void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Fast Cross team sum reduction (xteamr) helper function, see documentation
/// above.
void _INLINE_ATTR_ __kmpc_xteamr_ul_32x32_fast_sum(
    _UL v, _UL *r_ptr, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
    void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
    const uint64_t k, const uint32_t numteams,
    ompx::atomic::MemScopeTy Scope = ompx::atomic::system);
/// Intra-team reduction (iteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_iteamr_ul_32x32(_UL v, _UL *r_ptr,
                                          void (*_rf)(_UL *, _UL),
                                          void (*_rf_lds)(_RF_LDS _UL *,
                                                          _RF_LDS _UL *),
                                          const _UL rnv, const uint64_t k);

/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_d(double *val, double otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_f(float *val, float otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_f(_RF_LDS float *val, _RF_LDS float *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_h(_Float16 *val, _Float16 otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_h(_RF_LDS _Float16 *val, _RF_LDS _Float16 *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_bf(__bf16 *val, __bf16 otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_bf(_RF_LDS __bf16 *val, _RF_LDS __bf16 *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_cd(_CD *val, _CD otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_cd(_RF_LDS _CD *val, _RF_LDS _CD *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_cf(_CF *val, _CF otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_cf(_RF_LDS _CF *val, _RF_LDS _CF *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_s(short *val, short otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_s(_RF_LDS short *val, _RF_LDS short *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_us(_US *val, _US otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_us(_RF_LDS _US *val, _RF_LDS _US *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_i(int *val, int otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_i(_RF_LDS int *val, _RF_LDS int *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_ui(_UI *val, _UI otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_l(long *val, long otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_l(_RF_LDS long *val, _RF_LDS long *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_ul(_UL *val, _UL otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_d(double *val, double otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_lds_d(_RF_LDS double *val, _RF_LDS double *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_f(float *val, float otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_lds_f(_RF_LDS float *val, _RF_LDS float *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_h(_Float16 *val, _Float16 otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_lds_h(_RF_LDS _Float16 *val, _RF_LDS _Float16 *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_bf(__bf16 *val, __bf16 otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_lds_bf(_RF_LDS __bf16 *val, _RF_LDS __bf16 *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_s(short *val, short otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_lds_s(_RF_LDS short *val, _RF_LDS short *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_us(_US *val, _US otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_lds_us(_RF_LDS _US *val, _RF_LDS _US *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_i(int *val, int otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_lds_i(_RF_LDS int *val, _RF_LDS int *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_ui(_UI *val, _UI otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_l(long *val, long otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_lds_l(_RF_LDS long *val, _RF_LDS long *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_ul(_UL *val, _UL otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_max_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_d(double *val, double otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_lds_d(_RF_LDS double *val, _RF_LDS double *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_f(float *val, float otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_lds_f(_RF_LDS float *val, _RF_LDS float *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_h(_Float16 *val, _Float16 otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_lds_h(_RF_LDS _Float16 *val, _RF_LDS _Float16 *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_bf(__bf16 *val, __bf16 otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_lds_bf(_RF_LDS __bf16 *val, _RF_LDS __bf16 *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_s(short *val, short otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_lds_s(_RF_LDS short *val, _RF_LDS short *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_us(_US *val, _US otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_lds_us(_RF_LDS _US *val, _RF_LDS _US *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_i(int *val, int otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_lds_i(_RF_LDS int *val, _RF_LDS int *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_ui(_UI *val, _UI otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_l(long *val, long otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_lds_l(_RF_LDS long *val, _RF_LDS long *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_ul(_UL *val, _UL otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_min_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval);
} // end extern C

#undef _CD
#undef _CF
#undef _US
#undef _UI
#undef _UL
#undef _INLINE_ATTR_
#undef _RF_LDS

#endif // of ifndef OMPTARGET_DEVICERTL_XTEAMR_H
