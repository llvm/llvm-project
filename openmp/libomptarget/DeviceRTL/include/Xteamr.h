//===---------------- Xteamr.h - OpenMP interface ----------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2022 Advanced Micro Devices, All rights reserved.
// Notified per clause 4(b) of the license.
//
//===----------------------------------------------------------------------===//
//
// DeviceRTL Header file: Xteamr.h
//     External __kmpc headers for cross team reduction functions defined
//     in DeviceRTL/src/Xteamr.cpp. Clang generates a call to one of these
//     functions when it encounter a reduction. The specific function depends
//     on datatype, warpsize, and number of waves in the teamsize.  The number
//     of waves must be a power of 2 and the total number of threads must
//     be greater than or equal to the number of teams.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_XTEAMR_H
#define OMPTARGET_DEVICERTL_XTEAMR_H
#include "Types.h"

#define _CD double _Complex
#define _CF float _Complex
#define _UI unsigned int
#define _UL unsigned long
#define _INLINE_ATTR_ __attribute__((flatten, always_inline))
#define _RF_LDS volatile __attribute__((address_space(3)))

extern "C" {
/// External cross team reduction (xteamr) helper functions
///
/// The template for name of xteamr helper function is:
/// __kmpc_xteamr_<dtype>_<waves>x<WSZ> where
///    <dtype> is letter(s) representing data type, e.g. d=double
///    <waves> number of waves in thread block
///    <WSZ>   warp size, 32 or 64
/// So <waves> x <WSZ> is the number of threads per team.
/// Example: __kmpc_xteamr_d_16x64 is the reduction helper function
///          for all reductions with data type double using 1024 threads
///          per team.
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
void _INLINE_ATTR_ __kmpc_xteamr_d_16x64(
    double v, double *r_ptr, double *tvs, uint32_t *td,
    void (*_rf)(double *, double),
    void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_f_16x64(
    float v, float *r_ptr, float *tvs, uint32_t *td,
    void (*_rf)(float *, float),
    void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cd_16x64(
    _CD v, _CD *r_ptr, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
    void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cf_16x64(
    _CF v, _CF *r_ptr, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
    void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_i_16x64(
    int v, int *r_ptr, int *tvs, uint32_t *td, void (*_rf)(int *, int),
    void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ui_16x64(
    _UI v, _UI *r_ptr, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
    void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_l_16x64(
    long v, long *r_ptr, long *tvs, uint32_t *td, void (*_rf)(long *, long),
    void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ul_16x64(
    _UL v, _UL *r_ptr, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
    void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_d_32x32(
    double v, double *r_ptr, double *tvs, uint32_t *td,
    void (*_rf)(double *, double),
    void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_f_32x32(
    float v, float *r_ptr, float *tvs, uint32_t *td,
    void (*_rf)(float *, float),
    void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cd_32x32(
    _CD v, _CD *r_ptr, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
    void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cf_32x32(
    _CF v, _CF *r_ptr, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
    void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_i_32x32(
    int v, int *r_ptr, int *tvs, uint32_t *td, void (*_rf)(int *, int),
    void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ui_32x32(
    _UI v, _UI *r_ptr, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
    void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_l_32x32(
    long v, long *r_ptr, long *tvs, uint32_t *td, void (*_rf)(long *, long),
    void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ul_32x32(
    _UL v, _UL *r_ptr, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
    void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_d_8x64(
    double v, double *r_ptr, double *tvs, uint32_t *td,
    void (*_rf)(double *, double),
    void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_f_8x64(
    float v, float *r_ptr, float *tvs, uint32_t *td,
    void (*_rf)(float *, float),
    void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cd_8x64(
    _CD v, _CD *r_ptr, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
    void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cf_8x64(
    _CF v, _CF *r_ptr, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
    void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_i_8x64(
    int v, int *r_ptr, int *tvs, uint32_t *td, void (*_rf)(int *, int),
    void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ui_8x64(
    _UI v, _UI *r_ptr, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
    void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_l_8x64(
    long v, long *r_ptr, long *tvs, uint32_t *td, void (*_rf)(long *, long),
    void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ul_8x64(
    _UL v, _UL *r_ptr, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
    void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_d_16x32(
    double v, double *r_ptr, double *tvs, uint32_t *td,
    void (*_rf)(double *, double),
    void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_f_16x32(
    float v, float *r_ptr, float *tvs, uint32_t *td,
    void (*_rf)(float *, float),
    void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cd_16x32(
    _CD v, _CD *r_ptr, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
    void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cf_16x32(
    _CF v, _CF *r_ptr, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
    void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_i_16x32(
    int v, int *r_ptr, int *tvs, uint32_t *td, void (*_rf)(int *, int),
    void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ui_16x32(
    _UI v, _UI *r_ptr, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
    void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_l_16x32(
    long v, long *r_ptr, long *tvs, uint32_t *td, void (*_rf)(long *, long),
    void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ul_16x32(
    _UL v, _UL *r_ptr, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
    void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_d_4x64(
    double v, double *r_ptr, double *tvs, uint32_t *td,
    void (*_rf)(double *, double),
    void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_f_4x64(
    float v, float *r_ptr, float *tvs, uint32_t *td,
    void (*_rf)(float *, float),
    void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cd_4x64(
    _CD v, _CD *r_ptr, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
    void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cf_4x64(
    _CF v, _CF *r_ptr, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
    void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_i_4x64(
    int v, int *r_ptr, int *tvs, uint32_t *td, void (*_rf)(int *, int),
    void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ui_4x64(
    _UI v, _UI *r_ptr, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
    void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_l_4x64(
    long v, long *r_ptr, long *tvs, uint32_t *td, void (*_rf)(long *, long),
    void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ul_4x64(
    _UL v, _UL *r_ptr, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
    void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_d_8x32(
    double v, double *r_ptr, double *tvs, uint32_t *td,
    void (*_rf)(double *, double),
    void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_f_8x32(
    float v, float *r_ptr, float *tvs, uint32_t *td,
    void (*_rf)(float *, float),
    void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cd_8x32(
    _CD v, _CD *r_ptr, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
    void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_cf_8x32(
    _CF v, _CF *r_ptr, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
    void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_i_8x32(
    int v, int *r_ptr, int *tvs, uint32_t *td, void (*_rf)(int *, int),
    void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ui_8x32(
    _UI v, _UI *r_ptr, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
    void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_l_8x32(
    long v, long *r_ptr, long *tvs, uint32_t *td, void (*_rf)(long *, long),
    void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
    const uint64_t k, const uint32_t numteams);
/// Cross team reduction (xteamr) helper function, see documentation above.
void _INLINE_ATTR_ __kmpc_xteamr_ul_8x32(
    _UL v, _UL *r_ptr, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
    void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
    const uint64_t k, const uint32_t numteams);

/// Built-in pair reduction functions used as a function pointer in arguments
/// to cross team reduction (xteamr) helper functions defined above.
///
/// The template for the name of built-in pair reduction functions is
/// __kmpc_rfun_<fct>_<dtype> where
///    <fct>   is function name (e.g.sum,min,max)
///    <dtype> is letter(s) representing data type, e.g. d=double
///
/// All built-in pair reduction functions are defined in Xteamr.cpp.
/// Clang/flang code generation for C, C++, and FORTRAN use function pointers
/// to built-in pair reduction functions when generating a call to xteamr
/// helper functions.
///
/// \param Pointer to first TLS value where result is placed
/// \param The 2nd TLS value used in the pair reduction function
void __kmpc_rfun_sum_d(double *val, double otherval);

/// LDS Built-in pair reduction functions used as a function pointer in
/// arguments to cross team reduction (xteamr) helper functions.
/// The LDS pair reduction function only differs from the pair reduction
/// function in that the arguments use LDS storage.
///
/// The template for the name of LDS built-in pair reduction functions is
/// __kmpc_rfun_<fct>_lds_<dtype> where
///    <fct>   is function name (e.g.sum,min,max)
///    <dtype> is letter(s) representing data type, e.g. d=double
///
/// All built-in pair reduction functions are defined in Xteamr.cpp.
/// Clang/flang code generation for C, C++, and FORTRAN use function pointers
/// to built-in pair reduction functions when generating a call to xteamr
/// helper functions.
///
/// \param Pointer to the 1st value in LDS storage where result is placed.
/// \param Pointer to the 2nd value in LDS storage.
void __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval);

/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_f(float *val, float otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_f(_RF_LDS float *val, _RF_LDS float *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_cd(_CD *val, _CD otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_cd(_RF_LDS _CD *val, _RF_LDS _CD *otherval);
/// Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_cf(_CF *val, _CF otherval);
/// LDS Built-in pair reduction function, see documentation above.
void __kmpc_rfun_sum_lds_cf(_RF_LDS _CF *val, _RF_LDS _CF *otherval);
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
#undef _UI
#undef _UL
#undef _INLINE_ATTR_
#undef _RF_LDS

#endif // of ifndef OMPTARGET_DEVICERTL_XTEAMR_H
