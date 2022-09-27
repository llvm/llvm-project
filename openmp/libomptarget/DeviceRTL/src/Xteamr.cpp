//===---- Xteamr.cpp - OpenMP cross team helper functions ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains helper functions for cross team reductions
//
//===----------------------------------------------------------------------===//

#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

#define __XTEAM_SHARED_LDS volatile __attribute__((address_space(3)))

using namespace _OMP;

#pragma omp declare target
// Headers for specialized shfl_xor
double xteamr_shfl_xor_d(double var, const int lane_mask, const uint32_t width);
float xteamr_shfl_xor_f(float var, const int lane_mask, const uint32_t width);
int xteamr_shfl_xor_int(int var, const int lane_mask, const uint32_t width);
double _Complex xteamr_shfl_xor_cd(double _Complex var, const int lane_mask,
                                   const uint32_t width);
float _Complex xteamr_shfl_xor_cf(float _Complex var, const int lane_mask,
                                  const uint32_t width);

// Define the arch (amdgcn vs nvptx) variants of shfl

#pragma omp begin declare variant match(device = {arch(amdgcn)})

int xteamr_shfl_xor_int(int var, const int lane_mask, const uint32_t width) {
  int self = mapping::getThreadIdInWarp(); // __lane_id();
  int index = self ^ lane_mask;
  index = index >= ((self + width) & ~(width - 1)) ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}
float xteamr_shfl_xor_f(float var, const int lane_mask, const uint32_t width) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = xteamr_shfl_xor_int(tmp.i, lane_mask, width);
  return tmp.f;
}
double xteamr_shfl_xor_d(double var, const int lane_mask,
                         const uint32_t width) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");
  static_assert(sizeof(double) == sizeof(uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = xteamr_shfl_xor_int(tmp[0], lane_mask, width);
  tmp[1] = xteamr_shfl_xor_int(tmp[1], lane_mask, width);

  uint64_t tmp0 =
      (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
  double tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}

double _Complex xteamr_shfl_xor_cd(double _Complex var, const int lane_mask,
                                   const uint32_t width) {
  __real__(var) = xteamr_shfl_xor_d(__real__(var), lane_mask, width);
  __imag__(var) = xteamr_shfl_xor_d(__imag__(var), lane_mask, width);
  return var;
}
float _Complex xteamr_shfl_xor_cf(float _Complex var, const int lane_mask,
                                  const uint32_t width) {
  __real__(var) = xteamr_shfl_xor_f(__real__(var), lane_mask, width);
  __imag__(var) = xteamr_shfl_xor_f(__imag__(var), lane_mask, width);
  return var;
}
#pragma omp end declare variant

#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

int xteamr_shfl_xor_int(int var, const int lane_mask, const uint32_t width) {
  int c = ((32 - width) << 8) | 0x1f;
  return __nvvm_shfl_sync_bfly_i32(0xFFFFFFFF, var, lane_mask, c);
}
float xteamr_shfl_xor_f(float var, const int lane_mask, const uint32_t width) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = xteamr_shfl_xor_int(tmp.i, lane_mask, width);
  return tmp.f;
}
double xteamr_shfl_xor_d(double var, int laneMask, const uint32_t width) {
  unsigned lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
  hi = xteamr_shfl_xor_int(hi, laneMask, width);
  lo = xteamr_shfl_xor_int(lo, laneMask, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
  return var;
}
double _Complex xteamr_shfl_xor_cd(double _Complex var, const int lane_mask,
                                   const uint32_t width) {
  __real__(var) = xteamr_shfl_xor_d(__real__(var), lane_mask, width);
  __imag__(var) = xteamr_shfl_xor_d(__imag__(var), lane_mask, width);
  return var;
}
float _Complex xteamr_shfl_xor_cf(float _Complex var, const int lane_mask,
                                  const uint32_t width) {
  __real__(var) = xteamr_shfl_xor_f(__real__(var), lane_mask, width);
  __imag__(var) = xteamr_shfl_xor_f(__imag__(var), lane_mask, width);
  return var;
}
#pragma omp end declare variant
// } // end impl namespace

// tag dispatching of type specific shfl_xor, get_low, and get_high
struct _d_tag {};
struct _f_tag {};
struct _cd_tag {};
struct _cf_tag {};
struct _i_tag {};
struct _ui_tag {};
struct _l_tag {};
struct _ul_tag {};
template <typename T> struct __dispatch_tag;
template <> struct __dispatch_tag<double> {
  typedef _d_tag type;
};
template <> struct __dispatch_tag<float> {
  typedef _f_tag type;
};
template <> struct __dispatch_tag<double _Complex> {
  typedef _cd_tag type;
};
template <> struct __dispatch_tag<float _Complex> {
  typedef _cf_tag type;
};
template <> struct __dispatch_tag<int> {
  typedef _i_tag type;
};
template <> struct __dispatch_tag<unsigned int> {
  typedef _ui_tag type;
};
template <> struct __dispatch_tag<long> {
  typedef _l_tag type;
};
template <> struct __dispatch_tag<unsigned long> {
  typedef _ul_tag type;
};
template <const uint32_t _WSZ>
double xteamr_shfl_xor(_d_tag tag, double var, const int lane_mask) {
  return xteamr_shfl_xor_d(var, lane_mask, _WSZ);
}
template <const uint32_t _WSZ>
float xteamr_shfl_xor(_f_tag tag, float var, const int lane_mask) {
  return xteamr_shfl_xor_f(var, lane_mask, _WSZ);
}
template <const uint32_t _WSZ>
double _Complex xteamr_shfl_xor(_cd_tag tag, double _Complex var,
                                const int lane_mask) {
  return xteamr_shfl_xor_cd(var, lane_mask, _WSZ);
}
template <const uint32_t _WSZ>
float _Complex xteamr_shfl_xor(_cf_tag tag, float _Complex var,
                               const int lane_mask) {
  return xteamr_shfl_xor_cf(var, lane_mask, _WSZ);
}
template <const uint32_t _WSZ>
int xteamr_shfl_xor(_i_tag tag, int var, const int lane_mask) {
  return xteamr_shfl_xor_int(var, lane_mask, _WSZ);
}
template <const uint32_t _WSZ>
unsigned int xteamr_shfl_xor(_ui_tag tag, unsigned int var,
                             const int lane_mask) {
  return xteamr_shfl_xor_int(var, lane_mask, _WSZ);
}
template <const uint32_t _WSZ>
long xteamr_shfl_xor(_l_tag tag, long var, const int lane_mask) {
  return xteamr_shfl_xor_d(var, lane_mask, _WSZ);
}
template <const uint32_t _WSZ>
unsigned long xteamr_shfl_xor(_ul_tag tag, unsigned long var,
                              const int lane_mask) {
  return xteamr_shfl_xor_d(var, lane_mask, _WSZ);
}

template <typename T, const uint32_t _WSZ>
T xteamr_shfl_xor(T var, const int lane_mask) {
  typedef typename __dispatch_tag<T>::type tag;
  return xteamr_shfl_xor<_WSZ>(tag(), var, lane_mask);
}

template <typename T, const int32_t _NW, const int32_t _WSZ>
__attribute__((flatten, always_inline)) void _xteam_reduction(
    T val, T *r_ptr, T *team_vals, uint32_t *teams_done_ptr,
    void (*_rf)(T *, T),
    void (*_rf_lds)(__XTEAM_SHARED_LDS T *, __XTEAM_SHARED_LDS T *),
    const T inival) {

  constexpr int32_t _NT = _NW * _WSZ;
  const uint32_t omp_thread_num = mapping::getThreadIdInBlock();
  const uint32_t omp_team_num = mapping::getBlockId();
  const uint32_t wave_num = mapping::getWarpId();         // 0 15
  const uint32_t lane_num = mapping::getThreadIdInWarp(); //  0 63
  const uint32_t NumTeams = mapping::getNumberOfBlocks();
  static __XTEAM_SHARED_LDS T xwave_lds[_NW];
  static __XTEAM_SHARED_LDS bool __is_last_team;

  // Binary reduce each wave, then copy to xwave_lds[wave_num]
  for (unsigned int offset = _WSZ / 2; offset > 0; offset >>= 1)
    (*_rf)(&val, xteamr_shfl_xor<T, _WSZ>(val, offset));
  if (lane_num == 0)
    xwave_lds[wave_num] = val;

  // Binary reduce each wave's value into wave_lds[0] with lds memory.
  __kmpc_impl_syncthreads();
  for (unsigned int offset = _NW / 2; offset > 0; offset >>= 1) {
    if (omp_thread_num < offset)
      (*_rf_lds)(&(xwave_lds[omp_thread_num]),
                 &(xwave_lds[omp_thread_num + offset]));
  }
  __kmpc_impl_syncthreads();

  // Discover the last team to complete cross wave reduction
  // The team number of last team is nondeterministic.
  __is_last_team = false;
  if (omp_thread_num == 0) {
    team_vals[omp_team_num] = xwave_lds[0];
    uint32_t td = atomic::inc(teams_done_ptr, NumTeams - 1u, __ATOMIC_SEQ_CST);
    if (td == (NumTeams - 1u))
      __is_last_team = true;
  }

  // This sync needed, so all threads from last team know they are in the last
  // team.
  __kmpc_impl_syncthreads();

  if (__is_last_team) {
    // All threads from last completed team enter here.
    // All other teams exit.
    if (omp_thread_num < NumTeams)
      val = team_vals[omp_thread_num];
    else
      val = inival;

    // Reduce each wave into xwave_lds[wave_num]
    for (unsigned int offset = _WSZ / 2; offset > 0; offset >>= 1)
      (*_rf)(&val, xteamr_shfl_xor<T, _WSZ>(val, offset));
    if (lane_num == 0)
      xwave_lds[wave_num] = val;

    // Typically only 2 useable waves when <128 CUs. No gain to parallelizing
    // these last 2 reductions. So do these on thread 0 into lane 0's val.
    if (omp_thread_num == 0) {
      // FIXME: We know wave_lds[0] is done since wave_num==0 here. But do
      //        we need a sync here to ensure wave_lds[i!=0] is correct?
      unsigned int usableWaves = ((NumTeams - 1) / _WSZ) + 1;
      for (unsigned int kk = 1; kk < usableWaves; kk++)
        (*_rf_lds)(&xwave_lds[0], &xwave_lds[kk]);

      // Reduce with the original result value.
      xwave_lds[1] = *r_ptr;
      (*_rf_lds)(&xwave_lds[0], &xwave_lds[1]);
      *r_ptr = xwave_lds[0];
    }
  }
}

//  Calls to these __kmpc extern C functions are created in clang codegen
//  for FORTRAN, c, and C++. They may also be used for sumulation and testing.
//  The headers for these extern C functions are in ../include/Interface.h
//  The compiler builds the name based on data type,
//  number of waves in the team,and warpsize.
//
#define _EXT_ATTR extern "C" __attribute__((flatten, always_inline)) void
_EXT_ATTR
__kmpc_xteamr_d_16x64(double v, double *r_ptr, double *tvals, uint32_t *td_ptr,
                      void (*_rf)(double *, double),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS double *,
                                      __XTEAM_SHARED_LDS double *),
                      double iv) {
  _xteam_reduction<double, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_f_16x64(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                      void (*_rf)(float *, float),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS float *,
                                      __XTEAM_SHARED_LDS float *),
                      float iv) {
  _xteam_reduction<float, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}

_EXT_ATTR
__kmpc_xteamr_cd_16x64(double _Complex v, double _Complex *r_ptr, double _Complex *tvals, uint32_t *td_ptr,
                      void (*_rf)(double _Complex *, double _Complex),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS double _Complex *,
                                      __XTEAM_SHARED_LDS double _Complex *),
                      double _Complex iv) {
  _xteam_reduction<double _Complex, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_cf_16x64(float _Complex v, float _Complex *r_ptr, float _Complex *tvals, uint32_t *td_ptr,
                      void (*_rf)(float _Complex  *, float _Complex),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS float _Complex *,
                                      __XTEAM_SHARED_LDS float _Complex *),
                      float _Complex iv) {
  _xteam_reduction<float _Complex, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_i_16x64(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                      void (*_rf)(int *, int),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS int *,
                                      __XTEAM_SHARED_LDS int *),
                      int iv) {
  _xteam_reduction<int, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_ui_16x64(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                       uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS uint32_t *,
                                       __XTEAM_SHARED_LDS uint32_t *),
                       uint32_t iv) {
  _xteam_reduction<uint32_t, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_l_16x64(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                      void (*_rf)(long *, long),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS long *,
                                      __XTEAM_SHARED_LDS long *),
                      long iv) {
  _xteam_reduction<long, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_ul_16x64(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                       uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS uint64_t *,
                                       __XTEAM_SHARED_LDS uint64_t *),
                       uint64_t iv) {
  _xteam_reduction<uint64_t, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}

_EXT_ATTR
__kmpc_xteamr_d_8x64(double v, double *r_ptr, double *tvals, uint32_t *td_ptr,
                     void (*_rf)(double *, double),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS double *,
                                     __XTEAM_SHARED_LDS double *),
                     double iv) {
  _xteam_reduction<double, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_f_8x64(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                     void (*_rf)(float *, float),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS float *,
                                     __XTEAM_SHARED_LDS float *),
                     float iv) {
  _xteam_reduction<float, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_cd_8x64(double _Complex v, double _Complex *r_ptr, double _Complex *tvals, uint32_t *td_ptr,
                      void (*_rf)(double _Complex *, double _Complex),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS double _Complex *,
                                      __XTEAM_SHARED_LDS double _Complex *),
                      double _Complex iv) {
  _xteam_reduction<double _Complex, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_cf_8x64(float _Complex v, float _Complex *r_ptr, float _Complex *tvals, uint32_t *td_ptr,
                      void (*_rf)(float _Complex  *, float _Complex),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS float _Complex *,
                                      __XTEAM_SHARED_LDS float _Complex *),
                      float _Complex iv) {
  _xteam_reduction<float _Complex, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_i_8x64(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                     void (*_rf)(int *, int),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS int *,
                                     __XTEAM_SHARED_LDS int *),
                     int iv) {
  _xteam_reduction<int, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_ui_8x64(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                      uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS uint32_t *,
                                      __XTEAM_SHARED_LDS uint32_t *),
                      uint32_t iv) {
  _xteam_reduction<uint32_t, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_l_8x64(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                     void (*_rf)(long *, long),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS long *,
                                     __XTEAM_SHARED_LDS long *),
                     long iv) {
  _xteam_reduction<long, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_ul_8x64(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                      uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS uint64_t *,
                                      __XTEAM_SHARED_LDS uint64_t *),
                      uint64_t iv) {
  _xteam_reduction<uint64_t, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}

_EXT_ATTR
__kmpc_xteamr_d_4x64(double v, double *r_ptr, double *tvals, uint32_t *td_ptr,
                     void (*_rf)(double *, double),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS double *,
                                     __XTEAM_SHARED_LDS double *),
                     double iv) {
  _xteam_reduction<double, 4, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_f_4x64(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                     void (*_rf)(float *, float),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS float *,
                                     __XTEAM_SHARED_LDS float *),
                     float iv) {
  _xteam_reduction<float, 4, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_i_4x64(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                     void (*_rf)(int *, int),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS int *,
                                     __XTEAM_SHARED_LDS int *),
                     int iv) {
  _xteam_reduction<int, 4, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_ui_4x64(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                      uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS uint32_t *,
                                      __XTEAM_SHARED_LDS uint32_t *),
                      uint32_t iv) {
  _xteam_reduction<uint32_t, 4, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_l_4x64(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                     void (*_rf)(long *, long),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS long *,
                                     __XTEAM_SHARED_LDS long *),
                     long iv) {
  _xteam_reduction<long, 4, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_ul_4x64(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                      uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS uint64_t *,
                                      __XTEAM_SHARED_LDS uint64_t *),
                      uint64_t iv) {
  _xteam_reduction<uint64_t, 4, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}

_EXT_ATTR
__kmpc_xteamr_d_32x32(double v, double *r_ptr, double *tvals, uint32_t *td_ptr,
                      void (*_rf)(double *, double),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS double *,
                                      __XTEAM_SHARED_LDS double *),
                      double iv) {
  _xteam_reduction<double, 32, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_f_32x32(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                      void (*_rf)(float *, float),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS float *,
                                      __XTEAM_SHARED_LDS float *),
                      float iv) {
  _xteam_reduction<float, 32, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_i_32x32(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                      void (*_rf)(int *, int),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS int *,
                                      __XTEAM_SHARED_LDS int *),
                      int iv) {
  _xteam_reduction<int, 32, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_ui_32x32(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                       uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS uint32_t *,
                                       __XTEAM_SHARED_LDS uint32_t *),
                       uint32_t iv) {
  _xteam_reduction<uint32_t, 32, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_l_32x32(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                      void (*_rf)(long *, long),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS long *,
                                      __XTEAM_SHARED_LDS long *),
                      long iv) {
  _xteam_reduction<long, 32, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_ul_32x32(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                       uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS uint64_t *,
                                       __XTEAM_SHARED_LDS uint64_t *),
                       uint64_t iv) {
  _xteam_reduction<uint64_t, 32, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}

_EXT_ATTR
__kmpc_xteamr_d_16x32(double v, double *r_ptr, double *tvals, uint32_t *td_ptr,
                      void (*_rf)(double *, double),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS double *,
                                      __XTEAM_SHARED_LDS double *),
                      double iv) {
  _xteam_reduction<double, 16, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_f_16x32(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                      void (*_rf)(float *, float),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS float *,
                                      __XTEAM_SHARED_LDS float *),
                      float iv) {
  _xteam_reduction<float, 16, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_i_16x32(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                      void (*_rf)(int *, int),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS int *,
                                      __XTEAM_SHARED_LDS int *),
                      int iv) {
  _xteam_reduction<int, 16, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_ui_16x32(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                       uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS uint32_t *,
                                       __XTEAM_SHARED_LDS uint32_t *),
                       uint32_t iv) {
  _xteam_reduction<uint32_t, 16, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_l_16x32(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                      void (*_rf)(long *, long),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS long *,
                                      __XTEAM_SHARED_LDS long *),
                      long iv) {
  _xteam_reduction<long, 16, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_ul_16x32(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                       uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS uint64_t *,
                                       __XTEAM_SHARED_LDS uint64_t *),
                       uint64_t iv) {
  _xteam_reduction<uint64_t, 16, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}

_EXT_ATTR
__kmpc_xteamr_d_8x32(double v, double *r_ptr, double *tvals, uint32_t *td_ptr,
                     void (*_rf)(double *, double),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS double *,
                                     __XTEAM_SHARED_LDS double *),
                     double iv) {
  _xteam_reduction<double, 8, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_f_8x32(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                     void (*_rf)(float *, float),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS float *,
                                     __XTEAM_SHARED_LDS float *),
                     float iv) {
  _xteam_reduction<float, 8, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_i_8x32(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                     void (*_rf)(int *, int),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS int *,
                                     __XTEAM_SHARED_LDS int *),
                     int iv) {
  _xteam_reduction<int, 8, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_ui_8x32(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                      uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS uint32_t *,
                                      __XTEAM_SHARED_LDS uint32_t *),
                      uint32_t iv) {
  _xteam_reduction<uint32_t, 8, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_l_8x32(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                     void (*_rf)(long *, long),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS long *,
                                     __XTEAM_SHARED_LDS long *),
                     long iv) {
  _xteam_reduction<long, 8, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
_EXT_ATTR
__kmpc_xteamr_ul_8x32(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                      uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS uint64_t *,
                                      __XTEAM_SHARED_LDS uint64_t *),
                      uint64_t iv) {
  _xteam_reduction<uint64_t, 8, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv);
}
#undef _EXT_ATTR

#pragma omp end declare target
