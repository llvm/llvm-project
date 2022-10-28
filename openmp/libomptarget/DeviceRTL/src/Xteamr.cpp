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

#include "Xteamr.h"
#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

#define __XTEAM_SHARED_LDS volatile __attribute__((address_space(3)))

using namespace _OMP;

#pragma omp begin declare target device_type(nohost)

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
#pragma omp end declare variant

#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

int xteamr_shfl_xor_int(int var, const int lane_mask, const uint32_t width) {
  return __nvvm_shfl_sync_bfly_i32(0xFFFFFFFF, var, lane_mask, 0x1f);
}
double xteamr_shfl_xor_d(double var, int laneMask, const uint32_t width) {
  unsigned lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
  hi = xteamr_shfl_xor_int(hi, laneMask, width);
  lo = xteamr_shfl_xor_int(lo, laneMask, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
  return var;
}
#pragma omp end declare variant

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

/// Templated internal function used by all extern typed reductions
///
/// \param  Template typename parameter T
/// \param  Template parameter for number of waves, must be power of two
/// \param  Template parameter for warp size, 32 o 64
///
/// \param  Input thread local (TLS) value for warp shfl reduce
/// \param  Pointer to result value, also used in final reduction
/// \param  Global array of team values for this reduction only
/// \param  Pointer to atomically accessed teams done counter
/// \param  Function pointer to TLS pair reduction function
/// \param  Function pointer to LDS pair reduction function
/// \param  Reduction null value, used for partial waves
/// \param  The iteration value from 0 to (NumTeams*_NUM_THREADS)-1
/// \param  The number of teams participating in reduction

template <typename T, const int32_t _NW, const int32_t _WSZ>
__attribute__((flatten, always_inline)) void _xteam_reduction(
    T val, T *r_ptr, T *team_vals, uint32_t *teams_done_ptr,
    void (*_rf)(T *, T),
    void (*_rf_lds)(__XTEAM_SHARED_LDS T *, __XTEAM_SHARED_LDS T *),
    const T rnv, const uint64_t k, const uint32_t NumTeams) {

  // More efficient to derive these constants than get from mapped API
  constexpr uint32_t _NT = _NW * _WSZ;
  const uint32_t omp_thread_num = k % _NT;
  const uint32_t omp_team_num = k / _NT;
  const uint32_t wave_num = omp_thread_num / _WSZ;
  const uint32_t lane_num = omp_thread_num % _WSZ;

  static __XTEAM_SHARED_LDS T xwave_lds[_NW + 1];

// Cuda may restrict max threads, so clear unused wave values
#ifdef __NVPTX__
  if (_NW == 32) {
    if (omp_thread_num == 0) {
      for (uint32_t i = (omp_get_num_threads() / 32); i < _NW; i++)
        xwave_lds[i] = rnv;
    }
  }
#endif

  // Binary reduce each wave, then copy to xwave_lds[wave_num]
  for (unsigned int offset = _WSZ / 2; offset > 0; offset >>= 1)
    (*_rf)(&val, xteamr_shfl_xor<T, _WSZ>(val, offset));
  if (lane_num == 0)
    xwave_lds[wave_num] = val;

  // Binary reduce all wave values into wave_lds[0]
  _OMP::synchronize::threadsAligned();
  for (unsigned int offset = _NW / 2; offset > 0; offset >>= 1) {
    if (omp_thread_num < offset)
      (*_rf_lds)(&(xwave_lds[omp_thread_num]),
                 &(xwave_lds[omp_thread_num + offset]));
  }
  // No sync needed here from last reduction in LDS loop
  // because we only need xwave_lds[0] correct on thread 0.

  // Save the teams reduced value in team_vals global array
  // and atomically increment teams_done counter.
  static __XTEAM_SHARED_LDS uint32_t td;
  if (omp_thread_num == 0) {
    team_vals[omp_team_num] = xwave_lds[0];
    td = atomic::inc(teams_done_ptr, NumTeams - 1u, atomic::seq_cst);
  }

  // This sync needed so all threads from last team see the shared volatile
  // value td (teams done counter) so they know they are in the last team.
  _OMP::synchronize::threadsAligned();

  // If td counter reaches NumTeams-1, this is the last team.
  // The team number of this last team is nondeterministic.
  if (td == (NumTeams - 1u)) {

    // All threads from last completed team enter here.
    // All other teams exit the helper function.

    // To use TLS shfl reduce, copy team values to TLS val.
    // NumTeams must be <= _NUM_THREADS here.
    val = (omp_thread_num < NumTeams) ? team_vals[omp_thread_num] : rnv;

    // Need sync here to prepare for TLS shfl reduce.
    _OMP::synchronize::threadsAligned();

    // Reduce each wave into xwave_lds[wave_num]
    for (unsigned int offset = _WSZ / 2; offset > 0; offset >>= 1)
      (*_rf)(&val, xteamr_shfl_xor<T, _WSZ>(val, offset));
    if (lane_num == 0)
      xwave_lds[wave_num] = val;

    // To get final result, we know wave_lds[0] is done
    // Sync needed here to ensure wave_lds[i!=0] are correct.
    _OMP::synchronize::threadsAligned();

    // Typically only a few usable waves even for large GPUs.
    // No gain parallelizing these last few reductions.
    // So do reduction on thread 0 into lane 0's LDS val.
    if (omp_thread_num == 0) {
      unsigned int usableWaves = ((NumTeams - 1) / _WSZ) + 1;
      // Reduce with the original result value.
      xwave_lds[usableWaves] = *r_ptr;
      for (unsigned int kk = 1; kk <= usableWaves; kk++)
        (*_rf_lds)(&xwave_lds[0], &xwave_lds[kk]);

      *r_ptr = xwave_lds[0];
    }

    // This sync needed to prevent warps in last team from starting
    // if there was another reduction.
    _OMP::synchronize::threadsAligned();
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
                      const double iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<double, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                   numteams);
}
_EXT_ATTR
__kmpc_xteamr_f_16x64(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                      void (*_rf)(float *, float),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS float *,
                                      __XTEAM_SHARED_LDS float *),
                      const float iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<float, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                  numteams);
}

_EXT_ATTR
__kmpc_xteamr_cd_16x64(double _Complex v, double _Complex *r_ptr,
                       double _Complex *tvals, uint32_t *td_ptr,
                       void (*_rf)(double _Complex *, double _Complex),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS double _Complex *,
                                       __XTEAM_SHARED_LDS double _Complex *),
                       double _Complex iv, const uint64_t k,
                       const uint32_t numteams) {
  _xteam_reduction<double _Complex, 16, 64>(v, r_ptr, tvals, td_ptr, _rf,
                                            _rf_lds, iv, k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_cf_16x64(float _Complex v, float _Complex *r_ptr,
                       float _Complex *tvals, uint32_t *td_ptr,
                       void (*_rf)(float _Complex *, float _Complex),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS float _Complex *,
                                       __XTEAM_SHARED_LDS float _Complex *),
                       float _Complex iv, const uint64_t k,
                       const uint32_t numteams) {
  _xteam_reduction<float _Complex, 16, 64>(v, r_ptr, tvals, td_ptr, _rf,
                                           _rf_lds, iv, k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_i_16x64(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                      void (*_rf)(int *, int),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS int *,
                                      __XTEAM_SHARED_LDS int *),
                      const int iv, const uint64_t k, const uint32_t numteams) {
  _xteam_reduction<int, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                numteams);
}
_EXT_ATTR
__kmpc_xteamr_ui_16x64(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                       uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS uint32_t *,
                                       __XTEAM_SHARED_LDS uint32_t *),
                       const uint32_t iv, const uint64_t k,
                       const uint32_t numteams) {
  _xteam_reduction<uint32_t, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv,
                                     k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_l_16x64(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                      void (*_rf)(long *, long),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS long *,
                                      __XTEAM_SHARED_LDS long *),
                      const long iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<long, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                 numteams);
}
_EXT_ATTR
__kmpc_xteamr_ul_16x64(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                       uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS uint64_t *,
                                       __XTEAM_SHARED_LDS uint64_t *),
                       const uint64_t iv, const uint64_t k,
                       const uint32_t numteams) {
  _xteam_reduction<uint64_t, 16, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv,
                                     k, numteams);
}

_EXT_ATTR
__kmpc_xteamr_d_8x64(double v, double *r_ptr, double *tvals, uint32_t *td_ptr,
                     void (*_rf)(double *, double),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS double *,
                                     __XTEAM_SHARED_LDS double *),
                     const double iv, const uint64_t k,
                     const uint32_t numteams) {
  _xteam_reduction<double, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                  numteams);
}
_EXT_ATTR
__kmpc_xteamr_f_8x64(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                     void (*_rf)(float *, float),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS float *,
                                     __XTEAM_SHARED_LDS float *),
                     const float iv, const uint64_t k,
                     const uint32_t numteams) {
  _xteam_reduction<float, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                 numteams);
}
_EXT_ATTR
__kmpc_xteamr_cd_8x64(double _Complex v, double _Complex *r_ptr,
                      double _Complex *tvals, uint32_t *td_ptr,
                      void (*_rf)(double _Complex *, double _Complex),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS double _Complex *,
                                      __XTEAM_SHARED_LDS double _Complex *),
                      const double _Complex iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<double _Complex, 8, 64>(v, r_ptr, tvals, td_ptr, _rf,
                                           _rf_lds, iv, k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_cf_8x64(float _Complex v, float _Complex *r_ptr,
                      float _Complex *tvals, uint32_t *td_ptr,
                      void (*_rf)(float _Complex *, float _Complex),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS float _Complex *,
                                      __XTEAM_SHARED_LDS float _Complex *),
                      const float _Complex iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<float _Complex, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds,
                                          iv, k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_i_8x64(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                     void (*_rf)(int *, int),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS int *,
                                     __XTEAM_SHARED_LDS int *),
                     const int iv, const uint64_t k, const uint32_t numteams) {
  _xteam_reduction<int, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                               numteams);
}
_EXT_ATTR
__kmpc_xteamr_ui_8x64(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                      uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS uint32_t *,
                                      __XTEAM_SHARED_LDS uint32_t *),
                      const uint32_t iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<uint32_t, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv,
                                    k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_l_8x64(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                     void (*_rf)(long *, long),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS long *,
                                     __XTEAM_SHARED_LDS long *),
                     const long iv, const uint64_t k, const uint32_t numteams) {
  _xteam_reduction<long, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                numteams);
}
_EXT_ATTR
__kmpc_xteamr_ul_8x64(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                      uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS uint64_t *,
                                      __XTEAM_SHARED_LDS uint64_t *),
                      const uint64_t iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<uint64_t, 8, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv,
                                    k, numteams);
}

_EXT_ATTR
__kmpc_xteamr_d_4x64(double v, double *r_ptr, double *tvals, uint32_t *td_ptr,
                     void (*_rf)(double *, double),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS double *,
                                     __XTEAM_SHARED_LDS double *),
                     const double iv, const uint64_t k,
                     const uint32_t numteams) {
  _xteam_reduction<double, 4, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                  numteams);
}
_EXT_ATTR
__kmpc_xteamr_f_4x64(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                     void (*_rf)(float *, float),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS float *,
                                     __XTEAM_SHARED_LDS float *),
                     const float iv, const uint64_t k,
                     const uint32_t numteams) {
  _xteam_reduction<float, 4, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                 numteams);
}
_EXT_ATTR
__kmpc_xteamr_cd_4x64(double _Complex v, double _Complex *r_ptr,
                      double _Complex *tvals, uint32_t *td_ptr,
                      void (*_rf)(double _Complex *, double _Complex),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS double _Complex *,
                                      __XTEAM_SHARED_LDS double _Complex *),
                      const double _Complex iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<double _Complex, 4, 64>(v, r_ptr, tvals, td_ptr, _rf,
                                           _rf_lds, iv, k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_cf_4x64(float _Complex v, float _Complex *r_ptr,
                      float _Complex *tvals, uint32_t *td_ptr,
                      void (*_rf)(float _Complex *, float _Complex),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS float _Complex *,
                                      __XTEAM_SHARED_LDS float _Complex *),
                      const float _Complex iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<float _Complex, 4, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds,
                                          iv, k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_i_4x64(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                     void (*_rf)(int *, int),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS int *,
                                     __XTEAM_SHARED_LDS int *),
                     const int iv, const uint64_t k, const uint32_t numteams) {
  _xteam_reduction<int, 4, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                               numteams);
}
_EXT_ATTR
__kmpc_xteamr_ui_4x64(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                      uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS uint32_t *,
                                      __XTEAM_SHARED_LDS uint32_t *),
                      const uint32_t iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<uint32_t, 4, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv,
                                    k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_l_4x64(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                     void (*_rf)(long *, long),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS long *,
                                     __XTEAM_SHARED_LDS long *),
                     const long iv, const uint64_t k, const uint32_t numteams) {
  _xteam_reduction<long, 4, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                numteams);
}
_EXT_ATTR
__kmpc_xteamr_ul_4x64(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                      uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS uint64_t *,
                                      __XTEAM_SHARED_LDS uint64_t *),
                      const uint64_t iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<uint64_t, 4, 64>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv,
                                    k, numteams);
}

_EXT_ATTR
__kmpc_xteamr_d_32x32(double v, double *r_ptr, double *tvals, uint32_t *td_ptr,
                      void (*_rf)(double *, double),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS double *,
                                      __XTEAM_SHARED_LDS double *),
                      const double iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<double, 32, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                   numteams);
}
_EXT_ATTR
__kmpc_xteamr_f_32x32(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                      void (*_rf)(float *, float),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS float *,
                                      __XTEAM_SHARED_LDS float *),
                      const float iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<float, 32, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                  numteams);
}
_EXT_ATTR
__kmpc_xteamr_cd_32x32(double _Complex v, double _Complex *r_ptr,
                       double _Complex *tvals, uint32_t *td_ptr,
                       void (*_rf)(double _Complex *, double _Complex),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS double _Complex *,
                                       __XTEAM_SHARED_LDS double _Complex *),
                       const double _Complex iv, const uint64_t k,
                       const uint32_t numteams) {
  _xteam_reduction<double _Complex, 32, 32>(v, r_ptr, tvals, td_ptr, _rf,
                                            _rf_lds, iv, k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_cf_32x32(float _Complex v, float _Complex *r_ptr,
                       float _Complex *tvals, uint32_t *td_ptr,
                       void (*_rf)(float _Complex *, float _Complex),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS float _Complex *,
                                       __XTEAM_SHARED_LDS float _Complex *),
                       const float _Complex iv, const uint64_t k,
                       const uint32_t numteams) {
  _xteam_reduction<float _Complex, 32, 32>(v, r_ptr, tvals, td_ptr, _rf,
                                           _rf_lds, iv, k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_i_32x32(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                      void (*_rf)(int *, int),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS int *,
                                      __XTEAM_SHARED_LDS int *),
                      const int iv, const uint64_t k, const uint32_t numteams) {
  _xteam_reduction<int, 32, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                numteams);
}
_EXT_ATTR
__kmpc_xteamr_ui_32x32(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                       uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS uint32_t *,
                                       __XTEAM_SHARED_LDS uint32_t *),
                       const uint32_t iv, const uint64_t k,
                       const uint32_t numteams) {
  _xteam_reduction<uint32_t, 32, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv,
                                     k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_l_32x32(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                      void (*_rf)(long *, long),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS long *,
                                      __XTEAM_SHARED_LDS long *),
                      const long iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<long, 32, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                 numteams);
}
_EXT_ATTR
__kmpc_xteamr_ul_32x32(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                       uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS uint64_t *,
                                       __XTEAM_SHARED_LDS uint64_t *),
                       const uint64_t iv, const uint64_t k,
                       const uint32_t numteams) {
  _xteam_reduction<uint64_t, 32, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv,
                                     k, numteams);
}

_EXT_ATTR
__kmpc_xteamr_d_16x32(double v, double *r_ptr, double *tvals, uint32_t *td_ptr,
                      void (*_rf)(double *, double),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS double *,
                                      __XTEAM_SHARED_LDS double *),
                      const double iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<double, 16, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                   numteams);
}
_EXT_ATTR
__kmpc_xteamr_f_16x32(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                      void (*_rf)(float *, float),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS float *,
                                      __XTEAM_SHARED_LDS float *),
                      const float iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<float, 16, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                  numteams);
}
_EXT_ATTR
__kmpc_xteamr_cd_16x32(double _Complex v, double _Complex *r_ptr,
                       double _Complex *tvals, uint32_t *td_ptr,
                       void (*_rf)(double _Complex *, double _Complex),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS double _Complex *,
                                       __XTEAM_SHARED_LDS double _Complex *),
                       const double _Complex iv, const uint64_t k,
                       const uint32_t numteams) {
  _xteam_reduction<double _Complex, 16, 32>(v, r_ptr, tvals, td_ptr, _rf,
                                            _rf_lds, iv, k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_cf_16x32(float _Complex v, float _Complex *r_ptr,
                       float _Complex *tvals, uint32_t *td_ptr,
                       void (*_rf)(float _Complex *, float _Complex),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS float _Complex *,
                                       __XTEAM_SHARED_LDS float _Complex *),
                       const float _Complex iv, const uint64_t k,
                       const uint32_t numteams) {
  _xteam_reduction<float _Complex, 16, 32>(v, r_ptr, tvals, td_ptr, _rf,
                                           _rf_lds, iv, k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_i_16x32(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                      void (*_rf)(int *, int),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS int *,
                                      __XTEAM_SHARED_LDS int *),
                      const int iv, const uint64_t k, const uint32_t numteams) {
  _xteam_reduction<int, 16, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                numteams);
}
_EXT_ATTR
__kmpc_xteamr_ui_16x32(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                       uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS uint32_t *,
                                       __XTEAM_SHARED_LDS uint32_t *),
                       const uint32_t iv, const uint64_t k,
                       const uint32_t numteams) {
  _xteam_reduction<uint32_t, 16, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv,
                                     k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_l_16x32(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                      void (*_rf)(long *, long),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS long *,
                                      __XTEAM_SHARED_LDS long *),
                      const long iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<long, 16, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                 numteams);
}
_EXT_ATTR
__kmpc_xteamr_ul_16x32(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                       uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                       void (*_rf_lds)(__XTEAM_SHARED_LDS uint64_t *,
                                       __XTEAM_SHARED_LDS uint64_t *),
                       const uint64_t iv, const uint64_t k,
                       const uint32_t numteams) {
  _xteam_reduction<uint64_t, 16, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv,
                                     k, numteams);
}

_EXT_ATTR
__kmpc_xteamr_d_8x32(double v, double *r_ptr, double *tvals, uint32_t *td_ptr,
                     void (*_rf)(double *, double),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS double *,
                                     __XTEAM_SHARED_LDS double *),
                     const double iv, const uint64_t k,
                     const uint32_t numteams) {
  _xteam_reduction<double, 8, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                  numteams);
}
_EXT_ATTR
__kmpc_xteamr_f_8x32(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                     void (*_rf)(float *, float),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS float *,
                                     __XTEAM_SHARED_LDS float *),
                     const float iv, const uint64_t k,
                     const uint32_t numteams) {
  _xteam_reduction<float, 8, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                 numteams);
}
_EXT_ATTR
__kmpc_xteamr_cd_8x32(double _Complex v, double _Complex *r_ptr,
                      double _Complex *tvals, uint32_t *td_ptr,
                      void (*_rf)(double _Complex *, double _Complex),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS double _Complex *,
                                      __XTEAM_SHARED_LDS double _Complex *),
                      const double _Complex iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<double _Complex, 8, 32>(v, r_ptr, tvals, td_ptr, _rf,
                                           _rf_lds, iv, k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_cf_8x32(float _Complex v, float _Complex *r_ptr,
                      float _Complex *tvals, uint32_t *td_ptr,
                      void (*_rf)(float _Complex *, float _Complex),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS float _Complex *,
                                      __XTEAM_SHARED_LDS float _Complex *),
                      const float _Complex iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<float _Complex, 8, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds,
                                          iv, k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_i_8x32(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                     void (*_rf)(int *, int),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS int *,
                                     __XTEAM_SHARED_LDS int *),
                     const int iv, const uint64_t k, const uint32_t numteams) {
  _xteam_reduction<int, 8, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                               numteams);
}
_EXT_ATTR
__kmpc_xteamr_ui_8x32(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                      uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS uint32_t *,
                                      __XTEAM_SHARED_LDS uint32_t *),
                      const uint32_t iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<uint32_t, 8, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv,
                                    k, numteams);
}
_EXT_ATTR
__kmpc_xteamr_l_8x32(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                     void (*_rf)(long *, long),
                     void (*_rf_lds)(__XTEAM_SHARED_LDS long *,
                                     __XTEAM_SHARED_LDS long *),
                     const long iv, const uint64_t k, const uint32_t numteams) {
  _xteam_reduction<long, 8, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv, k,
                                numteams);
}
_EXT_ATTR
__kmpc_xteamr_ul_8x32(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                      uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                      void (*_rf_lds)(__XTEAM_SHARED_LDS uint64_t *,
                                      __XTEAM_SHARED_LDS uint64_t *),
                      const uint64_t iv, const uint64_t k,
                      const uint32_t numteams) {
  _xteam_reduction<uint64_t, 8, 32>(v, r_ptr, tvals, td_ptr, _rf, _rf_lds, iv,
                                    k, numteams);
}

// Built-in pair reduction functions used as function pointers for
// cross team reduction functions.

#define _RF_LDS volatile __attribute__((address_space(3)))

_EXT_ATTR __kmpc_rfun_sum_d(double *val, double otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_f(float *val, float otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_cd(double _Complex *val, double _Complex otherval) {
  *val += otherval;
}
_EXT_ATTR __kmpc_rfun_sum_lds_cd(_RF_LDS double _Complex *val,
                                 _RF_LDS double _Complex *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_cf(float _Complex *val, float _Complex otherval) {
  *val += otherval;
}
_EXT_ATTR __kmpc_rfun_sum_lds_cf(_RF_LDS float _Complex *val,
                                 _RF_LDS float _Complex *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_i(int *val, int otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_ui(unsigned int *val, unsigned int otherval) {
  *val += otherval;
}
_EXT_ATTR __kmpc_rfun_sum_lds_ui(_RF_LDS unsigned int *val,
                                 _RF_LDS unsigned int *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_l(long *val, long otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_ul(unsigned long *val, unsigned long otherval) {
  *val += otherval;
}
_EXT_ATTR __kmpc_rfun_sum_lds_ul(_RF_LDS unsigned long *val,
                                 _RF_LDS unsigned long *otherval) {
  *val += *otherval;
}

_EXT_ATTR __kmpc_rfun_min_d(double *val, double otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_f(float *val, float otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_i(int *val, int otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_ui(unsigned int *val, unsigned int otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_ui(_RF_LDS unsigned int *val,
                                 _RF_LDS unsigned int *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_l(long *val, long otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_ul(unsigned long *val, unsigned long otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_ul(_RF_LDS unsigned long *val,
                                 _RF_LDS unsigned long *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}

_EXT_ATTR __kmpc_rfun_max_d(double *val, double otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_f(float *val, float otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_i(int *val, int otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_ui(unsigned int *val, unsigned int otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_ui(_RF_LDS unsigned int *val,
                                 _RF_LDS unsigned int *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_l(long *val, long otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_ul(unsigned long *val, unsigned long otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_ul(_RF_LDS unsigned long *val,
                                 _RF_LDS unsigned long *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}

#undef _EXT_ATTR
#undef _RF_LDS

#pragma omp end declare target
