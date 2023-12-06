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

using namespace  ompx::mapping;

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
  int self = ompx::mapping::getThreadIdInWarp(); // __lane_id();
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

/// Templated internal function used by extern intra-team reductions
///
/// \param  Template typename parameter T
/// \param  Template parameter for number of waves, must be power of two
/// \param  Template parameter for warp size, 32 o 64
///
/// \param  Input thread local (TLS) value for warp shfl reduce
/// \param  Pointer to result value, also used in final reduction
/// \param  Function pointer to TLS pair reduction function
/// \param  Function pointer to LDS pair reduction function
/// \param  Reduction null value, used for partial waves
/// \param  The iteration value from 0 to (NumTeams*_NUM_THREADS)-1
///
template <typename T, const int32_t _NW, const int32_t _WSZ>
__attribute__((flatten, always_inline)) void _iteam_reduction(
    T val, T *r_ptr, void (*_rf)(T *, T),
    void (*_rf_lds)(__XTEAM_SHARED_LDS T *, __XTEAM_SHARED_LDS T *),
    const T rnv, const uint64_t k) {
  constexpr uint32_t _NT = _NW * _WSZ;
  const uint32_t omp_thread_num = k % _NT;
  const uint32_t wave_num = omp_thread_num / _WSZ;
  const uint32_t lane_num = omp_thread_num % _WSZ;
  static __XTEAM_SHARED_LDS T xwave_lds[_NW];

  // Binary reduce each wave, then copy to xwave_lds[wave_num]
  for (unsigned int offset = _WSZ / 2; offset > 0; offset >>= 1)
    (*_rf)(&val, xteamr_shfl_xor<T, _WSZ>(val, offset));
  if (lane_num == 0)
    xwave_lds[wave_num] = val;

  // Binary reduce all wave values into wave_lds[0]
  ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);
  for (unsigned int offset = _NW / 2; offset > 0; offset >>= 1) {
    if (omp_thread_num < offset)
      (*_rf_lds)(&(xwave_lds[omp_thread_num]),
                 &(xwave_lds[omp_thread_num + offset]));
  }

  // We only need xwave_lds[0] correct on thread 0.
  if (omp_thread_num == 0)
    *r_ptr = xwave_lds[0];

  ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);
}

/// Templated internal function used by all extern typed reductions
///
/// \param  Template typename parameter T
/// \param  Template parameter for maximum number of waves in this kernel.
/// \param  Template parameter for warp size, 32 or 64
/// \param  Template parameter if an atomic add should be used instead of
///         the 1-team-reduction round. Applies to sum reduction currently.
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

template <typename T, const int32_t _MaxNumWaves, const int32_t _WSZ,
          const bool _IS_FAST = false>
__attribute__((flatten, always_inline)) void _xteam_reduction(
    T val, T *r_ptr, T *team_vals, uint32_t *teams_done_ptr,
    void (*_rf)(T *, T),
    void (*_rf_lds)(__XTEAM_SHARED_LDS T *, __XTEAM_SHARED_LDS T *),
    const T rnv, const uint64_t k, const uint32_t NumTeams) {

  // More efficient to derive these constants than get from mapped API

  // Must be a power of 2.
  const uint32_t block_size = ompx::mapping::getNumberOfThreadsInBlock();

  const uint32_t number_of_waves = (block_size - 1) / _WSZ + 1;
  const uint32_t omp_thread_num = k % block_size;
  const uint32_t omp_team_num = k / block_size;
  const uint32_t wave_num = omp_thread_num / _WSZ;
  const uint32_t lane_num = omp_thread_num % _WSZ;

  static __XTEAM_SHARED_LDS T xwave_lds[_MaxNumWaves];

// Cuda may restrict max threads, so clear unused wave values
#ifdef __NVPTX__
  if (number_of_waves == 32) {
    if (omp_thread_num == 0) {
      for (uint32_t i = (omp_get_num_threads() / 32); i < number_of_waves; i++)
        xwave_lds[i] = rnv;
    }
  }
#endif

  // Binary reduce each wave, then copy to xwave_lds[wave_num]
  const uint32_t start_offset = block_size < _WSZ ? block_size / 2 : _WSZ / 2;
  for (unsigned int offset = start_offset; offset > 0; offset >>= 1)
    (*_rf)(&val, xteamr_shfl_xor<T, _WSZ>(val, offset));
  if (lane_num == 0)
    xwave_lds[wave_num] = val;

  // Binary reduce all wave values into wave_lds[0]
  for (unsigned int offset = number_of_waves / 2; offset > 0; offset >>= 1) {
    ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);
    if (omp_thread_num < offset)
      (*_rf_lds)(&(xwave_lds[omp_thread_num]),
                 &(xwave_lds[omp_thread_num + offset]));
  }

  if (_IS_FAST) {
    if (omp_thread_num == 0)
      ompx::atomic::add(r_ptr, xwave_lds[0], ompx::atomic::seq_cst);
  } else {
    // No sync needed here from last reduction in LDS loop
    // because we only need xwave_lds[0] correct on thread 0.

    // Save the teams reduced value in team_vals global array
    // and atomically increment teams_done counter.
    static __XTEAM_SHARED_LDS uint32_t td;
    if (omp_thread_num == 0) {
      team_vals[omp_team_num] = xwave_lds[0];
      td = ompx::atomic::inc(teams_done_ptr, NumTeams - 1u,
                             ompx::atomic::seq_cst,
                             ompx::atomic::MemScopeTy::device);
    }

    // This sync needed so all threads from last team see the shared volatile
    // value td (teams done counter) so they know they are in the last team.
    ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);

    // If td counter reaches NumTeams-1, this is the last team.
    // The team number of this last team is nondeterministic.
    if (td == (NumTeams - 1u)) {

      // All threads from last completed team enter here.
      // All other teams exit the helper function.

      // To use TLS shfl reduce, copy team values to TLS val.
      val = (omp_thread_num < NumTeams) ? team_vals[omp_thread_num] : rnv;

      // Need sync here to prepare for TLS shfl reduce.
      ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);

      // Reduce each wave into xwave_lds[wave_num]
      for (unsigned int offset = start_offset; offset > 0; offset >>= 1)
        (*_rf)(&val, xteamr_shfl_xor<T, _WSZ>(val, offset));
      if (lane_num == 0)
        xwave_lds[wave_num] = val;

      // Binary reduce all wave values into wave_lds[0]
      for (unsigned int offset = number_of_waves / 2; offset > 0;
           offset >>= 1) {
        ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);
        if (omp_thread_num < offset)
          (*_rf_lds)(&(xwave_lds[omp_thread_num]),
                     &(xwave_lds[omp_thread_num + offset]));
      }

      if (omp_thread_num == 0) {
        // Reduce with the original result value.
        val = xwave_lds[0];
        (*_rf)(&val, *r_ptr);

        // If more teams than threads, do non-parallel reduction of extra
        // team_vals. This loop iterates only if NumTeams > block_size.
        for (unsigned int offset = block_size; offset < NumTeams; offset++)
          (*_rf)(&val, team_vals[offset]);

        // Write over the external result value.
        *r_ptr = val;
      }

      // This sync needed to prevent warps in last team from starting
      // if there was another reduction.
      ompx::synchronize::threadsAligned(ompx::atomic::relaxed);
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
#define _CD double _Complex
#define _CF float _Complex
#define _UI unsigned int
#define _UL unsigned long
#define _LDS volatile __attribute__((address_space(3)))
_EXT_ATTR
__kmpc_xteamr_d_16x64(double v, double *r_p, double *tvs, uint32_t *td,
                      void (*rf)(double *, double),
                      void (*rflds)(_LDS double *, _LDS double *),
                      const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<double, 16, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_d_16x64_fast_sum(double v, double *r_p, double *tvs, uint32_t *td,
                               void (*rf)(double *, double),
                               void (*rflds)(_LDS double *, _LDS double *),
                               const double rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<double, 16, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k,
                                         nt);
}
_EXT_ATTR
__kmpc_iteamr_d_16x64(double v, double *r_p, void (*rf)(double *, double),
                      void (*rflds)(_LDS double *, _LDS double *),
                      const double rnv, const uint64_t k) {
  _iteam_reduction<double, 16, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_f_16x64(float v, float *r_p, float *tvs, uint32_t *td,
                      void (*rf)(float *, float),
                      void (*rflds)(_LDS float *, _LDS float *),
                      const float rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<float, 16, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_f_16x64_fast_sum(float v, float *r_p, float *tvs, uint32_t *td,
                               void (*rf)(float *, float),
                               void (*rflds)(_LDS float *, _LDS float *),
                               const float rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<float, 16, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_f_16x64(float v, float *r_p, void (*rf)(float *, float),
                      void (*rflds)(_LDS float *, _LDS float *),
                      const float rnv, const uint64_t k) {
  _iteam_reduction<float, 16, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cd_16x64(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                       void (*rf)(_CD *, _CD),
                       void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CD, 16, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cd_16x64_fast_sum(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                                void (*rf)(_CD *, _CD),
                                void (*rflds)(_LDS _CD *, _LDS _CD *),
                                const _CD rnv, const uint64_t k,
                                const uint32_t nt) {
  _xteam_reduction<_CD, 16, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cd_16x64(_CD v, _CD *r_p, void (*rf)(_CD *, _CD),
                       void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                       const uint64_t k) {
  _iteam_reduction<_CD, 16, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cf_16x64(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                       void (*rf)(_CF *, _CF),
                       void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CF, 16, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cf_16x64_fast_sum(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                                void (*rf)(_CF *, _CF),
                                void (*rflds)(_LDS _CF *, _LDS _CF *),
                                const _CF rnv, const uint64_t k,
                                const uint32_t nt) {
  _xteam_reduction<_CF, 16, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cf_16x64(_CF v, _CF *r_p, void (*rf)(_CF *, _CF),
                       void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                       const uint64_t k) {
  _iteam_reduction<_CF, 16, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_i_16x64(int v, int *r_p, int *tvs, uint32_t *td,
                      void (*rf)(int *, int),
                      void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<int, 16, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_i_16x64_fast_sum(int v, int *r_p, int *tvs, uint32_t *td,
                               void (*rf)(int *, int),
                               void (*rflds)(_LDS int *, _LDS int *),
                               const int rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<int, 16, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_i_16x64(int v, int *r_p, void (*rf)(int *, int),
                      void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                      const uint64_t k) {
  _iteam_reduction<int, 16, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ui_16x64(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                       void (*rf)(_UI *, _UI),
                       void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UI, 16, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ui_16x64_fast_sum(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                                void (*rf)(_UI *, _UI),
                                void (*rflds)(_LDS _UI *, _LDS _UI *),
                                const _UI rnv, const uint64_t k,
                                const uint32_t nt) {
  _xteam_reduction<_UI, 16, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ui_16x64(_UI v, _UI *r_p, void (*rf)(_UI *, _UI),
                       void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                       const uint64_t k) {
  _iteam_reduction<_UI, 16, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_l_16x64(long v, long *r_p, long *tvs, uint32_t *td,
                      void (*rf)(long *, long),
                      void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<long, 16, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_l_16x64_fast_sum(long v, long *r_p, long *tvs, uint32_t *td,
                               void (*rf)(long *, long),
                               void (*rflds)(_LDS long *, _LDS long *),
                               const long rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<long, 16, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_l_16x64(long v, long *r_p, void (*rf)(long *, long),
                      void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                      const uint64_t k) {
  _iteam_reduction<long, 16, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ul_16x64(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                       void (*rf)(_UL *, _UL),
                       void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UL, 16, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ul_16x64_fast_sum(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                                void (*rf)(_UL *, _UL),
                                void (*rflds)(_LDS _UL *, _LDS _UL *),
                                const _UL rnv, const uint64_t k,
                                const uint32_t nt) {
  _xteam_reduction<_UL, 16, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ul_16x64(_UL v, _UL *r_p, void (*rf)(_UL *, _UL),
                       void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                       const uint64_t k) {
  _iteam_reduction<_UL, 16, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_d_8x64(double v, double *r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<double, 8, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_d_8x64_fast_sum(double v, double *r_p, double *tvs, uint32_t *td,
                              void (*rf)(double *, double),
                              void (*rflds)(_LDS double *, _LDS double *),
                              const double rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<double, 8, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_d_8x64(double v, double *r_p, void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k) {
  _iteam_reduction<double, 8, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_f_8x64(float v, float *r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<float, 8, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_f_8x64_fast_sum(float v, float *r_p, float *tvs, uint32_t *td,
                              void (*rf)(float *, float),
                              void (*rflds)(_LDS float *, _LDS float *),
                              const float rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<float, 8, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_f_8x64(float v, float *r_p, void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k) {
  _iteam_reduction<float, 8, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cd_8x64(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CD, 8, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cd_8x64_fast_sum(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                               void (*rf)(_CD *, _CD),
                               void (*rflds)(_LDS _CD *, _LDS _CD *),
                               const _CD rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CD, 8, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cd_8x64(_CD v, _CD *r_p, void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k) {
  _iteam_reduction<_CD, 8, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cf_8x64(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CF, 8, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cf_8x64_fast_sum(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                               void (*rf)(_CF *, _CF),
                               void (*rflds)(_LDS _CF *, _LDS _CF *),
                               const _CF rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CF, 8, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cf_8x64(_CF v, _CF *r_p, void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k) {
  _iteam_reduction<_CF, 8, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_i_8x64(int v, int *r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<int, 8, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_i_8x64_fast_sum(int v, int *r_p, int *tvs, uint32_t *td,
                              void (*rf)(int *, int),
                              void (*rflds)(_LDS int *, _LDS int *),
                              const int rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<int, 8, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_i_8x64(int v, int *r_p, void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k) {
  _iteam_reduction<int, 8, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ui_8x64(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UI, 8, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ui_8x64_fast_sum(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                               void (*rf)(_UI *, _UI),
                               void (*rflds)(_LDS _UI *, _LDS _UI *),
                               const _UI rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UI, 8, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ui_8x64(_UI v, _UI *r_p, void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k) {
  _iteam_reduction<_UI, 8, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_l_8x64(long v, long *r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<long, 8, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_l_8x64_fast_sum(long v, long *r_p, long *tvs, uint32_t *td,
                              void (*rf)(long *, long),
                              void (*rflds)(_LDS long *, _LDS long *),
                              const long rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<long, 8, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_l_8x64(long v, long *r_p, void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k) {
  _iteam_reduction<long, 8, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ul_8x64(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UL, 8, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ul_8x64_fast_sum(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                               void (*rf)(_UL *, _UL),
                               void (*rflds)(_LDS _UL *, _LDS _UL *),
                               const _UL rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UL, 8, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ul_8x64(_UL v, _UL *r_p, void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k) {
  _iteam_reduction<_UL, 8, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_d_4x64(double v, double *r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<double, 4, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_d_4x64_fast_sum(double v, double *r_p, double *tvs, uint32_t *td,
                              void (*rf)(double *, double),
                              void (*rflds)(_LDS double *, _LDS double *),
                              const double rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<double, 4, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_d_4x64(double v, double *r_p, void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k) {
  _iteam_reduction<double, 4, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_f_4x64(float v, float *r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<float, 4, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_f_4x64_fast_sum(float v, float *r_p, float *tvs, uint32_t *td,
                              void (*rf)(float *, float),
                              void (*rflds)(_LDS float *, _LDS float *),
                              const float rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<float, 4, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_f_4x64(float v, float *r_p, void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k) {
  _iteam_reduction<float, 4, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cd_4x64(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CD, 4, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cd_4x64_fast_sum(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                               void (*rf)(_CD *, _CD),
                               void (*rflds)(_LDS _CD *, _LDS _CD *),
                               const _CD rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CD, 4, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cd_4x64(_CD v, _CD *r_p, void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k) {
  _iteam_reduction<_CD, 4, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cf_4x64(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CF, 4, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cf_4x64_fast_sum(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                               void (*rf)(_CF *, _CF),
                               void (*rflds)(_LDS _CF *, _LDS _CF *),
                               const _CF rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CF, 4, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cf_4x64(_CF v, _CF *r_p, void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k) {
  _iteam_reduction<_CF, 4, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_i_4x64(int v, int *r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<int, 4, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_i_4x64_fast_sum(int v, int *r_p, int *tvs, uint32_t *td,
                              void (*rf)(int *, int),
                              void (*rflds)(_LDS int *, _LDS int *),
                              const int rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<int, 4, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_i_4x64(int v, int *r_p, void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k) {
  _iteam_reduction<int, 4, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ui_4x64(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UI, 4, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ui_4x64_fast_sum(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                               void (*rf)(_UI *, _UI),
                               void (*rflds)(_LDS _UI *, _LDS _UI *),
                               const _UI rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UI, 4, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ui_4x64(_UI v, _UI *r_p, void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k) {
  _iteam_reduction<_UI, 4, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_l_4x64(long v, long *r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<long, 4, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_l_4x64_fast_sum(long v, long *r_p, long *tvs, uint32_t *td,
                              void (*rf)(long *, long),
                              void (*rflds)(_LDS long *, _LDS long *),
                              const long rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<long, 4, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_l_4x64(long v, long *r_p, void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k) {
  _iteam_reduction<long, 4, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ul_4x64(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UL, 4, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ul_4x64_fast_sum(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                               void (*rf)(_UL *, _UL),
                               void (*rflds)(_LDS _UL *, _LDS _UL *),
                               const _UL rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UL, 4, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ul_4x64(_UL v, _UL *r_p, void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k) {
  _iteam_reduction<_UL, 4, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_d_2x64(double v, double *r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<double, 2, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_d_2x64_fast_sum(double v, double *r_p, double *tvs, uint32_t *td,
                              void (*rf)(double *, double),
                              void (*rflds)(_LDS double *, _LDS double *),
                              const double rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<double, 2, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_d_2x64(double v, double *r_p, void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k) {
  _iteam_reduction<double, 2, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_f_2x64(float v, float *r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<float, 2, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_f_2x64_fast_sum(float v, float *r_p, float *tvs, uint32_t *td,
                              void (*rf)(float *, float),
                              void (*rflds)(_LDS float *, _LDS float *),
                              const float rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<float, 2, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_f_2x64(float v, float *r_p, void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k) {
  _iteam_reduction<float, 2, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cd_2x64(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CD, 2, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cd_2x64_fast_sum(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                               void (*rf)(_CD *, _CD),
                               void (*rflds)(_LDS _CD *, _LDS _CD *),
                               const _CD rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CD, 2, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cd_2x64(_CD v, _CD *r_p, void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k) {
  _iteam_reduction<_CD, 2, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cf_2x64(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CF, 2, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cf_2x64_fast_sum(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                               void (*rf)(_CF *, _CF),
                               void (*rflds)(_LDS _CF *, _LDS _CF *),
                               const _CF rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CF, 2, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cf_2x64(_CF v, _CF *r_p, void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k) {
  _iteam_reduction<_CF, 2, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_i_2x64(int v, int *r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<int, 2, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_i_2x64_fast_sum(int v, int *r_p, int *tvs, uint32_t *td,
                              void (*rf)(int *, int),
                              void (*rflds)(_LDS int *, _LDS int *),
                              const int rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<int, 2, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_i_2x64(int v, int *r_p, void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k) {
  _iteam_reduction<int, 2, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ui_2x64(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UI, 2, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ui_2x64_fast_sum(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                               void (*rf)(_UI *, _UI),
                               void (*rflds)(_LDS _UI *, _LDS _UI *),
                               const _UI rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UI, 2, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ui_2x64(_UI v, _UI *r_p, void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k) {
  _iteam_reduction<_UI, 2, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_l_2x64(long v, long *r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<long, 2, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_l_2x64_fast_sum(long v, long *r_p, long *tvs, uint32_t *td,
                              void (*rf)(long *, long),
                              void (*rflds)(_LDS long *, _LDS long *),
                              const long rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<long, 2, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_l_2x64(long v, long *r_p, void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k) {
  _iteam_reduction<long, 2, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ul_2x64(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UL, 2, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ul_2x64_fast_sum(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                               void (*rf)(_UL *, _UL),
                               void (*rflds)(_LDS _UL *, _LDS _UL *),
                               const _UL rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UL, 2, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ul_2x64(_UL v, _UL *r_p, void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k) {
  _iteam_reduction<_UL, 2, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_d_1x64(double v, double *r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<double, 1, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_d_1x64_fast_sum(double v, double *r_p, double *tvs, uint32_t *td,
                              void (*rf)(double *, double),
                              void (*rflds)(_LDS double *, _LDS double *),
                              const double rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<double, 1, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_d_1x64(double v, double *r_p, void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k) {
  _iteam_reduction<double, 1, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_f_1x64(float v, float *r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<float, 1, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_f_1x64_fast_sum(float v, float *r_p, float *tvs, uint32_t *td,
                              void (*rf)(float *, float),
                              void (*rflds)(_LDS float *, _LDS float *),
                              const float rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<float, 1, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_f_1x64(float v, float *r_p, void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k) {
  _iteam_reduction<float, 1, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cd_1x64(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CD, 1, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cd_1x64_fast_sum(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                               void (*rf)(_CD *, _CD),
                               void (*rflds)(_LDS _CD *, _LDS _CD *),
                               const _CD rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CD, 1, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cd_1x64(_CD v, _CD *r_p, void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k) {
  _iteam_reduction<_CD, 1, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cf_1x64(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CF, 1, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cf_1x64_fast_sum(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                               void (*rf)(_CF *, _CF),
                               void (*rflds)(_LDS _CF *, _LDS _CF *),
                               const _CF rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CF, 1, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cf_1x64(_CF v, _CF *r_p, void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k) {
  _iteam_reduction<_CF, 1, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_i_1x64(int v, int *r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<int, 1, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_i_1x64_fast_sum(int v, int *r_p, int *tvs, uint32_t *td,
                              void (*rf)(int *, int),
                              void (*rflds)(_LDS int *, _LDS int *),
                              const int rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<int, 1, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_i_1x64(int v, int *r_p, void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k) {
  _iteam_reduction<int, 1, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ui_1x64(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UI, 1, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ui_1x64_fast_sum(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                               void (*rf)(_UI *, _UI),
                               void (*rflds)(_LDS _UI *, _LDS _UI *),
                               const _UI rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UI, 1, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ui_1x64(_UI v, _UI *r_p, void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k) {
  _iteam_reduction<_UI, 1, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_l_1x64(long v, long *r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<long, 1, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_l_1x64_fast_sum(long v, long *r_p, long *tvs, uint32_t *td,
                              void (*rf)(long *, long),
                              void (*rflds)(_LDS long *, _LDS long *),
                              const long rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<long, 1, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_l_1x64(long v, long *r_p, void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k) {
  _iteam_reduction<long, 1, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ul_1x64(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UL, 1, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ul_1x64_fast_sum(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                               void (*rf)(_UL *, _UL),
                               void (*rflds)(_LDS _UL *, _LDS _UL *),
                               const _UL rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UL, 1, 64, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ul_1x64(_UL v, _UL *r_p, void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k) {
  _iteam_reduction<_UL, 1, 64>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_d_32x32(double v, double *r_p, double *tvs, uint32_t *td,
                      void (*rf)(double *, double),
                      void (*rflds)(_LDS double *, _LDS double *),
                      const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<double, 32, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_d_32x32_fast_sum(double v, double *r_p, double *tvs, uint32_t *td,
                               void (*rf)(double *, double),
                               void (*rflds)(_LDS double *, _LDS double *),
                               const double rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<double, 32, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k,
                                         nt);
}
_EXT_ATTR
__kmpc_iteamr_d_32x32(double v, double *r_p, void (*rf)(double *, double),
                      void (*rflds)(_LDS double *, _LDS double *),
                      const double rnv, const uint64_t k) {
  _iteam_reduction<double, 32, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_f_32x32(float v, float *r_p, float *tvs, uint32_t *td,
                      void (*rf)(float *, float),
                      void (*rflds)(_LDS float *, _LDS float *),
                      const float rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<float, 32, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_f_32x32_fast_sum(float v, float *r_p, float *tvs, uint32_t *td,
                               void (*rf)(float *, float),
                               void (*rflds)(_LDS float *, _LDS float *),
                               const float rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<float, 32, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_f_32x32(float v, float *r_p, void (*rf)(float *, float),
                      void (*rflds)(_LDS float *, _LDS float *),
                      const float rnv, const uint64_t k) {
  _iteam_reduction<float, 32, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cd_32x32(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                       void (*rf)(_CD *, _CD),
                       void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CD, 32, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cd_32x32_fast_sum(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                                void (*rf)(_CD *, _CD),
                                void (*rflds)(_LDS _CD *, _LDS _CD *),
                                const _CD rnv, const uint64_t k,
                                const uint32_t nt) {
  _xteam_reduction<_CD, 32, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cd_32x32(_CD v, _CD *r_p, void (*rf)(_CD *, _CD),
                       void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                       const uint64_t k) {
  _iteam_reduction<_CD, 32, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cf_32x32(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                       void (*rf)(_CF *, _CF),
                       void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CF, 32, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cf_32x32_fast_sum(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                                void (*rf)(_CF *, _CF),
                                void (*rflds)(_LDS _CF *, _LDS _CF *),
                                const _CF rnv, const uint64_t k,
                                const uint32_t nt) {
  _xteam_reduction<_CF, 32, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cf_32x32(_CF v, _CF *r_p, void (*rf)(_CF *, _CF),
                       void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                       const uint64_t k) {
  _iteam_reduction<_CF, 32, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_i_32x32(int v, int *r_p, int *tvs, uint32_t *td,
                      void (*rf)(int *, int),
                      void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<int, 32, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_i_32x32_fast_sum(int v, int *r_p, int *tvs, uint32_t *td,
                               void (*rf)(int *, int),
                               void (*rflds)(_LDS int *, _LDS int *),
                               const int rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<int, 32, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_i_32x32(int v, int *r_p, void (*rf)(int *, int),
                      void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                      const uint64_t k) {
  _iteam_reduction<int, 32, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ui_32x32(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                       void (*rf)(_UI *, _UI),
                       void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UI, 32, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ui_32x32_fast_sum(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                                void (*rf)(_UI *, _UI),
                                void (*rflds)(_LDS _UI *, _LDS _UI *),
                                const _UI rnv, const uint64_t k,
                                const uint32_t nt) {
  _xteam_reduction<_UI, 32, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ui_32x32(_UI v, _UI *r_p, void (*rf)(_UI *, _UI),
                       void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                       const uint64_t k) {
  _iteam_reduction<_UI, 32, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_l_32x32(long v, long *r_p, long *tvs, uint32_t *td,
                      void (*rf)(long *, long),
                      void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<long, 32, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_l_32x32_fast_sum(long v, long *r_p, long *tvs, uint32_t *td,
                               void (*rf)(long *, long),
                               void (*rflds)(_LDS long *, _LDS long *),
                               const long rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<long, 32, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_l_32x32(long v, long *r_p, void (*rf)(long *, long),
                      void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                      const uint64_t k) {
  _iteam_reduction<long, 32, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ul_32x32(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                       void (*rf)(_UL *, _UL),
                       void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UL, 32, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ul_32x32_fast_sum(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                                void (*rf)(_UL *, _UL),
                                void (*rflds)(_LDS _UL *, _LDS _UL *),
                                const _UL rnv, const uint64_t k,
                                const uint32_t nt) {
  _xteam_reduction<_UL, 32, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ul_32x32(_UL v, _UL *r_p, void (*rf)(_UL *, _UL),
                       void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                       const uint64_t k) {
  _iteam_reduction<_UL, 32, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_d_16x32(double v, double *r_p, double *tvs, uint32_t *td,
                      void (*rf)(double *, double),
                      void (*rflds)(_LDS double *, _LDS double *),
                      const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<double, 16, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_d_16x32_fast_sum(double v, double *r_p, double *tvs, uint32_t *td,
                               void (*rf)(double *, double),
                               void (*rflds)(_LDS double *, _LDS double *),
                               const double rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<double, 16, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k,
                                         nt);
}
_EXT_ATTR
__kmpc_iteamr_d_16x32(double v, double *r_p, void (*rf)(double *, double),
                      void (*rflds)(_LDS double *, _LDS double *),
                      const double rnv, const uint64_t k) {
  _iteam_reduction<double, 16, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_f_16x32(float v, float *r_p, float *tvs, uint32_t *td,
                      void (*rf)(float *, float),
                      void (*rflds)(_LDS float *, _LDS float *),
                      const float rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<float, 16, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_f_16x32_fast_sum(float v, float *r_p, float *tvs, uint32_t *td,
                               void (*rf)(float *, float),
                               void (*rflds)(_LDS float *, _LDS float *),
                               const float rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<float, 16, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_f_16x32(float v, float *r_p, void (*rf)(float *, float),
                      void (*rflds)(_LDS float *, _LDS float *),
                      const float rnv, const uint64_t k) {
  _iteam_reduction<float, 16, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cd_16x32(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                       void (*rf)(_CD *, _CD),
                       void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CD, 16, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cd_16x32_fast_sum(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                                void (*rf)(_CD *, _CD),
                                void (*rflds)(_LDS _CD *, _LDS _CD *),
                                const _CD rnv, const uint64_t k,
                                const uint32_t nt) {
  _xteam_reduction<_CD, 16, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cd_16x32(_CD v, _CD *r_p, void (*rf)(_CD *, _CD),
                       void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                       const uint64_t k) {
  _iteam_reduction<_CD, 16, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cf_16x32(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                       void (*rf)(_CF *, _CF),
                       void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CF, 16, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cf_16x32_fast_sum(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                                void (*rf)(_CF *, _CF),
                                void (*rflds)(_LDS _CF *, _LDS _CF *),
                                const _CF rnv, const uint64_t k,
                                const uint32_t nt) {
  _xteam_reduction<_CF, 16, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cf_16x32(_CF v, _CF *r_p, void (*rf)(_CF *, _CF),
                       void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                       const uint64_t k) {
  _iteam_reduction<_CF, 16, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_i_16x32(int v, int *r_p, int *tvs, uint32_t *td,
                      void (*rf)(int *, int),
                      void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<int, 16, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_i_16x32_fast_sum(int v, int *r_p, int *tvs, uint32_t *td,
                               void (*rf)(int *, int),
                               void (*rflds)(_LDS int *, _LDS int *),
                               const int rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<int, 16, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_i_16x32(int v, int *r_p, void (*rf)(int *, int),
                      void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                      const uint64_t k) {
  _iteam_reduction<int, 16, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ui_16x32(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                       void (*rf)(_UI *, _UI),
                       void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UI, 16, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ui_16x32_fast_sum(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                                void (*rf)(_UI *, _UI),
                                void (*rflds)(_LDS _UI *, _LDS _UI *),
                                const _UI rnv, const uint64_t k,
                                const uint32_t nt) {
  _xteam_reduction<_UI, 16, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ui_16x32(_UI v, _UI *r_p, void (*rf)(_UI *, _UI),
                       void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                       const uint64_t k) {
  _iteam_reduction<_UI, 16, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_l_16x32(long v, long *r_p, long *tvs, uint32_t *td,
                      void (*rf)(long *, long),
                      void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<long, 16, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_l_16x32_fast_sum(long v, long *r_p, long *tvs, uint32_t *td,
                               void (*rf)(long *, long),
                               void (*rflds)(_LDS long *, _LDS long *),
                               const long rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<long, 16, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_l_16x32(long v, long *r_p, void (*rf)(long *, long),
                      void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                      const uint64_t k) {
  _iteam_reduction<long, 16, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ul_16x32(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                       void (*rf)(_UL *, _UL),
                       void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UL, 16, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ul_16x32_fast_sum(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                                void (*rf)(_UL *, _UL),
                                void (*rflds)(_LDS _UL *, _LDS _UL *),
                                const _UL rnv, const uint64_t k,
                                const uint32_t nt) {
  _xteam_reduction<_UL, 16, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ul_16x32(_UL v, _UL *r_p, void (*rf)(_UL *, _UL),
                       void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                       const uint64_t k) {
  _iteam_reduction<_UL, 16, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_d_8x32(double v, double *r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<double, 8, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_d_8x32_fast_sum(double v, double *r_p, double *tvs, uint32_t *td,
                              void (*rf)(double *, double),
                              void (*rflds)(_LDS double *, _LDS double *),
                              const double rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<double, 8, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_d_8x32(double v, double *r_p, void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k) {
  _iteam_reduction<double, 8, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_f_8x32(float v, float *r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<float, 8, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_f_8x32_fast_sum(float v, float *r_p, float *tvs, uint32_t *td,
                              void (*rf)(float *, float),
                              void (*rflds)(_LDS float *, _LDS float *),
                              const float rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<float, 8, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_f_8x32(float v, float *r_p, void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k) {
  _iteam_reduction<float, 8, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cd_8x32(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CD, 8, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cd_8x32_fast_sum(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                               void (*rf)(_CD *, _CD),
                               void (*rflds)(_LDS _CD *, _LDS _CD *),
                               const _CD rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CD, 8, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cd_8x32(_CD v, _CD *r_p, void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k) {
  _iteam_reduction<_CD, 8, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cf_8x32(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CF, 8, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cf_8x32_fast_sum(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                               void (*rf)(_CF *, _CF),
                               void (*rflds)(_LDS _CF *, _LDS _CF *),
                               const _CF rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CF, 8, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cf_8x32(_CF v, _CF *r_p, void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k) {
  _iteam_reduction<_CF, 8, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_i_8x32(int v, int *r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<int, 8, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_i_8x32_fast_sum(int v, int *r_p, int *tvs, uint32_t *td,
                              void (*rf)(int *, int),
                              void (*rflds)(_LDS int *, _LDS int *),
                              const int rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<int, 8, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_i_8x32(int v, int *r_p, void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k) {
  _iteam_reduction<int, 8, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ui_8x32(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UI, 8, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ui_8x32_fast_sum(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                               void (*rf)(_UI *, _UI),
                               void (*rflds)(_LDS _UI *, _LDS _UI *),
                               const _UI rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UI, 8, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ui_8x32(_UI v, _UI *r_p, void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k) {
  _iteam_reduction<_UI, 8, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_l_8x32(long v, long *r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<long, 8, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_l_8x32_fast_sum(long v, long *r_p, long *tvs, uint32_t *td,
                              void (*rf)(long *, long),
                              void (*rflds)(_LDS long *, _LDS long *),
                              const long rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<long, 8, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_l_8x32(long v, long *r_p, void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k) {
  _iteam_reduction<long, 8, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ul_8x32(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UL, 8, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ul_8x32_fast_sum(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                               void (*rf)(_UL *, _UL),
                               void (*rflds)(_LDS _UL *, _LDS _UL *),
                               const _UL rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UL, 8, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ul_8x32(_UL v, _UL *r_p, void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k) {
  _iteam_reduction<_UL, 8, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_d_4x32(double v, double *r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<double, 4, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_d_4x32_fast_sum(double v, double *r_p, double *tvs, uint32_t *td,
                              void (*rf)(double *, double),
                              void (*rflds)(_LDS double *, _LDS double *),
                              const double rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<double, 4, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_d_4x32(double v, double *r_p, void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k) {
  _iteam_reduction<double, 4, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_f_4x32(float v, float *r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<float, 4, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_f_4x32_fast_sum(float v, float *r_p, float *tvs, uint32_t *td,
                              void (*rf)(float *, float),
                              void (*rflds)(_LDS float *, _LDS float *),
                              const float rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<float, 4, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_f_4x32(float v, float *r_p, void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k) {
  _iteam_reduction<float, 4, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cd_4x32(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CD, 4, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cd_4x32_fast_sum(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                               void (*rf)(_CD *, _CD),
                               void (*rflds)(_LDS _CD *, _LDS _CD *),
                               const _CD rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CD, 4, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cd_4x32(_CD v, _CD *r_p, void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k) {
  _iteam_reduction<_CD, 4, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cf_4x32(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CF, 4, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cf_4x32_fast_sum(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                               void (*rf)(_CF *, _CF),
                               void (*rflds)(_LDS _CF *, _LDS _CF *),
                               const _CF rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CF, 4, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cf_4x32(_CF v, _CF *r_p, void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k) {
  _iteam_reduction<_CF, 4, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_i_4x32(int v, int *r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<int, 4, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_i_4x32_fast_sum(int v, int *r_p, int *tvs, uint32_t *td,
                              void (*rf)(int *, int),
                              void (*rflds)(_LDS int *, _LDS int *),
                              const int rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<int, 4, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_i_4x32(int v, int *r_p, void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k) {
  _iteam_reduction<int, 4, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ui_4x32(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UI, 4, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ui_4x32_fast_sum(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                               void (*rf)(_UI *, _UI),
                               void (*rflds)(_LDS _UI *, _LDS _UI *),
                               const _UI rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UI, 4, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ui_4x32(_UI v, _UI *r_p, void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k) {
  _iteam_reduction<_UI, 4, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_l_4x32(long v, long *r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<long, 4, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_l_4x32_fast_sum(long v, long *r_p, long *tvs, uint32_t *td,
                              void (*rf)(long *, long),
                              void (*rflds)(_LDS long *, _LDS long *),
                              const long rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<long, 4, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_l_4x32(long v, long *r_p, void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k) {
  _iteam_reduction<long, 4, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ul_4x32(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UL, 4, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ul_4x32_fast_sum(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                               void (*rf)(_UL *, _UL),
                               void (*rflds)(_LDS _UL *, _LDS _UL *),
                               const _UL rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UL, 4, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ul_4x32(_UL v, _UL *r_p, void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k) {
  _iteam_reduction<_UL, 4, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_d_2x32(double v, double *r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<double, 2, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_d_2x32_fast_sum(double v, double *r_p, double *tvs, uint32_t *td,
                              void (*rf)(double *, double),
                              void (*rflds)(_LDS double *, _LDS double *),
                              const double rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<double, 2, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_d_2x32(double v, double *r_p, void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k) {
  _iteam_reduction<double, 2, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_f_2x32(float v, float *r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<float, 2, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_f_2x32_fast_sum(float v, float *r_p, float *tvs, uint32_t *td,
                              void (*rf)(float *, float),
                              void (*rflds)(_LDS float *, _LDS float *),
                              const float rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<float, 2, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_f_2x32(float v, float *r_p, void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k) {
  _iteam_reduction<float, 2, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cd_2x32(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CD, 2, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cd_2x32_fast_sum(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                               void (*rf)(_CD *, _CD),
                               void (*rflds)(_LDS _CD *, _LDS _CD *),
                               const _CD rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CD, 2, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cd_2x32(_CD v, _CD *r_p, void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k) {
  _iteam_reduction<_CD, 2, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cf_2x32(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CF, 2, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cf_2x32_fast_sum(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                               void (*rf)(_CF *, _CF),
                               void (*rflds)(_LDS _CF *, _LDS _CF *),
                               const _CF rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CF, 2, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cf_2x32(_CF v, _CF *r_p, void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k) {
  _iteam_reduction<_CF, 2, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_i_2x32(int v, int *r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<int, 2, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_i_2x32_fast_sum(int v, int *r_p, int *tvs, uint32_t *td,
                              void (*rf)(int *, int),
                              void (*rflds)(_LDS int *, _LDS int *),
                              const int rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<int, 2, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_i_2x32(int v, int *r_p, void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k) {
  _iteam_reduction<int, 2, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ui_2x32(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UI, 2, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ui_2x32_fast_sum(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                               void (*rf)(_UI *, _UI),
                               void (*rflds)(_LDS _UI *, _LDS _UI *),
                               const _UI rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UI, 2, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ui_2x32(_UI v, _UI *r_p, void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k) {
  _iteam_reduction<_UI, 2, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_l_2x32(long v, long *r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<long, 2, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_l_2x32_fast_sum(long v, long *r_p, long *tvs, uint32_t *td,
                              void (*rf)(long *, long),
                              void (*rflds)(_LDS long *, _LDS long *),
                              const long rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<long, 2, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_l_2x32(long v, long *r_p, void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k) {
  _iteam_reduction<long, 2, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ul_2x32(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UL, 2, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ul_2x32_fast_sum(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                               void (*rf)(_UL *, _UL),
                               void (*rflds)(_LDS _UL *, _LDS _UL *),
                               const _UL rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UL, 2, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ul_2x32(_UL v, _UL *r_p, void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k) {
  _iteam_reduction<_UL, 2, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_d_1x32(double v, double *r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_reduction<double, 1, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_d_1x32_fast_sum(double v, double *r_p, double *tvs, uint32_t *td,
                              void (*rf)(double *, double),
                              void (*rflds)(_LDS double *, _LDS double *),
                              const double rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<double, 1, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_d_1x32(double v, double *r_p, void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k) {
  _iteam_reduction<double, 1, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_f_1x32(float v, float *r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<float, 1, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_f_1x32_fast_sum(float v, float *r_p, float *tvs, uint32_t *td,
                              void (*rf)(float *, float),
                              void (*rflds)(_LDS float *, _LDS float *),
                              const float rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<float, 1, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_f_1x32(float v, float *r_p, void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k) {
  _iteam_reduction<float, 1, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cd_1x32(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CD, 1, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cd_1x32_fast_sum(_CD v, _CD *r_p, _CD *tvs, uint32_t *td,
                               void (*rf)(_CD *, _CD),
                               void (*rflds)(_LDS _CD *, _LDS _CD *),
                               const _CD rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CD, 1, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cd_1x32(_CD v, _CD *r_p, void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k) {
  _iteam_reduction<_CD, 1, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_cf_1x32(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_CF, 1, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_cf_1x32_fast_sum(_CF v, _CF *r_p, _CF *tvs, uint32_t *td,
                               void (*rf)(_CF *, _CF),
                               void (*rflds)(_LDS _CF *, _LDS _CF *),
                               const _CF rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_CF, 1, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_cf_1x32(_CF v, _CF *r_p, void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k) {
  _iteam_reduction<_CF, 1, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_i_1x32(int v, int *r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<int, 1, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_i_1x32_fast_sum(int v, int *r_p, int *tvs, uint32_t *td,
                              void (*rf)(int *, int),
                              void (*rflds)(_LDS int *, _LDS int *),
                              const int rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<int, 1, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_i_1x32(int v, int *r_p, void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k) {
  _iteam_reduction<int, 1, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ui_1x32(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UI, 1, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ui_1x32_fast_sum(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                               void (*rf)(_UI *, _UI),
                               void (*rflds)(_LDS _UI *, _LDS _UI *),
                               const _UI rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UI, 1, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ui_1x32(_UI v, _UI *r_p, void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k) {
  _iteam_reduction<_UI, 1, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_l_1x32(long v, long *r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_reduction<long, 1, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_l_1x32_fast_sum(long v, long *r_p, long *tvs, uint32_t *td,
                              void (*rf)(long *, long),
                              void (*rflds)(_LDS long *, _LDS long *),
                              const long rnv, const uint64_t k,
                              const uint32_t nt) {
  _xteam_reduction<long, 1, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_l_1x32(long v, long *r_p, void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k) {
  _iteam_reduction<long, 1, 32>(v, r_p, rf, rflds, rnv, k);
}
_EXT_ATTR
__kmpc_xteamr_ul_1x32(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_reduction<_UL, 1, 32>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteamr_ul_1x32_fast_sum(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                               void (*rf)(_UL *, _UL),
                               void (*rflds)(_LDS _UL *, _LDS _UL *),
                               const _UL rnv, const uint64_t k,
                               const uint32_t nt) {
  _xteam_reduction<_UL, 1, 32, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_iteamr_ul_1x32(_UL v, _UL *r_p, void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k) {
  _iteam_reduction<_UL, 1, 32>(v, r_p, rf, rflds, rnv, k);
}
#undef _CD
#undef _CF
#undef _UI
#undef _UL
#undef _LDS

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
