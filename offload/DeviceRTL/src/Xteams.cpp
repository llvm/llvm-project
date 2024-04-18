//===---- Xteams.cpp - OpenMP cross team helper functions ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains helper functions for cross team scan
//
//===----------------------------------------------------------------------===//

#include "Xteams.h"
#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

#define __XTEAM_SHARED_LDS volatile __attribute__((address_space(3)))

using namespace ompx::mapping;

#pragma omp begin declare target device_type(nohost)


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

// Returns true if num is an odd power of two 
bool is_odd_power(uint32_t num) {
  bool is_odd = false;
  while(num != 1) {
    num >>= 1;
    is_odd = !is_odd;
  }
  return is_odd;
}

// Returns the smallest power of two which is >= `num`
uint32_t get_ceiled_num(uint32_t num) {
  // return num;
  uint32_t ceil_num = 1;
  while(ceil_num < num) 
    ceil_num <<= 1;
  return ceil_num;
}

/// Templated internal function used by all extern typed scans
///
/// \param  Template typename parameter T
/// \param  Template parameter for number of waves, must be power of two
/// \param  Template parameter for warp size, 32 o 64
///
/// \param val Input thread local (TLS) value for intra team scan
/// \param storage Pointer to global shared storage used by all the threads
/// \param r_array Pointer to result scan array (output)
/// \param team_vals Global array storing reduction computed after per team scan
/// \param teams_done_ptr Pointer to atomically access teams done counter
/// \param _rf Function pointer to TLS pair reduction function
/// \param _rf_lds Function pointer to LDS pair reduction function
/// \param rnv Reduction null value (e.g. 0 for addition)
/// \param k The iteration value from 0 to (NumTeams*_NUM_THREADS)-1
/// \param NumTeams The number of teams 

template <typename T, const int32_t _NW, const int32_t _WSZ>
__attribute__((flatten, always_inline)) void _xteam_scan(
    T val, T* storage, T* r_array, T *team_vals, 
    uint32_t *teams_done_ptr, void (*_rf)(T *, T),
    void (*_rf_lds)(__XTEAM_SHARED_LDS T *, __XTEAM_SHARED_LDS T *),
    const T rnv, const uint64_t k, const uint32_t NumTeams) {

  // More efficient to derive these constants than get from mapped API
  constexpr uint32_t _NT = _NW * _WSZ;      // number of threads within a team
  const uint32_t omp_thread_num = k % _NT;  // thread ID within a team
  const uint32_t omp_team_num = k / _NT;    // team ID
  const uint32_t total_num_threads = NumTeams * _NT;
  uint32_t first = 0;

  // Computing Scan within each Team (Intra-Team Scan)
  ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);

  for(int offset = 1; offset < _NT; offset <<= 1) {
    if(omp_thread_num >= offset) 
      (*_rf)(&val, storage[first + k - offset]);   // val += storage[first + k - offset];
    first = total_num_threads - first;
    storage[first + k] = val;
    ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);
  }

  // The offset value which is required to access the computed team-wise scan 
  // based upon the workgroup size.
  uint32_t offset = is_odd_power(_NT) ? total_num_threads : 0;
  storage[k] = storage[offset + k];  

  // The teams_done_ptr will be read using this
  static __XTEAM_SHARED_LDS uint32_t td;
  if(omp_thread_num == 0) {
    // store the team-level reduction in team_vals[]
    team_vals[omp_team_num] = storage[omp_team_num*_NT + _NT - 1];
    td = ompx::atomic::inc(teams_done_ptr, NumTeams - 1u, ompx::atomic::seq_cst,
                           ompx::atomic::MemScopeTy::device);
  }

  // This sync is needed because all threads of the last team which reaches
  // this part of code need to know that they are in the last team by 
  // reading the shared volatile value `td`.
  ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);

  // If td counter reaches NumTeams-1, this is the last team. Threads of the
  // last team enter here.
  if (td == (NumTeams - 1u)) {
    // Shared memory for the last team to compute scan of the Intra-Team reductions.
    // Assuming that NumTeams <= _NT
    // TODO: This assumption needs to be get rid of by introducing some serial 
    // work here. This is required to support arbitrary NumTeams. This is the
    // reason why we do not test for teamsize 64 yet.
    static __XTEAM_SHARED_LDS T partial_sums[2*_NT + 1]; 
    
    // To make sure the scan algorithm works, ceiling the NumTeams to the next power 
    // of two is required.
    const uint32_t ceiledNumTeams = get_ceiled_num(NumTeams);
    
    // preparing `val` to hold the per team reductions from Intra-Team scan
    // for Cross-Team Scan operation
    val = omp_thread_num < ceiledNumTeams ? team_vals[omp_thread_num] : rnv;
    partial_sums[omp_thread_num] = val;
    first = 0;
    
    // Computing Scan across teams (Cross-Team Scan)
    ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);

    for(int offset = 1; offset < ceiledNumTeams; offset <<= 1) {
      if(omp_thread_num >= offset) 
        (*_rf)(&val, partial_sums[first + omp_thread_num - offset]); // val += partial_sums[first + omp_thread_num - offset]
      first = ceiledNumTeams - first;
      partial_sums[first + omp_thread_num] = val;
      ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);
    }

    // updating the `team_vals` to hold the cross-team scanned result 
    if(omp_thread_num < ceiledNumTeams) {
      // The offset required to access the computed scan of Intra-Team reductions
      offset = is_odd_power(ceiledNumTeams) ? ceiledNumTeams : 0;
      team_vals[omp_thread_num] = partial_sums[offset + omp_thread_num]; 
    }
  }
}

//  Calls to these __kmpc extern C functions will be created in clang codegen
//  for C and C++. They may also be used for simulation and testing.
//  The headers for these extern C functions are in ../include/Xteams.h
//  The compiler builds the name based on the data type,
//  number of waves in the team and warpsize.

#define _EXT_ATTR extern "C" __attribute__((flatten, always_inline)) void
#define _CD double _Complex
#define _CF float _Complex
#define _UI unsigned int
#define _UL unsigned long
#define _LDS volatile __attribute__((address_space(3)))
_EXT_ATTR
__kmpc_xteams_d_16x64(double v, double* storage, double* r_p, double *tvs, uint32_t *td,
                      void (*rf)(double *, double),
                      void (*rflds)(_LDS double *, _LDS double *),
                      const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_scan<double, 16, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_f_16x64(float v, float* storage, float* r_p, float *tvs, uint32_t *td,
                      void (*rf)(float *, float),
                      void (*rflds)(_LDS float *, _LDS float *),
                      const float rnv, const uint64_t k, const uint32_t nt) {
  _xteam_scan<float, 16, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cd_16x64(_CD v, _CD* storage, _CD* r_p, _CD *tvs, uint32_t *td,
                       void (*rf)(_CD *, _CD),
                       void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CD, 16, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cf_16x64(_CF v, _CF* storage, _CF* r_p, _CF *tvs, uint32_t *td,
                       void (*rf)(_CF *, _CF),
                       void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CF, 16, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_i_16x64(int v, int* storage, int* r_p, int *tvs, uint32_t *td,
                      void (*rf)(int *, int),
                      void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<int, 16, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ui_16x64(_UI v, _UI* storage, _UI* r_p, _UI *tvs, uint32_t *td,
                       void (*rf)(_UI *, _UI),
                       void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UI, 16, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_l_16x64(long v, long* storage, long* r_p, long *tvs, uint32_t *td,
                      void (*rf)(long *, long),
                      void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<long, 16, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ul_16x64(_UL v, _UL* storage, _UL* r_p, _UL *tvs, uint32_t *td,
                       void (*rf)(_UL *, _UL),
                       void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UL, 16, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_d_8x64(double v, double* storage, double* r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_scan<double, 8, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_f_8x64(float v, float* storage, float* r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<float, 8, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cd_8x64(_CD v, _CD* storage, _CD* r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CD, 8, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cf_8x64(_CF v, _CF* storage, _CF* r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CF, 8, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_i_8x64(int v, int* storage, int* r_p, int* tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<int, 8, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ui_8x64(_UI v, _UI* storage, _UI* r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UI, 8, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_l_8x64(long v, long* storage, long* r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<long, 8, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ul_8x64(_UL v, _UL* storage, _UL* r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UL, 8, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_d_4x64(double v, double* storage, double* r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_scan<double, 4, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_f_4x64(float v, float* storage, float* r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<float, 4, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cd_4x64(_CD v, _CD* storage, _CD* r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CD, 4, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cf_4x64(_CF v, _CF* storage, _CF* r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CF, 4, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_i_4x64(int v, int* storage, int* r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<int, 4, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ui_4x64(_UI v, _UI* storage, _UI* r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UI, 4, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_l_4x64(long v, long* storage, long* r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<long, 4, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ul_4x64(_UL v, _UL* storage, _UL* r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UL, 4, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_d_2x64(double v, double* storage, double* r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_scan<double, 2, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_f_2x64(float v, float* storage, float* r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<float, 2, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cd_2x64(_CD v, _CD* storage, _CD* r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CD, 2, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cf_2x64(_CF v, _CF* storage, _CF* r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CF, 2, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_i_2x64(int v, int* storage, int* r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<int, 2, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ui_2x64(_UI v, _UI* storage, _UI* r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UI, 2, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_l_2x64(long v, long* storage, long* r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<long, 2, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ul_2x64(_UL v, _UL* storage, _UL* r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UL, 2, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_d_1x64(double v, double* storage, double* r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_scan<double, 1, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_f_1x64(float v, float* storage, float* r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<float, 1, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cd_1x64(_CD v, _CD* storage, _CD* r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CD, 1, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cf_1x64(_CF v, _CF* storage, _CF* r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CF, 1, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_i_1x64(int v, int* storage, int* r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<int, 1, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ui_1x64(_UI v, _UI* storage, _UI* r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UI, 1, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_l_1x64(long v, long* storage, long* r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<long, 1, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ul_1x64(_UL v, _UL* storage, _UL* r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UL, 1, 64>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_d_32x32(double v, double* storage, double* r_p, double *tvs, uint32_t *td,
                      void (*rf)(double *, double),
                      void (*rflds)(_LDS double *, _LDS double *),
                      const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_scan<double, 32, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_f_32x32(float v, float* storage, float* r_p, float *tvs, uint32_t *td,
                      void (*rf)(float *, float),
                      void (*rflds)(_LDS float *, _LDS float *),
                      const float rnv, const uint64_t k, const uint32_t nt) {
  _xteam_scan<float, 32, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cd_32x32(_CD v, _CD* storage, _CD* r_p, _CD *tvs, uint32_t *td,
                       void (*rf)(_CD *, _CD),
                       void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CD, 32, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cf_32x32(_CF v, _CF* storage, _CF* r_p, _CF *tvs, uint32_t *td,
                       void (*rf)(_CF *, _CF),
                       void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CF, 32, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_i_32x32(int v, int* storage, int* r_p, int *tvs, uint32_t *td,
                      void (*rf)(int *, int),
                      void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<int, 32, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ui_32x32(_UI v, _UI* storage, _UI* r_p, _UI *tvs, uint32_t *td,
                       void (*rf)(_UI *, _UI),
                       void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UI, 32, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_l_32x32(long v, long* storage, long* r_p, long *tvs, uint32_t *td,
                      void (*rf)(long *, long),
                      void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<long, 32, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ul_32x32(_UL v, _UL* storage, _UL* r_p, _UL *tvs, uint32_t *td,
                       void (*rf)(_UL *, _UL),
                       void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UL, 32, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_d_16x32(double v, double* storage, double* r_p, double *tvs, uint32_t *td,
                      void (*rf)(double *, double),
                      void (*rflds)(_LDS double *, _LDS double *),
                      const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_scan<double, 16, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_f_16x32(float v, float* storage, float* r_p, float *tvs, uint32_t *td,
                      void (*rf)(float *, float),
                      void (*rflds)(_LDS float *, _LDS float *),
                      const float rnv, const uint64_t k, const uint32_t nt) {
  _xteam_scan<float, 16, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cd_16x32(_CD v, _CD* storage, _CD* r_p, _CD *tvs, uint32_t *td,
                       void (*rf)(_CD *, _CD),
                       void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CD, 16, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cf_16x32(_CF v, _CF* storage, _CF* r_p, _CF *tvs, uint32_t *td,
                       void (*rf)(_CF *, _CF),
                       void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CF, 16, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_i_16x32(int v, int* storage, int* r_p, int *tvs, uint32_t *td,
                      void (*rf)(int *, int),
                      void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<int, 16, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ui_16x32(_UI v, _UI* storage, _UI* r_p, _UI *tvs, uint32_t *td,
                       void (*rf)(_UI *, _UI),
                       void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UI, 16, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_l_16x32(long v, long* storage, long* r_p, long *tvs, uint32_t *td,
                      void (*rf)(long *, long),
                      void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<long, 16, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ul_16x32(_UL v, _UL* storage, _UL* r_p, _UL *tvs, uint32_t *td,
                       void (*rf)(_UL *, _UL),
                       void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                       const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UL, 16, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_d_8x32(double v, double* storage, double* r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_scan<double, 8, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_f_8x32(float v, float* storage, float* r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<float, 8, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cd_8x32(_CD v, _CD* storage, _CD* r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CD, 8, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cf_8x32(_CF v, _CF* storage, _CF* r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CF, 8, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_i_8x32(int v, int* storage, int* r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<int, 8, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ui_8x32(_UI v, _UI* storage, _UI* r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UI, 8, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_l_8x32(long v, long* storage, long* r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<long, 8, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ul_8x32(_UL v, _UL* storage, _UL* r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UL, 8, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_d_4x32(double v, double* storage, double* r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_scan<double, 4, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_f_4x32(float v, float* storage, float* r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<float, 4, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cd_4x32(_CD v, _CD* storage, _CD* r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CD, 4, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cf_4x32(_CF v, _CF* storage, _CF* r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CF, 4, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_i_4x32(int v, int* storage, int* r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<int, 4, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ui_4x32(_UI v, _UI* storage, _UI* r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UI, 4, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_l_4x32(long v, long* storage, long* r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<long, 4, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ul_4x32(_UL v, _UL* storage, _UL* r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UL, 4, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_d_2x32(double v, double* storage, double* r_p, double *tvs, uint32_t *td,
                     void (*rf)(double *, double),
                     void (*rflds)(_LDS double *, _LDS double *),
                     const double rnv, const uint64_t k, const uint32_t nt) {
  _xteam_scan<double, 2, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_f_2x32(float v, float* storage, float* r_p, float *tvs, uint32_t *td,
                     void (*rf)(float *, float),
                     void (*rflds)(_LDS float *, _LDS float *), const float rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<float, 2, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cd_2x32(_CD v, _CD* storage, _CD* r_p, _CD *tvs, uint32_t *td,
                      void (*rf)(_CD *, _CD),
                      void (*rflds)(_LDS _CD *, _LDS _CD *), const _CD rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CD, 2, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_cf_2x32(_CF v, _CF* storage, _CF* r_p, _CF *tvs, uint32_t *td,
                      void (*rf)(_CF *, _CF),
                      void (*rflds)(_LDS _CF *, _LDS _CF *), const _CF rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_CF, 2, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_i_2x32(int v, int* storage, int* r_p, int *tvs, uint32_t *td,
                     void (*rf)(int *, int),
                     void (*rflds)(_LDS int *, _LDS int *), const int rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<int, 2, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ui_2x32(_UI v, _UI* storage, _UI* r_p, _UI *tvs, uint32_t *td,
                      void (*rf)(_UI *, _UI),
                      void (*rflds)(_LDS _UI *, _LDS _UI *), const _UI rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UI, 2, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_l_2x32(long v, long* storage, long* r_p, long *tvs, uint32_t *td,
                     void (*rf)(long *, long),
                     void (*rflds)(_LDS long *, _LDS long *), const long rnv,
                     const uint64_t k, const uint32_t nt) {
  _xteam_scan<long, 2, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
_EXT_ATTR
__kmpc_xteams_ul_2x32(_UL v, _UL* storage, _UL* r_p, _UL *tvs, uint32_t *td,
                      void (*rf)(_UL *, _UL),
                      void (*rflds)(_LDS _UL *, _LDS _UL *), const _UL rnv,
                      const uint64_t k, const uint32_t nt) {
  _xteam_scan<_UL, 2, 32>(v, storage, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
#undef _CF
#undef _UI
#undef _UL
#undef _LDS
#undef _EXT_ATTR

#pragma omp end declare target
