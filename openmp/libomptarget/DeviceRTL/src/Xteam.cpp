//===---- Xteam.cpp - OpenMP cross team helper functions ---- C++ -*-===//
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

#define __XTEAM_MAX_FLOAT (__builtin_inff())
#define __XTEAM_LOW_FLOAT -__XTEAM_MAX_FLOAT
#define __XTEAM_MAX_DOUBLE (__builtin_huge_val())
#define __XTEAM_LOW_DOUBLE -__XTEAM_MAX_DOUBLE
// end FIXME above FIXME
#define __XTEAM_MAX_INT 2147483647
#define __XTEAM_LOW_INT (-__XTEAM_MAX_INT - 1)
// FIXME  Research!
#define __XTEAM_MAX_UINT 2147483647
#define __XTEAM_LOW_UINT 0
// FIXME  Research!
#define __XTEAM_MAX_LONG 2147483647
// FIXME  Research!
#define __XTEAM_LOW_LONG __XTEAM_LOW_INT
// FIXME  Research!
#define __XTEAM_MAX_ULONG 2147483647
#define __XTEAM_LOW_ULONG 0
#define __XTEAM_NTHREADS 1024
#define __XTEAM_MAXW_PERTEAM 32
#define __XTEAM_SHARED volatile __attribute__((address_space(3)))

using namespace _OMP;

#pragma omp declare target

// This is the XTEAM memory interface which is for cross team communication.
// Currently only enough global memory for a single double value per team is
// statically created here. For future growth, please make slotsz so team slot
// remains 8-byte aligned.
constexpr uint64_t __xteam_mem_slotsz = 16; // support double _Complex
// FIXME: It would be better to use actual number of team_procs for a specific
//   GPU than 256. Future support will allocate  __xteam_mem_ptr using device
//   malloc to only allocate enough space for actual number of team_procs.
constexpr uint64_t __xteam_mem_max_team_procs = 256;
constexpr uint64_t __xteam_mem_sz =
    __xteam_mem_slotsz * __xteam_mem_max_team_procs;
// static __attribute__((address_space(1), aligned(64)))
char __xteam_mem_ptr[__xteam_mem_sz] __attribute__((aligned(64)));
void __xteam_set_mem(uint64_t team_proc_num, void *data, uint64_t length,
                     uint64_t offset = 0) {
  __builtin_memcpy(
      &__xteam_mem_ptr[(team_proc_num * __xteam_mem_slotsz) + offset], data,
      length);
}
void __xteam_get_mem(uint64_t team_proc_num, void *data, uint64_t length,
                     uint64_t offset = 0) {
  __builtin_memcpy(
      data, &__xteam_mem_ptr[(team_proc_num * __xteam_mem_slotsz) + offset],
      length);
}

// Headers for specialized shfl_xor
double __shfl_xor_d(double var, int lane_mask,
                    int width = mapping::getWarpSize());
float __shfl_xor_f(float var, int lane_mask,
                   int width = mapping::getWarpSize());
int __shfl_xor_int(int var, int lane_mask, int width = mapping::getWarpSize());
double _Complex __shfl_xor_cd(double _Complex var, int lane_mask,
                              int width = mapping::getWarpSize());
float _Complex __shfl_xor_cf(float _Complex var, int lane_mask,
                             int width = mapping::getWarpSize());

// Define the arch variants of shfl

#pragma omp begin declare variant match(device = {arch(amdgcn)})

int __shfl_xor_int(int var, int lane_mask, int width = mapping::getWarpSize()) {
  int self = mapping::getThreadIdInWarp(); // __lane_id();
  int index = self ^ lane_mask;
  index = index >= ((self + width) & ~(width - 1)) ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}
float __shfl_xor_f(float var, int lane_mask,
                   int width = mapping::getWarpSize()) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_xor_int(tmp.i, lane_mask, width);
  return tmp.f;
}
double __shfl_xor_d(double var, int lane_mask,
                    int width = mapping::getWarpSize()) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");
  static_assert(sizeof(double) == sizeof(uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_xor_int(tmp[0], lane_mask, width);
  tmp[1] = __shfl_xor_int(tmp[1], lane_mask, width);

  uint64_t tmp0 =
      (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
  double tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}

double _Complex __shfl_xor_cd(double _Complex var, int lane_mask,
                              int width = mapping::getWarpSize()) {
  __real__(var) = __shfl_xor_d(__real__(var), lane_mask, width);
  __imag__(var) = __shfl_xor_d(__imag__(var), lane_mask, width);
  return var;
}
float _Complex __shfl_xor_cf(float _Complex var, int lane_mask,
                             int width = mapping::getWarpSize()) {
  __real__(var) = __shfl_xor_f(__real__(var), lane_mask, width);
  __imag__(var) = __shfl_xor_f(__imag__(var), lane_mask, width);
  return var;
}
#pragma omp end declare variant

#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

int __shfl_xor_int(int var, int lane_mask, int width) {
  int c = ((32 - width) << 8) | 0x1f;
  return __nvvm_shfl_sync_bfly_i32(0xFFFFFFFF, var, lane_mask, c);
}
float __shfl_xor_f(float var, int lane_mask,
                   int width = mapping::getWarpSize()) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_xor_int(tmp.i, lane_mask, width);
  return tmp.f;
}
double __shfl_xor_d(double var, int laneMask,
                    int width = mapping::getWarpSize()) {
  unsigned lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
  hi = __shfl_xor_int(hi, laneMask, width);
  lo = __shfl_xor_int(lo, laneMask, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
  return var;
}
double _Complex __shfl_xor_cd(double _Complex var, int lane_mask,
                              int width = mapping::getWarpSize()) {
  __real__(var) = __shfl_xor_d(__real__(var), lane_mask, width);
  __imag__(var) = __shfl_xor_d(__imag__(var), lane_mask, width);
  return var;
}
float _Complex __shfl_xor_cf(float _Complex var, int lane_mask,
                             int width = mapping::getWarpSize()) {
  __real__(var) = __shfl_xor_f(__real__(var), lane_mask, width);
  __imag__(var) = __shfl_xor_f(__imag__(var), lane_mask, width);
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
double __shfl_xor(_d_tag tag, double var, int lane_mask, int width) {
  return __shfl_xor_d(var, lane_mask, width);
}
float __shfl_xor(_f_tag tag, float var, int lane_mask, int width) {
  return __shfl_xor_f(var, lane_mask, width);
}
double _Complex __shfl_xor(_cd_tag tag, double _Complex var, int lane_mask,
                           int width) {
  return __shfl_xor_cd(var, lane_mask, width);
}
float _Complex __shfl_xor(_cf_tag tag, float _Complex var, int lane_mask,
                          int width) {
  return __shfl_xor_cf(var, lane_mask, width);
}
int __shfl_xor(_i_tag tag, int var, int lane_mask, int width) {
  return __shfl_xor_int(var, lane_mask, width);
}
unsigned int __shfl_xor(_ui_tag tag, unsigned int var, int lane_mask,
                        int width) {
  return __shfl_xor_int(var, lane_mask, width);
}
long __shfl_xor(_l_tag tag, long var, int lane_mask, int width) {
  return __shfl_xor_d(var, lane_mask, width);
}
unsigned long __shfl_xor(_ul_tag tag, unsigned long var, int lane_mask,
                         int width) {
  return __shfl_xor_d(var, lane_mask, width);
}
template <typename T> T __shfl_xor(T var, int lane_mask, int width) {
  typedef typename __dispatch_tag<T>::type tag;
  return __shfl_xor(tag(), var, lane_mask, width);
}

double __get_low(_d_tag) { return __XTEAM_LOW_DOUBLE; }
float __get_low(_f_tag) { return __XTEAM_LOW_FLOAT; }
int __get_low(_i_tag) { return __XTEAM_LOW_INT; }
long __get_low(_l_tag) { return __XTEAM_LOW_LONG; }
unsigned int __get_low(_ui_tag) { return __XTEAM_LOW_UINT; }
unsigned long __get_low(_ul_tag) { return __XTEAM_LOW_ULONG; }
template <typename T> T __get_low() {
  typedef typename __dispatch_tag<T>::type tag;
  return __get_low(tag());
}

double __get_max(_d_tag) { return __XTEAM_MAX_DOUBLE; }
float __get_max(_f_tag) { return __XTEAM_MAX_FLOAT; }
int __get_max(_i_tag) { return __XTEAM_MAX_INT; }
long __get_max(_l_tag) { return __XTEAM_MAX_LONG; }
unsigned int __get_max(_ui_tag) { return __XTEAM_MAX_UINT; }
unsigned long __get_max(_ul_tag) { return __XTEAM_MAX_ULONG; }
template <typename T> T __get_max() {
  typedef typename __dispatch_tag<T>::type tag;
  return __get_max(tag());
}

static uint32_t teams_done = 0;
static volatile bool SHARED(__is_last_team);

template <typename T> void __local_xteam_sum(T inval, T *result_value) {
  T val;
#pragma omp allocate(val) allocator(omp_thread_mem_alloc)
  val = inval;

  const int32_t omp_thread_num = mapping::getThreadIdInBlock();
  const int32_t omp_team_num = mapping::getBlockId();
  const int32_t wave_num = mapping::getWarpId();         // 0 15
  const int32_t lane_num = mapping::getThreadIdInWarp(); //  0 63
  const int32_t wsz = mapping::getWarpSize();
  constexpr int32_t NumThreads = __XTEAM_NTHREADS;
  const int32_t NumTeams = mapping::getNumberOfBlocks();
  const int32_t num_waves = NumThreads / wsz;
  static __XTEAM_SHARED T psums[__XTEAM_MAXW_PERTEAM];

  // FIXME: This may be deletable
  if (omp_thread_num == 0) {
    T teamval = T(0);
    __xteam_set_mem(omp_team_num, &teamval, sizeof(T), 0);
  }

  // Reduce each wavefront to psums[wave_num]
  __kmpc_impl_syncthreads();
  for (unsigned int offset = wsz / 2; offset > 0; offset >>= 1)
    val += __shfl_xor<T>(val, offset, wsz);

  if (lane_num == 0)
    psums[wave_num] = val;

  for (unsigned int offset = num_waves / 2; offset > 0; offset >>= 1) {
    __kmpc_impl_syncthreads();
    if (wave_num < offset) {
      psums[wave_num] += psums[wave_num + offset];
    }
  }
  __is_last_team = false;
  if (omp_thread_num == 0) {
    T teamval = psums[0];
    __xteam_set_mem(omp_team_num, &teamval, sizeof(T), 0);
    uint32_t td = atomic::inc(&teams_done, NumTeams - 1u, __ATOMIC_SEQ_CST);
    if (td == (NumTeams - 1u))
      __is_last_team = true;
  }

  // Sync so all threads from last team know they are in the last team
  __kmpc_impl_syncthreads();

  if (__is_last_team) {
    // All threads from last completed team enter here.
    val = T(0);
    if (omp_thread_num < NumTeams) {
      T teamval;
      __xteam_get_mem(omp_thread_num, &teamval, sizeof(T), 0);
      val = teamval;
    }
    for (unsigned int offset = wsz / 2; offset > 0; offset >>= 1) {
      val += __shfl_xor<T>(val, offset, wsz);
    }
    if (lane_num == 0) {
      psums[wave_num] = val;
    }
    if (omp_thread_num == 0) {
      unsigned int usableWaves = ((NumTeams - 1) / wsz) + 1;
      for (unsigned int kk = 1; kk < usableWaves; kk++)
        psums[0] += psums[kk];
      *result_value = psums[0];
    }
  }
}

template <typename T> void __local_xteam_max(T inval, T *result_value) {
  T val;
#pragma omp allocate(val) allocator(omp_thread_mem_alloc)
  val = inval;

  const int32_t omp_thread_num = mapping::getThreadIdInBlock();
  const int32_t omp_team_num = mapping::getBlockId();
  const int32_t wave_num = mapping::getWarpId();         // 0 15
  const int32_t lane_num = mapping::getThreadIdInWarp(); //  0 63
  const int32_t wsz = mapping::getWarpSize();
  constexpr int32_t NumThreads = __XTEAM_NTHREADS;
  const int32_t NumTeams = mapping::getNumberOfBlocks();
  const int32_t num_waves = NumThreads / wsz;
  static __XTEAM_SHARED T psums[__XTEAM_MAXW_PERTEAM];

  if (omp_thread_num == 0) {
    T teamval = __get_low<T>();
    __xteam_set_mem(omp_team_num, &teamval, sizeof(T), 0);
  }

  // Reduce each wavefront to psums[wave_num]
  __kmpc_impl_syncthreads();
  for (unsigned int offset = wsz / 2; offset > 0; offset >>= 1) {
    T otherval = __shfl_xor<T>(val, offset, wsz);
    if (otherval > val)
      val = otherval;
  }

  if (lane_num == 0)
    psums[wave_num] = val;

  for (unsigned int offset = num_waves / 2; offset > 0; offset >>= 1) {
    __kmpc_impl_syncthreads();
    if (wave_num < offset) {
      T otherval = psums[wave_num + offset];
      if (otherval > psums[wave_num])
        psums[wave_num] = otherval;
    }
  }
  __is_last_team = false;
  if (omp_thread_num == 0) {
    T teamval = psums[0];
    __xteam_set_mem(omp_team_num, &teamval, sizeof(T), 0);
    uint32_t td = atomic::inc(&teams_done, NumTeams - 1u, __ATOMIC_SEQ_CST);
    if (td == (NumTeams - 1u))
      __is_last_team = true;
  }

  // Sync so all threads from last team know they are in the last team
  __kmpc_impl_syncthreads();

  if (__is_last_team) {
    // All threads from last completed team enter here.
    val = __get_low<T>();
    if (omp_thread_num < NumTeams) {
      T teamval;
      __xteam_get_mem(omp_thread_num, &teamval, sizeof(T), 0);
      val = teamval;
    }
    for (unsigned int offset = wsz / 2; offset > 0; offset >>= 1) {
      T otherval = __shfl_xor<T>(val, offset, wsz);
      if (otherval > val)
        val = otherval;
    }
    if (lane_num == 0) {
      psums[wave_num] = val;
    }
    if (omp_thread_num == 0) {
      unsigned int usableWaves = ((NumTeams - 1) / wsz) + 1;
      for (unsigned int kk = 1; kk < usableWaves; kk++) {
        T otherval = psums[kk];
        if (otherval > psums[0])
          psums[0] = otherval;
      }
      *result_value = psums[0];
    }
  }
}

template <typename T> void __local_xteam_min(T inval, T *result_value) {
  T val;
#pragma omp allocate(val) allocator(omp_thread_mem_alloc)
  val = inval;

  const int32_t omp_thread_num = mapping::getThreadIdInBlock();
  const int32_t omp_team_num = mapping::getBlockId();
  const int32_t wave_num = mapping::getWarpId();         // 0 15
  const int32_t lane_num = mapping::getThreadIdInWarp(); //  0 63
  const int32_t wsz = mapping::getWarpSize();
  constexpr int32_t NumThreads = __XTEAM_NTHREADS;
  const int32_t NumTeams = mapping::getNumberOfBlocks();
  const int32_t num_waves = NumThreads / wsz;
  static __XTEAM_SHARED T psums[__XTEAM_MAXW_PERTEAM];

  if (omp_thread_num == 0) {
    T teamval = __get_max<T>();
    __xteam_set_mem(omp_team_num, &teamval, sizeof(T), 0);
  }

  // Reduce each wavefront to psums[wave_num]
  __kmpc_impl_syncthreads();
  for (unsigned int offset = wsz / 2; offset > 0; offset >>= 1) {
    T otherval = __shfl_xor<T>(val, offset, wsz);
    if (otherval < val)
      val = otherval;
  }

  if (lane_num == 0)
    psums[wave_num] = val;

  for (unsigned int offset = num_waves / 2; offset > 0; offset >>= 1) {
    __kmpc_impl_syncthreads();
    if (wave_num < offset) {
      T otherval = psums[wave_num + offset];
      if (otherval < psums[wave_num])
        psums[wave_num] = otherval;
    }
  }
  __is_last_team = false;
  if (omp_thread_num == 0) {
    T teamval = psums[0];
    __xteam_set_mem(omp_team_num, &teamval, sizeof(T), 0);
    uint32_t td = atomic::inc(&teams_done, NumTeams - 1u, __ATOMIC_SEQ_CST);
    if (td == (NumTeams - 1u))
      __is_last_team = true;
  }

  // Sync so all threads from last team know they are in the last team
  __kmpc_impl_syncthreads();

  if (__is_last_team) {
    // All threads from last completed team enter here.
    val = __get_max<T>();
    if (omp_thread_num < NumTeams) {
      T teamval;
      __xteam_get_mem(omp_thread_num, &teamval, sizeof(T), 0);
      val = teamval;
    }
    for (unsigned int offset = wsz / 2; offset > 0; offset >>= 1) {
      T otherval = __shfl_xor<T>(val, offset, wsz);
      if (otherval < val)
        val = otherval;
    }
    if (lane_num == 0) {
      psums[wave_num] = val;
    }
    if (omp_thread_num == 0) {
      unsigned int usableWaves = ((NumTeams - 1) / wsz) + 1;
      for (unsigned int kk = 1; kk < usableWaves; kk++) {
        T otherval = psums[kk];
        psums[0] = (otherval < psums[0]) ? otherval : psums[0];
      }
      *result_value = psums[0];
    }
  }
}

//  Calls to these __kmpc extern C functions are created in clang codegen
//  for FORTRAH, c, and C++. They may also be used for sumulation and teeting.
//  The headers for these extern C fns are in ../include/Interface.h
extern "C" {
void __kmpc_xteam_sum_d(double inval, double *result_value) {
  __local_xteam_sum<double>(inval, result_value);
}
void __kmpc_xteam_sum_f(float inval, float *result_value) {
  __local_xteam_sum<float>(inval, result_value);
}
void __kmpc_xteam_sum_cd(double _Complex inval, double _Complex *result_value) {
  __local_xteam_sum<double _Complex>(inval, result_value);
}
void __kmpc_xteam_sum_cf(float _Complex inval, float _Complex *result_value) {
  __local_xteam_sum<float _Complex>(inval, result_value);
}
void __kmpc_xteam_sum_i(int inval, int *result_value) {
  __local_xteam_sum<int>(inval, result_value);
}
void __kmpc_xteam_sum_ui(unsigned int inval, unsigned int *result_value) {
  __local_xteam_sum<unsigned int>(inval, result_value);
}
void __kmpc_xteam_sum_l(long inval, long *result_value) {
  __local_xteam_sum<long>(inval, result_value);
}
void __kmpc_xteam_sum_ul(unsigned long inval, unsigned long *result_value) {
  __local_xteam_sum<unsigned long>(inval, result_value);
}

// Note: One may not compare/order complex numbers so no complex max or min

void __kmpc_xteam_max_d(double inval, double *result_value) {
  __local_xteam_max<double>(inval, result_value);
}
void __kmpc_xteam_max_f(float inval, float *result_value) {
  __local_xteam_max<float>(inval, result_value);
}
void __kmpc_xteam_max_i(int inval, int *result_value) {
  __local_xteam_max<int>(inval, result_value);
}
void __kmpc_xteam_max_ui(unsigned int inval, unsigned int *result_value) {
  __local_xteam_max<unsigned int>(inval, result_value);
}
void __kmpc_xteam_max_l(long inval, long *result_value) {
  __local_xteam_max<long>(inval, result_value);
}
void __kmpc_xteam_max_ul(unsigned long inval, unsigned long *result_value) {
  __local_xteam_max<unsigned long>(inval, result_value);
}

void __kmpc_xteam_min_d(double inval, double *result_value) {
  __local_xteam_min<double>(inval, result_value);
}
void __kmpc_xteam_min_f(float inval, float *result_value) {
  __local_xteam_min<float>(inval, result_value);
}
void __kmpc_xteam_min_i(int inval, int *result_value) {
  __local_xteam_min<int>(inval, result_value);
}
void __kmpc_xteam_min_ui(unsigned int inval, unsigned int *result_value) {
  __local_xteam_min<unsigned int>(inval, result_value);
}
void __kmpc_xteam_min_l(long inval, long *result_value) {
  __local_xteam_min<long>(inval, result_value);
}
void __kmpc_xteam_min_ul(unsigned long inval, unsigned long *result_value) {
  __local_xteam_min<unsigned long>(inval, result_value);
}

} // end extern "C"

#pragma omp end declare target
