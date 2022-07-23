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

using namespace _OMP;

#pragma omp declare target

// Support for __xteam_mem interface which is for cross team communication.
// Currently only enough global memory for a single double value per team is
// statically created. For future growth , please make slotsz so team slot
// remains 8-byte aligned.
constexpr uint64_t __xteam_mem_slotsz = 8;
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
#pragma omp end declare variant
// } // end impl namespace

static uint32_t teams_done = 0;
static volatile bool SHARED(__is_last_team);

extern "C" {
void __kmpc_xteam_sum_d(double inval, double *result_value) {
  double val;
#pragma omp allocate(val) allocator(omp_thread_mem_alloc)
  val = inval;

  const int32_t omp_thread_num = mapping::getThreadIdInBlock();
  const int32_t omp_team_num = mapping::getBlockId();
  const int32_t wave_num = mapping::getWarpId();         // 0 15
  const int32_t lane_num = mapping::getThreadIdInWarp(); //  0 63
  const int32_t wsz = mapping::getWarpSize();
  const int32_t NumThreads = 1024; // omp_get_num_threads() is wrong here
  const int32_t NumTeams = mapping::getNumberOfBlocks();
  const int32_t num_waves = NumThreads / wsz;
  // Allocate enough share for possible 32 waves per team (nvidia)
  static volatile __attribute__((address_space(3))) double psums[32];

  if (omp_thread_num == 0) {
    double teamval = 0.0;
    __xteam_set_mem(omp_team_num, &teamval, sizeof(double), 0);
  }

  // Reduce each wavefront to psums[wave_num]
  __kmpc_impl_syncthreads();
  for (unsigned int offset = wsz / 2; offset > 0; offset >>= 1)
    val += __shfl_xor_d(val, offset, wsz);

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
    double teamval = psums[0];
    __xteam_set_mem(omp_team_num, &teamval, sizeof(double), 0);
    uint32_t td = atomic::inc(&teams_done, NumTeams - 1u, __ATOMIC_SEQ_CST);
    if (td == (NumTeams - 1u))
      __is_last_team = true;
  }

  // Sync so all threads from last team know they are in the last team
  __kmpc_impl_syncthreads();

  if (__is_last_team) {
    // All threads from last completed team enter here.
    val = double(0);
    if (omp_thread_num < NumTeams) {
      double teamval;
      __xteam_get_mem(omp_thread_num, &teamval, sizeof(double), 0);
      val = teamval;
    }
    for (unsigned int offset = wsz / 2; offset > 0; offset >>= 1) {
      val += __shfl_xor_d(val, offset, wsz);
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
void __kmpc_xteam_sum_f(float inval, float *result_value) {
  float val;
#pragma omp allocate(val) allocator(omp_thread_mem_alloc)
  val = inval;

  const int32_t omp_thread_num = mapping::getThreadIdInBlock();
  const int32_t omp_team_num = mapping::getBlockId();
  const int32_t wave_num = mapping::getWarpId();         // 0 15
  const int32_t lane_num = mapping::getThreadIdInWarp(); //  0 63
  const int32_t wsz = mapping::getWarpSize();
  const int32_t NumThreads = 1024; // omp_get_num_threads() is wrong here
  const int32_t NumTeams = mapping::getNumberOfBlocks();
  const int32_t num_waves = NumThreads / wsz;
  // Allocate enough share for possible 32 waves per team (nvidia)
  static volatile __attribute__((address_space(3))) float psums[32];

  if (omp_thread_num == 0) {
    float teamval = float(0);
    __xteam_set_mem(omp_team_num, &teamval, sizeof(float), 0);
  }

  // Reduce each wavefront to psums[wave_num]
  __kmpc_impl_syncthreads();
  for (unsigned int offset = wsz / 2; offset > 0; offset >>= 1)
    val += __shfl_xor_d(val, offset, wsz);

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
    float teamval = psums[0];
    __xteam_set_mem(omp_team_num, &teamval, sizeof(float), 0);
    uint32_t td = atomic::inc(&teams_done, NumTeams - 1u, __ATOMIC_SEQ_CST);
    if (td == (NumTeams - 1u))
      __is_last_team = true;
  }

  // Sync so all threads from last team know they are in the last team
  __kmpc_impl_syncthreads();

  if (__is_last_team) {
    // All threads from last completed team enter here.
    val = float(0);
    if (omp_thread_num < NumTeams) {
      float teamval;
      __xteam_get_mem(omp_thread_num, &teamval, sizeof(float), 0);
      val = teamval;
    }
    for (unsigned int offset = wsz / 2; offset > 0; offset >>= 1) {
      val += __shfl_xor_d(val, offset, wsz);
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

} // end extern "C"

#pragma omp end declare target
