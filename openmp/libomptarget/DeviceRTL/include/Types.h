//===---------- Types.h - OpenMP types ---------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_TYPES_H
#define OMPTARGET_TYPES_H

#include "DevRTLExtras.h"

// Tell the compiler that we do not have any "call-like" inline assembly in the
// device rutime. That means we cannot have inline assembly which will call
// another function but only inline assembly that performs some operation or
// side-effect and then continues execution with something on the existing call
// stack.
//
// TODO: Find a good place for this
#pragma omp assumes ext_no_call_asm

/// Base type declarations for freestanding mode
///
///{
using int8_t = char;
using uint8_t = unsigned char;
using int16_t = short;
using uint16_t = unsigned short;
using int32_t = int;
using uint32_t = unsigned int;
using int64_t = long;
using uint64_t = unsigned long;
using size_t = decltype(sizeof(char));
// TODO: Properly implement this
using intptr_t = int64_t;
using uintptr_t = uint64_t;

static_assert(sizeof(int8_t) == 1, "type size mismatch");
static_assert(sizeof(uint8_t) == 1, "type size mismatch");
static_assert(sizeof(int16_t) == 2, "type size mismatch");
static_assert(sizeof(uint16_t) == 2, "type size mismatch");
static_assert(sizeof(int32_t) == 4, "type size mismatch");
static_assert(sizeof(uint32_t) == 4, "type size mismatch");
static_assert(sizeof(int64_t) == 8, "type size mismatch");
static_assert(sizeof(uint64_t) == 8, "type size mismatch");
///}

enum omp_proc_bind_t {
  omp_proc_bind_false = 0,
  omp_proc_bind_true = 1,
  omp_proc_bind_master = 2,
  omp_proc_bind_close = 3,
  omp_proc_bind_spread = 4
};

enum omp_sched_t {
  omp_sched_static = 1,  /* chunkSize >0 */
  omp_sched_dynamic = 2, /* chunkSize >0 */
  omp_sched_guided = 3,  /* chunkSize >0 */
  omp_sched_auto = 4,    /* no chunkSize */
};

enum kmp_sched_t {
  kmp_sched_static_chunk = 33,
  kmp_sched_static_nochunk = 34,
  kmp_sched_dynamic = 35,
  kmp_sched_guided = 36,
  kmp_sched_runtime = 37,
  kmp_sched_auto = 38,

  kmp_sched_static_balanced_chunk = 45,

  kmp_sched_static_ordered = 65,
  kmp_sched_static_nochunk_ordered = 66,
  kmp_sched_dynamic_ordered = 67,
  kmp_sched_guided_ordered = 68,
  kmp_sched_runtime_ordered = 69,
  kmp_sched_auto_ordered = 70,

  kmp_sched_distr_static_chunk = 91,
  kmp_sched_distr_static_nochunk = 92,
  kmp_sched_distr_static_chunk_sched_static_chunkone = 93,

  kmp_sched_default = kmp_sched_static_nochunk,
  kmp_sched_unordered_first = kmp_sched_static_chunk,
  kmp_sched_unordered_last = kmp_sched_auto,
  kmp_sched_ordered_first = kmp_sched_static_ordered,
  kmp_sched_ordered_last = kmp_sched_auto_ordered,
  kmp_sched_distribute_first = kmp_sched_distr_static_chunk,
  kmp_sched_distribute_last =
      kmp_sched_distr_static_chunk_sched_static_chunkone,

  /* Support for OpenMP 4.5 monotonic and nonmonotonic schedule modifiers.
   * Since we need to distinguish the three possible cases (no modifier,
   * monotonic modifier, nonmonotonic modifier), we need separate bits for
   * each modifier. The absence of monotonic does not imply nonmonotonic,
   * especially since 4.5 says that the behaviour of the "no modifier" case
   * is implementation defined in 4.5, but will become "nonmonotonic" in 5.0.
   *
   * Since we're passing a full 32 bit value, we can use a couple of high
   * bits for these flags; out of paranoia we avoid the sign bit.
   *
   * These modifiers can be or-ed into non-static schedules by the compiler
   * to pass the additional information. They will be stripped early in the
   * processing in __kmp_dispatch_init when setting up schedules, so
   * most of the code won't ever see schedules with these bits set.
   */
  kmp_sched_modifier_monotonic = (1 << 29),
  /**< Set if the monotonic schedule modifier was present */
  kmp_sched_modifier_nonmonotonic = (1 << 30),
/**< Set if the nonmonotonic schedule modifier was present */

#define SCHEDULE_WITHOUT_MODIFIERS(s)                                          \
  (enum kmp_sched_t)(                                                          \
      (s) & ~(kmp_sched_modifier_nonmonotonic | kmp_sched_modifier_monotonic))
#define SCHEDULE_HAS_MONOTONIC(s) (((s)&kmp_sched_modifier_monotonic) != 0)
#define SCHEDULE_HAS_NONMONOTONIC(s)                                           \
  (((s)&kmp_sched_modifier_nonmonotonic) != 0)
#define SCHEDULE_HAS_NO_MODIFIERS(s)                                           \
  (((s) & (kmp_sched_modifier_nonmonotonic | kmp_sched_modifier_monotonic)) == \
   0)

};

struct TaskDescriptorTy;
using TaskFnTy = int32_t (*)(int32_t global_tid, TaskDescriptorTy *taskDescr);
struct TaskDescriptorTy {
  void *Payload;
  TaskFnTy TaskFn;
};

#pragma omp begin declare variant match(device = {arch(amdgcn)})
using LaneMaskTy = uint64_t;
#pragma omp end declare variant

#pragma omp begin declare variant match(                                       \
        device = {arch(amdgcn)}, implementation = {extension(match_none)})
using LaneMaskTy = uint64_t;
#pragma omp end declare variant

namespace lanes {
enum : LaneMaskTy { All = ~(LaneMaskTy)0 };
} // namespace lanes

/// The ident structure that describes a source location. The struct is
/// identical to the one in the kmp.h file. We maintain the same data structure
/// for compatibility.
struct IdentTy {
  int32_t reserved_1;  /**<  might be used in Fortran; see above  */
  int32_t flags;       /**<  also f.flags; KMP_IDENT_xxx flags; KMP_IDENT_KMPC
                            identifies this union member  */
  int32_t reserved_2;  /**<  not really used in Fortran any more; see above */
  int32_t reserved_3;  /**<  source[4] in Fortran, do not use for C++  */
  char const *psource; /**<  String describing the source location.
                       The string is composed of semi-colon separated fields
                       which describe the source file, the function and a pair
                       of line numbers that delimit the construct. */
};

using __kmpc_impl_lanemask_t = LaneMaskTy;

using ParallelRegionFnTy = void *;

using CriticalNameTy = int32_t[8];

struct omp_lock_t {
  void *Lock;
};

using InterWarpCopyFnTy = void (*)(void *src, int32_t warp_num);
using ShuffleReductFnTy = void (*)(void *rhsData, int16_t lane_id,
                                   int16_t lane_offset, int16_t shortCircuit);
using ListGlobalFnTy = void (*)(void *buffer, int idx, void *reduce_data);

/// Macros for allocating variables in different address spaces.
///{

// Follows the pattern in interface.h
typedef enum omp_allocator_handle_t {
  omp_null_allocator = 0,
  omp_default_mem_alloc = 1,
  omp_large_cap_mem_alloc = 2,
  omp_const_mem_alloc = 3,
  omp_high_bw_mem_alloc = 4,
  omp_low_lat_mem_alloc = 5,
  omp_cgroup_mem_alloc = 6,
  omp_pteam_mem_alloc = 7,
  omp_thread_mem_alloc = 8,
  KMP_ALLOCATOR_MAX_HANDLE = ~(0U)
} omp_allocator_handle_t;

#define __PRAGMA(STR) _Pragma(#STR)
#define OMP_PRAGMA(STR) __PRAGMA(omp STR)

#define SHARED(NAME)                                                           \
  NAME [[clang::loader_uninitialized]];                                        \
  OMP_PRAGMA(allocate(NAME) allocator(omp_pteam_mem_alloc))

// TODO: clang should use address space 5 for omp_thread_mem_alloc, but right
//       now that's not the case.
#define THREAD_LOCAL(NAME)                                                     \
  [[clang::address_space(5)]] NAME [[clang::loader_uninitialized]]

// TODO: clang should use address space 4 for omp_const_mem_alloc, maybe it
//       does?
#define CONSTANT(NAME)                                                         \
  [[clang::address_space(4)]] NAME [[clang::loader_uninitialized]]

///}

#endif
