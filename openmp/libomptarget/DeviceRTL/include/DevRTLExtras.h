//===---------- DevRTLExtras.h - OpenMP types --------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Additional OpenMP type definitions, in conjunction with Types.h.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_DEVICERTL_INCLUDE_DEVRTLEXTRAS_H
#define OPENMP_LIBOMPTARGET_DEVICERTL_INCLUDE_DEVRTLEXTRAS_H

/// Base type declarations for freestanding mode
///
///{
using uint64_t = unsigned long;
// TODO: Properly implement this
using uintptr_t = uint64_t;
///}

/// Macros for allocating variables in different address spaces.
///{

// Follows the pattern in interface.h
// Same definitions as in host runtime
// TODO: move these definitions to a common
// place between host and device runtimes (e.g. in LLVM)
typedef enum omp_memspace_handle_t {
  omp_default_mem_space = 0,
  omp_large_cap_mem_space = 1,
  omp_const_mem_space = 2,
  omp_high_bw_mem_space = 3,
  omp_low_lat_mem_space = 4,
  llvm_omp_target_host_mem_space = 100,
  llvm_omp_target_shared_mem_space = 101,
  llvm_omp_target_device_mem_space = 102,
  KMP_MEMSPACE_MAX_HANDLE = ~(0u)
} omp_memspace_handle_t;

typedef enum {
  omp_atk_sync_hint = 1,
  omp_atk_alignment = 2,
  omp_atk_access = 3,
  omp_atk_pool_size = 4,
  omp_atk_fallback = 5,
  omp_atk_fb_data = 6,
  omp_atk_pinned = 7,
  omp_atk_partition = 8
} omp_alloctrait_key_t;

typedef enum {
  omp_atv_false = 0,
  omp_atv_true = 1,
  omp_atv_contended = 3,
  omp_atv_uncontended = 4,
  omp_atv_serialized = 5,
  omp_atv_sequential = omp_atv_serialized, // (deprecated)
  omp_atv_private = 6,
  omp_atv_all = 7,
  omp_atv_thread = 8,
  omp_atv_pteam = 9,
  omp_atv_cgroup = 10,
  omp_atv_default_mem_fb = 11,
  omp_atv_null_fb = 12,
  omp_atv_abort_fb = 13,
  omp_atv_allocator_fb = 14,
  omp_atv_environment = 15,
  omp_atv_nearest = 16,
  omp_atv_blocked = 17,
  omp_atv_interleaved = 18
} omp_alloctrait_value_t;
#define omp_atv_default ((uintptr_t)-1)

typedef struct {
  omp_alloctrait_key_t key;
  uintptr_t value;
} omp_alloctrait_t;

// Attribute to keep alive certain definition for the bitcode library.
#ifdef LIBOMPTARGET_BC_TARGET
#define KEEP_ALIVE __attribute__((used, retain))
#else
#define KEEP_ALIVE
#endif

///}

#endif // OPENMP_LIBOMPTARGET_DEVICERTL_INCLUDE_DEVRTLEXTRAS_H
