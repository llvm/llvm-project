//===--------- allocator.h - OpenMP target memory allocator ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Macros for allocating variables in different address spaces.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_ALLOCATOR_H
#define OMPTARGET_ALLOCATOR_H

#if _OPENMP
// Follows the pattern in interface.h
// Clang sema checks this type carefully, needs to closely match that from omp.h
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

#define EXTERN_SHARED(NAME)                                                    \
  NAME;                                                                        \
  OMP_PRAGMA(allocate(NAME) allocator(omp_pteam_mem_alloc))
#endif

#endif // OMPTARGET_ALLOCATOR_H
