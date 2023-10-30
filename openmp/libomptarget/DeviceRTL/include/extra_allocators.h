//===---------- extra_allocators.h - OpenMP interface -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Additional OpenMP interface definitions, in conjunction with Interface.h.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_DEVICERTL_INCLUDE_EXTRA_ALLOCATORS_H
#define OPENMP_LIBOMPTARGET_DEVICERTL_INCLUDE_EXTRA_ALLOCATORS_H

#include "Types.h"
#include "Xteamr.h"

extern "C" {
/// Tasking
///
///{
void omp_fulfill_event(uint64_t);
///}

/// OpenMP 5.1 Memory Management routines (from libomp)
/// OpenMP allocator API is currently unimplemented, including traits.
/// All allocation routines will directly call the global memory allocation
/// routine and, consequently, omp_free will call device memory deallocation.
///
/// {
omp_allocator_handle_t omp_init_allocator(omp_memspace_handle_t m, int ntraits,
                                          omp_alloctrait_t traits[]);

void omp_destroy_allocator(omp_allocator_handle_t allocator);

void omp_set_default_allocator(omp_allocator_handle_t a);

omp_allocator_handle_t omp_get_default_allocator(void);

void *omp_alloc(uint64_t size,
                omp_allocator_handle_t allocator = omp_null_allocator);

void *omp_aligned_alloc(uint64_t align, uint64_t size,
                        omp_allocator_handle_t allocator = omp_null_allocator);

void *omp_calloc(uint64_t nmemb, uint64_t size,
                 omp_allocator_handle_t allocator = omp_null_allocator);

void *omp_aligned_calloc(uint64_t align, uint64_t nmemb, uint64_t size,
                         omp_allocator_handle_t allocator = omp_null_allocator);

void *omp_realloc(void *ptr, uint64_t size,
                  omp_allocator_handle_t allocator = omp_null_allocator,
                  omp_allocator_handle_t free_allocator = omp_null_allocator);

void omp_free(void *ptr, omp_allocator_handle_t allocator = omp_null_allocator);
/// }

/// CUDA exposes a native malloc/free API, while ROCm does not.
//// Any re-definitions of malloc/free delete the native CUDA
//// but they are necessary
#ifdef __AMDGCN__
void *malloc(uint64_t Size);
void free(void *Ptr);
#endif
} // extern "C"

extern "C" {
/// External interface to get the block size
uint32_t __kmpc_get_hardware_num_blocks();

/// Synchronization
///
///{
void __kmpc_impl_syncthreads();

void __kmpc_flush_acquire(IdentTy *Loc);

void __kmpc_flush_release(IdentTy *Loc);

void __kmpc_flush_acqrel(IdentTy *Loc);
///}

/// Tasking
///
///{
void *__kmpc_task_allow_completion_event(IdentTy *loc_ref, uint32_t gtid,
                                         TaskDescriptorTy *task);
///}

/// __init_ThreadDSTPtrPtr is defined in Workshare.cpp to initialize
/// the static LDS global variable ThreadDSTPtrPtr to 0.
/// It is called in Kernel.cpp at the end of initializeRuntime().
void __init_ThreadDSTPtrPtr();
} // extern "C"

/// Extra API exposed by ROCm
extern "C" {
int omp_ext_get_warp_id(void);
int omp_ext_get_lane_id(void);
int omp_ext_get_master_thread_id(void);
int omp_ext_get_smid(void);
int omp_ext_is_spmd_mode(void);
unsigned long long omp_ext_get_active_threads_mask(void);
} // extern "C"

#endif // OPENMP_LIBOMPTARGET_DEVICERTL_INCLUDE_EXTRA_ALLOCATORS_H
