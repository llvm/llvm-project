//===------------- target_interface.h - Target interfaces --------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains interfaces that must be implemented by each target.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_TARGET_INTERFACE_H_
#define _OMPTARGET_TARGET_INTERFACE_H_

#include <stdint.h>

#include "DeviceEnvironment.h"
#include "target_impl.h"

// Calls to the NVPTX layer (assuming 1D layout)
EXTERN int __kmpc_get_hardware_thread_id_in_block();
EXTERN int GetBlockIdInKernel();
EXTERN NOINLINE int __kmpc_get_hardware_num_blocks();
EXTERN NOINLINE int __kmpc_get_hardware_num_threads_in_block();
EXTERN unsigned GetWarpId();
EXTERN unsigned GetLaneId();

// Atomics
uint32_t __kmpc_atomic_add(uint32_t *, uint32_t);
uint32_t __kmpc_atomic_inc(uint32_t *, uint32_t);
uint32_t __kmpc_atomic_max(uint32_t *, uint32_t);
uint32_t __kmpc_atomic_exchange(uint32_t *, uint32_t);
uint32_t __kmpc_atomic_cas(uint32_t *, uint32_t, uint32_t);
static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "");
unsigned long long __kmpc_atomic_exchange(unsigned long long *,
                                          unsigned long long);
unsigned long long __kmpc_atomic_add(unsigned long long *, unsigned long long);

// Locks
EXTERN void __kmpc_impl_init_lock(omp_lock_t *lock);
EXTERN void __kmpc_impl_destroy_lock(omp_lock_t *lock);
EXTERN void __kmpc_impl_set_lock(omp_lock_t *lock);
EXTERN void __kmpc_impl_unset_lock(omp_lock_t *lock);
EXTERN int __kmpc_impl_test_lock(omp_lock_t *lock);

EXTERN void __kmpc_impl_threadfence();
EXTERN void __kmpc_impl_threadfence_block();
EXTERN void __kmpc_impl_threadfence_system();

EXTERN double __kmpc_impl_get_wtick();
EXTERN double __kmpc_impl_get_wtime();

EXTERN void __kmpc_impl_unpack(uint64_t val, uint32_t &lo, uint32_t &hi);
EXTERN uint64_t __kmpc_impl_pack(uint32_t lo, uint32_t hi);
EXTERN __kmpc_impl_lanemask_t __kmpc_impl_lanemask_lt();
EXTERN __kmpc_impl_lanemask_t __kmpc_impl_lanemask_gt();
EXTERN uint32_t __kmpc_impl_smid();

EXTERN __kmpc_impl_lanemask_t __kmpc_impl_activemask();

EXTERN void __kmpc_impl_syncthreads();
EXTERN void __kmpc_impl_syncwarp(__kmpc_impl_lanemask_t Mask);

// Kernel initialization
EXTERN void __kmpc_impl_target_init();

// Memory
EXTERN void *__kmpc_impl_malloc(size_t);
EXTERN void __kmpc_impl_free(void *);

// Barrier until num_threads arrive.
EXTERN void __kmpc_impl_named_sync(uint32_t num_threads);

extern DeviceEnvironmentTy omptarget_device_environment;

#endif // _OMPTARGET_TARGET_INTERFACE_H_
