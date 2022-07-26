//===- Synchronization.h - OpenMP synchronization utilities ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
// Notified per clause 4(b) of the license.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_SYNCHRONIZATION_H
#define OMPTARGET_DEVICERTL_SYNCHRONIZATION_H

#include "Types.h"

namespace _OMP {

namespace synchronize {

/// Initialize the synchronization machinery. Must be called by all threads.
void init(bool IsSPMD);

/// Synchronize all threads in a warp identified by \p Mask.
void warp(LaneMaskTy Mask);

/// Synchronize all threads in a block.
void threads();

#pragma omp declare target

/// Flags used by master and workers to synchronize in generic state machine.
extern bool volatile omptarget_workers_done;
#pragma omp allocate(omptarget_workers_done) allocator(omp_pteam_mem_alloc)

extern bool volatile omptarget_master_ready;
#pragma omp allocate(omptarget_master_ready) allocator(omp_pteam_mem_alloc)

#pragma omp end declare target

/// Synchronize workers with master at the beginning of a parallel region
/// in generic mode.
void workersStartBarrier();

/// Synchronize workers with master at the end of a parallel region
/// in generic mode.
void workersDoneBarrier();

/// Synchronizing threads is allowed even if they all hit different instances of
/// `synchronize::threads()`. However, `synchronize::threadsAligned()` is more
/// restrictive in that it requires all threads to hit the same instance.
///{
#pragma omp begin assumes ext_aligned_barrier

/// Synchronize all threads in a block, they are are reaching the same
/// instruction (hence all threads in the block are "aligned").
void threadsAligned();

#pragma omp end assumes
///}

} // namespace synchronize

namespace fence {

/// Memory fence with \p Ordering semantics for the team.
void team(int Ordering);

/// Memory fence with \p Ordering semantics for the contention group.
void kernel(int Ordering);

/// Memory fence with \p Ordering semantics for the system.
void system(int Ordering);

} // namespace fence

namespace atomic {

/// Atomically load \p Addr with \p Ordering semantics.
uint32_t load(uint32_t *Addr, int Ordering);

/// Atomically store \p V to \p Addr with \p Ordering semantics.
void store(uint32_t *Addr, uint32_t V, int Ordering);

/// Atomically increment \p *Addr and wrap at \p V with \p Ordering semantics.
uint32_t inc(uint32_t *Addr, uint32_t V, int Ordering);

/// Atomically add \p V to \p *Addr with \p Ordering semantics.
uint32_t add(uint32_t *Addr, uint32_t V, int Ordering);

/// Atomically add \p V to \p *Addr with \p Ordering semantics.
uint64_t add(uint64_t *Addr, uint64_t V, int Ordering);

} // namespace atomic

} // namespace _OMP

#endif
