//===- Synchronization.cpp - OpenMP Device synchronization API ---- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Include all synchronization.
//
//===----------------------------------------------------------------------===//

#include "Synchronization.h"

#include "Debug.h"
#include "DeviceTypes.h"
#include "DeviceUtils.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"

using namespace ompx;

namespace impl {

/// Atomics
///
///{
///}

/// AMDGCN Implementation
///
///{
#ifdef __AMDGPU__

[[clang::loader_uninitialized]] static Local<uint32_t> namedBarrierTracker;

void namedBarrierInit() {
  // Don't have global ctors, and shared memory is not zero init
  atomic::store(&namedBarrierTracker, 0u, atomic::release);
}

void namedBarrier() {
  uint32_t NumThreads = omp_get_num_threads();
  // assert(NumThreads % 32 == 0);

  uint32_t WarpSize = mapping::getWarpSize();
  uint32_t NumWaves = NumThreads / WarpSize;

  fence::team(atomic::acquire);

  // named barrier implementation for amdgcn.
  // Uses two 16 bit unsigned counters. One for the number of waves to have
  // reached the barrier, and one to count how many times the barrier has been
  // passed. These are packed in a single atomically accessed 32 bit integer.
  // Low bits for the number of waves, assumed zero before this call.
  // High bits to count the number of times the barrier has been passed.

  // precondition: NumWaves != 0;
  // invariant: NumWaves * WarpSize == NumThreads;
  // precondition: NumWaves < 0xffffu;

  // Increment the low 16 bits once, using the lowest active thread.
  if (mapping::isLeaderInWarp()) {
    uint32_t load = atomic::add(&namedBarrierTracker, 1,
                                atomic::relaxed); // commutative

    // Record the number of times the barrier has been passed
    uint32_t generation = load & 0xffff0000u;

    if ((load & 0x0000ffffu) == (NumWaves - 1)) {
      // Reached NumWaves in low bits so this is the last wave.
      // Set low bits to zero and increment high bits
      load += 0x00010000u; // wrap is safe
      load &= 0xffff0000u; // because bits zeroed second

      // Reset the wave counter and release the waiting waves
      atomic::store(&namedBarrierTracker, load, atomic::relaxed);
    } else {
      // more waves still to go, spin until generation counter changes
      do {
        __builtin_amdgcn_s_sleep(0);
        load = atomic::load(&namedBarrierTracker, atomic::relaxed);
      } while ((load & 0xffff0000u) == generation);
    }
  }
  fence::team(atomic::release);
}

void syncWarp(__kmpc_impl_lanemask_t) {
  // This is a no-op on current AMDGPU hardware but it is used by the optimizer
  // to enforce convergent behaviour between control flow graphs.
  __builtin_amdgcn_wave_barrier();
}

void syncThreadsAligned(atomic::OrderingTy Ordering) {
  synchronize::threads(Ordering);
}

// TODO: Don't have wavefront lane locks. Possibly can't have them.
void unsetLock(omp_lock_t *) { __builtin_trap(); }
int testLock(omp_lock_t *) { __builtin_trap(); }
void initLock(omp_lock_t *) { __builtin_trap(); }
void destroyLock(omp_lock_t *) { __builtin_trap(); }
void setLock(omp_lock_t *) { __builtin_trap(); }

constexpr uint32_t UNSET = 0;
constexpr uint32_t SET = 1;

void unsetCriticalLock(omp_lock_t *Lock) {
  [[maybe_unused]] uint32_t before =
      atomicExchange((uint32_t *)Lock, UNSET, atomic::acq_rel);
}

void setCriticalLock(omp_lock_t *Lock) {
  uint64_t LowestActiveThread = utils::ffs(mapping::activemask()) - 1;
  if (mapping::getThreadIdInWarp() == LowestActiveThread) {
    fence::kernel(atomic::release);
    while (
        !cas((uint32_t *)Lock, UNSET, SET, atomic::relaxed, atomic::relaxed)) {
      __builtin_amdgcn_s_sleep(32);
    }
    fence::kernel(atomic::acquire);
  }
}

#endif
///}

/// NVPTX Implementation
///
///{
#ifdef __NVPTX__

void namedBarrierInit() {}

void namedBarrier() {
  uint32_t NumThreads = omp_get_num_threads();
  ASSERT(NumThreads % 32 == 0, nullptr);

  // The named barrier for active parallel threads of a team in an L1 parallel
  // region to synchronize with each other.
  constexpr int BarrierNo = 7;
  __nvvm_barrier_sync_cnt(BarrierNo, NumThreads);
}

void syncThreadsAligned(atomic::OrderingTy Ordering) { __syncthreads(); }

constexpr uint32_t OMP_SPIN = 1000;
constexpr uint32_t UNSET = 0;
constexpr uint32_t SET = 1;

void unsetLock(omp_lock_t *Lock) {
  [[maybe_unused]] uint32_t before = atomicExchange(
      reinterpret_cast<uint32_t *>(Lock), UNSET, atomic::seq_cst);
}

int testLock(omp_lock_t *Lock) {
  return atomic::add(reinterpret_cast<uint32_t *>(Lock), 0u, atomic::seq_cst);
}

void initLock(omp_lock_t *Lock) { unsetLock(Lock); }

void destroyLock(omp_lock_t *Lock) { unsetLock(Lock); }

void setLock(omp_lock_t *Lock) {
  // TODO: not sure spinning is a good idea here..
  while (atomic::cas(reinterpret_cast<uint32_t *>(Lock), UNSET, SET,
                     atomic::seq_cst, atomic::seq_cst) != UNSET) {
    int32_t start = __nvvm_read_ptx_sreg_clock();
    int32_t now;
    for (;;) {
      now = __nvvm_read_ptx_sreg_clock();
      int32_t cycles = now > start ? now - start : now + (0xffffffff - start);
      if (cycles >= OMP_SPIN * mapping::getBlockIdInKernel()) {
        break;
      }
    }
  } // wait for 0 to be the read value
}

void unsetCriticalLock(omp_lock_t *Lock) { unsetLock(Lock); }

void setCriticalLock(omp_lock_t *Lock) { setLock(Lock); }

#endif
///}

} // namespace impl

void synchronize::init(bool IsSPMD) {
  if (!IsSPMD)
    impl::namedBarrierInit();
}

void synchronize::threadsAligned(atomic::OrderingTy Ordering) {
  impl::syncThreadsAligned(Ordering);
}

void unsetCriticalLock(omp_lock_t *Lock) { impl::unsetLock(Lock); }

void setCriticalLock(omp_lock_t *Lock) { impl::setLock(Lock); }

extern "C" {
void __kmpc_ordered(IdentTy *Loc, int32_t TId) {}

void __kmpc_end_ordered(IdentTy *Loc, int32_t TId) {}

int32_t __kmpc_cancel_barrier(IdentTy *Loc, int32_t TId) {
  __kmpc_barrier(Loc, TId);
  return 0;
}

void __kmpc_barrier(IdentTy *Loc, int32_t TId) {
  if (mapping::isSPMDMode())
    return __kmpc_barrier_simple_spmd(Loc, TId);

  // Generic parallel regions are run with multiple of the warp size or single
  // threaded, in the latter case we need to stop here.
  if (omp_get_num_threads() == 1)
    return __kmpc_flush(Loc);

  impl::namedBarrier();
}

[[clang::noinline]] void __kmpc_barrier_simple_spmd(IdentTy *Loc, int32_t TId) {
  synchronize::threadsAligned(atomic::OrderingTy::seq_cst);
}

[[clang::noinline]] void __kmpc_barrier_simple_generic(IdentTy *Loc,
                                                       int32_t TId) {
  synchronize::threads(atomic::OrderingTy::seq_cst);
}

int32_t __kmpc_master(IdentTy *Loc, int32_t TId) {
  return omp_get_thread_num() == 0;
}

void __kmpc_end_master(IdentTy *Loc, int32_t TId) {}

int32_t __kmpc_masked(IdentTy *Loc, int32_t TId, int32_t Filter) {
  return omp_get_thread_num() == Filter;
}

void __kmpc_end_masked(IdentTy *Loc, int32_t TId) {}

int32_t __kmpc_single(IdentTy *Loc, int32_t TId) {
  return __kmpc_master(Loc, TId);
}

void __kmpc_end_single(IdentTy *Loc, int32_t TId) {
  // The barrier is explicitly called.
}

void __kmpc_flush(IdentTy *Loc) { fence::kernel(atomic::seq_cst); }

uint64_t __kmpc_warp_active_thread_mask(void) { return mapping::activemask(); }

void __kmpc_syncwarp(uint64_t Mask) { synchronize::warp(Mask); }

void __kmpc_critical(IdentTy *Loc, int32_t TId, CriticalNameTy *Name) {
  impl::setCriticalLock(reinterpret_cast<omp_lock_t *>(Name));
}

void __kmpc_end_critical(IdentTy *Loc, int32_t TId, CriticalNameTy *Name) {
  impl::unsetCriticalLock(reinterpret_cast<omp_lock_t *>(Name));
}

void omp_init_lock(omp_lock_t *Lock) { impl::initLock(Lock); }

void omp_destroy_lock(omp_lock_t *Lock) { impl::destroyLock(Lock); }

void omp_set_lock(omp_lock_t *Lock) { impl::setLock(Lock); }

void omp_unset_lock(omp_lock_t *Lock) { impl::unsetLock(Lock); }

int omp_test_lock(omp_lock_t *Lock) { return impl::testLock(Lock); }

void ompx_sync_block(int Ordering) {
  impl::syncThreadsAligned(atomic::OrderingTy(Ordering));
}
void ompx_sync_block_acq_rel() {
  impl::syncThreadsAligned(atomic::OrderingTy::acq_rel);
}
void ompx_sync_block_divergent(int Ordering) {
  synchronize::threads(atomic::OrderingTy(Ordering));
}
} // extern "C"
