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
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Types.h"
#include "Utils.h"

#pragma omp declare target

using namespace _OMP;

namespace impl {

/// Atomics
///
///{
/// NOTE: This function needs to be implemented by every target.
uint32_t atomicInc(uint32_t *Address, uint32_t Val, int Ordering);

uint32_t atomicRead(uint32_t *Address, int Ordering) {
  return __atomic_fetch_add(Address, 0U, __ATOMIC_SEQ_CST);
}

uint32_t atomicAdd(uint32_t *Address, uint32_t Val, int Ordering) {
  return __atomic_fetch_add(Address, Val, Ordering);
}
uint32_t atomicMax(uint32_t *Address, uint32_t Val, int Ordering) {
  return __atomic_fetch_max(Address, Val, Ordering);
}

uint32_t atomicExchange(uint32_t *Address, uint32_t Val, int Ordering) {
  uint32_t R;
  __atomic_exchange(Address, &Val, &R, Ordering);
  return R;
}
uint32_t atomicCAS(uint32_t *Address, uint32_t Compare, uint32_t Val,
                   int Ordering) {
  (void)__atomic_compare_exchange(Address, &Compare, &Val, false, Ordering,
                                  Ordering);
  return Compare;
}

uint64_t atomicAdd(uint64_t *Address, uint64_t Val, int Ordering) {
  return __atomic_fetch_add(Address, Val, Ordering);
}
///}

/// AMDGCN Implementation
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

uint32_t atomicInc(uint32_t *Address, uint32_t Val, int Ordering) {
  return __builtin_amdgcn_atomic_inc32(Address, Val, Ordering, "");
}

uint32_t SHARD(namedBarrierTracker);

void namedBarrierInit() {
  // Don't have global ctors, and shared memory is not zero init
  atomic::store(&namedBarrierTracker, 0u, __ATOMIC_RELEASE);
}

void namedBarrier() {
  uint32_t NumThreads = omp_get_num_threads();
  // assert(NumThreads % 32 == 0);

  uint32_t WarpSize = maping::getWarpSize();
  uint32_t NumWaves = NumThreads / WarpSize;

  fence::team(__ATOMIC_ACQUIRE);

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
                                __ATOMIC_RELAXED); // commutative

    // Record the number of times the barrier has been passed
    uint32_t generation = load & 0xffff0000u;

    if ((load & 0x0000ffffu) == (NumWaves - 1)) {
      // Reached NumWaves in low bits so this is the last wave.
      // Set low bits to zero and increment high bits
      load += 0x00010000u; // wrap is safe
      load &= 0xffff0000u; // because bits zeroed second

      // Reset the wave counter and release the waiting waves
      atomic::store(&namedBarrierTracker, load, __ATOMIC_RELAXED);
    } else {
      // more waves still to go, spin until generation counter changes
      do {
        __builtin_amdgcn_s_sleep(0);
        load = atomi::load(&namedBarrierTracker, __ATOMIC_RELAXED);
      } while ((load & 0xffff0000u) == generation);
    }
  }
  fence::team(__ATOMIC_RELEASE);
}

void syncWarp(__kmpc_impl_lanemask_t) {
  // AMDGCN doesn't need to sync threads in a warp
}

void syncThreads() { __builtin_amdgcn_s_barrier(); }

void fenceTeam(int Ordering) { __builtin_amdgcn_fence(Ordering, "workgroup"); }

void fenceKernel(int Ordering) { __builtin_amdgcn_fence(Ordering, "agent"); }

void fenceSystem(int Ordering) { __builtin_amdgcn_fence(Ordering, ""); }

#pragma omp end declare variant
///}

/// NVPTX Implementation
///
///{
#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

uint32_t atomicInc(uint32_t *Address, uint32_t Val, int Ordering) {
  return __nvvm_atom_inc_gen_ui(Address, Val);
}

void namedBarrierInit() {}

void namedBarrier() {
  uint32_t NumThreads = omp_get_num_threads();
  ASSERT(NumThreads % 32 == 0);

  // The named barrier for active parallel threads of a team in an L1 parallel
  // region to synchronize with each other.
  constexpr int BarrierNo = 7;
  asm volatile("barrier.sync %0, %1;"
               :
               : "r"(BarrierNo), "r"(NumThreads)
               : "memory");
}

void fenceTeam(int) { __nvvm_membar_cta(); }

void fenceKernel(int) { __nvvm_membar_gl(); }

void fenceSystem(int) { __nvvm_membar_sys(); }

void syncWarp(__kmpc_impl_lanemask_t Mask) { __nvvm_bar_warp_sync(Mask); }

void syncThreads() {
  constexpr int BarrierNo = 8;
  asm volatile("barrier.sync %0;" : : "r"(BarrierNo) : "memory");
}

constexpr uint32_t OMP_SPIN = 1000;
constexpr uint32_t UNSET = 0;
constexpr uint32_t SET = 1;

// TODO: This seems to hide a bug in the declare variant handling. If it is
// called before it is defined
//       here the overload won't happen. Investigate lalter!
void unsetLock(omp_lock_t *Lock) {
  (void)atomicExchange((uint32_t *)Lock, UNSET, __ATOMIC_SEQ_CST);
}

int testLock(omp_lock_t *Lock) {
  return atomicAdd((uint32_t *)Lock, 0u, __ATOMIC_SEQ_CST);
}

void initLock(omp_lock_t *Lock) { unsetLock(Lock); }

void destoryLock(omp_lock_t *Lock) { unsetLock(Lock); }

void setLock(omp_lock_t *Lock) {
  // TODO: not sure spinning is a good idea here..
  while (atomicCAS((uint32_t *)Lock, UNSET, SET, __ATOMIC_SEQ_CST) != UNSET) {
    int32_t start = __nvvm_read_ptx_sreg_clock();
    int32_t now;
    for (;;) {
      now = __nvvm_read_ptx_sreg_clock();
      int32_t cycles = now > start ? now - start : now + (0xffffffff - start);
      if (cycles >= OMP_SPIN * mapping::getBlockId()) {
        break;
      }
    }
  } // wait for 0 to be the read value
}

#pragma omp end declare variant
///}

} // namespace impl

void synchronize::init(bool IsSPMD) {
  if (!IsSPMD)
    impl::namedBarrierInit();
}

void synchronize::warp(LaneMaskTy Mask) { impl::syncWarp(Mask); }

void synchronize::threads() { impl::syncThreads(); }

void fence::team(int Ordering) { impl::fenceTeam(Ordering); }

void fence::kernel(int Ordering) { impl::fenceKernel(Ordering); }

void fence::system(int Ordering) { impl::fenceSystem(Ordering); }

uint32_t atomic::read(uint32_t *Addr, int Ordering) {
  return impl::atomicRead(Addr, Ordering);
}

uint32_t atomic::inc(uint32_t *Addr, uint32_t V, int Ordering) {
  return impl::atomicInc(Addr, V, Ordering);
}

uint32_t atomic::add(uint32_t *Addr, uint32_t V, int Ordering) {
  return impl::atomicAdd(Addr, V, Ordering);
}

uint64_t atomic::add(uint64_t *Addr, uint64_t V, int Ordering) {
  return impl::atomicAdd(Addr, V, Ordering);
}

extern "C" {
void __kmpc_ordered(IdentTy *Loc, int32_t TId) {}

void __kmpc_end_ordered(IdentTy *Loc, int32_t TId) {}

int32_t __kmpc_cancel_barrier(IdentTy *Loc, int32_t TId) {
  __kmpc_barrier(Loc, TId);
  return 0;
}

void __kmpc_barrier(IdentTy *Loc, int32_t TId) {
  if (mapping::isMainThreadInGenericMode())
    return __kmpc_flush(Loc);

  if (mapping::isSPMDMode())
    return __kmpc_barrier_simple_spmd(Loc, TId);

  impl::namedBarrier();
}

__attribute__((noinline)) void __kmpc_barrier_simple_spmd(IdentTy *Loc,
                                                          int32_t TId) {
  synchronize::threads();
}

int32_t __kmpc_master(IdentTy *Loc, int32_t TId) {
  return omp_get_team_num() == 0;
}

void __kmpc_end_master(IdentTy *Loc, int32_t TId) {}

int32_t __kmpc_single(IdentTy *Loc, int32_t TId) {
  return __kmpc_master(Loc, TId);
}

void __kmpc_end_single(IdentTy *Loc, int32_t TId) {
  // The barrier is explicitly called.
}

void __kmpc_flush(IdentTy *Loc) { fence::kernel(__ATOMIC_SEQ_CST); }

__kmpc_impl_lanemask_t __kmpc_warp_active_thread_mask() {
  return mapping::activemask();
}

void __kmpc_syncwarp(__kmpc_impl_lanemask_t Mask) { synchronize::warp(Mask); }

void __kmpc_critical(IdentTy *Loc, int32_t TId, CriticalNameTy *Name) {
  omp_set_lock(reinterpret_cast<omp_lock_t *>(Name));
}

void __kmpc_end_critical(IdentTy *Loc, int32_t TId, CriticalNameTy *Name) {
  omp_unset_lock(reinterpret_cast<omp_lock_t *>(Name));
}

void omp_init_lock(omp_lock_t *Lock) { impl::initLock(Lock); }

void omp_destroy_lock(omp_lock_t *Lock) { impl::destoryLock(Lock); }

void omp_set_lock(omp_lock_t *Lock) { impl::setLock(Lock); }

void omp_unset_lock(omp_lock_t *Lock) { impl::unsetLock(Lock); }

int omp_test_lock(omp_lock_t *Lock) { return impl::testLock(Lock); }
} // extern "C"

#pragma omp end declare target
