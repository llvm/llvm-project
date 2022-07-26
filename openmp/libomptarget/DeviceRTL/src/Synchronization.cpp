//===- Synchronization.cpp - OpenMP Device synchronization API ---- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
// Notified per clause 4(b) of the license.
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

#pragma omp begin declare target device_type(nohost)

using namespace _OMP;

namespace impl {

/// Atomics
///
///{
/// NOTE: This function needs to be implemented by every target.
uint32_t atomicInc(uint32_t *Address, uint32_t Val, int Ordering);

uint32_t atomicLoad(uint32_t *Address, int Ordering) {
  return __atomic_fetch_add(Address, 0U, __ATOMIC_SEQ_CST);
}

void atomicStore(uint32_t *Address, uint32_t Val, int Ordering) {
  __atomic_store_n(Address, Val, Ordering);
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

float unsafeAtomicAdd(float *addr, float value);
///}

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

// Forward declarations defined to be defined for AMDGCN and NVPTX.
uint32_t atomicInc(uint32_t *A, uint32_t V, int Ordering);
void namedBarrierInit();
void namedBarrier();
void fenceTeam(int Ordering);
void fenceKernel(int Ordering);
void fenceSystem(int Ordering);
void syncWarp(__kmpc_impl_lanemask_t);
void syncThreads();
void syncThreadsAligned() { syncThreads(); }
void unsetLock(omp_lock_t *);
int testLock(omp_lock_t *);
void initLock(omp_lock_t *);
void destroyLock(omp_lock_t *);
void setLock(omp_lock_t *);

/// AMDGCN Implementation
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

uint32_t atomicInc(uint32_t *A, uint32_t V, int Ordering) {
  // builtin_amdgcn_atomic_inc32 should expand to this switch when
  // passed a runtime value, but does not do so yet. Workaround here.
  switch (Ordering) {
  default:
    __builtin_unreachable();
  case __ATOMIC_RELAXED:
    return __builtin_amdgcn_atomic_inc32(A, V, __ATOMIC_RELAXED, "");
  case __ATOMIC_ACQUIRE:
    return __builtin_amdgcn_atomic_inc32(A, V, __ATOMIC_ACQUIRE, "");
  case __ATOMIC_RELEASE:
    return __builtin_amdgcn_atomic_inc32(A, V, __ATOMIC_RELEASE, "");
  case __ATOMIC_ACQ_REL:
    return __builtin_amdgcn_atomic_inc32(A, V, __ATOMIC_ACQ_REL, "");
  case __ATOMIC_SEQ_CST:
    return __builtin_amdgcn_atomic_inc32(A, V, __ATOMIC_SEQ_CST, "");
  }
}

uint32_t SHARED(namedBarrierTracker);

void namedBarrierInit() {
  // Don't have global ctors, and shared memory is not zero init
  atomic::store(&namedBarrierTracker, 0u, __ATOMIC_RELEASE);
}

void namedBarrier() {
  uint32_t NumThreads = omp_get_num_threads();
  // assert(NumThreads % 32 == 0);

  uint32_t WarpSize = mapping::getWarpSize();
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
        load = atomic::load(&namedBarrierTracker, __ATOMIC_RELAXED);
      } while ((load & 0xffff0000u) == generation);
    }
  }
  fence::team(__ATOMIC_RELEASE);
}

// sema checking of amdgcn_fence is aggressive. Intention is to patch clang
// so that it is usable within a template environment and so that a runtime
// value of the memory order is expanded to this switch within clang/llvm.
void fenceTeam(int Ordering) {
  switch (Ordering) {
  default:
    __builtin_unreachable();
  case __ATOMIC_ACQUIRE:
    return __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  case __ATOMIC_RELEASE:
    return __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  case __ATOMIC_ACQ_REL:
    return __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
  case __ATOMIC_SEQ_CST:
    return __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
  }
}
void fenceKernel(int Ordering) {
  switch (Ordering) {
  default:
    __builtin_unreachable();
  case __ATOMIC_ACQUIRE:
    return __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  case __ATOMIC_RELEASE:
    return __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  case __ATOMIC_ACQ_REL:
    return __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent");
  case __ATOMIC_SEQ_CST:
    return __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
  }
}
void fenceSystem(int Ordering) {
  switch (Ordering) {
  default:
    __builtin_unreachable();
  case __ATOMIC_ACQUIRE:
    return __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  case __ATOMIC_RELEASE:
    return __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
  case __ATOMIC_ACQ_REL:
    return __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "");
  case __ATOMIC_SEQ_CST:
    return __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
  }
}

void syncWarp(__kmpc_impl_lanemask_t) {
  // AMDGCN doesn't need to sync threads in a warp
}

void syncThreads() { __builtin_amdgcn_s_barrier(); }
void syncThreadsAligned() { syncThreads(); }

void initLock(omp_lock_t *Lock) { unsetLock(Lock); }

void destroyLock(omp_lock_t *Lock) { unsetLock(Lock); }

void setLock(omp_lock_t *Lock) {
  uint64_t lowestActiveThread = utils::ffs(mapping::activemask()) - 1;
  if (mapping::getThreadIdInWarp() == lowestActiveThread) {
    while (atomicCAS((uint32_t *)Lock, UNSET, SET, __ATOMIC_SEQ_CST) != UNSET) {
      __builtin_amdgcn_s_sleep(0);
    }
  }
  // test_lock will now return true for any thread in the warp
}

#if defined(__gfx90a__) && __has_builtin(__builtin_amdgcn_is_shared) &&        \
    __has_builtin(__builtin_amdgcn_is_private) &&                              \
    __has_builtin(__builtin_amdgcn_ds_atomic_fadd_f32) &&                      \
    __has_builtin(__builtin_amdgcn_global_atomic_fadd_f32)
// This function is called for gfx90a only and single precision
// floating point type
float unsafeAtomicAdd(float *addr, float value) {
  if (__builtin_amdgcn_is_shared(
          (const __attribute__((address_space(0))) void *)addr))
    return __builtin_amdgcn_ds_atomic_fadd_f32(
        (const __attribute__((address_space(3))) float *)addr, value);
  else if (__builtin_amdgcn_is_private(
               (const __attribute__((address_space(0))) void *)addr)) {
    float temp = *addr;
    *addr = temp + value;
    return temp;
  }
  return __builtin_amdgcn_global_atomic_fadd_f32(
      (const __attribute__((address_space(1))) float *)addr, value);
}
#endif // if defined(gfx90a) &&

bool volatile omptarget_workers_done [[clang::loader_uninitialized]];
#pragma omp allocate(omptarget_workers_done) allocator(omp_pteam_mem_alloc)

bool volatile omptarget_master_ready [[clang::loader_uninitialized]];
#pragma omp allocate(omptarget_master_ready) allocator(omp_pteam_mem_alloc)

void workersStartBarrier() {
#ifdef __AMDGCN__
  synchronize::omptarget_workers_done = true;
  synchronize::threads();
  while (!synchronize::omptarget_master_ready)
    synchronize::threads();
  synchronize::omptarget_workers_done = false;
#else
  synchronize::threads();
#endif
}

void workersDoneBarrier() {
  // This worker termination logic permits full barriers in reductions
  // by keeping the master thread waiting at another barrier till
  // all workers are finished.
#ifdef __AMDGCN__
  if (mapping::getThreadIdInBlock() == 0)
    synchronize::omptarget_workers_done = true;
#endif
  synchronize::threads();
}

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

void syncThreadsAligned() { __syncthreads(); }

void workersStartBarrier() { syncThreads(); }

void workersDoneBarrier() { syncThreads(); }

constexpr uint32_t OMP_SPIN = 1000;

void initLock(omp_lock_t *Lock) { unsetLock(Lock); }

void destroyLock(omp_lock_t *Lock) { unsetLock(Lock); }

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

float unsafeAtomicAdd(float *addr, float value) { return 0.0; }

#pragma omp end declare variant
///}

} // namespace impl

void synchronize::init(bool IsSPMD) {
  if (!IsSPMD)
    impl::namedBarrierInit();
}

void synchronize::warp(LaneMaskTy Mask) { impl::syncWarp(Mask); }

void synchronize::threads() { impl::syncThreads(); }

void synchronize::threadsAligned() { impl::syncThreadsAligned(); }

void synchronize::workersStartBarrier() { impl::workersStartBarrier(); }

void synchronize::workersDoneBarrier() { impl::workersDoneBarrier(); }

void fence::team(int Ordering) { impl::fenceTeam(Ordering); }

void fence::kernel(int Ordering) { impl::fenceKernel(Ordering); }

void fence::system(int Ordering) { impl::fenceSystem(Ordering); }

uint32_t atomic::load(uint32_t *Addr, int Ordering) {
  return impl::atomicLoad(Addr, Ordering);
}

void atomic::store(uint32_t *Addr, uint32_t V, int Ordering) {
  impl::atomicStore(Addr, V, Ordering);
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
void __kmpc_ordered(IdentTy *Loc, int32_t TId) { FunctionTracingRAII(); }

void __kmpc_end_ordered(IdentTy *Loc, int32_t TId) { FunctionTracingRAII(); }

int32_t __kmpc_cancel_barrier(IdentTy *Loc, int32_t TId) {
  FunctionTracingRAII();
  __kmpc_barrier(Loc, TId);
  return 0;
}

void __kmpc_barrier(IdentTy *Loc, int32_t TId) {
  FunctionTracingRAII();
  if (mapping::isMainThreadInGenericMode())
    return __kmpc_flush(Loc);

  if (mapping::isSPMDMode())
    return __kmpc_barrier_simple_spmd(Loc, TId);

  impl::namedBarrier();
}

void __kmpc_impl_syncthreads() { synchronize::threads(); }

__attribute__((noinline)) void __kmpc_barrier_simple_spmd(IdentTy *Loc,
                                                          int32_t TId) {
  FunctionTracingRAII();
  synchronize::threadsAligned();
}

void __kmpc_barrier_simple_generic(IdentTy *Loc, int32_t TId) {
  FunctionTracingRAII();
  synchronize::threads();
}

int32_t __kmpc_master(IdentTy *Loc, int32_t TId) {
  FunctionTracingRAII();
  return omp_get_thread_num() == 0;
}

void __kmpc_end_master(IdentTy *Loc, int32_t TId) { FunctionTracingRAII(); }

int32_t __kmpc_single(IdentTy *Loc, int32_t TId) {
  FunctionTracingRAII();
  return __kmpc_master(Loc, TId);
}

void __kmpc_end_single(IdentTy *Loc, int32_t TId) {
  FunctionTracingRAII();
  // The barrier is explicitly called.
}

void __kmpc_flush(IdentTy *Loc) {
  FunctionTracingRAII();
  fence::kernel(__ATOMIC_SEQ_CST);
}

uint64_t __kmpc_warp_active_thread_mask(void) {
  FunctionTracingRAII();
  return mapping::activemask();
}

void __kmpc_syncwarp(uint64_t Mask) {
  FunctionTracingRAII();
  synchronize::warp(Mask);
}

void __kmpc_critical(IdentTy *Loc, int32_t TId, CriticalNameTy *Name) {
  FunctionTracingRAII();
  omp_set_lock(reinterpret_cast<omp_lock_t *>(Name));
}

void __kmpc_end_critical(IdentTy *Loc, int32_t TId, CriticalNameTy *Name) {
  FunctionTracingRAII();
  omp_unset_lock(reinterpret_cast<omp_lock_t *>(Name));
}

void omp_init_lock(omp_lock_t *Lock) { impl::initLock(Lock); }

void omp_destroy_lock(omp_lock_t *Lock) { impl::destroyLock(Lock); }

void omp_set_lock(omp_lock_t *Lock) { impl::setLock(Lock); }

void omp_unset_lock(omp_lock_t *Lock) { impl::unsetLock(Lock); }

int omp_test_lock(omp_lock_t *Lock) { return impl::testLock(Lock); }

float __kmpc_unsafeAtomicAdd(float *addr, float value) {
  return impl::unsafeAtomicAdd(addr, value);
}
} // extern "C"

#pragma omp end declare target
