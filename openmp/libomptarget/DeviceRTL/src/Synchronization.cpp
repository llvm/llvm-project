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

#pragma omp begin declare target device_type(nohost)

using namespace ompx;

namespace impl {

/// Atomics
///
///{
/// NOTE: This function needs to be implemented by every target.
uint32_t atomicInc(uint32_t *Address, uint32_t Val, atomic::OrderingTy Ordering,
                   atomic::MemScopeTy MemScope);

template <typename Ty>
Ty atomicAdd(Ty *Address, Ty Val, atomic::OrderingTy Ordering) {
  return __scoped_atomic_fetch_add(Address, Val, Ordering,
                                   __MEMORY_SCOPE_DEVICE);
}

template <typename Ty>
Ty atomicMul(Ty *Address, Ty V, atomic::OrderingTy Ordering) {
  Ty TypedCurrentVal, TypedResultVal, TypedNewVal;
  bool Success;
  do {
    TypedCurrentVal = atomic::load(Address, Ordering);
    TypedNewVal = TypedCurrentVal * V;
    Success = atomic::cas(Address, TypedCurrentVal, TypedNewVal, Ordering,
                          atomic::relaxed);
  } while (!Success);
  return TypedResultVal;
}

template <typename Ty> Ty atomicLoad(Ty *Address, atomic::OrderingTy Ordering) {
  return atomicAdd(Address, Ty(0), Ordering);
}

template <typename Ty>
void atomicStore(Ty *Address, Ty Val, atomic::OrderingTy Ordering) {
  __scoped_atomic_store_n(Address, Val, Ordering, __MEMORY_SCOPE_DEVICE);
}

template <typename Ty>
bool atomicCAS(Ty *Address, Ty ExpectedV, Ty DesiredV,
               atomic::OrderingTy OrderingSucc,
               atomic::OrderingTy OrderingFail) {
  return __scoped_atomic_compare_exchange(Address, &ExpectedV, &DesiredV, false,
                                          OrderingSucc, OrderingFail,
                                          __MEMORY_SCOPE_DEVICE);
}

template <typename Ty>
Ty atomicMin(Ty *Address, Ty Val, atomic::OrderingTy Ordering) {
  return __scoped_atomic_fetch_min(Address, Val, Ordering,
                                   __MEMORY_SCOPE_DEVICE);
}

template <typename Ty>
Ty atomicMax(Ty *Address, Ty Val, atomic::OrderingTy Ordering) {
  return __scoped_atomic_fetch_max(Address, Val, Ordering,
                                   __MEMORY_SCOPE_DEVICE);
}

// TODO: Implement this with __atomic_fetch_max and remove the duplication.
template <typename Ty, typename STy, typename UTy>
Ty atomicMinFP(Ty *Address, Ty Val, atomic::OrderingTy Ordering) {
  if (Val >= 0)
    return atomicMin((STy *)Address, utils::convertViaPun<STy>(Val), Ordering);
  return atomicMax((UTy *)Address, utils::convertViaPun<UTy>(Val), Ordering);
}

template <typename Ty, typename STy, typename UTy>
Ty atomicMaxFP(Ty *Address, Ty Val, atomic::OrderingTy Ordering) {
  if (Val >= 0)
    return atomicMax((STy *)Address, utils::convertViaPun<STy>(Val), Ordering);
  return atomicMin((UTy *)Address, utils::convertViaPun<UTy>(Val), Ordering);
}

template <typename Ty>
Ty atomicOr(Ty *Address, Ty Val, atomic::OrderingTy Ordering) {
  return __scoped_atomic_fetch_or(Address, Val, Ordering,
                                  __MEMORY_SCOPE_DEVICE);
}

template <typename Ty>
Ty atomicAnd(Ty *Address, Ty Val, atomic::OrderingTy Ordering) {
  return __scoped_atomic_fetch_and(Address, Val, Ordering,
                                   __MEMORY_SCOPE_DEVICE);
}

template <typename Ty>
Ty atomicXOr(Ty *Address, Ty Val, atomic::OrderingTy Ordering) {
  return __scoped_atomic_fetch_xor(Address, Val, Ordering,
                                   __MEMORY_SCOPE_DEVICE);
}

uint32_t atomicExchange(uint32_t *Address, uint32_t Val,
                        atomic::OrderingTy Ordering) {
  uint32_t R;
  __scoped_atomic_exchange(Address, &Val, &R, Ordering, __MEMORY_SCOPE_DEVICE);
  return R;
}
///}

// Forward declarations defined to be defined for AMDGCN and NVPTX.
uint32_t atomicInc(uint32_t *A, uint32_t V, atomic::OrderingTy Ordering,
                   atomic::MemScopeTy MemScope);
void namedBarrierInit();
void namedBarrier();
void fenceTeam(atomic::OrderingTy Ordering);
void fenceKernel(atomic::OrderingTy Ordering);
void fenceSystem(atomic::OrderingTy Ordering);
void syncWarp(__kmpc_impl_lanemask_t);
void syncThreads(atomic::OrderingTy Ordering);
void syncThreadsAligned(atomic::OrderingTy Ordering) { syncThreads(Ordering); }
void unsetLock(omp_lock_t *);
int testLock(omp_lock_t *);
void initLock(omp_lock_t *);
void destroyLock(omp_lock_t *);
void setLock(omp_lock_t *);
void unsetCriticalLock(omp_lock_t *);
void setCriticalLock(omp_lock_t *);

/// AMDGCN Implementation
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

uint32_t atomicInc(uint32_t *A, uint32_t V, atomic::OrderingTy Ordering,
                   atomic::MemScopeTy MemScope) {
  // builtin_amdgcn_atomic_inc32 should expand to this switch when
  // passed a runtime value, but does not do so yet. Workaround here.

#define ScopeSwitch(ORDER)                                                     \
  switch (MemScope) {                                                          \
  case atomic::MemScopeTy::all:                                                \
    return __builtin_amdgcn_atomic_inc32(A, V, ORDER, "");                     \
  case atomic::MemScopeTy::device:                                             \
    return __builtin_amdgcn_atomic_inc32(A, V, ORDER, "agent");                \
  case atomic::MemScopeTy::cgroup:                                             \
    return __builtin_amdgcn_atomic_inc32(A, V, ORDER, "workgroup");            \
  }

#define Case(ORDER)                                                            \
  case ORDER:                                                                  \
    ScopeSwitch(ORDER)

  switch (Ordering) {
  default:
    __builtin_unreachable();
    Case(atomic::relaxed);
    Case(atomic::aquire);
    Case(atomic::release);
    Case(atomic::acq_rel);
    Case(atomic::seq_cst);
#undef Case
#undef ScopeSwitch
  }
}

uint32_t SHARED(namedBarrierTracker);

void namedBarrierInit() {
  // Don't have global ctors, and shared memory is not zero init
  atomic::store(&namedBarrierTracker, 0u, atomic::release);
}

void namedBarrier() {
  uint32_t NumThreads = omp_get_num_threads();
  // assert(NumThreads % 32 == 0);

  uint32_t WarpSize = mapping::getWarpSize();
  uint32_t NumWaves = NumThreads / WarpSize;

  fence::team(atomic::aquire);

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

// sema checking of amdgcn_fence is aggressive. Intention is to patch clang
// so that it is usable within a template environment and so that a runtime
// value of the memory order is expanded to this switch within clang/llvm.
void fenceTeam(atomic::OrderingTy Ordering) {
  switch (Ordering) {
  default:
    __builtin_unreachable();
  case atomic::aquire:
    return __builtin_amdgcn_fence(atomic::aquire, "workgroup");
  case atomic::release:
    return __builtin_amdgcn_fence(atomic::release, "workgroup");
  case atomic::acq_rel:
    return __builtin_amdgcn_fence(atomic::acq_rel, "workgroup");
  case atomic::seq_cst:
    return __builtin_amdgcn_fence(atomic::seq_cst, "workgroup");
  }
}
void fenceKernel(atomic::OrderingTy Ordering) {
  switch (Ordering) {
  default:
    __builtin_unreachable();
  case atomic::aquire:
    return __builtin_amdgcn_fence(atomic::aquire, "agent");
  case atomic::release:
    return __builtin_amdgcn_fence(atomic::release, "agent");
  case atomic::acq_rel:
    return __builtin_amdgcn_fence(atomic::acq_rel, "agent");
  case atomic::seq_cst:
    return __builtin_amdgcn_fence(atomic::seq_cst, "agent");
  }
}
void fenceSystem(atomic::OrderingTy Ordering) {
  switch (Ordering) {
  default:
    __builtin_unreachable();
  case atomic::aquire:
    return __builtin_amdgcn_fence(atomic::aquire, "");
  case atomic::release:
    return __builtin_amdgcn_fence(atomic::release, "");
  case atomic::acq_rel:
    return __builtin_amdgcn_fence(atomic::acq_rel, "");
  case atomic::seq_cst:
    return __builtin_amdgcn_fence(atomic::seq_cst, "");
  }
}

void syncWarp(__kmpc_impl_lanemask_t) {
  // This is a no-op on current AMDGPU hardware but it is used by the optimizer
  // to enforce convergent behaviour between control flow graphs.
  __builtin_amdgcn_wave_barrier();
}

void syncThreads(atomic::OrderingTy Ordering) {
  if (Ordering != atomic::relaxed)
    fenceTeam(Ordering == atomic::acq_rel ? atomic::release : atomic::seq_cst);

  __builtin_amdgcn_s_barrier();

  if (Ordering != atomic::relaxed)
    fenceTeam(Ordering == atomic::acq_rel ? atomic::aquire : atomic::seq_cst);
}
void syncThreadsAligned(atomic::OrderingTy Ordering) { syncThreads(Ordering); }

// TODO: Don't have wavefront lane locks. Possibly can't have them.
void unsetLock(omp_lock_t *) { __builtin_trap(); }
int testLock(omp_lock_t *) { __builtin_trap(); }
void initLock(omp_lock_t *) { __builtin_trap(); }
void destroyLock(omp_lock_t *) { __builtin_trap(); }
void setLock(omp_lock_t *) { __builtin_trap(); }

constexpr uint32_t UNSET = 0;
constexpr uint32_t SET = 1;

void unsetCriticalLock(omp_lock_t *Lock) {
  (void)atomicExchange((uint32_t *)Lock, UNSET, atomic::acq_rel);
}

void setCriticalLock(omp_lock_t *Lock) {
  uint64_t LowestActiveThread = utils::ffs(mapping::activemask()) - 1;
  if (mapping::getThreadIdInWarp() == LowestActiveThread) {
    fenceKernel(atomic::release);
    while (!atomicCAS((uint32_t *)Lock, UNSET, SET, atomic::relaxed,
                      atomic::relaxed)) {
      __builtin_amdgcn_s_sleep(32);
    }
    fenceKernel(atomic::aquire);
  }
}

#pragma omp end declare variant
///}

/// NVPTX Implementation
///
///{
#pragma omp begin declare variant match(                                       \
        device = {arch(nvptx, nvptx64)},                                       \
            implementation = {extension(match_any)})

uint32_t atomicInc(uint32_t *Address, uint32_t Val, atomic::OrderingTy Ordering,
                   atomic::MemScopeTy MemScope) {
  return __nvvm_atom_inc_gen_ui(Address, Val);
}

void namedBarrierInit() {}

void namedBarrier() {
  uint32_t NumThreads = omp_get_num_threads();
  ASSERT(NumThreads % 32 == 0, nullptr);

  // The named barrier for active parallel threads of a team in an L1 parallel
  // region to synchronize with each other.
  constexpr int BarrierNo = 7;
  __nvvm_barrier_sync_cnt(BarrierNo, NumThreads);
}

void fenceTeam(atomic::OrderingTy) { __nvvm_membar_cta(); }

void fenceKernel(atomic::OrderingTy) { __nvvm_membar_gl(); }

void fenceSystem(atomic::OrderingTy) { __nvvm_membar_sys(); }

void syncWarp(__kmpc_impl_lanemask_t Mask) { __nvvm_bar_warp_sync(Mask); }

void syncThreads(atomic::OrderingTy Ordering) {
  constexpr int BarrierNo = 8;
  __nvvm_barrier_sync(BarrierNo);
}

void syncThreadsAligned(atomic::OrderingTy Ordering) { __syncthreads(); }

constexpr uint32_t OMP_SPIN = 1000;
constexpr uint32_t UNSET = 0;
constexpr uint32_t SET = 1;

// TODO: This seems to hide a bug in the declare variant handling. If it is
// called before it is defined
//       here the overload won't happen. Investigate lalter!
void unsetLock(omp_lock_t *Lock) {
  (void)atomicExchange((uint32_t *)Lock, UNSET, atomic::seq_cst);
}

int testLock(omp_lock_t *Lock) {
  return atomicAdd((uint32_t *)Lock, 0u, atomic::seq_cst);
}

void initLock(omp_lock_t *Lock) { unsetLock(Lock); }

void destroyLock(omp_lock_t *Lock) { unsetLock(Lock); }

void setLock(omp_lock_t *Lock) {
  // TODO: not sure spinning is a good idea here..
  while (atomicCAS((uint32_t *)Lock, UNSET, SET, atomic::seq_cst,
                   atomic::seq_cst) != UNSET) {
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

#pragma omp end declare variant
///}

} // namespace impl

void synchronize::init(bool IsSPMD) {
  if (!IsSPMD)
    impl::namedBarrierInit();
}

void synchronize::warp(LaneMaskTy Mask) { impl::syncWarp(Mask); }

void synchronize::threads(atomic::OrderingTy Ordering) {
  impl::syncThreads(Ordering);
}

void synchronize::threadsAligned(atomic::OrderingTy Ordering) {
  impl::syncThreadsAligned(Ordering);
}

void fence::team(atomic::OrderingTy Ordering) { impl::fenceTeam(Ordering); }

void fence::kernel(atomic::OrderingTy Ordering) { impl::fenceKernel(Ordering); }

void fence::system(atomic::OrderingTy Ordering) { impl::fenceSystem(Ordering); }

#define ATOMIC_COMMON_OP(TY)                                                   \
  TY atomic::add(TY *Addr, TY V, atomic::OrderingTy Ordering) {                \
    return impl::atomicAdd(Addr, V, Ordering);                                 \
  }                                                                            \
  TY atomic::mul(TY *Addr, TY V, atomic::OrderingTy Ordering) {                \
    return impl::atomicMul(Addr, V, Ordering);                                 \
  }                                                                            \
  TY atomic::load(TY *Addr, atomic::OrderingTy Ordering) {                     \
    return impl::atomicLoad(Addr, Ordering);                                   \
  }                                                                            \
  bool atomic::cas(TY *Addr, TY ExpectedV, TY DesiredV,                        \
                   atomic::OrderingTy OrderingSucc,                            \
                   atomic::OrderingTy OrderingFail) {                          \
    return impl::atomicCAS(Addr, ExpectedV, DesiredV, OrderingSucc,            \
                           OrderingFail);                                      \
  }

#define ATOMIC_FP_ONLY_OP(TY, STY, UTY)                                        \
  TY atomic::min(TY *Addr, TY V, atomic::OrderingTy Ordering) {                \
    return impl::atomicMinFP<TY, STY, UTY>(Addr, V, Ordering);                 \
  }                                                                            \
  TY atomic::max(TY *Addr, TY V, atomic::OrderingTy Ordering) {                \
    return impl::atomicMaxFP<TY, STY, UTY>(Addr, V, Ordering);                 \
  }                                                                            \
  void atomic::store(TY *Addr, TY V, atomic::OrderingTy Ordering) {            \
    impl::atomicStore(reinterpret_cast<UTY *>(Addr),                           \
                      utils::convertViaPun<UTY>(V), Ordering);                 \
  }

#define ATOMIC_INT_ONLY_OP(TY)                                                 \
  TY atomic::min(TY *Addr, TY V, atomic::OrderingTy Ordering) {                \
    return impl::atomicMin<TY>(Addr, V, Ordering);                             \
  }                                                                            \
  TY atomic::max(TY *Addr, TY V, atomic::OrderingTy Ordering) {                \
    return impl::atomicMax<TY>(Addr, V, Ordering);                             \
  }                                                                            \
  TY atomic::bit_or(TY *Addr, TY V, atomic::OrderingTy Ordering) {             \
    return impl::atomicOr(Addr, V, Ordering);                                  \
  }                                                                            \
  TY atomic::bit_and(TY *Addr, TY V, atomic::OrderingTy Ordering) {            \
    return impl::atomicAnd(Addr, V, Ordering);                                 \
  }                                                                            \
  TY atomic::bit_xor(TY *Addr, TY V, atomic::OrderingTy Ordering) {            \
    return impl::atomicXOr(Addr, V, Ordering);                                 \
  }                                                                            \
  void atomic::store(TY *Addr, TY V, atomic::OrderingTy Ordering) {            \
    impl::atomicStore(Addr, V, Ordering);                                      \
  }

#define ATOMIC_FP_OP(TY, STY, UTY)                                             \
  ATOMIC_FP_ONLY_OP(TY, STY, UTY)                                              \
  ATOMIC_COMMON_OP(TY)

#define ATOMIC_INT_OP(TY)                                                      \
  ATOMIC_INT_ONLY_OP(TY)                                                       \
  ATOMIC_COMMON_OP(TY)

// This needs to be kept in sync with the header. Also the reason we don't use
// templates here.
ATOMIC_INT_OP(int8_t)
ATOMIC_INT_OP(int16_t)
ATOMIC_INT_OP(int32_t)
ATOMIC_INT_OP(int64_t)
ATOMIC_INT_OP(uint8_t)
ATOMIC_INT_OP(uint16_t)
ATOMIC_INT_OP(uint32_t)
ATOMIC_INT_OP(uint64_t)
ATOMIC_FP_OP(float, int32_t, uint32_t)
ATOMIC_FP_OP(double, int64_t, uint64_t)

#undef ATOMIC_INT_ONLY_OP
#undef ATOMIC_FP_ONLY_OP
#undef ATOMIC_COMMON_OP
#undef ATOMIC_INT_OP
#undef ATOMIC_FP_OP

uint32_t atomic::inc(uint32_t *Addr, uint32_t V, atomic::OrderingTy Ordering,
                     atomic::MemScopeTy MemScope) {
  return impl::atomicInc(Addr, V, Ordering, MemScope);
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
  if (mapping::isMainThreadInGenericMode())
    return __kmpc_flush(Loc);

  if (mapping::isSPMDMode())
    return __kmpc_barrier_simple_spmd(Loc, TId);

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
  impl::syncThreads(atomic::OrderingTy(Ordering));
}
} // extern "C"

#pragma omp end declare target
