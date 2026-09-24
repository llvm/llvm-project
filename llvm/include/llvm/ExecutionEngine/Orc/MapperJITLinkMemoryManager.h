//===--------------- MapperJITLinkMemoryManager.h -*- C++ -*---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements JITLinkMemoryManager using MemoryMapper
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_MAPPERJITLINKMEMORYMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_MAPPERJITLINKMEMORYMANAGER_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/MemoryMapper.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {
namespace orc {

class LLVM_ABI MapperJITLinkMemoryManager
    : public jitlink::JITLinkMemoryManager {
public:
  /// If ColocatePerJITDylib is true each JITDylib draws from its own pool of
  /// reservations, so all of a JITDylib's objects are colocated (kept within
  /// range of each other for direct, e.g. 32-bit PC-relative, references). If
  /// false (the default) a single shared pool is used for all JITDylibs.
  MapperJITLinkMemoryManager(size_t ReservationGranularity,
                             std::unique_ptr<MemoryMapper> Mapper,
                             bool ColocatePerJITDylib = false);

  template <class MemoryMapperType, class... Args>
  static Expected<std::unique_ptr<MapperJITLinkMemoryManager>>
  CreateWithMapper(size_t ReservationGranularity, Args &&...A) {
    auto Mapper = MemoryMapperType::Create(std::forward<Args>(A)...);
    if (!Mapper)
      return Mapper.takeError();

    return std::make_unique<MapperJITLinkMemoryManager>(ReservationGranularity,
                                                        std::move(*Mapper));
  }

  /// The default per-slab reservation size used by
  /// createColocatingInProcessMemoryManager: 1 GiB on 64-bit hosts, 10 MiB on
  /// 32-bit hosts. This is address space that is reserved, not committed.
  static constexpr size_t defaultSlabSize() {
    if constexpr (sizeof(void *) >= 8)
      return size_t(1) << 30; // 1 GiB on 64-bit hosts.
    else
      return size_t(10) << 20; // 10 MiB on 32-bit hosts.
  }

  /// The reservation granularity passed to the constructor: each slab
  /// reserved from the executor address space is a multiple of this.
  size_t reservationUnits() const { return ReservationUnits; }

  void allocate(const jitlink::JITLinkDylib *JD, jitlink::LinkGraph &G,
                OnAllocatedFunction OnAllocated) override;
  // synchronous overload
  using JITLinkMemoryManager::allocate;

  void deallocate(std::vector<FinalizedAlloc> Allocs,
                  OnDeallocatedFunction OnDeallocated) override;
  // synchronous overload
  using JITLinkMemoryManager::deallocate;

  /// Policy invoked before reserving an additional slab for a JITDylib (i.e.
  /// when its existing reservations can't satisfy a request). JD and
  /// RequestedSize identify the JITDylib and reservation size behind the
  /// request, for use in diagnostics. Returning an Error fails the
  /// triggering allocation.
  using SlabGrowthPolicy = unique_function<Error(
      const jitlink::JITLinkDylib *JD, size_t RequestedSize)>;

  /// Restrict each JITDylib to a single reservation ("slab"). Once a
  /// JITDylib's slab is full, further allocations for it fail rather than
  /// silently spilling into another, possibly out-of-range, slab.
  void denySlabGrowth();

  /// Allow each JITDylib to grow into additional reservations on demand (the
  /// default). Objects in different slabs of the same JITDylib are not
  /// guaranteed to be within range of each other.
  ///
  /// Note: LLJITBuilder::setColocatingSlabAllocator() overrides this default
  /// to denySlabGrowth() when no SlabGrowthPolicy is given.
  void allowSlabGrowth();

  /// Set a custom slab-growth policy. See SlabGrowthPolicy above.
  void setSlabPolicy(SlabGrowthPolicy Policy);

  /// Called when a registered JITDylib is destroyed (see
  /// jitlink::JITLinkDylib::notifyOnDestruction). Frees its pool so a new
  /// JITDylib at the same (reused) address doesn't inherit stale state.
  void notifyDestroying(jitlink::JITLinkDylib &JD) override;

private:
  class InFlightAlloc;

  using AvailableMemoryMap = IntervalMap<ExecutorAddr, bool>;

  // Returns the pool of reserved-but-not-yet-allocated ranges for the given
  // key, creating it on first use. The key is the JITDylib when colocating
  // per-JITDylib, otherwise nullptr (a single shared pool). Must be called with
  // Mutex held.
  AvailableMemoryMap &getAvailableMemory(const jitlink::JITLinkDylib *Key);

  std::mutex Mutex;

  // We reserve multiples of this from the executor address space
  size_t ReservationUnits;

  // When true, each JITDylib gets its own pool of reservations (so a
  // JITDylib's objects are colocated); when false a single nullptr-keyed pool
  // is shared by all JITDylibs.
  bool ColocatePerJITDylib;

  // Policy consulted before reserving an additional slab for a pool (i.e. when
  // a pool already owns a reservation but none of its free ranges fit the
  // request). Returning an Error fails the allocation. Defaults to allowing
  // additional slabs (preserving the historical behavior).
  SlabGrowthPolicy OnSlabGrow = [](const jitlink::JITLinkDylib *,
                                   size_t) -> Error {
    return Error::success();
  };

  AvailableMemoryMap::Allocator AMAllocator;

  // Per-key pool state: the reserved-but-not-yet-allocated ranges available
  // for reuse, and whether the pool already owns at least one reservation
  // (used to detect when a further reservation would be another slab for it).
  // Both live in one map, keyed the same way, so they can't get out of sync
  // with each other.
  struct PoolInfo {
    // The unique_ptr keeps each IntervalMap at a stable address as the map
    // grows, so references returned by getAvailableMemory() stay valid.
    std::unique_ptr<AvailableMemoryMap> AvailPool;
    bool Reserved = false;
  };
  DenseMap<const jitlink::JITLinkDylib *, PoolInfo> Pools;

  // Ranges that have been reserved and already allocated: base address -> size.
  DenseMap<ExecutorAddr, ExecutorAddrDiff> UsedMemory;

  // Base address -> the pool key the allocation was drawn from, so that
  // deallocate() can return the range to the right pool.
  DenseMap<ExecutorAddr, const jitlink::JITLinkDylib *> AllocPoolKey;

  std::unique_ptr<MemoryMapper> Mapper;
};

/// Create an in-process slab allocator that colocates each JITDylib's objects
/// within its own reservation(s). If ReservationGranularity is not given,
/// MapperJITLinkMemoryManager::defaultSlabSize() is used.
LLVM_ABI Expected<std::unique_ptr<MapperJITLinkMemoryManager>>
createColocatingInProcessMemoryManager(
    std::optional<size_t> ReservationGranularity = std::nullopt);

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_MAPPERJITLINKMEMORYMANAGER_H
