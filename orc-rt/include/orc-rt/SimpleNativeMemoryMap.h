//===- SimpleNativeMemoryMap.h -- Mem via standard host OS APIs -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SimpleNativeMemoryMap and related APIs.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SIMPLENATIVEMEMORYMAP_H
#define ORC_RT_SIMPLENATIVEMEMORYMAP_H

#include "orc-rt/AllocAction.h"
#include "orc-rt/BootstrapInfo.h"
#include "orc-rt/Error.h"
#include "orc-rt/MemoryFlags.h"
#include "orc-rt/Service.h"
#include "orc-rt/SimpleSymbolTable.h"
#include "orc-rt/move_only_function.h"
#include "orc-rt/sps-ci/SimpleNativeMemoryMapSPSCI.h"

#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace orc_rt {

class Session;

/// JIT'd memory management backend.
///
/// Intances can:
/// 1. Reserve address space.
/// 2. Initialize memory regions within reserved memory (copying content,
///    applying permissions, running finalize actions, and recording
///    deallocate actions).
/// 3. Deinitialize memory regions within reserved memory (running
///    deallocate actions and making memory available for future
///    initialize calls (if the system permits this).
/// 4. Release address space, deinitializing any remaining initialized
///    regions, and returning the address space to the system for reuse (if
///    the system permits).
class SimpleNativeMemoryMap : public Service {
public:
  /// Create a SimpleNativeMemoryMap, adding associated symbols to the given
  /// SimpleSymbolTable (typically the BootstrapInfo table).
  static Expected<std::unique_ptr<SimpleNativeMemoryMap>>
  Create(Session &S, SimpleSymbolTable &ST,
         const char *InstanceName = "orc_rt_SimpleNativeMemoryMap_Instance",
         SimpleSymbolTable::MutatorFn AddInterface =
             sps_ci::addSimpleNativeMemoryMap);

  /// Convenience constructor that adds default symbols to the given
  /// BootstrapInfo's symbols map.
  static Expected<std::unique_ptr<SimpleNativeMemoryMap>>
  Create(Session &S, BootstrapInfo &BI) {
    return Create(S, BI.symbols());
  }
  /// SimpleNativeMemoryMap is not copyable / moveable.
  SimpleNativeMemoryMap(const SimpleNativeMemoryMap &) = delete;
  SimpleNativeMemoryMap &operator=(const SimpleNativeMemoryMap &) = delete;
  SimpleNativeMemoryMap(SimpleNativeMemoryMap &&) = delete;
  SimpleNativeMemoryMap &operator=(SimpleNativeMemoryMap &&) = delete;

  /// Reserves a slab of contiguous address space for allocation.
  ///
  /// Returns the base address of the allocated memory.
  using OnReserveCompleteFn = move_only_function<void(Expected<void *>)>;
  void reserve(OnReserveCompleteFn &&OnComplete, size_t Size);

  /// Release a slab of contiguous address space back to the system.
  using OnReleaseCompleteFn = move_only_function<void(Error)>;
  void release(OnReleaseCompleteFn &&OnComplete, void *Addrs);

  /// Convenience method to release multiple slabs with one call. This can be
  /// used to save on interprocess communication at the cost of less expressive
  /// errors.
  void releaseMultiple(OnReleaseCompleteFn &&OnComplete,
                       std::vector<void *> Addrs);

  struct InitializeRequest {
    struct Segment {
      AllocGroup AG;
      char *Address = nullptr;
      size_t Size = 0;
      span<const char> Content;
    };

    std::vector<Segment> Segments;
    std::vector<AllocActionPair> AAPs;
  };

  /// Writes content into the requested ranges, applies permissions, and
  /// performs allocation actions.
  using OnInitializeCompleteFn = move_only_function<void(Expected<void *>)>;
  void initialize(OnInitializeCompleteFn &&OnComplete, InitializeRequest IR);

  /// Runs deallocation actions and resets memory permissions for the requested
  /// memory.
  using OnDeinitializeCompleteFn = move_only_function<void(Error)>;
  void deinitialize(OnDeinitializeCompleteFn &&OnComplete, void *Base);

  /// Convenience method to deinitialize multiple regions with one call. This
  /// can be used to save on interprocess communication at the cost of less
  /// expressive errors.
  void deinitializeMultiple(OnDeinitializeCompleteFn &&OnComplete,
                            std::vector<void *> Bases);

  void onDetach(Service::OnCompleteFn OnComplete) override;
  void onShutdown(Service::OnCompleteFn OnComplete) override;

private:
  SimpleNativeMemoryMap(Session &S) : S(S) {}

  struct SlabInfo {
    SlabInfo(size_t Size) : Size(Size) {}
    size_t Size;
    std::unordered_map<void *, std::vector<AllocAction>> DeallocActions;
  };

  void releaseNext(OnReleaseCompleteFn &&OnComplete, std::vector<void *> Addrs,
                   bool AnyError, Error LastErr);
  void deinitializeNext(OnDeinitializeCompleteFn &&OnComplete,
                        std::vector<void *> Bases, bool AnyError,
                        Error LastErr);
  void shutdownNext(OnCompleteFn OnComplete, std::vector<void *> Bases);
  Error makeBadSlabError(void *Base, const char *Op);
  SlabInfo *findSlabInfoFor(void *Base);
  Error recordDeallocActions(void *Base,
                             std::vector<AllocAction> DeallocActions);

  Session &S;
  std::mutex M;
  std::map<void *, SlabInfo> Slabs;
};

} // namespace orc_rt

#endif // ORC_RT_SIMPLENATIVEMEMORYMAP_H
