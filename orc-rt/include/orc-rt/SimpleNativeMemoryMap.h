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
#include "orc-rt/Error.h"
#include "orc-rt/MemoryFlags.h"
#include "orc-rt/ResourceManager.h"
#include "orc-rt/SPSWrapperFunction.h"
#include "orc-rt/move_only_function.h"

#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace orc_rt {

/// JIT'd memory management backend.
///
/// Intances can:
/// 1. Reserve address space.
/// 2. Finalize memory regions within reserved memory (copying content,
///    applying permissions, running finalize actions, and recording
///    deallocate actions).
/// 3. Deallocate memory regions within reserved memory (running
///    deallocate actions and making memory available for future
///    finalize calls (if the system permits this).
/// 4. Release address space, deallocating any not-yet-deallocated finalized
///    regions, and returning the address space to the system for reuse (if
///    the system permits).
class SimpleNativeMemoryMap : public ResourceManager {
public:
  /// Reserves a slab of contiguous address space for allocation.
  ///
  /// Returns the base address of the allocated memory.
  using OnReserveCompleteFn = move_only_function<void(Expected<void *>)>;
  void reserve(OnReserveCompleteFn &&OnComplete, size_t Size);

  /// Release a slab of contiguous address space back to the system.
  using OnReleaseCompleteFn = move_only_function<void(Error)>;
  void release(OnReleaseCompleteFn &&OnComplete, void *Addr);

  struct FinalizeRequest {
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
  using OnFinalizeCompleteFn = move_only_function<void(Expected<void *>)>;
  void finalize(OnFinalizeCompleteFn &&OnComplete, FinalizeRequest FR);

  /// Runs deallocation actions and resets memory permissions for the requested
  /// memory.
  using OnDeallocateCompleteFn = move_only_function<void(Error)>;
  void deallocate(OnDeallocateCompleteFn &&OnComplete, void *Base);

  void detach(ResourceManager::OnCompleteFn OnComplete) override;
  void shutdown(ResourceManager::OnCompleteFn OnComplete) override;

private:
  struct SlabInfo {
    SlabInfo(size_t Size) : Size(Size) {}
    size_t Size;
    std::unordered_map<void *, std::vector<AllocAction>> DeallocActions;
  };

  void shutdownNext(OnCompleteFn OnComplete, std::vector<void *> Bases);
  Error makeBadSlabError(void *Base, const char *Op);
  SlabInfo *findSlabInfoFor(void *Base);
  Error recordDeallocActions(void *Base,
                             std::vector<AllocAction> DeallocActions);

  std::mutex M;
  std::map<void *, SlabInfo> Slabs;
};

} // namespace orc_rt

ORC_RT_SPS_INTERFACE void orc_rt_SimpleNativeMemoryMap_reserve_sps_wrapper(
    orc_rt_SessionRef Session, void *CallCtx,
    orc_rt_WrapperFunctionReturn Return, orc_rt_WrapperFunctionBuffer ArgBytes);

ORC_RT_SPS_INTERFACE void orc_rt_SimpleNativeMemoryMap_release_sps_wrapper(
    orc_rt_SessionRef Session, void *CallCtx,
    orc_rt_WrapperFunctionReturn Return, orc_rt_WrapperFunctionBuffer ArgBytes);

ORC_RT_SPS_INTERFACE void orc_rt_SimpleNativeMemoryMap_finalize_sps_wrapper(
    orc_rt_SessionRef Session, void *CallCtx,
    orc_rt_WrapperFunctionReturn Return, orc_rt_WrapperFunctionBuffer ArgBytes);

ORC_RT_SPS_INTERFACE void orc_rt_SimpleNativeMemoryMap_deallocate_sps_wrapper(
    orc_rt_SessionRef Session, void *CallCtx,
    orc_rt_WrapperFunctionReturn Return, orc_rt_WrapperFunctionBuffer ArgBytes);

#endif // ORC_RT_SIMPLENATIVEMEMORYMAP_H
