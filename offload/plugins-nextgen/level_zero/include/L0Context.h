//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero Context abstraction.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0CONTEXT_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0CONTEXT_H

#include "L0Memory.h"
#include "L0Queue.h"
#include "PluginInterface.h"

namespace llvm::omp::target::plugin {

class L0DeviceTy;

/// Plugin-side context for Level Zero. Owns a ze_context_handle_t that is
/// scoped to the set of devices grouped by the user through olCreateContext.
class LevelZeroPluginContextTy final : public PluginContextTy {
public:
  LevelZeroPluginContextTy(GenericPluginTy &Plugin,
                           llvm::ArrayRef<GenericDeviceTy *> Devices,
                           ze_driver_handle_t Driver,
                           ze_context_handle_t ZeContext, bool OwnsZeContext)
      : PluginContextTy(Plugin, Devices), Driver(Driver), ZeContext(ZeContext),
        OwnsZeContext(OwnsZeContext), QueueCache(*this) {}

  ~LevelZeroPluginContextTy() override = default;

  Error deinit() override;

  ze_driver_handle_t getZeDriver() const { return Driver; }
  ze_context_handle_t getZeContext() const { return ZeContext; }

  Error initAsyncInfoImpl(GenericDeviceTy &Device,
                          AsyncInfoWrapperTy &AsyncInfoWrapper) override;

  llvm::Expected<void *> allocate(GenericDeviceTy &Device, int64_t Size,
                                  void *HostPtr, TargetAllocTy Kind,
                                  size_t Alignment,
                                  GenericProfilerTy *ProfilerPtr) override;
  llvm::Error deallocate(GenericDeviceTy &Device, void *Ptr, TargetAllocTy Kind,
                         GenericProfilerTy *ProfilerPtr) override;
  Expected<PluginAllocInfoTy> getAllocInfo(const void *Ptr) override;

  Expected<void *> getNativeHandle() override { return ZeContext; }

  /// Get kernel indirect access flags from all allocators in this context.
  ze_kernel_indirect_access_flags_t getIndirectFlags() {
    ze_kernel_indirect_access_flags_t Flags = 0;
    for (auto &[Device, Allocator] : DeviceAllocators)
      Flags |= Allocator->getIndirectFlags();
    if (HostAllocator)
      Flags |= HostAllocator->getIndirectFlags();
    return Flags;
  }

  /// Initialize per-plugin-context memory allocators. Runs the pool
  /// probe L0 calls up-front so the first user allocation is not delayed.
  Error initAllocators();

  /// Pop an idle queue for \p Device from the cache, or create a new one.
  Expected<L0QueueTy *> takeCachedQueue(L0DeviceTy *Device) {
    return QueueCache.getQueue(*Device);
  }

  /// Return an idle queue to the cache.
  void returnCachedQueue(L0QueueTy *Queue) { QueueCache.releaseQueue(Queue); }

private:
  ze_driver_handle_t Driver;
  ze_context_handle_t ZeContext;
  bool OwnsZeContext;

  L0QueueCacheTy QueueCache;

  /// Per-plugin-context allocators; scoped to this context's ze_context.
  llvm::DenseMap<L0DeviceTy *, std::unique_ptr<MemAllocatorTy>>
      DeviceAllocators;
  std::unique_ptr<MemAllocatorTy> HostAllocator;
};

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0CONTEXT_H
