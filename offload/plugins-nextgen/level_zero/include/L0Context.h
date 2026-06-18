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

#include "L0Event.h"
#include "L0Memory.h"
#include "PerThreadTable.h"
#include "PluginInterface.h"

#include <mutex>
#include <unordered_map>

namespace llvm::omp::target::plugin {

class LevelZeroPluginTy;

class L0ContextTLSTy {
  StagingBufferTy StagingBuffer;

public:
  StagingBufferTy &getStagingBuffer() { return StagingBuffer; }
  const StagingBufferTy &getStagingBuffer() const { return StagingBuffer; }

  Error deinit() { return StagingBuffer.clear(); }
};

struct L0ContextTLSTableTy
    : public PerThreadContainer<
          std::unordered_map<ze_context_handle_t, L0ContextTLSTy>> {
  Error deinit() {
    return PerThreadTable::deinit(
        [](L0ContextTLSTy &Entry) -> auto { return Entry.deinit(); });
  }
};

/// Driver- and context-specific resources for a single ze_context_handle_t.
class L0ContextTy : public PluginContextTy {
  /// Level Zero Driver handle.
  ze_driver_handle_t zeDriver = nullptr;

  /// Common Level Zero context.
  ze_context_handle_t zeContext = nullptr;

  /// True when this instance is responsible for destroying `zeContext`.
  bool OwnsZeContext = false;

  /// API version supported by the Level Zero driver.
  ze_api_version_t APIVersion = ZE_API_VERSION_CURRENT;

  /// Imported external pointers. Track this only for user-directed
  /// imports/releases.
  llvm::DenseMap<uintptr_t, size_t> ImportedPtrs;

  /// Common event pool.
  EventPoolTy EventPool;

  /// Host Memory allocator for this context.
  MemAllocatorTy HostMemAllocator;

  /// Per-device allocators, parallel to `Devices`. Empty unless this instance
  /// was constructed with explicit devices.
  llvm::SmallVector<std::unique_ptr<MemAllocatorTy>> DeviceAllocators;

public:
  /// Named constants for checking the imported external pointer regions.
  static constexpr int32_t ImportNotExist = -1;
  static constexpr int32_t ImportUnknown = 0;
  static constexpr int32_t ImportExist = 1;

  /// Create context, initialize event pool and extension functions.
  L0ContextTy(LevelZeroPluginTy &Plugin, ze_driver_handle_t zeDriver,
              int32_t /*DriverId*/);

  /// Adopt an existing ze_context_handle_t. Pair with initWithDevices() to
  /// initialize the event pool, extension functions, and per-device pools.
  L0ContextTy(LevelZeroPluginTy &Plugin, ze_driver_handle_t zeDriver,
              ze_context_handle_t AdoptedZeContext, bool OwnsZeContext,
              llvm::ArrayRef<GenericDeviceTy *> Devices);

  L0ContextTy(const L0ContextTy &) = delete;
  L0ContextTy(L0ContextTy &&) = delete;
  L0ContextTy &operator=(const L0ContextTy &) = delete;
  L0ContextTy &operator=(const L0ContextTy &&) = delete;

  ~L0ContextTy() override;

  Error init();
  Error initWithDevices();
  Error deinit();

  LevelZeroPluginTy &getPlugin() const;

  Expected<void *> allocate(GenericDeviceTy &Device, int64_t Size,
                            TargetAllocTy Kind) override;
  Error deallocate(void *Ptr) override;
  Expected<PluginAllocInfoTy> getAllocInfo(const void *Ptr) override;

  StagingBufferTy &getStagingBuffer();

  /// Add imported external pointer region.
  void addImported(void *Ptr, size_t Size) {
    (void)ImportedPtrs.try_emplace(reinterpret_cast<uintptr_t>(Ptr), Size);
  }

  /// Remove imported external pointer region.
  void removeImported(void *Ptr) {
    (void)ImportedPtrs.erase(reinterpret_cast<uintptr_t>(Ptr));
  }
  /// Check if imported regions contain the specified region.
  int32_t checkImported(void *Ptr, size_t Size) const {
    uintptr_t LB = reinterpret_cast<uintptr_t>(Ptr);
    uintptr_t UB = LB + Size;
    // We do not expect a large number of user-directed imports, so use simple
    // logic.
    for (auto &I : ImportedPtrs) {
      uintptr_t ILB = I.first;
      uintptr_t IUB = ILB + I.second;
      if (LB >= ILB && UB <= IUB)
        return ImportExist;
      if ((LB >= ILB && LB < IUB) || (UB > ILB && UB <= IUB))
        return ImportUnknown;
    }
    return ImportNotExist;
  }

  ze_driver_handle_t getZeDriver() const { return zeDriver; }

  /// Return context associated with the driver.
  ze_context_handle_t getZeContext() const { return zeContext; }

  /// Return driver API version.
  ze_api_version_t getDriverAPIVersion() const { return APIVersion; }

  /// Return the event pool of this driver.
  EventPoolTy &getEventPool() { return EventPool; }
  const EventPoolTy &getEventPool() const { return EventPool; }

  bool supportsLargeMem() const {
    // Large memory support is available since API version 1.1.
    return getDriverAPIVersion() >= ZE_API_VERSION_1_1;
  }

  const MemAllocatorTy &getHostMemAllocator() const { return HostMemAllocator; }
  MemAllocatorTy &getHostMemAllocator() { return HostMemAllocator; }

  /// Level Zero extension function pointer for kernel argument size query.
  ze_result_t(ZE_APICALL *zexKernelGetArgumentSize)(
      ze_kernel_handle_t hKernel, uint32_t argIndex,
      uint32_t *pArgSize) = nullptr;

  ze_context_handle_t(ZE_APICALL *zeDriverGetDefaultContext)(
      ze_driver_handle_t hDriver) = nullptr;
};

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0CONTEXT_H
