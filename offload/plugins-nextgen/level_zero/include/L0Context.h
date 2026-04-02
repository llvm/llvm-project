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
#include "PerThreadTable.h"

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

/// Driver and context-specific resources. We assume a single context per
/// driver.
class L0ContextTy {
  /// The plugin that created this context.
  LevelZeroPluginTy &Plugin;

  /// Level Zero Driver handle.
  ze_driver_handle_t zeDriver = nullptr;

  /// Common Level Zero context.
  ze_context_handle_t zeContext = nullptr;

  /// API version supported by the Level Zero driver.
  ze_api_version_t APIVersion = ZE_API_VERSION_CURRENT;

  /// Imported external pointers. Track this only for user-directed
  /// imports/releases.
  llvm::DenseMap<uintptr_t, size_t> ImportedPtrs;

  /// Common event pool.
  EventPoolTy EventPool;

  /// Host Memory allocator for this driver.
  MemAllocatorTy HostMemAllocator;

public:
  /// Named constants for checking the imported external pointer regions.
  static constexpr int32_t ImportNotExist = -1;
  static constexpr int32_t ImportUnknown = 0;
  static constexpr int32_t ImportExist = 1;

  /// Create context, initialize event pool and extension functions.
  L0ContextTy(LevelZeroPluginTy &Plugin, ze_driver_handle_t zeDriver,
              int32_t DriverId)
      : Plugin(Plugin), zeDriver(zeDriver) {}

  L0ContextTy(const L0ContextTy &) = delete;
  L0ContextTy(L0ContextTy &&) = delete;
  L0ContextTy &operator=(const L0ContextTy &) = delete;
  L0ContextTy &operator=(const L0ContextTy &&) = delete;

  /// Release resources.
  ~L0ContextTy() = default;

  Error init();
  Error deinit();

  LevelZeroPluginTy &getPlugin() const { return Plugin; }

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
};

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0CONTEXT_H
