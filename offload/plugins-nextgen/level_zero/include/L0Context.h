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

#include "APIHelpers.h"
#include "L0Compat.h"
#include "L0Event.h"
#include "L0Memory.h"
#include "PerThreadTable.h"
#include "level_zero/ze_api.h"

namespace llvm::omp::target::plugin {

class LevelZeroPluginTy;
class LevelZeroPluginContextTy;

class L0ContextTLSTy {
  StagingBufferTy StagingBuffer;

public:
  StagingBufferTy &getStagingBuffer() { return StagingBuffer; }
  const StagingBufferTy &getStagingBuffer() const { return StagingBuffer; }

  Error deinit() { return StagingBuffer.clear(); }
};

// Helper for managing Level Zero APIs.
// It provides two interfaces - by default it tries to call the function
// directly - either through dlopen or directly linked (see L0DynWrapper.cpp).
// It is also possible to call through an internal function pointer, which
// can be populated using `loadExperimental` using
// `zeDriverGetExtensionFunctionAddress`.
// `addFallbackFunction`. It was implemented in order to support different
// versions of level zero software stack and different kinds of drivers.
template <auto Fn, auto UnsupportedValue = ZE_RESULT_ERROR_UNSUPPORTED_FEATURE>
class ZeDispatcher {
public:
  constexpr ZeDispatcher() = default;

  [[nodiscard]]
  bool available() const {
    if (FuncPtr != nullptr)
      return true;

    return api_helper::canCall<Fn>();
  }

  explicit operator bool() const { return available(); }

  template <typename... Args>
  decltype(auto) operator()(Args &&...ArgsList) const {
    // Need to cast the type to avoid mismatch of return type deduction
    using ReturnTy = std::invoke_result_t<decltype(Fn), Args...>;
    if (FuncPtr != nullptr)
      return FuncPtr(std::forward<Args>(ArgsList)...);

    if (!api_helper::canCall<Fn>())
      return static_cast<ReturnTy>(UnsupportedValue);

    return Fn(std::forward<Args>(ArgsList)...);
  }

  bool loadExperimental(ze_driver_handle_t zeDriver, const char *FuncName) {
    assert(!api_helper::canCall<Fn>() &&
           "ZeDispatcher::loadExperimental called without "
           "ZeDispatcher::available check!");

    ze_result_t Result = ZE_RESULT_SUCCESS;
    CALL_ZE_RET(Result, zeDriverGetExtensionFunctionAddress, zeDriver, FuncName,
                reinterpret_cast<void **>(&FuncPtr));

    if (Result != ZE_RESULT_SUCCESS || FuncPtr == nullptr)
      return false;

    return true;
  }

private:
  decltype(Fn) FuncPtr = nullptr;
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

  /// Default plugin-side context used by the libomptarget path.
  std::unique_ptr<LevelZeroPluginContextTy> DefaultUserCtx;

public:
  /// Named constants for checking the imported external pointer regions.
  static constexpr int32_t ImportNotExist = -1;
  static constexpr int32_t ImportUnknown = 0;
  static constexpr int32_t ImportExist = 1;

  /// Create context, initialize event pool and extension functions.
  L0ContextTy(LevelZeroPluginTy &Plugin, ze_driver_handle_t zeDriver,
              int32_t DriverId);

  L0ContextTy(const L0ContextTy &) = delete;
  L0ContextTy(L0ContextTy &&) = delete;
  L0ContextTy &operator=(const L0ContextTy &) = delete;
  L0ContextTy &operator=(const L0ContextTy &&) = delete;

  /// Release resources.
  ~L0ContextTy();

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

  /// Return the default plugin-side context used by the libomptarget path.
  LevelZeroPluginContextTy &getDefaultUserCtx() const {
    return *DefaultUserCtx;
  }

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

  std::atomic<bool> AppendLaunchKernelWithArgsSupported = true;

  ZeDispatcher<zeCommandListAppendLaunchKernelWithArguments>
      LaunchKernelWithArguments;
  ZeDispatcher<zexKernelGetArgumentSize> KernelGetArgumentSize;
  ZeDispatcher<zeCommandListAppendHostFunction> CommandListAppendHostFunction;
  ZeDispatcher<zeDriverGetDefaultContext, nullptr> DriverGetDefaultContext;
};

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0CONTEXT_H
