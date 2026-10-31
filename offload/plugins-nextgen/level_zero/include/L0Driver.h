//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero Driver abstraction.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0DRIVER_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0DRIVER_H

#include "APIHelpers.h"
#include "L0Compat.h"
#include "L0Defs.h"
#include "L0Event.h"
#include "level_zero/ze_api.h"

#include <atomic>

namespace llvm::omp::target::plugin {

class LevelZeroPluginTy;
class LevelZeroPluginContextTy;

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
    return api_helper::canCall<Fn>() || ExperimentalFallbackPtr != nullptr;
  }

  explicit operator bool() const { return available(); }

  template <typename... Args> decltype(auto) operator()(Args &&...ArgsList) {
    // Need to cast the type to avoid mismatch of return type deduction
    using ReturnTy = std::invoke_result_t<decltype(Fn), Args...>;

    if (!api_helper::canCall<Fn>() ||
        AvailableButUnsupported.load(std::memory_order_relaxed)) {
      if (ExperimentalFallbackPtr != nullptr) {
        return ExperimentalFallbackPtr(std::forward<Args>(ArgsList)...);
      }

      return static_cast<ReturnTy>(UnsupportedValue);
    }

    auto Ret = Fn(std::forward<Args>(ArgsList)...);

    if (Ret == UnsupportedValue) {
      AvailableButUnsupported.store(true, std::memory_order_relaxed);

      if (ExperimentalFallbackPtr != nullptr)
        return ExperimentalFallbackPtr(std::forward<Args>(ArgsList)...);
    }

    return Ret;
  }

  bool loadExperimental(ze_driver_handle_t zeDriver, const char *FuncName) {
    ze_result_t Result = ZE_RESULT_SUCCESS;
    CALL_ZE(Result, zeDriverGetExtensionFunctionAddress, zeDriver, FuncName,
            reinterpret_cast<void **>(&ExperimentalFallbackPtr));

    if (Result != ZE_RESULT_SUCCESS || ExperimentalFallbackPtr == nullptr) {
      ODBG(OLDT_ZeDispatcher) << "Failed to load experimental variant of "
                              << __PRETTY_FUNCTION__ << "using: " << FuncName;
      return false;
    }

    ODBG(OLDT_ZeDispatcher) << "Loaded experimental variant of "
                            << __PRETTY_FUNCTION__ << " using: " << FuncName;
    return true;
  }

private:
  std::atomic<bool> AvailableButUnsupported = false;
  decltype(Fn) ExperimentalFallbackPtr = nullptr;
};

/// Per-driver state: driver handle and version, extension dispatchers, the
/// event pool, and a driver-wide ze_context. The ze_context is shared with
/// the default plugin context used by the libomptarget path. Contexts created
/// through olCreateContext are separate LevelZeroPluginContextTy objects (see
/// L0Context.h).
class L0DriverTy {
  /// The plugin that created this driver.
  LevelZeroPluginTy &Plugin;

  /// Level Zero Driver handle.
  ze_driver_handle_t zeDriver = nullptr;

  uint32_t DriverId;

  /// Driver-wide Level Zero context.
  ze_context_handle_t zeContext = nullptr;

  /// API version supported by the Level Zero driver.
  ze_api_version_t APIVersion = ZE_API_VERSION_CURRENT;

  /// Version of the Level Zero driver.
  std::string DriverVersion;

  Expected<std::string> tryGetIntelDriverVersion();

  /// Imported external pointers. Track this only for user-directed
  /// imports/releases.
  llvm::DenseMap<uintptr_t, size_t> ImportedPtrs;

  /// Common event pool.
  EventPoolTy EventPool;

  /// Default plugin-side context used by the libomptarget path.
  std::unique_ptr<LevelZeroPluginContextTy> DefaultUserCtx;

public:
  /// Named constants for checking the imported external pointer regions.
  static constexpr int32_t ImportNotExist = -1;
  static constexpr int32_t ImportUnknown = 0;
  static constexpr int32_t ImportExist = 1;

  L0DriverTy(LevelZeroPluginTy &Plugin, ze_driver_handle_t zeDriver,
             int32_t DriverId);

  L0DriverTy(const L0DriverTy &) = delete;
  L0DriverTy(L0DriverTy &&) = delete;
  L0DriverTy &operator=(const L0DriverTy &) = delete;
  L0DriverTy &operator=(const L0DriverTy &&) = delete;

  /// Release resources.
  ~L0DriverTy();

  /// Create the driver-wide context, initialize the event pool, extension
  /// functions and the default plugin context.
  Error init();
  Error deinit();

  LevelZeroPluginTy &getPlugin() const { return Plugin; }

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

  uint32_t getDriverId() const { return DriverId; }

  /// Return the driver-wide context.
  ze_context_handle_t getZeContext() const { return zeContext; }

  /// Return the default plugin-side context used by the libomptarget path.
  LevelZeroPluginContextTy &getDefaultUserCtx() const {
    return *DefaultUserCtx;
  }

  /// Return driver API version.
  ze_api_version_t getDriverAPIVersion() const { return APIVersion; }

  /// Return driver version.
  const std::string &getDriverVersion() const { return DriverVersion; }

  /// Return the event pool of this driver.
  EventPoolTy &getEventPool() { return EventPool; }
  const EventPoolTy &getEventPool() const { return EventPool; }

  bool supportsLargeMem() const {
    // Large memory support is available since API version 1.1.
    return getDriverAPIVersion() >= ZE_API_VERSION_1_1;
  }

  std::atomic<bool> AppendLaunchKernelWithArgsSupported = true;

  ZeDispatcher<zeCommandListAppendLaunchKernelWithArguments>
      LaunchKernelWithArguments;
  ZeDispatcher<zeCommandListAppendHostFunction> CommandListAppendHostFunction;
  ZeDispatcher<zeDriverGetDefaultContext, nullptr> DriverGetDefaultContext;
  ZeDispatcher<zeIntelGetDriverVersionString> IntelGetDriverVersionString;
};

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0DRIVER_H
