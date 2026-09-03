//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Level Zero Context abstraction.
//
//===----------------------------------------------------------------------===//

#include "L0Context.h"
#include "L0Plugin.h"

namespace llvm::omp::target::plugin {

/// Compose a driver version string from the packed value reported by
/// zeDriverGetProperties. Used when zeIntelGetDriverVersionString is missing.
static Expected<std::string>
getDriverVersionFromProperties(ze_driver_handle_t zeDriver) {
  ze_driver_properties_t DriverProperties{};
  DriverProperties.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
  CALL_ZE_RET_ERROR(zeDriverGetProperties, zeDriver, &DriverProperties);

  const uint32_t PackedVersion = DriverProperties.driverVersion;
  return std::to_string((PackedVersion & 0xFF000000) >> 24) + "." +
         std::to_string((PackedVersion & 0x00FF0000) >> 16) + "." +
         std::to_string(PackedVersion & 0x0000FFFF);
}

L0ContextTy::L0ContextTy(LevelZeroPluginTy &Plugin, ze_driver_handle_t zeDriver,
                         int32_t DriverId)
    : Plugin(Plugin), zeDriver(zeDriver), DriverId(DriverId) {}

L0ContextTy::~L0ContextTy() = default;

Expected<std::string> L0ContextTy::tryGetIntelDriverVersion() {
  size_t VersionSize = 0;
  CALL_ZE_RET_ERROR(IntelGetDriverVersionString, zeDriver, nullptr,
                    &VersionSize);
  std::string Version(VersionSize, '\0');
  CALL_ZE_RET_ERROR(IntelGetDriverVersionString, zeDriver, Version.data(),
                    &VersionSize);
  if (!Version.empty() && Version.back() == '\0')
    Version.pop_back();
  return Version;
}

Error L0ContextTy::init() {
  auto CleanupOnError = [&]() {
    if (zeContext) {
      zeContextDestroy(zeContext);
      zeContext = nullptr;
    }
  };
  CALL_ZE_RET_ERROR(zeDriverGetApiVersion, zeDriver, &APIVersion);
  ODBG(OLDT_Init) << "Driver API version is "
                  << llvm::format(PRIx32, APIVersion);

  ze_context_desc_t Desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CALL_ZE_RET_ERROR(zeContextCreate, zeDriver, &Desc, &zeContext);

  const auto &Options = Plugin.getOptions();
  bool UseCounterBasedEvents = Options.CommandMode == CommandModeTy::InOrder ||
                               Options.CommandMode == CommandModeTy::Sync;
  if (UseCounterBasedEvents)
    ODBG(OLDT_Init) << "Using counter-based events for "
                    << (Options.CommandMode == CommandModeTy::InOrder
                            ? "InOrder"
                            : "Sync")
                    << " command mode";

  if (auto Err = EventPool.init(zeContext, UseCounterBasedEvents,
                                /* Flags */ 0)) {
    CleanupOnError();
    return Err;
  }
  if (auto Err = HostMemAllocator.initHostPool(*this, Plugin.getOptions())) {
    if (auto DeinitErr = EventPool.deinit())
      Err = joinErrors(std::move(Err), std::move(DeinitErr));
    CleanupOnError();
    return Err;
  }

  ODBG(OLDT_Init) << "APIs supported by the context with dlopen: ";
  ODBG(OLDT_Init) << "  zeCommandListAppendLaunchKernelWithArguments: "
                  << (LaunchKernelWithArguments.available() ? "yes" : "no");
  ODBG(OLDT_Init) << "  zexKernelGetArgumentSize: "
                  << (KernelGetArgumentSize.available() ? "yes" : "no");
  ODBG(OLDT_Init) << "  zeCommandListAppendHostFunction: "
                  << (CommandListAppendHostFunction.available() ? "yes" : "no");
  ODBG(OLDT_Init) << "  zeDriverGetDefaultContext: "
                  << (DriverGetDefaultContext.available() ? "yes" : "no");

  if (!LaunchKernelWithArguments)
    LaunchKernelWithArguments.loadExperimental(
        zeDriver, "zeCommandListAppendLaunchKernelWithArguments");

  if (!KernelGetArgumentSize)
    KernelGetArgumentSize.loadExperimental(zeDriver,
                                           "zexKernelGetArgumentSize");

  if (!CommandListAppendHostFunction)
    CommandListAppendHostFunction.loadExperimental(
        zeDriver, "zeCommandListAppendHostFunction");
  if (!CommandListAppendHostFunction)
    // Try again with a name used in compute runtime 25.35 to 25.48
    CommandListAppendHostFunction.loadExperimental(
        zeDriver, "zexCommandListAppendHostFunction");

  if (!DriverGetDefaultContext)
    DriverGetDefaultContext.loadExperimental(zeDriver,
                                             "zeDriverGetDefaultContext");

  if (!IntelGetDriverVersionString)
    IntelGetDriverVersionString.loadExperimental(
        zeDriver, "zeIntelGetDriverVersionString");

  ODBG(OLDT_Init) << "APIs supported by the context with added extensions: ";
  ODBG(OLDT_Init) << "  zeCommandListAppendLaunchKernelWithArguments: "
                  << (LaunchKernelWithArguments.available() ? "yes" : "no");
  ODBG(OLDT_Init) << "  zexKernelGetArgumentSize: "
                  << (KernelGetArgumentSize.available() ? "yes" : "no");
  ODBG(OLDT_Init) << "  zeCommandListAppendHostFunction: "
                  << (CommandListAppendHostFunction.available() ? "yes" : "no");
  ODBG(OLDT_Init) << "  zeDriverGetDefaultContext: "
                  << (DriverGetDefaultContext.available() ? "yes" : "no");

  auto DriverVersionOrErr = IntelGetDriverVersionString.available()
                                ? tryGetIntelDriverVersion()
                                : getDriverVersionFromProperties(zeDriver);
  if (!DriverVersionOrErr)
    return DriverVersionOrErr.takeError();
  DriverVersion = std::move(*DriverVersionOrErr);
  ODBG(OLDT_Init) << "Driver version is " << DriverVersion;

  DefaultUserCtx = std::make_unique<LevelZeroPluginContextTy>(
      Plugin, /*Devices=*/llvm::ArrayRef<GenericDeviceTy *>{}, zeDriver,
      zeContext, /*OwnsZeContext=*/false);

  return Plugin::success();
}

Error L0ContextTy::deinit() {
  // Release the default context (drains its queue cache) before zeContext.
  if (DefaultUserCtx) {
    if (auto Err = DefaultUserCtx->deinit())
      return Err;
    DefaultUserCtx.reset();
  }
  if (auto Err = EventPool.deinit())
    return Err;
  if (auto Err = HostMemAllocator.deinit())
    return Err;
  if (zeContext)
    CALL_ZE_RET_ERROR(zeContextDestroy, zeContext);
  return Plugin::success();
}

StagingBufferTy &L0ContextTy::getStagingBuffer() {
  auto &TLS = Plugin.getContextTLS(getZeContext());
  auto &Buffer = TLS.getStagingBuffer();
  const auto &Options = Plugin.getOptions();
  if (!Buffer.initialized())
    Buffer.init(getZeContext(), Options.StagingBufferSize,
                Options.StagingBufferCount);
  return Buffer;
}

} // namespace llvm::omp::target::plugin
