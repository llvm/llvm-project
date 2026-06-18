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
#include "L0Device.h"
#include "L0Plugin.h"

namespace llvm::omp::target::plugin {

L0ContextTy::L0ContextTy(LevelZeroPluginTy &Plugin, ze_driver_handle_t zeDriver,
                         int32_t /*DriverId*/)
    : PluginContextTy(Plugin, /*Devices=*/{}), zeDriver(zeDriver),
      OwnsZeContext(true) {}

L0ContextTy::L0ContextTy(LevelZeroPluginTy &Plugin, ze_driver_handle_t zeDriver,
                         ze_context_handle_t AdoptedZeContext,
                         bool OwnsZeContext,
                         llvm::ArrayRef<GenericDeviceTy *> Devices)
    : PluginContextTy(Plugin, Devices), zeDriver(zeDriver),
      zeContext(AdoptedZeContext), OwnsZeContext(OwnsZeContext) {}

L0ContextTy::~L0ContextTy() {
  if (zeContext)
    consumeError(deinit());
}

LevelZeroPluginTy &L0ContextTy::getPlugin() const {
  return static_cast<LevelZeroPluginTy &>(Plugin);
}

Error L0ContextTy::init() {
  auto cleanupOnError = [&]() {
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
    cleanupOnError();
    return Err;
  }
  if (auto Err =
          HostMemAllocator.initHostPool(*this, getPlugin().getOptions())) {
    if (auto DeinitErr = EventPool.deinit())
      Err = joinErrors(std::move(Err), std::move(DeinitErr));
    cleanupOnError();
    return Err;
  }

  ze_result_t RC;
  CALL_ZE(RC, zeDriverGetExtensionFunctionAddress, zeDriver,
          "zexKernelGetArgumentSize", (void **)&zexKernelGetArgumentSize);
  if (RC != ZE_RESULT_SUCCESS)
    zexKernelGetArgumentSize = nullptr;

  CALL_ZE(RC, zeDriverGetExtensionFunctionAddress, zeDriver,
          "zeDriverGetDefaultContext", (void **)&zeDriverGetDefaultContext);
  if (RC != ZE_RESULT_SUCCESS)
    zeDriverGetDefaultContext = nullptr;

  return Plugin::success();
}

Error L0ContextTy::initWithDevices() {
  CALL_ZE_RET_ERROR(zeDriverGetApiVersion, zeDriver, &APIVersion);

  if (auto Err = EventPool.init(zeContext, 0))
    return Err;

  const auto &Options = getPlugin().getOptions();
  if (auto Err = HostMemAllocator.initHostPool(*this, Options))
    return Err;

  DeviceAllocators.reserve(Devices.size());
  for (auto *D : Devices) {
    auto &L0Dev = static_cast<L0DeviceTy &>(*D);
    auto Alloc = std::make_unique<MemAllocatorTy>();
    if (auto Err = Alloc->initDevicePools(L0Dev, Options, /*Context=*/this))
      return Err;
    Alloc->updateMaxAllocSize(L0Dev);
    DeviceAllocators.push_back(std::move(Alloc));
  }
  return Plugin::success();
}

Error L0ContextTy::deinit() {
  if (auto Err = EventPool.deinit())
    return Err;
  if (auto Err = HostMemAllocator.deinit())
    return Err;
  for (auto &A : DeviceAllocators)
    if (auto Err = A->deinit())
      return Err;
  DeviceAllocators.clear();
  if (zeContext) {
    if (OwnsZeContext)
      CALL_ZE_RET_ERROR(zeContextDestroy, zeContext);
    zeContext = nullptr;
  }
  return Plugin::success();
}

Expected<void *> L0ContextTy::allocate(GenericDeviceTy &Device, int64_t Size,
                                       TargetAllocTy Kind) {
  auto &L0Dev = static_cast<L0DeviceTy &>(Device);
  MemAllocatorTy *Allocator = nullptr;
  if (Kind == TARGET_ALLOC_HOST) {
    Allocator = &HostMemAllocator;
  } else {
    for (int i = 0; i < (int)Devices.size(); ++i) {
      if (Devices[i] == &L0Dev) {
        Allocator = DeviceAllocators[i].get();
        break;
      }
    }
  }
  if (!Allocator)
    return Plugin::error(ErrorCode::INVALID_DEVICE,
                         "device is not part of this plugin context");

  return Allocator->alloc(Size, /*Align=*/0, Kind, /*Offset=*/0,
                          /*UserAlloc=*/true, /*DevMalloc=*/false,
                          /*MemAdvice=*/0, AllocOptionTy::ALLOC_OPT_NONE);
}

Error L0ContextTy::deallocate(void *Ptr) {
  if (HostMemAllocator.getAllocInfo(Ptr))
    return HostMemAllocator.dealloc(Ptr);
  for (auto &Allocator : DeviceAllocators)
    if (Allocator->getAllocInfo(Ptr))
      return Allocator->dealloc(Ptr);
  return Plugin::error(
      ErrorCode::INVALID_CONTEXT,
      "address is not a known allocation in the given context");
}

Expected<PluginAllocInfoTy> L0ContextTy::getAllocInfo(const void *Ptr) {
  auto Build = [](GenericDeviceTy *Device, const MemAllocInfoTy *Info) {
    return PluginAllocInfoTy{Device, static_cast<TargetAllocTy>(Info->Kind),
                             Info->Base, Info->ReqSize};
  };

  // OL_MEM_INFO_DEVICE is rejected at the API layer for host allocations,
  // so the owner is unused here; report null and let the caller decide.
  if (auto *Info = HostMemAllocator.getAllocInfoContaining(Ptr))
    return Build(/*Device=*/nullptr, Info);
  for (int i = 0; i < (int)Devices.size(); ++i) {
    if (auto *Info = DeviceAllocators[i]->getAllocInfoContaining(Ptr))
      return Build(Devices[i], Info);
  }
  return Plugin::error(ErrorCode::NOT_FOUND,
                       "allocated memory information not found in context");
}

StagingBufferTy &L0ContextTy::getStagingBuffer() {
  auto &TLS = getPlugin().getContextTLS(getZeContext());
  auto &Buffer = TLS.getStagingBuffer();
  const auto &Options = getPlugin().getOptions();
  if (!Buffer.initialized())
    Buffer.init(getZeContext(), Options.StagingBufferSize,
                Options.StagingBufferCount);
  return Buffer;
}

} // namespace llvm::omp::target::plugin
