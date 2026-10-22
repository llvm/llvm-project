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

using namespace error;

Error LevelZeroPluginContextTy::initAsyncInfoImpl(
    GenericDeviceTy &Device, AsyncInfoWrapperTy &AsyncInfoWrapper) {
  auto &L0Device = static_cast<L0DeviceTy &>(Device);
  auto QueueOrErr = L0Device.getOrCreateQueue(AsyncInfoWrapper, this);
  return QueueOrErr ? Plugin::success() : QueueOrErr.takeError();
}

Error LevelZeroPluginContextTy::deinit() {
  if (auto Err = QueueCache.deinit())
    return Err;
  // Tear down allocators before their ze_context.
  for (auto &KV : DeviceAllocators)
    if (auto Err = KV.second->deinit())
      return Err;
  DeviceAllocators.clear();
  if (auto Err = HostAllocator.deinit())
    return Err;
  if (OwnsZeContext && ZeContext) {
    CALL_ZE_RET_ERROR(zeContextDestroy, ZeContext);
    ZeContext = nullptr;
    OwnsZeContext = false;
  }
  return Plugin::success();
}

Error LevelZeroPluginContextTy::initAllocators() {
  const auto &Options = static_cast<LevelZeroPluginTy &>(Plugin).getOptions();
  for (auto *D : Devices) {
    auto &L0Device = static_cast<L0DeviceTy &>(*D);
    auto Alloc = std::make_unique<MemAllocatorTy>();
    if (auto Err = Alloc->initDevicePools(L0Device, Options, ZeContext))
      return Err;
    DeviceAllocators.try_emplace(&L0Device, std::move(Alloc));
  }
  if (auto Err = HostAllocator.initHostPool(Driver, Options, ZeContext))
    return Err;
  // Host MaxAllocSize = min over devices. The default context has no devices,
  // so each device lowers it from L0DeviceTy::initImpl instead.
  for (auto *D : Devices)
    HostAllocator.updateMaxAllocSize(static_cast<L0DeviceTy &>(*D));
  return Plugin::success();
}

Expected<void *> LevelZeroPluginContextTy::allocate(
    GenericDeviceTy &Device, int64_t Size, void * /*HostPtr*/,
    TargetAllocTy Kind, size_t Alignment, GenericProfilerTy * /*ProfilerPtr*/) {
  MemAllocatorTy *Allocator = nullptr;
  int32_t ResolvedKind = Kind;
  if (Kind == TARGET_ALLOC_HOST) {
    Allocator = &HostAllocator;
  } else {
    if (ResolvedKind == TARGET_ALLOC_DEFAULT)
      ResolvedKind = TARGET_ALLOC_DEVICE;
    auto &L0Device = static_cast<L0DeviceTy &>(Device);
    auto It = DeviceAllocators.find(&L0Device);
    if (It == DeviceAllocators.end())
      return Plugin::error(ErrorCode::INVALID_DEVICE,
                           "device is not part of this context");
    Allocator = It->second.get();
  }
  return Allocator->alloc(Size, Alignment, ResolvedKind, /*Offset=*/0,
                          /*UserAlloc=*/true, /*DevMalloc=*/false,
                          /*MemAdvice=*/
                          std::numeric_limits<uint32_t>::max(),
                          AllocOptionTy::ALLOC_OPT_NONE);
}

Error LevelZeroPluginContextTy::deallocate(
    GenericDeviceTy &Device, void *Ptr, TargetAllocTy Kind,
    GenericProfilerTy * /*ProfilerPtr*/) {
  if (Kind == TARGET_ALLOC_HOST)
    return HostAllocator.dealloc(Ptr);
  auto &L0Device = static_cast<L0DeviceTy &>(Device);
  auto It = DeviceAllocators.find(&L0Device);
  if (It == DeviceAllocators.end())
    return Plugin::error(ErrorCode::INVALID_DEVICE,
                         "device is not part of this context");
  return It->second->dealloc(Ptr);
}

Expected<PluginAllocInfoTy>
LevelZeroPluginContextTy::getAllocInfo(const void *Ptr) {
  void *Raw = const_cast<void *>(Ptr);
  for (const auto &[Device, Allocator] : DeviceAllocators) {
    if (auto *Info = Allocator->getAllocInfo(Raw))
      return PluginAllocInfoTy{Device, static_cast<TargetAllocTy>(Info->Kind),
                               Info->Base, Info->ReqSize};
  }
  if (auto *Info = HostAllocator.getAllocInfo(Raw))
    return PluginAllocInfoTy{nullptr, static_cast<TargetAllocTy>(Info->Kind),
                             Info->Base, Info->ReqSize};

  return Plugin::error(error::ErrorCode::NOT_FOUND,
                       "pointer is not a known allocation in this context");
}

} // namespace llvm::omp::target::plugin
