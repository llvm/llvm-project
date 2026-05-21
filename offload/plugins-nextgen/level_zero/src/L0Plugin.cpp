//===--- Target RTLs Implementation ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for SPIR-V/Xe machine.
//
//===----------------------------------------------------------------------===//

#include <level_zero/zes_api.h>

#include "L0Device.h"
#include "L0Interop.h"
#include "L0Kernel.h"
#include "L0Plugin.h"
#include "L0Trace.h"

#include "llvm/Object/OffloadBinary.h"

namespace llvm::omp::target::plugin {

using namespace llvm::omp::target;
using namespace error;

Expected<int32_t> LevelZeroPluginTy::findDevices() {
  CALL_ZE_RET_ERROR(zeInit, ZE_INIT_FLAG_GPU_ONLY);
  uint32_t NumDrivers = 0;
  CALL_ZE_RET_ERROR(zeDriverGet, &NumDrivers, nullptr);
  if (NumDrivers == 0) {
    ODBG(OLDT_Init) << "Cannot find any drivers.";
    return 0;
  }

  // We expect multiple drivers on Windows to support different device types,
  // so we need to maintain multiple drivers and contexts in general.
  llvm::SmallVector<ze_driver_handle_t> FoundDrivers(NumDrivers);
  CALL_ZE_RET_ERROR(zeDriverGet, &NumDrivers, FoundDrivers.data());

  struct RootInfoTy {
    uint32_t OrderId;
    ze_device_handle_t ZeDevice;
    L0ContextTy *Driver;
    bool IsDiscrete;
  };
  llvm::SmallVector<RootInfoTy> RootDevices;

  uint32_t OrderId = 0;
  for (uint32_t DriverId = 0; DriverId < NumDrivers; DriverId++) {
    const auto &Driver = FoundDrivers[DriverId];
    uint32_t DeviceCount = 0;
    ze_result_t RC;
    CALL_ZE(RC, zeDeviceGet, Driver, &DeviceCount, nullptr);
    if (RC != ZE_RESULT_SUCCESS || DeviceCount == 0) {
      ODBG(OLDT_Init) << "Cannot find any devices from driver " << Driver
                      << ".";
      continue;
    }
    // We have a driver that supports at least one device.
    ContextList.emplace_back(*this, Driver, DriverId);
    auto &DrvInfo = ContextList.back();
    if (auto Err = DrvInfo.init()) {
      // Remove the partially initialized context from the list
      ContextList.pop_back();
      return std::move(Err);
    }
    llvm::SmallVector<ze_device_handle_t> FoundDevices(DeviceCount);
    CALL_ZE_RET_ERROR(zeDeviceGet, Driver, &DeviceCount, FoundDevices.data());

    for (auto &zeDevice : FoundDevices)
      RootDevices.push_back(
          {OrderId++, zeDevice, &DrvInfo, L0DeviceTy::isDiscrete(zeDevice)});
  }

  // Move discrete devices to the front.
  std::sort(RootDevices.begin(), RootDevices.end(),
            [](const RootInfoTy &A, const RootInfoTy &B) {
              // If both are discrete, order by OrderId.
              // If both are not discrete, order by OrderId.
              // Otherwise, discrete goes first.

              if (A.IsDiscrete && B.IsDiscrete)
                return A.OrderId < B.OrderId;
              if (!A.IsDiscrete && !B.IsDiscrete)
                return A.OrderId < B.OrderId;
              return A.IsDiscrete;
            });

  for (size_t RootId = 0; RootId < RootDevices.size(); RootId++) {
    const auto ZeDevice = RootDevices[RootId].ZeDevice;
    auto *RootDriver = RootDevices[RootId].Driver;
    DetectedDevices.push_back(DeviceInfoTy{
        {ZeDevice, static_cast<int32_t>(RootId), -1, -1}, RootDriver});
  }
  int32_t NumDevices = DetectedDevices.size();

  ODBG_OS(OLDT_Init, [&](llvm::raw_ostream &O) {
    O << "Found " << NumDevices << " devices.\n"
      << "List of devices (DeviceID[.SubID[.CCSID]])\n";
    for (auto &DeviceInfo : DetectedDevices)
      O << "-- Device " << DeviceInfo.Id.RootId
        << (DeviceInfo.Id.SubId < 0
                ? ""
                : ("." + std::to_string(DeviceInfo.Id.SubId)))
        << (DeviceInfo.Id.CCSId < 0
                ? ""
                : ("." + std::to_string(DeviceInfo.Id.CCSId)))
        << "\n";
  });
  return NumDevices;
}

Expected<int32_t> LevelZeroPluginTy::initImpl() {
  ODBG(OLDT_Init) << "Level0 NG plugin initialization";
  // Process options before anything else.
  Options.init();
  return findDevices();
}

Error LevelZeroPluginTy::deinitImpl() {
  ODBG(OLDT_Deinit) << "Deinit Level0 plugin!";
  if (auto Err = ContextTLSTable.deinit())
    return Err;
  {
    std::lock_guard<std::mutex> Lock(DefaultContextsMutex);
    DefaultContexts.clear();
  }
  for (auto &Context : ContextList)
    if (auto Err = Context.deinit())
      return Err;
  ContextList.clear();
  ODBG(OLDT_Deinit) << "Level0 plugin deinitialized successfully";
  return Plugin::success();
}

GenericDeviceTy *LevelZeroPluginTy::createDevice(GenericPluginTy &Plugin,
                                                 int32_t DeviceId,
                                                 int32_t NumDevices) {
  auto &DeviceInfo = DetectedDevices[DeviceId];
  auto RootId = DeviceInfo.Id.RootId;
  auto SubId = DeviceInfo.Id.SubId;
  auto CCSId = DeviceInfo.Id.CCSId;
  auto zeDevice = DeviceInfo.Id.zeId;
  auto *zeDriver = DeviceInfo.Driver;

  std::string IdStr = std::to_string(RootId) +
                      (SubId < 0 ? "" : "." + std::to_string(SubId)) +
                      (CCSId < 0 ? "" : "." + std::to_string(CCSId));

  return new L0DeviceTy(static_cast<LevelZeroPluginTy &>(Plugin), DeviceId,
                        NumDevices, zeDevice, *zeDriver, std::move(IdStr),
                        CCSId < 0 ? 0 : CCSId /* ComputeIndex */);
}

GenericGlobalHandlerTy *LevelZeroPluginTy::createGlobalHandler() {
  return new L0GlobalHandlerTy();
}

Expected<std::unique_ptr<PluginContextTy>>
LevelZeroPluginTy::createPluginContext(
    llvm::ArrayRef<GenericDeviceTy *> Devices) {
  if (Devices.empty())
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "createPluginContext called with no devices");

  auto &First = static_cast<L0DeviceTy &>(*Devices[0]);
  L0ContextTy &DriverCtx = First.getL0Context();
  ze_driver_handle_t Driver = DriverCtx.getZeDriver();
  for (auto *D : Devices.drop_front()) {
    auto &L0D = static_cast<L0DeviceTy &>(*D);
    if (&L0D.getL0Context() != &DriverCtx)
      return Plugin::error(
          ErrorCode::INVALID_ARGUMENT,
          "All devices in an L0 context must share the same driver");
  }

  // When the user requests every device on the driver, share the L0 default
  // context so we interop cleanly with other L0 clients on the same driver.
  // Partial device sets get their own fresh context.
  size_t DriverDeviceCount = 0;
  for (int i = 0; i < (int)getNumDevices(); ++i) {
    auto &L0D = static_cast<const L0DeviceTy &>(getDevice(i));
    if (&L0D.getL0Context() == &DriverCtx)
      ++DriverDeviceCount;
  }
  const bool IsFullPlatform = (Devices.size() == DriverDeviceCount);

  ze_context_handle_t ZeContext = nullptr;
  bool OwnsZeContext = false;
  if (IsFullPlatform && DriverCtx.zeDriverGetDefaultContext)
    ZeContext = DriverCtx.zeDriverGetDefaultContext(Driver);
  if (!ZeContext) {
    ze_context_desc_t Desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    CALL_ZE_RET_ERROR(zeContextCreate, Driver, &Desc, &ZeContext);
    OwnsZeContext = true;
  }

  auto Ctx = std::make_unique<L0ContextTy>(*this, Driver, ZeContext,
                                           OwnsZeContext, Devices);
  if (auto Err = Ctx->initWithDevices())
    return std::move(Err);
  return std::unique_ptr<PluginContextTy>(std::move(Ctx));
}

Expected<std::unique_ptr<PluginContextTy>>
LevelZeroPluginTy::createDefaultPluginContext() {
  // The L0 default is per-driver and built lazily by getDefaultContext; the
  // plugin-level DefaultContext is never read. Return a harmless empty
  // placeholder so generic init has something to store.
  return std::make_unique<DefaultPluginContextTy>(*this);
}

Expected<PluginContextTy *>
LevelZeroPluginTy::getDefaultContext(GenericDeviceTy &Device) {
  auto &L0Dev = static_cast<L0DeviceTy &>(Device);
  L0ContextTy *Driver = &L0Dev.getL0Context();

  std::lock_guard<std::mutex> Lock(DefaultContextsMutex);
  auto It = DefaultContexts.find(Driver);
  if (It != DefaultContexts.end())
    return It->second.get();

  llvm::SmallVector<GenericDeviceTy *> DriverDevices;
  for (int i = 0; i < (int)getNumDevices(); ++i) {
    auto &D = static_cast<L0DeviceTy &>(getDevice(i));
    if (&D.getL0Context() == Driver)
      DriverDevices.push_back(&D);
  }

  // Reuse createPluginContext so the default wraps the driver's default
  // ze_context (full-driver case) and gets the same per-device pools as a
  // user-created L0 context.
  auto CtxOrErr = createPluginContext(DriverDevices);
  if (!CtxOrErr)
    return CtxOrErr.takeError();
  PluginContextTy *Raw = CtxOrErr->get();
  DefaultContexts[Driver] = std::move(*CtxOrErr);
  return Raw;
}

Error LevelZeroPluginTy::flushQueueImpl(omp_interop_val_t *Interop) {
  return Plugin::success();
}

Expected<bool> LevelZeroPluginTy::isELFCompatible(uint32_t DeviceId,
                                                  StringRef Image) const {
  uint64_t MajorVer, MinorVer;
  return isValidOneOmpImage(Image, MajorVer, MinorVer);
}

// We only need to check for formats other than ELF here.
Expected<bool> LevelZeroPluginTy::isImageCompatible(StringRef Image) const {
  switch (identify_magic(Image)) {
  case file_magic::spirv_object:
    // Handle SPIRV objects directly
    return true;
  case file_magic::offload_binary: {
    // Handle OffloadBinary format
    MemoryBufferRef Buffer(Image, "offload_binary");
    auto BinariesOrErr = OffloadBinary::create(Buffer);
    if (!BinariesOrErr)
      return BinariesOrErr.takeError();

    auto &Binaries = *BinariesOrErr;
    if (Binaries.size() != 1)
      return false;

    const OffloadBinary *InnerBinary = Binaries[0].get();
    ImageKind ImageKind = InnerBinary->getImageKind();
    llvm::Triple Triple(InnerBinary->getTriple());

    if (Triple.getArch() != getTripleArch())
      return false;

    if (ImageKind != llvm::object::IMG_SPIRV &&
        ImageKind != llvm::object::IMG_Object)
      return false;

    return true;
  }
  default:
    // Unknown format
    return false;
  }
}

Error LevelZeroPluginTy::syncBarrierImpl(omp_interop_val_t *Interop) {
  if (!Interop) {
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "Invalid/inconsistent OpenMP interop " DPxMOD "\n",
                         DPxPTR(Interop));
  }
  if (!Interop->async_info || !Interop->async_info->Queue)
    return Plugin::success();

  const auto L0 = static_cast<L0Interop::Property *>(Interop->rtl_property);

  ODBG(OLDT_Sync) << "LevelZeroPluginTy::sync_barrier: Synchronizing "
                  << Interop << " with ImmCmdList barrier";
  auto ImmCmdList = L0->ImmCmdList;

  CALL_ZE_RET_ERROR(zeCommandListHostSynchronize, ImmCmdList, L0DefaultTimeout);

  return Plugin::success();
}

Error LevelZeroPluginTy::asyncBarrierImpl(omp_interop_val_t *Interop) {
  if (!Interop) {
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "Invalid/inconsistent OpenMP interop " DPxMOD "\n",
                         DPxPTR(Interop));
  }
  if (!Interop->async_info || !Interop->async_info->Queue)
    return Plugin::success();

  const auto L0 = static_cast<L0Interop::Property *>(Interop->rtl_property);
  if (Interop->attrs.inorder)
    return Plugin::success();

  ODBG(OLDT_Sync) << "LevelZeroPluginTy::async_barrier: Appending ImmCmdList "
                  << "barrier to " << Interop;
  auto ImmCmdList = L0->ImmCmdList;
  CALL_ZE_RET_ERROR(zeCommandListAppendBarrier, ImmCmdList, nullptr, 0,
                    nullptr);

  return Plugin::success();
}

} // namespace llvm::omp::target::plugin

extern "C" {
llvm::omp::target::plugin::GenericPluginTy *createPlugin_level_zero() {
  return new llvm::omp::target::plugin::LevelZeroPluginTy();
}
}
