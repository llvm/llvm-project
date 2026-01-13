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

namespace llvm::omp::target::plugin {

using namespace llvm::omp::target;
using namespace error;

#pragma clang diagnostic ignored "-Wglobal-constructors"
// Common data across all possible plugin instantiations.
L0OptionsTy LevelZeroPluginTy::Options;

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
    if (auto Err = DrvInfo.init())
      return std::move(Err);
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
  if (auto Err = DeviceTLSTable.deinit())
    return Err;
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

Error LevelZeroPluginTy::flushQueueImpl(omp_interop_val_t *Interop) {
  return Plugin::success();
}

Expected<bool> LevelZeroPluginTy::isELFCompatible(uint32_t DeviceId,
                                                  StringRef Image) const {
  uint64_t MajorVer, MinorVer;
  return isValidOneOmpImage(Image, MajorVer, MinorVer);
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
  const auto device_id = Interop->device_id;
  auto &l0Device = getDeviceFromId(device_id);

  // We can synchronize both L0 & SYCL objects with the same ze command.
  if (l0Device.useImmForInterop()) {
    ODBG(OLDT_Sync) << "LevelZeroPluginTy::sync_barrier: Synchronizing "
                    << Interop << " with ImmCmdList barrier";
    auto ImmCmdList = L0->ImmCmdList;

    CALL_ZE_RET_ERROR(zeCommandListHostSynchronize, ImmCmdList,
                      L0DefaultTimeout);
  } else {
    ODBG(OLDT_Sync) << "LevelZeroPluginTy::sync_barrier: Synchronizing "
                    << Interop << " with queue synchronize";
    auto CmdQueue = L0->CommandQueue;
    CALL_ZE_RET_ERROR(zeCommandQueueSynchronize, CmdQueue, L0DefaultTimeout);
  }

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
  const auto device_id = Interop->device_id;
  if (Interop->attrs.inorder)
    return Plugin::success();

  auto &l0Device = getDeviceFromId(device_id);
  if (l0Device.useImmForInterop()) {
    ODBG(OLDT_Sync) << "LevelZeroPluginTy::async_barrier: Appending ImmCmdList "
                    << "barrier to " << Interop;
    auto ImmCmdList = L0->ImmCmdList;
    CALL_ZE_RET_ERROR(zeCommandListAppendBarrier, ImmCmdList, nullptr, 0,
                      nullptr);
  } else {
#if 0
    // TODO: re-enable once we have a way to delay the CmdList reset .
    ODBG(OLDT_Sync) << "LevelZeroPluginTy::async_barrier: Appending CmdList "
                   << "barrier to " << Interop;
    auto CmdQueue = L0->CommandQueue;
    ze_command_list_handle_t CmdList = l0Device.getCmdList();
    CALL_ZE_RET_ERROR(zeCommandListAppendBarrier, CmdList, nullptr, 0, nullptr);
    CALL_ZE_RET_ERROR(zeCommandListClose, CmdList);
    CALL_ZE_RET_ERROR(zeCommandQueueExecuteCommandLists, CmdQueue, 1, &CmdList,
                      nullptr);
    CALL_ZE_RET_ERROR(zeCommandListReset, CmdList);
#else
    return syncBarrierImpl(Interop);
#endif
  }

  return Plugin::success();
}

} // namespace llvm::omp::target::plugin

extern "C" {
llvm::omp::target::plugin::GenericPluginTy *createPlugin_level_zero() {
  return new llvm::omp::target::plugin::LevelZeroPluginTy();
}
}
