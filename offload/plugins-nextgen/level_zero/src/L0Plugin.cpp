//===--- Target RTLs Implementation ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for SPIR-V/Xe machine
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
// Common data across all possible plugin instantiations
L0OptionsTy LevelZeroPluginTy::Options;

Expected<int32_t> LevelZeroPluginTy::findDevices() {
  CALL_ZE_RET_ERROR(zeInit, ZE_INIT_FLAG_GPU_ONLY);
  uint32_t NumDrivers = 0;
  CALL_ZE_RET_ERROR(zeDriverGet, &NumDrivers, nullptr);
  if (NumDrivers == 0) {
    DP("Cannot find any drivers.\n");
    return 0;
  }

  // We expect multiple drivers on Windows to support different device types,
  // so we need to maintain multiple drivers and contexts in general.
  llvm::SmallVector<ze_driver_handle_t> FoundDrivers(NumDrivers);
  CALL_ZE_RET_ERROR(zeDriverGet, &NumDrivers, FoundDrivers.data());

  struct RootInfoTy {
    uint32_t OrderId;
    ze_device_handle_t zeDevice;
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
      DP("Cannot find any devices from driver " DPxMOD ".\n", DPxPTR(Driver));
      continue;
    }
    // We have a driver that supports at least one device
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

  // move discrete devices to the front
  std::sort(RootDevices.begin(), RootDevices.end(),
            [](const RootInfoTy &A, const RootInfoTy &B) {
              // if both are discrete, order by OrderId
              // if both are not discrete, order by OrderId
              // Otherwise, discrete goes first

              if (A.IsDiscrete && B.IsDiscrete)
                return A.OrderId < B.OrderId;
              if (!A.IsDiscrete && !B.IsDiscrete)
                return A.OrderId < B.OrderId;
              return A.IsDiscrete;
            });

  for (size_t RootId = 0; RootId < RootDevices.size(); RootId++) {
    const auto zeDevice = RootDevices[RootId].zeDevice;
    auto *RootDriver = RootDevices[RootId].Driver;
    DetectedDevices.push_back(DeviceInfoTy{
        {zeDevice, static_cast<int32_t>(RootId), -1, -1}, RootDriver});
  }
  int32_t NumDevices = DetectedDevices.size();

  DP("Found %" PRIu32 " devices.\n", NumDevices);
  DP("List of devices (DeviceID[.SubID[.CCSID]])\n");
  for (auto &DeviceInfo : DetectedDevices) {
    (void)DeviceInfo; // to avoid unused variable warning in non-debug builds
    DP("-- Device %" PRIu32 "%s%s (zeDevice=" PRIu64 ") from Driver %p\n",
       DeviceInfo.Id.RootId,
       (DeviceInfo.Id.SubId < 0
            ? ""
            : ("." + std::to_string(DeviceInfo.Id.SubId)).c_str()),
       (DeviceInfo.Id.CCSId < 0
            ? ""
            : ("." + std::to_string(DeviceInfo.Id.CCSId)).c_str()),
       DPxPTR(DeviceInfo.Id.zeId), DPxPTR(DeviceInfo.Driver));
  }
  return NumDevices;
}

Expected<int32_t> LevelZeroPluginTy::initImpl() {
  DP("Level0 NG plugin initialization\n");
  // process options before anything else
  Options.init();
  return findDevices();
}

Error LevelZeroPluginTy::deinitImpl() {
  DP("Deinit Level0 plugin!\n");
  if (auto Err = ContextTLSTable.deinit())
    return Err;
  DeviceTLSTable.clear();
  ThreadTLSTable.clear();
  for (auto &Context : ContextList)
    if (auto Err = Context.deinit())
      return Err;
  ContextList.clear();
  DP("Level0 plugin deinitialized successfully\n");
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

  auto *NewDevice = new L0DeviceTy(
      static_cast<LevelZeroPluginTy &>(Plugin), DeviceId, NumDevices, zeDevice,
      *zeDriver, std::move(IdStr), CCSId < 0 ? 0 : CCSId /* ComputeIndex */);
  if (NewDevice && getDebugLevel() > 0) {
    DP("Device %" PRIi32 " information\n", DeviceId);
    NewDevice->reportDeviceInfo();
  }
  return NewDevice;
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

  // L0 object
  const auto L0 = static_cast<L0Interop::Property *>(Interop->rtl_property);
  const auto device_id = Interop->device_id;
  auto &l0Device = getDeviceFromId(device_id);

  // We can synchronize both L0 & SYCL objects with the same ze command
  if (l0Device.useImmForInterop()) {
    DP("LevelZeroPluginTy::sync_barrier: Synchronizing " DPxMOD
       " with ImmCmdList barrier\n",
       DPxPTR(Interop));
    auto ImmCmdList = L0->ImmCmdList;

    CALL_ZE_RET_ERROR(zeCommandListHostSynchronize, ImmCmdList, UINT64_MAX);
  } else {
    DP("LevelZeroPluginTy::sync_barrier: Synchronizing " DPxMOD
       " with queue synchronize\n",
       DPxPTR(Interop));
    auto CmdQueue = L0->CommandQueue;
    CALL_ZE_RET_ERROR(zeCommandQueueSynchronize, CmdQueue, UINT64_MAX);
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
    DP("LevelZeroPluginTy::async_barrier: Appending ImmCmdList barrier "
       "to " DPxMOD "\n",
       DPxPTR(Interop));
    auto ImmCmdList = L0->ImmCmdList;
    CALL_ZE_RET_ERROR(zeCommandListAppendBarrier, ImmCmdList, nullptr, 0,
                      nullptr);
  } else {
#if 0
    // TODO: re-enable once we have a way to delay the CmdList reset 
    DP("LevelZeroPluginTy::async_barrier: Appending CmdList barrier to " DPxMOD
       "\n",
       DPxPTR(Interop));
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
