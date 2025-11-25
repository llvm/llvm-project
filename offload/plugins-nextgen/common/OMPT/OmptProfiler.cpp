//===- OmptProfiler.cpp - OMPT impl of GenericProfilerTy --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of OmptProfilerTy
//
//===----------------------------------------------------------------------===//

#include "OmptProfiler.h"
#include "OpenMP/OMPT/Interface.h"
#include "PluginInterface.h"
#include "Shared/Debug.h"

using namespace llvm::omp::target;

void ompt::OmptProfilerTy::handleInit(plugin::GenericDeviceTy *Device,
                                      plugin::GenericPluginTy *Plugin) {
  auto DeviceId = Device->getDeviceId();
  auto DevicePtr = reinterpret_cast<ompt_device_t *>(Device);
  ompt::setDeviceId(DevicePtr, Plugin->getUserId(DeviceId));

  if (ompt::Initialized) {
    bool ExpectedStatus = false;
    if (OmptInitialized.compare_exchange_strong(ExpectedStatus, true))
      performOmptCallback(device_initialize, Plugin->getUserId(DeviceId),
                          /*type=*/Device->getComputeUnitKind().c_str(),
                          /*device=*/DevicePtr,
                          /*lookup=*/ompt::lookupDeviceTracingFn,
                          /*documentation=*/nullptr);
  }
}

void ompt::OmptProfilerTy::handleDeinit(
    plugin::GenericDeviceTy *Device, target::plugin::GenericPluginTy *Plugin) {
  auto DeviceId = Device->getDeviceId();

  if (ompt::Initialized) {
    bool ExpectedStatus = true;
    if (OmptInitialized.compare_exchange_strong(ExpectedStatus, false))
      performOmptCallback(device_finalize, Plugin->getUserId(DeviceId));
  }
  ompt::removeDeviceId(reinterpret_cast<ompt_device_t *>(Device));
}

void ompt::OmptProfilerTy::handleLoadBinary(plugin::GenericDeviceTy *Device,
                                            plugin::GenericPluginTy *Plugin,
                                            const StringRef InputTgtImage) {

  if (!ompt::Initialized)
    return;

  auto DeviceId = Device->getDeviceId();
  size_t Bytes = InputTgtImage.size();
  performOmptCallback(
      device_load, Plugin->getUserId(DeviceId),
      /*FileName=*/nullptr, /*FileOffset=*/0, /*VmaInFile=*/nullptr,
      /*ImgSize=*/Bytes,
      /*HostAddr=*/const_cast<unsigned char *>(InputTgtImage.bytes_begin()),
      /*DeviceAddr=*/nullptr, /* FIXME: ModuleId */ 0);
}

void ompt::OmptProfilerTy::handleDataAlloc(uint64_t StartNanos,
                                           uint64_t EndNanos, void *HostPtr,
                                           uint64_t Size, void *Data) {
  ompt::setOmptTimestamp(StartNanos, EndNanos);
}

void ompt::OmptProfilerTy::handleDataDelete(uint64_t StartNanos,
                                            uint64_t EndNanos, void *TgtPtr,
                                            void *Data) {
  ompt::setOmptTimestamp(StartNanos, EndNanos);
}

void ompt::OmptProfilerTy::handlePreKernelLaunch(
    plugin::GenericDeviceTy *Device, uint32_t NumBlocks[3],
    __tgt_async_info *AI) {
  if (!ompt::isTracedDevice(getDeviceId(Device)))
    return;

  if (AI->ProfilerData == nullptr)
    return;

  auto ProfilerSpecificData =
      reinterpret_cast<ompt::OmptEventInfoTy *>(AI->ProfilerData);
  assert(ProfilerSpecificData && "Invalid ProfilerSpecificData");
  // Set number of granted teams for OMPT
  setOmptGrantedNumTeams(NumBlocks[0]);
  ProfilerSpecificData->NumTeams = NumBlocks[0];
}

void ompt::OmptProfilerTy::handleKernelCompletion(uint64_t StartNanos,
                                                  uint64_t EndNanos,
                                                  void *Data) {

  if (!isProfilingEnabled())
    return;

  /// Empty data means no tracing in OMPT
  /// offload/include/OpenMP/OMPT/Interface.h line 492
  if (!Data)
    return;

  DP("OMPT-Async: Time kernel for asynchronous execution: Start %lu "
     "End %lu\n",
     StartNanos, EndNanos);

  auto OmptEventInfo = reinterpret_cast<ompt::OmptEventInfoTy *>(Data);
  assert(OmptEventInfo && "Invalid OmptEventInfo");
  assert(OmptEventInfo->TraceRecord && "Invalid TraceRecord");

  ompt::RegionInterface.stopTargetSubmitTraceAsync(OmptEventInfo->TraceRecord,
                                                   OmptEventInfo->NumTeams,
                                                   StartNanos, EndNanos);

  // Done processing, our responsibility to free the memory
  freeProfilerDataEntry(OmptEventInfo);
}

void ompt::OmptProfilerTy::handleDataTransfer(uint64_t StartNanos,
                                              uint64_t EndNanos, void *Data) {

  if (!isProfilingEnabled())
    return;

  /// Empty data means no tracing in OMPT
  /// offload/include/OpenMP/OMPT/Interface.h line 492
  if (!Data)
    return;

  DP("OMPT-Async: Time data for asynchronous execution: Start %lu "
     "End %lu\n",
     StartNanos, EndNanos);

  auto OmptEventInfo = reinterpret_cast<ompt::OmptEventInfoTy *>(Data);
  assert(OmptEventInfo && "Invalid OmptEventInfo");
  assert(OmptEventInfo->TraceRecord && "Invalid TraceRecord");

  ompt::RegionInterface.stopTargetDataMovementTraceAsync(
      OmptEventInfo->TraceRecord, StartNanos, EndNanos);

  // Done processing, our responsibility to free the memory
  freeProfilerDataEntry(OmptEventInfo);
}

bool ompt::OmptProfilerTy::isProfilingEnabled() { return ompt::TracingActive; }

void ompt::OmptProfilerTy::setTimeConversionFactorsImpl(double Slope,
                                                        double Offset) {
  DP("Using Time Slope: %f and Offset: %f \n", Slope, Offset);
  setOmptHostToDeviceRate(Slope, Offset);
}
