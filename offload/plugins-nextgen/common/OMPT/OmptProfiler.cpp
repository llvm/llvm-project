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
#include "PluginInterface.h"
#include "Shared/Debug.h"

void llvm::omp::target::ompt::OmptProfilerTy::handleInit(
    llvm::omp::target::plugin::GenericDeviceTy *Device,
    llvm::omp::target::plugin::GenericPluginTy *Plugin) {
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

void llvm::omp::target::ompt::OmptProfilerTy::handleDeinit(
    llvm::omp::target::plugin::GenericDeviceTy *Device,
    llvm::omp::target::plugin::GenericPluginTy *Plugin) {
  auto DeviceId = Device->getDeviceId();

  if (ompt::Initialized) {
    bool ExpectedStatus = true;
    if (OmptInitialized.compare_exchange_strong(ExpectedStatus, false))
      performOmptCallback(device_finalize, Plugin->getUserId(DeviceId));
  }
  ompt::removeDeviceId(reinterpret_cast<ompt_device_t *>(Device));
}

void llvm::omp::target::ompt::OmptProfilerTy::handleLoadBinary(
    llvm::omp::target::plugin::GenericDeviceTy *Device,
    llvm::omp::target::plugin::GenericPluginTy *Plugin,
    const StringRef InputTgtImage) {
  auto DeviceId = Device->getDeviceId();

  if (ompt::Initialized) {
    size_t Bytes = InputTgtImage.size();
    performOmptCallback(
        device_load, Plugin->getUserId(DeviceId),
        /*FileName=*/nullptr, /*FileOffset=*/0, /*VmaInFile=*/nullptr,
        /*ImgSize=*/Bytes,
        /*HostAddr=*/const_cast<unsigned char *>(InputTgtImage.bytes_begin()),
        /*DeviceAddr=*/nullptr, /* FIXME: ModuleId */ 0);
  }
}

void llvm::omp::target::ompt::OmptProfilerTy::handleDataAlloc(
    uint64_t StartNanos, uint64_t EndNanos, void *HostPtr, uint64_t Size,
    void *Data) {
  ompt::setOmptTimestamp(StartNanos, EndNanos);
}

void llvm::omp::target::ompt::OmptProfilerTy::handleDataDelete(
    uint64_t StartNanos, uint64_t EndNanos, void *TgtPtr, void *Data) {
  ompt::setOmptTimestamp(StartNanos, EndNanos);
}

void llvm::omp::target::ompt::OmptProfilerTy::handlePreKernelLaunch(
    llvm::omp::target::plugin::GenericDeviceTy *Device, uint32_t NumBlocks[3],
    __tgt_async_info *AI) {
  OMPT_IF_TRACING_ENABLED(
      if (llvm::omp::target::ompt::isTracedDevice(getDeviceId(Device))) {
        if (AI->ProfilerData != nullptr) {
          auto ProfilerSpecificData = reinterpret_cast<ompt::OmptEventInfoTy *>(AI->ProfilerData);
          // Set number of granted teams for OMPT
          setOmptGrantedNumTeams(NumBlocks[0]);
          ProfilerSpecificData->NumTeams = NumBlocks[0];
        }
      });
}

void llvm::omp::target::ompt::OmptProfilerTy::handleKernelCompletion(
    uint64_t StartNanos, uint64_t EndNanos, void *Data) {

  if (!isProfilingEnabled())
    return;

  DP("OMPT-Async: Time kernel for asynchronous execution (Plugin): Start %lu "
     "End %lu\n",
     StartNanos, EndNanos);

  auto OmptEventInfo = reinterpret_cast<ompt::OmptEventInfoTy *>(Data);
  assert(OmptEventInfo && "Invalid OmptEventInfo");
  assert(OmptEventInfo->TraceRecord && "Invalid TraceRecord");

  llvm::omp::target::ompt::RegionInterface.stopTargetSubmitTraceAsync(
      OmptEventInfo->TraceRecord, OmptEventInfo->NumTeams, StartNanos,
      EndNanos);

  // Done processing, our responsibility to free the memory
  freeProfilerDataEntry(OmptEventInfo);
}

void llvm::omp::target::ompt::OmptProfilerTy::handleDataTransfer(
    uint64_t StartNanos, uint64_t EndNanos, void *Data) {

  if (!isProfilingEnabled())
    return;

  auto OmptEventInfo = reinterpret_cast<ompt::OmptEventInfoTy *>(Data);
  assert(OmptEventInfo && "Invalid OmptEventInfo");
  assert(OmptEventInfo->TraceRecord && "Invalid TraceRecord");

  llvm::omp::target::ompt::RegionInterface.stopTargetDataMovementTraceAsync(
      OmptEventInfo->TraceRecord, StartNanos, EndNanos);

  // Done processing, our responsibility to free the memory
  freeProfilerDataEntry(OmptEventInfo);
}

bool llvm::omp::target::ompt::OmptProfilerTy::isProfilingEnabled() {
  return llvm::omp::target::ompt::TracingActive;
}

void llvm::omp::target::ompt::OmptProfilerTy::setTimeConversionFactorsImpl(
    double Slope, double Offset) {
  DP("Using Time Slope: %f and Offset: %f \n", Slope, Offset);
  setOmptHostToDeviceRate(Slope, Offset);
}
