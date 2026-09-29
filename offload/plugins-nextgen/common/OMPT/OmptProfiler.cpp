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

#include <memory>

using namespace llvm::omp::target;

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
  if (!ompt::isTracedDevice(
          ompt::getDeviceId(reinterpret_cast<ompt_device_t *>(Device))))
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

  // Null data means no trace record was assigned for this event, see
  // TracerInterfaceRAII in OpenMP/OMPT/Interface.h.
  if (!Data)
    return;

  ODBG(ODT_Tool) << "OMPT-Async: Time kernel for asynchronous execution: Start "
                 << StartNanos << " End " << EndNanos;

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

  // Null data means no trace record was assigned for this event, see
  // TracerInterfaceRAII in OpenMP/OMPT/Interface.h.
  if (!Data)
    return;

  ODBG(ODT_Tool) << "OMPT-Async: Time data for asynchronous execution: Start "
                 << StartNanos << " End " << EndNanos;

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
  ODBG(ODT_Tool) << "Using Time Slope: " << Slope << " and Offset: " << Offset;
  setOmptHostToDeviceRate(Slope, Offset);
}
