//===- InternalEventOperators.cpp - Operator implementations ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the internal event operators, like comparators.
///
//===----------------------------------------------------------------------===//

#include "InternalEvent.h"

namespace omptest {

namespace internal {

bool operator==(const ParallelBegin &Expected, const ParallelBegin &Observed) {
  return Expected.NumThreads == Observed.NumThreads;
}

bool operator==(const Work &Expected, const Work &Observed) {
  bool isSameWorkType = (Expected.WorkType == Observed.WorkType);
  bool isSameEndpoint = (Expected.Endpoint == Observed.Endpoint);
  bool isSameParallelData =
      (Expected.ParallelData == std::numeric_limits<ompt_data_t *>::min()) ||
      (Expected.ParallelData == Observed.ParallelData);
  bool isSameTaskData =
      (Expected.TaskData == std::numeric_limits<ompt_data_t *>::min()) ||
      (Expected.TaskData == Observed.TaskData);
  bool isSameCount = (Expected.Count == std::numeric_limits<uint64_t>::min()) ||
                     (Expected.Count == Observed.Count);
  return isSameWorkType && isSameEndpoint && isSameParallelData &&
         isSameTaskData && isSameCount;
}

bool operator==(const ImplicitTask &Expected, const ImplicitTask &Observed) {
  bool isSameEndpoint = (Expected.Endpoint == Observed.Endpoint);
  bool isSameActualParallelism =
      (Expected.ActualParallelism ==
       std::numeric_limits<unsigned int>::min()) ||
      (Expected.ActualParallelism == Observed.ActualParallelism);
  bool isSameIndex =
      (Expected.Index == std::numeric_limits<unsigned int>::min()) ||
      (Expected.Index == Observed.Index);
  return isSameEndpoint && isSameActualParallelism && isSameIndex;
}

bool operator==(const SyncRegion &Expected, const SyncRegion &Observed) {
  bool isSameKind = (Expected.Kind == Observed.Kind);
  bool isSameEndpoint = (Expected.Endpoint == Observed.Endpoint);
  bool isSameParallelData =
      (Expected.ParallelData == std::numeric_limits<ompt_data_t *>::min()) ||
      (Expected.ParallelData == Observed.ParallelData);
  bool isSameTaskData =
      (Expected.TaskData == std::numeric_limits<ompt_data_t *>::min()) ||
      (Expected.TaskData == Observed.TaskData);
  return isSameKind && isSameEndpoint && isSameParallelData && isSameTaskData;
}

bool operator==(const Target &Expected, const Target &Observed) {
  bool isSameKind = (Expected.Kind == Observed.Kind);
  bool isSameEndpoint = (Expected.Endpoint == Observed.Endpoint);
  bool isSameDeviceNum =
      (Expected.DeviceNum == std::numeric_limits<int>::min()) ||
      (Expected.DeviceNum == Observed.DeviceNum);
  return isSameKind && isSameEndpoint && isSameDeviceNum;
}

bool operator==(const TargetEmi &Expected, const TargetEmi &Observed) {
  bool isSameKind = (Expected.Kind == Observed.Kind);
  bool isSameEndpoint = (Expected.Endpoint == Observed.Endpoint);
  bool isSameDeviceNum =
      (Expected.DeviceNum == std::numeric_limits<int>::min()) ||
      (Expected.DeviceNum == Observed.DeviceNum);
  return isSameKind && isSameEndpoint && isSameDeviceNum;
}

bool operator==(const TargetDataOp &Expected, const TargetDataOp &Observed) {
  bool isSameOpType = (Expected.OpType == Observed.OpType);
  bool isSameSize = (Expected.Bytes == std::numeric_limits<size_t>::min()) ||
                    (Expected.Bytes == Observed.Bytes);
  bool isSameSrcAddr =
      (Expected.SrcAddr == std::numeric_limits<void *>::min()) ||
      (Expected.SrcAddr == Observed.SrcAddr);
  bool isSameDstAddr =
      (Expected.DstAddr == std::numeric_limits<void *>::min()) ||
      (Expected.DstAddr == Observed.DstAddr);
  bool isSameSrcDeviceNum =
      (Expected.SrcDeviceNum == std::numeric_limits<int>::min()) ||
      (Expected.SrcDeviceNum == Observed.SrcDeviceNum);
  bool isSameDstDeviceNum =
      (Expected.DstDeviceNum == std::numeric_limits<int>::min()) ||
      (Expected.DstDeviceNum == Observed.DstDeviceNum);
  return isSameOpType && isSameSize && isSameSrcAddr && isSameDstAddr &&
         isSameSrcDeviceNum && isSameDstDeviceNum;
}

bool operator==(const TargetDataOpEmi &Expected,
                const TargetDataOpEmi &Observed) {
  bool isSameOpType = (Expected.OpType == Observed.OpType);
  bool isSameEndpoint = (Expected.Endpoint == Observed.Endpoint);
  bool isSameSize = (Expected.Bytes == std::numeric_limits<size_t>::min()) ||
                    (Expected.Bytes == Observed.Bytes);
  bool isSameSrcAddr =
      (Expected.SrcAddr == std::numeric_limits<void *>::min()) ||
      (Expected.SrcAddr == Observed.SrcAddr);
  bool isSameDstAddr =
      (Expected.DstAddr == std::numeric_limits<void *>::min()) ||
      (Expected.DstAddr == Observed.DstAddr);
  bool isSameSrcDeviceNum =
      (Expected.SrcDeviceNum == std::numeric_limits<int>::min()) ||
      (Expected.SrcDeviceNum == Observed.SrcDeviceNum);
  bool isSameDstDeviceNum =
      (Expected.DstDeviceNum == std::numeric_limits<int>::min()) ||
      (Expected.DstDeviceNum == Observed.DstDeviceNum);
  return isSameOpType && isSameEndpoint && isSameSize && isSameSrcAddr &&
         isSameDstAddr && isSameSrcDeviceNum && isSameDstDeviceNum;
}

bool operator==(const TargetSubmit &Expected, const TargetSubmit &Observed) {
  bool isSameReqNumTeams =
      (Expected.RequestedNumTeams == Observed.RequestedNumTeams);
  return isSameReqNumTeams;
}

bool operator==(const TargetSubmitEmi &Expected,
                const TargetSubmitEmi &Observed) {
  bool isSameReqNumTeams =
      (Expected.RequestedNumTeams == Observed.RequestedNumTeams);
  bool isSameEndpoint = (Expected.Endpoint == Observed.Endpoint);
  return isSameReqNumTeams && isSameEndpoint;
}

bool operator==(const DeviceInitialize &Expected,
                const DeviceInitialize &Observed) {
  bool isSameDeviceNum = (Expected.DeviceNum == Observed.DeviceNum);
  bool isSameType =
      (Expected.Type == std::numeric_limits<const char *>::min()) ||
      ((Expected.Type == Observed.Type) ||
       (strcmp(Expected.Type, Observed.Type) == 0));
  bool isSameDevice =
      (Expected.Device == std::numeric_limits<ompt_device_t *>::min()) ||
      (Expected.Device == Observed.Device);
  return isSameDeviceNum && isSameType && isSameDevice;
}

bool operator==(const DeviceFinalize &Expected,
                const DeviceFinalize &Observed) {
  bool isSameDeviceNum =
      (Expected.DeviceNum == std::numeric_limits<int>::min()) ||
      (Expected.DeviceNum == Observed.DeviceNum);
  return isSameDeviceNum;
}

bool operator==(const DeviceLoad &Expected, const DeviceLoad &Observed) {
  bool isSameDeviceNum =
      (Expected.DeviceNum == std::numeric_limits<int>::min()) ||
      (Expected.DeviceNum == Observed.DeviceNum);
  bool isSameSize = (Expected.Bytes == std::numeric_limits<size_t>::min()) ||
                    (Expected.Bytes == Observed.Bytes);
  return isSameDeviceNum && isSameSize;
}

bool operator==(const BufferRequest &Expected, const BufferRequest &Observed) {
  bool isSameDeviceNum =
      (Expected.DeviceNum == std::numeric_limits<int>::min()) ||
      (Expected.DeviceNum == Observed.DeviceNum);
  bool isSameSize = (Expected.Bytes == std::numeric_limits<size_t *>::min()) ||
                    (Expected.Bytes == Observed.Bytes);
  return isSameDeviceNum && isSameSize;
}

bool operator==(const BufferComplete &Expected,
                const BufferComplete &Observed) {
  bool isSameDeviceNum =
      (Expected.DeviceNum == std::numeric_limits<int>::min()) ||
      (Expected.DeviceNum == Observed.DeviceNum);
  bool isSameSize = (Expected.Bytes == std::numeric_limits<size_t>::min()) ||
                    (Expected.Bytes == Observed.Bytes);
  return isSameDeviceNum && isSameSize;
}

bool operator==(const BufferRecord &Expected, const BufferRecord &Observed) {
  bool isSameType = (Expected.Record.type == Observed.Record.type);
  bool isSameTargetId =
      (Expected.Record.target_id == std::numeric_limits<ompt_id_t>::min()) ||
      (Expected.Record.target_id == Observed.Record.target_id);
  if (!(isSameType && isSameTargetId))
    return false;
  bool isEqual = true;
  ompt_device_time_t ObservedDurationNs =
      Observed.Record.record.target_data_op.end_time - Observed.Record.time;
  switch (Expected.Record.type) {
  case ompt_callback_target:
    isEqual &= (Expected.Record.record.target.kind ==
                std::numeric_limits<ompt_target_t>::min()) ||
               (Expected.Record.record.target.kind ==
                Observed.Record.record.target.kind);
    isEqual &= (Expected.Record.record.target.endpoint ==
                std::numeric_limits<ompt_scope_endpoint_t>::min()) ||
               (Expected.Record.record.target.endpoint ==
                Observed.Record.record.target.endpoint);
    isEqual &= (Expected.Record.record.target.device_num ==
                std::numeric_limits<int>::min()) ||
               (Expected.Record.record.target.device_num ==
                Observed.Record.record.target.device_num);
    break;
  case ompt_callback_target_data_op:
    isEqual &= (Expected.Record.record.target_data_op.optype ==
                std::numeric_limits<ompt_target_data_op_t>::min()) ||
               (Expected.Record.record.target_data_op.optype ==
                Observed.Record.record.target_data_op.optype);
    isEqual &= (Expected.Record.record.target_data_op.bytes ==
                std::numeric_limits<size_t>::min()) ||
               (Expected.Record.record.target_data_op.bytes ==
                Observed.Record.record.target_data_op.bytes);
    isEqual &= (Expected.Record.record.target_data_op.src_addr ==
                std::numeric_limits<void *>::min()) ||
               (Expected.Record.record.target_data_op.src_addr ==
                Observed.Record.record.target_data_op.src_addr);
    isEqual &= (Expected.Record.record.target_data_op.dest_addr ==
                std::numeric_limits<void *>::min()) ||
               (Expected.Record.record.target_data_op.dest_addr ==
                Observed.Record.record.target_data_op.dest_addr);
    isEqual &= (Expected.Record.record.target_data_op.src_device_num ==
                std::numeric_limits<int>::min()) ||
               (Expected.Record.record.target_data_op.src_device_num ==
                Observed.Record.record.target_data_op.src_device_num);
    isEqual &= (Expected.Record.record.target_data_op.dest_device_num ==
                std::numeric_limits<int>::min()) ||
               (Expected.Record.record.target_data_op.dest_device_num ==
                Observed.Record.record.target_data_op.dest_device_num);
    isEqual &= (Expected.Record.record.target_data_op.host_op_id ==
                std::numeric_limits<ompt_id_t>::min()) ||
               (Expected.Record.record.target_data_op.host_op_id ==
                Observed.Record.record.target_data_op.host_op_id);
    isEqual &= (Expected.Record.record.target_data_op.codeptr_ra ==
                std::numeric_limits<void *>::min()) ||
               (Expected.Record.record.target_data_op.codeptr_ra ==
                Observed.Record.record.target_data_op.codeptr_ra);
    if (Expected.Record.record.target_data_op.end_time !=
        std::numeric_limits<ompt_device_time_t>::min()) {
      isEqual &=
          ObservedDurationNs <= Expected.Record.record.target_data_op.end_time;
    }
    isEqual &= ObservedDurationNs >= Expected.Record.time;
    break;
  case ompt_callback_target_submit:
    ObservedDurationNs =
        Observed.Record.record.target_kernel.end_time - Observed.Record.time;
    isEqual &= (Expected.Record.record.target_kernel.requested_num_teams ==
                std::numeric_limits<unsigned int>::min()) ||
               (Expected.Record.record.target_kernel.requested_num_teams ==
                Observed.Record.record.target_kernel.requested_num_teams);
    isEqual &= (Expected.Record.record.target_kernel.granted_num_teams ==
                std::numeric_limits<unsigned int>::min()) ||
               (Expected.Record.record.target_kernel.granted_num_teams ==
                Observed.Record.record.target_kernel.granted_num_teams);
    isEqual &= (Expected.Record.record.target_kernel.host_op_id ==
                std::numeric_limits<ompt_id_t>::min()) ||
               (Expected.Record.record.target_kernel.host_op_id ==
                Observed.Record.record.target_kernel.host_op_id);
    if (Expected.Record.record.target_kernel.end_time !=
        std::numeric_limits<ompt_device_time_t>::min()) {
      isEqual &=
          ObservedDurationNs <= Expected.Record.record.target_kernel.end_time;
    }
    isEqual &= ObservedDurationNs >= Expected.Record.time;
    break;
  default:
    assert(false && "Encountered invalid record type");
  }
  return isEqual;
}

} // namespace internal

} // namespace omptest
