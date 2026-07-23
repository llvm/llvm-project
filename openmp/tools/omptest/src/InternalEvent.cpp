//===- InternalEvent.cpp - Internal event implementation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements internal event representation methods and helper functions.
///
//===----------------------------------------------------------------------===//

#include "InternalEvent.h"

#include <iomanip>
#include <sstream>

using namespace omptest;
using namespace util;

std::string util::makeHexString(uint64_t Data, bool IsPointer, size_t MinBytes,
                                bool ShowHexBase) {
  if (Data == 0 && IsPointer)
    return "(nil)";

  thread_local std::ostringstream os;
  // Clear the content of the stream
  os.str(std::string());

  // Manually prefixing "0x" will make the use of std::setfill more easy
  if (ShowHexBase)
    os << "0x";

  // Default to 32bit (8 hex digits) width, if exceeding 64bit or zero value
  size_t NumDigits = (MinBytes > 0 && MinBytes < 9) ? (MinBytes << 1) : 8;

  if (MinBytes > 0)
    os << std::setfill('0') << std::setw(NumDigits);

  os << std::hex << Data;
  return os.str();
}

std::string internal::AssertionSyncPoint::toString() const {
  std::string S{"Assertion SyncPoint: '"};
  S.append(Name).append(1, '\'');
  return S;
}

std::string internal::ThreadBegin::toString() const {
  std::string S{"OMPT Callback ThreadBegin: "};
  S.append("ThreadType=").append(std::to_string(ThreadType));
  return S;
}

std::string internal::ThreadEnd::toString() const {
  std::string S{"OMPT Callback ThreadEnd"};
  return S;
}

std::string internal::ParallelBegin::toString() const {
  std::string S{"OMPT Callback ParallelBegin: "};
  S.append("NumThreads=").append(std::to_string(NumThreads));
  return S;
}

std::string internal::ParallelEnd::toString() const {
  // TODO: Should we expose more detailed info here?
  std::string S{"OMPT Callback ParallelEnd"};
  return S;
}

std::string internal::Work::toString() const {
  std::string S{"OMPT Callback Work: "};
  S.append("work_type=").append(std::to_string(WorkType));
  S.append(" endpoint=").append(std::to_string(Endpoint));
  S.append(" parallel_data=").append(makeHexString((uint64_t)ParallelData));
  S.append(" task_data=").append(makeHexString((uint64_t)TaskData));
  S.append(" count=").append(std::to_string(Count));
  S.append(" codeptr=").append(makeHexString((uint64_t)CodeptrRA));
  return S;
}

std::string internal::Dispatch::toString() const {
  std::string S{"OMPT Callback Dispatch: "};
  S.append("parallel_data=").append(makeHexString((uint64_t)ParallelData));
  S.append(" task_data=").append(makeHexString((uint64_t)TaskData));
  S.append(" kind=").append(std::to_string(Kind));
  // TODO Check what to print for instance in all different cases
  if (Kind == ompt_dispatch_iteration) {
    S.append(" instance=[it=")
        .append(std::to_string(Instance.value))
        .append(1, ']');
  } else if (Kind == ompt_dispatch_section) {
    S.append(" instance=[ptr=")
        .append(makeHexString((uint64_t)Instance.ptr))
        .append(1, ']');
  } else if ((Kind == ompt_dispatch_ws_loop_chunk ||
              Kind == ompt_dispatch_taskloop_chunk ||
              Kind == ompt_dispatch_distribute_chunk) &&
             Instance.ptr != nullptr) {
    auto Chunk = static_cast<ompt_dispatch_chunk_t *>(Instance.ptr);
    S.append(" instance=[chunk=(start=")
        .append(std::to_string(Chunk->start))
        .append(", iterations=")
        .append(std::to_string(Chunk->iterations))
        .append(")]");
  }
  return S;
}

std::string internal::TaskCreate::toString() const {
  std::string S{"OMPT Callback TaskCreate: "};
  S.append("encountering_task_data=")
      .append(makeHexString((uint64_t)EncounteringTaskData));
  S.append(" encountering_task_frame=")
      .append(makeHexString((uint64_t)EncounteringTaskFrame));
  S.append(" new_task_data=").append(makeHexString((uint64_t)NewTaskData));
  S.append(" flags=").append(std::to_string(Flags));
  S.append(" has_dependences=").append(std::to_string(HasDependences));
  S.append(" codeptr=").append(makeHexString((uint64_t)CodeptrRA));
  return S;
}

std::string internal::ImplicitTask::toString() const {
  std::string S{"OMPT Callback ImplicitTask: "};
  S.append("endpoint=").append(std::to_string(Endpoint));
  S.append(" parallel_data=").append(makeHexString((uint64_t)ParallelData));
  S.append(" task_data=").append(makeHexString((uint64_t)TaskData));
  S.append(" actual_parallelism=").append(std::to_string(ActualParallelism));
  S.append(" index=").append(std::to_string(Index));
  S.append(" flags=").append(std::to_string(Flags));
  return S;
}

std::string internal::SyncRegion::toString() const {
  std::string S{"OMPT Callback SyncRegion: "};
  S.append("kind=").append(std::to_string(Kind));
  S.append(" endpoint=").append(std::to_string(Endpoint));
  S.append(" parallel_data=").append(makeHexString((uint64_t)ParallelData));
  S.append(" task_data=").append(makeHexString((uint64_t)TaskData));
  S.append(" codeptr=").append(makeHexString((uint64_t)CodeptrRA));
  return S;
}

std::string internal::Target::toString() const {
  // TODO Should we canonicalize the string prefix (use "OMPT ..." everywhere)?
  std::string S{"Callback Target: target_id="};
  S.append(std::to_string(TargetId));
  S.append(" kind=").append(std::to_string(Kind));
  S.append(" endpoint=").append(std::to_string(Endpoint));
  S.append(" device_num=").append(std::to_string(DeviceNum));
  S.append(" code=").append(makeHexString((uint64_t)CodeptrRA));
  return S;
}

std::string internal::TargetEmi::toString() const {
  // TODO Should we canonicalize the string prefix (use "OMPT ..." everywhere)?
  std::string S{"Callback Target EMI: kind="};
  S.append(std::to_string(Kind));
  S.append(" endpoint=").append(std::to_string(Endpoint));
  S.append(" device_num=").append(std::to_string(DeviceNum));
  S.append(" task_data=").append(makeHexString((uint64_t)TaskData));
  S.append(" (")
      .append(makeHexString((uint64_t)(TaskData) ? TaskData->value : 0,
                            /*IsPointer=*/false))
      .append(1, ')');
  S.append(" target_task_data=")
      .append(makeHexString((uint64_t)TargetTaskData));
  S.append(" (")
      .append(
          makeHexString((uint64_t)(TargetTaskData) ? TargetTaskData->value : 0,
                        /*IsPointer=*/false))
      .append(1, ')');
  S.append(" target_data=").append(makeHexString((uint64_t)TargetData));
  S.append(" (")
      .append(makeHexString((uint64_t)(TargetData) ? TargetData->value : 0,
                            /*IsPointer=*/false))
      .append(1, ')');
  S.append(" code=").append(makeHexString((uint64_t)CodeptrRA));
  return S;
}

std::string internal::TargetDataOp::toString() const {
  std::string S{"  Callback DataOp: target_id="};
  S.append(std::to_string(TargetId));
  S.append(" host_op_id=").append(std::to_string(HostOpId));
  S.append(" optype=").append(std::to_string(OpType));
  S.append(" src=").append(makeHexString((uint64_t)SrcAddr));
  S.append(" src_device_num=").append(std::to_string(SrcDeviceNum));
  S.append(" dest=").append(makeHexString((uint64_t)DstAddr));
  S.append(" dest_device_num=").append(std::to_string(DstDeviceNum));
  S.append(" bytes=").append(std::to_string(Bytes));
  S.append(" code=").append(makeHexString((uint64_t)CodeptrRA));
  return S;
}

std::string internal::TargetDataOpEmi::toString() const {
  std::string S{"  Callback DataOp EMI: endpoint="};
  S.append(std::to_string(Endpoint));
  S.append(" optype=").append(std::to_string(OpType));
  S.append(" target_task_data=")
      .append(makeHexString((uint64_t)TargetTaskData));
  S.append(" (")
      .append(
          makeHexString((uint64_t)(TargetTaskData) ? TargetTaskData->value : 0,
                        /*IsPointer=*/false))
      .append(1, ')');
  S.append(" target_data=").append(makeHexString((uint64_t)TargetData));
  S.append(" (")
      .append(makeHexString((uint64_t)(TargetData) ? TargetData->value : 0,
                            /*IsPointer=*/false))
      .append(1, ')');
  S.append(" host_op_id=").append(makeHexString((uint64_t)HostOpId));
  S.append(" (")
      .append(makeHexString((uint64_t)(HostOpId) ? (*HostOpId) : 0,
                            /*IsPointer=*/false))
      .append(1, ')');
  S.append(" src=").append(makeHexString((uint64_t)SrcAddr));
  S.append(" src_device_num=").append(std::to_string(SrcDeviceNum));
  S.append(" dest=").append(makeHexString((uint64_t)DstAddr));
  S.append(" dest_device_num=").append(std::to_string(DstDeviceNum));
  S.append(" bytes=").append(std::to_string(Bytes));
  S.append(" code=").append(makeHexString((uint64_t)CodeptrRA));
  return S;
}

std::string internal::TargetSubmit::toString() const {
  std::string S{"  Callback Submit: target_id="};
  S.append(std::to_string(TargetId));
  S.append(" host_op_id=").append(std::to_string(HostOpId));
  S.append(" req_num_teams=").append(std::to_string(RequestedNumTeams));
  return S;
}

std::string internal::TargetSubmitEmi::toString() const {
  std::string S{"  Callback Submit EMI: endpoint="};
  S.append(std::to_string(Endpoint));
  S.append(" req_num_teams=").append(std::to_string(RequestedNumTeams));
  S.append(" target_data=").append(makeHexString((uint64_t)TargetData));
  S.append(" (")
      .append(makeHexString((uint64_t)(TargetData) ? TargetData->value : 0,
                            /*IsPointer=*/false))
      .append(1, ')');
  S.append(" host_op_id=").append(makeHexString((uint64_t)HostOpId));
  S.append(" (")
      .append(makeHexString((uint64_t)(HostOpId) ? (*HostOpId) : 0,
                            /*IsPointer=*/false))
      .append(1, ')');
  return S;
}

std::string internal::DeviceInitialize::toString() const {
  std::string S{"Callback Init: device_num="};
  S.append(std::to_string(DeviceNum));
  S.append(" type=").append((Type) ? Type : "(null)");
  S.append(" device=").append(makeHexString((uint64_t)Device));
  S.append(" lookup=").append(makeHexString((uint64_t)LookupFn));
  S.append(" doc=").append(makeHexString((uint64_t)DocStr));
  return S;
}

std::string internal::DeviceFinalize::toString() const {
  std::string S{"Callback Fini: device_num="};
  S.append(std::to_string(DeviceNum));
  return S;
}

std::string internal::DeviceLoad::toString() const {
  std::string S{"Callback Load: device_num:"};
  S.append(std::to_string(DeviceNum));
  S.append(" module_id:").append(std::to_string(ModuleId));
  S.append(" filename:").append((Filename == nullptr) ? "(null)" : Filename);
  S.append(" host_addr:").append(makeHexString((uint64_t)HostAddr));
  S.append(" device_addr:").append(makeHexString((uint64_t)DeviceAddr));
  S.append(" bytes:").append(std::to_string(Bytes));
  return S;
}

std::string internal::BufferRequest::toString() const {
  std::string S{"Allocated "};
  S.append(std::to_string((Bytes != nullptr) ? *Bytes : 0))
      .append(" bytes at ");
  S.append(makeHexString((Buffer != nullptr) ? (uint64_t)*Buffer : 0));
  S.append(" in buffer request callback");
  return S;
}

std::string internal::BufferComplete::toString() const {
  std::string S{"Executing buffer complete callback: "};
  S.append(std::to_string(DeviceNum)).append(1, ' ');
  S.append(makeHexString((uint64_t)Buffer)).append(1, ' ');
  S.append(std::to_string(Bytes)).append(1, ' ');
  S.append(makeHexString((uint64_t)Begin)).append(1, ' ');
  S.append(std::to_string(BufferOwned));
  return S;
}

std::string internal::BufferRecord::toString() const {
  std::string S{""};
  std::string T{""};
  S.append("rec=").append(makeHexString((uint64_t)RecordPtr));
  S.append(" type=").append(std::to_string(Record.type));

  T.append("time=").append(std::to_string(Record.time));
  T.append(" thread_id=").append(std::to_string(Record.thread_id));
  T.append(" target_id=").append(std::to_string(Record.target_id));

  switch (Record.type) {
  case ompt_callback_target:
  case ompt_callback_target_emi: {
    // Handle Target Record
    ompt_record_target_t TR = Record.record.target;
    S.append(" (Target task) ").append(T);
    S.append(" kind=").append(std::to_string(TR.kind));
    S.append(" endpoint=").append(std::to_string(TR.endpoint));
    S.append(" device=").append(std::to_string(TR.device_num));
    S.append(" task_id=").append(std::to_string(TR.task_id));
    S.append(" codeptr=").append(makeHexString((uint64_t)TR.codeptr_ra));
    break;
  }
  case ompt_callback_target_data_op:
  case ompt_callback_target_data_op_emi: {
    // Handle Target DataOp Record
    ompt_record_target_data_op_t TDR = Record.record.target_data_op;
    S.append(" (Target data op) ").append(T);
    S.append(" host_op_id=").append(std::to_string(TDR.host_op_id));
    S.append(" optype=").append(std::to_string(TDR.optype));
    S.append(" src_addr=").append(makeHexString((uint64_t)TDR.src_addr));
    S.append(" src_device=").append(std::to_string(TDR.src_device_num));
    S.append(" dest_addr=").append(makeHexString((uint64_t)TDR.dest_addr));
    S.append(" dest_device=").append(std::to_string(TDR.dest_device_num));
    S.append(" bytes=").append(std::to_string(TDR.bytes));
    S.append(" end_time=").append(std::to_string(TDR.end_time));
    S.append(" duration=").append(std::to_string(TDR.end_time - Record.time));
    S.append(" ns codeptr=").append(makeHexString((uint64_t)TDR.codeptr_ra));
    break;
  }
  case ompt_callback_target_submit:
  case ompt_callback_target_submit_emi: {
    // Handle Target Kernel Record
    ompt_record_target_kernel_t TKR = Record.record.target_kernel;
    S.append(" (Target kernel) ").append(T);
    S.append(" host_op_id=").append(std::to_string(TKR.host_op_id));
    S.append(" requested_num_teams=")
        .append(std::to_string(TKR.requested_num_teams));
    S.append(" granted_num_teams=")
        .append(std::to_string(TKR.granted_num_teams));
    S.append(" end_time=").append(std::to_string(TKR.end_time));
    S.append(" duration=").append(std::to_string(TKR.end_time - Record.time));
    S.append(" ns");
    break;
  }
  default:
    S.append(" (unsupported record type)");
    break;
  }

  return S;
}

std::string internal::BufferRecordDeallocation::toString() const {
  std::string S{"Deallocated "};
  S.append(makeHexString((uint64_t)Buffer));
  return S;
}
