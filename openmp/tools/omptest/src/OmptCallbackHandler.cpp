//===- OmptCallbackHandler.cpp - OMPT Callback handling impl. ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the OMPT callback handling implementations.
///
//===----------------------------------------------------------------------===//

#include "OmptCallbackHandler.h"

using namespace omptest;

OmptCallbackHandler *Handler = nullptr;

OmptCallbackHandler &OmptCallbackHandler::get() {
  if (Handler == nullptr)
    Handler = new OmptCallbackHandler();

  return *Handler;
}

void OmptCallbackHandler::subscribe(OmptListener *Listener) {
  Subscribers.push_back(Listener);
}

void OmptCallbackHandler::clearSubscribers() {
  replay();

  Subscribers.clear();
}

void OmptCallbackHandler::replay() {
  if (!RecordAndReplay)
    return;

  for (auto &E : RecordedEvents)
    for (const auto &S : Subscribers)
      S->notify(std::move(E));
}

void OmptCallbackHandler::handleThreadBegin(ompt_thread_t ThreadType,
                                            ompt_data_t *ThreadData) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::ThreadBegin(
        "Thread Begin", "", ObserveState::Generated, ThreadType));
    return;
  }

  // Initial thread event likely to preceed assertion registration, so skip
  if (ThreadType == ompt_thread_initial)
    return;
  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::ThreadBegin(
        "Thread Begin", "", ObserveState::Generated, ThreadType));
}

void OmptCallbackHandler::handleThreadEnd(ompt_data_t *ThreadData) {
  if (RecordAndReplay) {
    recordEvent(
        OmptAssertEvent::ThreadEnd("Thread End", "", ObserveState::Generated));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(
        OmptAssertEvent::ThreadEnd("Thread End", "", ObserveState::Generated));
}

void OmptCallbackHandler::handleTaskCreate(
    ompt_data_t *EncounteringTaskData,
    const ompt_frame_t *EncounteringTaskFrame, ompt_data_t *NewTaskData,
    int Flags, int HasDependences, const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TaskCreate(
        "Task Create", "", ObserveState::Generated, EncounteringTaskData,
        EncounteringTaskFrame, NewTaskData, Flags, HasDependences, CodeptrRA));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TaskCreate(
        "Task Create", "", ObserveState::Generated, EncounteringTaskData,
        EncounteringTaskFrame, NewTaskData, Flags, HasDependences, CodeptrRA));
}

void OmptCallbackHandler::handleTaskSchedule(ompt_data_t *PriorTaskData,
                                             ompt_task_status_t PriorTaskStatus,
                                             ompt_data_t *NextTaskData) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TaskSchedule("Task Schedule", "",
                                              ObserveState::Generated));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TaskSchedule("Task Schedule", "",
                                            ObserveState::Generated));
}

void OmptCallbackHandler::handleImplicitTask(ompt_scope_endpoint_t Endpoint,
                                             ompt_data_t *ParallelData,
                                             ompt_data_t *TaskData,
                                             unsigned int ActualParallelism,
                                             unsigned int Index, int Flags) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::ImplicitTask(
        "Implicit Task", "", ObserveState::Generated, Endpoint, ParallelData,
        TaskData, ActualParallelism, Index, Flags));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::ImplicitTask(
        "Implicit Task", "", ObserveState::Generated, Endpoint, ParallelData,
        TaskData, ActualParallelism, Index, Flags));
}

void OmptCallbackHandler::handleParallelBegin(
    ompt_data_t *EncounteringTaskData,
    const ompt_frame_t *EncounteringTaskFrame, ompt_data_t *ParallelData,
    unsigned int RequestedParallelism, int Flags, const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::ParallelBegin(
        "Parallel Begin", "", ObserveState::Generated, RequestedParallelism));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::ParallelBegin(
        "Parallel Begin", "", ObserveState::Generated, RequestedParallelism));
}

void OmptCallbackHandler::handleParallelEnd(ompt_data_t *ParallelData,
                                            ompt_data_t *EncounteringTaskData,
                                            int Flags, const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::ParallelEnd("Parallel End", "",
                                             ObserveState::Generated));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::ParallelEnd("Parallel End", "",
                                           ObserveState::Generated));
}

void OmptCallbackHandler::handleDeviceInitialize(
    int DeviceNum, const char *Type, ompt_device_t *Device,
    ompt_function_lookup_t LookupFn, const char *DocumentationStr) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::DeviceInitialize(
        "Device Inititalize", "", ObserveState::Generated, DeviceNum, Type,
        Device, LookupFn, DocumentationStr));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::DeviceInitialize(
        "Device Inititalize", "", ObserveState::Generated, DeviceNum, Type,
        Device, LookupFn, DocumentationStr));
}

void OmptCallbackHandler::handleDeviceFinalize(int DeviceNum) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::DeviceFinalize(
        "Device Finalize", "", ObserveState::Generated, DeviceNum));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::DeviceFinalize(
        "Device Finalize", "", ObserveState::Generated, DeviceNum));
}

void OmptCallbackHandler::handleTarget(ompt_target_t Kind,
                                       ompt_scope_endpoint_t Endpoint,
                                       int DeviceNum, ompt_data_t *TaskData,
                                       ompt_id_t TargetId,
                                       const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::Target("Target", "", ObserveState::Generated,
                                        Kind, Endpoint, DeviceNum, TaskData,
                                        TargetId, CodeptrRA));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::Target("Target", "", ObserveState::Generated,
                                      Kind, Endpoint, DeviceNum, TaskData,
                                      TargetId, CodeptrRA));
}

void OmptCallbackHandler::handleTargetEmi(ompt_target_t Kind,
                                          ompt_scope_endpoint_t Endpoint,
                                          int DeviceNum, ompt_data_t *TaskData,
                                          ompt_data_t *TargetTaskData,
                                          ompt_data_t *TargetData,
                                          const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TargetEmi(
        "Target EMI", "", ObserveState::Generated, Kind, Endpoint, DeviceNum,
        TaskData, TargetTaskData, TargetData, CodeptrRA));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TargetEmi(
        "Target EMI", "", ObserveState::Generated, Kind, Endpoint, DeviceNum,
        TaskData, TargetTaskData, TargetData, CodeptrRA));
}

void OmptCallbackHandler::handleTargetSubmit(ompt_id_t TargetId,
                                             ompt_id_t HostOpId,
                                             unsigned int RequestedNumTeams) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TargetSubmit("Target Submit", "",
                                              ObserveState::Generated, TargetId,
                                              HostOpId, RequestedNumTeams));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TargetSubmit("Target Submit", "",
                                            ObserveState::Generated, TargetId,
                                            HostOpId, RequestedNumTeams));
}

void OmptCallbackHandler::handleTargetSubmitEmi(
    ompt_scope_endpoint_t Endpoint, ompt_data_t *TargetData,
    ompt_id_t *HostOpId, unsigned int RequestedNumTeams) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TargetSubmitEmi(
        "Target Submit EMI", "", ObserveState::Generated, Endpoint, TargetData,
        HostOpId, RequestedNumTeams));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TargetSubmitEmi(
        "Target Submit EMI", "", ObserveState::Generated, Endpoint, TargetData,
        HostOpId, RequestedNumTeams));
}

void OmptCallbackHandler::handleTargetDataOp(
    ompt_id_t TargetId, ompt_id_t HostOpId, ompt_target_data_op_t OpType,
    void *SrcAddr, int SrcDeviceNum, void *DstAddr, int DstDeviceNum,
    size_t Bytes, const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TargetDataOp(
        "Target Data Op", "", ObserveState::Generated, TargetId, HostOpId,
        OpType, SrcAddr, SrcDeviceNum, DstAddr, DstDeviceNum, Bytes,
        CodeptrRA));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TargetDataOp(
        "Target Data Op", "", ObserveState::Generated, TargetId, HostOpId,
        OpType, SrcAddr, SrcDeviceNum, DstAddr, DstDeviceNum, Bytes,
        CodeptrRA));
}

void OmptCallbackHandler::handleTargetDataOpEmi(
    ompt_scope_endpoint_t Endpoint, ompt_data_t *TargetTaskData,
    ompt_data_t *TargetData, ompt_id_t *HostOpId, ompt_target_data_op_t OpType,
    void *SrcAddr, int SrcDeviceNum, void *DstAddr, int DstDeviceNum,
    size_t Bytes, const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TargetDataOpEmi(
        "Target Data Op EMI", "", ObserveState::Generated, Endpoint,
        TargetTaskData, TargetData, HostOpId, OpType, SrcAddr, SrcDeviceNum,
        DstAddr, DstDeviceNum, Bytes, CodeptrRA));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TargetDataOpEmi(
        "Target Data Op EMI", "", ObserveState::Generated, Endpoint,
        TargetTaskData, TargetData, HostOpId, OpType, SrcAddr, SrcDeviceNum,
        DstAddr, DstDeviceNum, Bytes, CodeptrRA));
}

void OmptCallbackHandler::handleDeviceLoad(int DeviceNum, const char *Filename,
                                           int64_t OffsetInFile,
                                           void *VmaInFile, size_t Bytes,
                                           void *HostAddr, void *DeviceAddr,
                                           uint64_t ModuleId) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::DeviceLoad(
        "Device Load", "", ObserveState::Generated, DeviceNum, Filename,
        OffsetInFile, VmaInFile, Bytes, HostAddr, DeviceAddr, ModuleId));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::DeviceLoad(
        "Device Load", "", ObserveState::Generated, DeviceNum, Filename,
        OffsetInFile, VmaInFile, Bytes, HostAddr, DeviceAddr, ModuleId));
}

void OmptCallbackHandler::handleDeviceUnload(int DeviceNum, uint64_t ModuleId) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::DeviceUnload("Device Unload", "",
                                              ObserveState::Generated));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::DeviceUnload("Device Unload", "",
                                            ObserveState::Generated));
}

void OmptCallbackHandler::handleBufferRequest(int DeviceNum,
                                              ompt_buffer_t **Buffer,
                                              size_t *Bytes) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::BufferRequest("Buffer Request", "",
                                               ObserveState::Generated,
                                               DeviceNum, Buffer, Bytes));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::BufferRequest("Buffer Request", "",
                                             ObserveState::Generated, DeviceNum,
                                             Buffer, Bytes));
}

void OmptCallbackHandler::handleBufferComplete(int DeviceNum,
                                               ompt_buffer_t *Buffer,
                                               size_t Bytes,
                                               ompt_buffer_cursor_t Begin,
                                               int BufferOwned) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::BufferComplete(
        "Buffer Complete", "", ObserveState::Generated, DeviceNum, Buffer,
        Bytes, Begin, BufferOwned));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::BufferComplete(
        "Buffer Complete", "", ObserveState::Generated, DeviceNum, Buffer,
        Bytes, Begin, BufferOwned));
}

void OmptCallbackHandler::handleBufferRecord(ompt_record_ompt_t *Record) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::BufferRecord("Buffer Record", "",
                                              ObserveState::Generated, Record));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::BufferRecord("Buffer Record", "",
                                            ObserveState::Generated, Record));
}

void OmptCallbackHandler::handleBufferRecordDeallocation(
    ompt_buffer_t *Buffer) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::BufferRecordDeallocation(
        "Buffer Deallocation", "", ObserveState::Generated, Buffer));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::BufferRecordDeallocation(
        "Buffer Deallocation", "", ObserveState::Generated, Buffer));
}

void OmptCallbackHandler::handleWork(ompt_work_t WorkType,
                                     ompt_scope_endpoint_t Endpoint,
                                     ompt_data_t *ParallelData,
                                     ompt_data_t *TaskData, uint64_t Count,
                                     const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::Work("Work", "", ObserveState::Generated,
                                      WorkType, Endpoint, ParallelData,
                                      TaskData, Count, CodeptrRA));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::Work("Work", "", ObserveState::Generated,
                                    WorkType, Endpoint, ParallelData, TaskData,
                                    Count, CodeptrRA));
}

void OmptCallbackHandler::handleSyncRegion(ompt_sync_region_t Kind,
                                           ompt_scope_endpoint_t Endpoint,
                                           ompt_data_t *ParallelData,
                                           ompt_data_t *TaskData,
                                           const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::SyncRegion(
        "SyncRegion", "", ObserveState::Generated, Kind, Endpoint, ParallelData,
        TaskData, CodeptrRA));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::SyncRegion(
        "SyncRegion", "", ObserveState::Generated, Kind, Endpoint, ParallelData,
        TaskData, CodeptrRA));
}

void OmptCallbackHandler::handleDispatch(ompt_data_t *ParallelData,
                                         ompt_data_t *TaskData,
                                         ompt_dispatch_t Kind,
                                         ompt_data_t Instance) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::Dispatch("Dispatch", "",
                                          ObserveState::Generated, ParallelData,
                                          TaskData, Kind, Instance));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::Dispatch("Dispatch", "", ObserveState::Generated,
                                        ParallelData, TaskData, Kind,
                                        Instance));
}

void OmptCallbackHandler::handleAssertionSyncPoint(
    const std::string &SyncPointName) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::AssertionSyncPoint(
        "Assertion SyncPoint", "", ObserveState::Generated, SyncPointName));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::AssertionSyncPoint(
        "Assertion SyncPoint", "", ObserveState::Generated, SyncPointName));
}

void OmptCallbackHandler::recordEvent(OmptAssertEvent &&Event) {
  RecordedEvents.emplace_back(std::forward<OmptAssertEvent>(Event));
}
