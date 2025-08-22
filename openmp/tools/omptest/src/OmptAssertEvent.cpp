//===- OmptAssertEvent.cpp - Assertion event implementations ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements assertion event CTORs, for generally all observable events.
///
//===----------------------------------------------------------------------===//

#include "OmptAssertEvent.h"
#include <omp-tools.h>

using namespace omptest;

const char *omptest::to_string(ObserveState State) {
  switch (State) {
  case ObserveState::Generated:
    return "Generated";
  case ObserveState::Always:
    return "Always";
  case ObserveState::Never:
    return "Never";
  default:
    assert(false && "Requested string representation for unknown ObserveState");
    return "UNKNOWN";
  }
}

OmptAssertEvent::OmptAssertEvent(const std::string &Name,
                                 const std::string &Group,
                                 const ObserveState &Expected,
                                 internal::InternalEvent *IE)
    : Name(Name), Group(Group), ExpectedState(Expected), TheEvent(IE) {}

OmptAssertEvent OmptAssertEvent::AssertionSyncPoint(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, const std::string &SyncPointName) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::AssertionSyncPoint(SyncPointName));
}

OmptAssertEvent
OmptAssertEvent::AssertionSuspend(const std::string &Name,
                                  const std::string &Group,
                                  const ObserveState &Expected) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::AssertionSuspend());
}

OmptAssertEvent OmptAssertEvent::ThreadBegin(const std::string &Name,
                                             const std::string &Group,
                                             const ObserveState &Expected,
                                             ompt_thread_t ThreadType) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::ThreadBegin(ThreadType));
}

OmptAssertEvent OmptAssertEvent::ThreadEnd(const std::string &Name,
                                           const std::string &Group,
                                           const ObserveState &Expected) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected, new internal::ThreadEnd());
}

OmptAssertEvent OmptAssertEvent::ParallelBegin(const std::string &Name,
                                               const std::string &Group,
                                               const ObserveState &Expected,
                                               int NumThreads) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::ParallelBegin(NumThreads));
}

OmptAssertEvent OmptAssertEvent::ParallelEnd(const std::string &Name,
                                             const std::string &Group,
                                             const ObserveState &Expected,
                                             ompt_data_t *ParallelData,
                                             ompt_data_t *EncounteringTaskData,
                                             int Flags, const void *CodeptrRA) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::ParallelEnd(ParallelData,
                                                   EncounteringTaskData, Flags,
                                                   CodeptrRA));
}

OmptAssertEvent
OmptAssertEvent::Work(const std::string &Name, const std::string &Group,
                      const ObserveState &Expected, ompt_work_t WorkType,
                      ompt_scope_endpoint_t Endpoint, ompt_data_t *ParallelData,
                      ompt_data_t *TaskData, uint64_t Count,
                      const void *CodeptrRA) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::Work(WorkType, Endpoint, ParallelData,
                                            TaskData, Count, CodeptrRA));
}

OmptAssertEvent
OmptAssertEvent::Dispatch(const std::string &Name, const std::string &Group,
                          const ObserveState &Expected,
                          ompt_data_t *ParallelData, ompt_data_t *TaskData,
                          ompt_dispatch_t Kind, ompt_data_t Instance) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(
      EName, EGroup, Expected,
      new internal::Dispatch(ParallelData, TaskData, Kind, Instance));
}

OmptAssertEvent OmptAssertEvent::TaskCreate(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_data_t *EncounteringTaskData,
    const ompt_frame_t *EncounteringTaskFrame, ompt_data_t *NewTaskData,
    int Flags, int HasDependences, const void *CodeptrRA) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(
      EName, EGroup, Expected,
      new internal::TaskCreate(EncounteringTaskData, EncounteringTaskFrame,
                               NewTaskData, Flags, HasDependences, CodeptrRA));
}

OmptAssertEvent OmptAssertEvent::TaskSchedule(const std::string &Name,
                                              const std::string &Group,
                                              const ObserveState &Expected) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected, new internal::TaskSchedule());
}

OmptAssertEvent OmptAssertEvent::ImplicitTask(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_scope_endpoint_t Endpoint,
    ompt_data_t *ParallelData, ompt_data_t *TaskData,
    unsigned int ActualParallelism, unsigned int Index, int Flags) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::ImplicitTask(Endpoint, ParallelData,
                                                    TaskData, ActualParallelism,
                                                    Index, Flags));
}

OmptAssertEvent OmptAssertEvent::SyncRegion(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_sync_region_t Kind,
    ompt_scope_endpoint_t Endpoint, ompt_data_t *ParallelData,
    ompt_data_t *TaskData, const void *CodeptrRA) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::SyncRegion(Kind, Endpoint, ParallelData,
                                                  TaskData, CodeptrRA));
}

OmptAssertEvent
OmptAssertEvent::Target(const std::string &Name, const std::string &Group,
                        const ObserveState &Expected, ompt_target_t Kind,
                        ompt_scope_endpoint_t Endpoint, int DeviceNum,
                        ompt_data_t *TaskData, ompt_id_t TargetId,
                        const void *CodeptrRA) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::Target(Kind, Endpoint, DeviceNum,
                                              TaskData, TargetId, CodeptrRA));
}

OmptAssertEvent
OmptAssertEvent::TargetEmi(const std::string &Name, const std::string &Group,
                           const ObserveState &Expected, ompt_target_t Kind,
                           ompt_scope_endpoint_t Endpoint, int DeviceNum,
                           ompt_data_t *TaskData, ompt_data_t *TargetTaskData,
                           ompt_data_t *TargetData, const void *CodeptrRA) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::TargetEmi(Kind, Endpoint, DeviceNum,
                                                 TaskData, TargetTaskData,
                                                 TargetData, CodeptrRA));
}

OmptAssertEvent OmptAssertEvent::TargetDataOp(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_id_t TargetId, ompt_id_t HostOpId,
    ompt_target_data_op_t OpType, void *SrcAddr, int SrcDeviceNum,
    void *DstAddr, int DstDeviceNum, size_t Bytes, const void *CodeptrRA) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::TargetDataOp(
                             TargetId, HostOpId, OpType, SrcAddr, SrcDeviceNum,
                             DstAddr, DstDeviceNum, Bytes, CodeptrRA));
}

OmptAssertEvent OmptAssertEvent::TargetDataOp(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_target_data_op_t OpType, size_t Bytes,
    void *SrcAddr, void *DstAddr, int SrcDeviceNum, int DstDeviceNum,
    ompt_id_t TargetId, ompt_id_t HostOpId, const void *CodeptrRA) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::TargetDataOp(
                             TargetId, HostOpId, OpType, SrcAddr, SrcDeviceNum,
                             DstAddr, DstDeviceNum, Bytes, CodeptrRA));
}

OmptAssertEvent OmptAssertEvent::TargetDataOpEmi(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_scope_endpoint_t Endpoint,
    ompt_data_t *TargetTaskData, ompt_data_t *TargetData, ompt_id_t *HostOpId,
    ompt_target_data_op_t OpType, void *SrcAddr, int SrcDeviceNum,
    void *DstAddr, int DstDeviceNum, size_t Bytes, const void *CodeptrRA) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(
      EName, EGroup, Expected,
      new internal::TargetDataOpEmi(Endpoint, TargetTaskData, TargetData,
                                    HostOpId, OpType, SrcAddr, SrcDeviceNum,
                                    DstAddr, DstDeviceNum, Bytes, CodeptrRA));
}

OmptAssertEvent OmptAssertEvent::TargetDataOpEmi(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_target_data_op_t OpType,
    ompt_scope_endpoint_t Endpoint, size_t Bytes, void *SrcAddr, void *DstAddr,
    int SrcDeviceNum, int DstDeviceNum, ompt_data_t *TargetTaskData,
    ompt_data_t *TargetData, ompt_id_t *HostOpId, const void *CodeptrRA) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(
      EName, EGroup, Expected,
      new internal::TargetDataOpEmi(Endpoint, TargetTaskData, TargetData,
                                    HostOpId, OpType, SrcAddr, SrcDeviceNum,
                                    DstAddr, DstDeviceNum, Bytes, CodeptrRA));
}

OmptAssertEvent OmptAssertEvent::TargetSubmit(const std::string &Name,
                                              const std::string &Group,
                                              const ObserveState &Expected,
                                              ompt_id_t TargetId,
                                              ompt_id_t HostOpId,
                                              unsigned int RequestedNumTeams) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(
      EName, EGroup, Expected,
      new internal::TargetSubmit(TargetId, HostOpId, RequestedNumTeams));
}

OmptAssertEvent OmptAssertEvent::TargetSubmit(const std::string &Name,
                                              const std::string &Group,
                                              const ObserveState &Expected,
                                              unsigned int RequestedNumTeams,
                                              ompt_id_t TargetId,
                                              ompt_id_t HostOpId) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(
      EName, EGroup, Expected,
      new internal::TargetSubmit(TargetId, HostOpId, RequestedNumTeams));
}

OmptAssertEvent OmptAssertEvent::TargetSubmitEmi(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_scope_endpoint_t Endpoint,
    ompt_data_t *TargetData, ompt_id_t *HostOpId,
    unsigned int RequestedNumTeams) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::TargetSubmitEmi(Endpoint, TargetData,
                                                       HostOpId,
                                                       RequestedNumTeams));
}

OmptAssertEvent OmptAssertEvent::TargetSubmitEmi(const std::string &Name,
                                                 const std::string &Group,
                                                 const ObserveState &Expected,
                                                 unsigned int RequestedNumTeams,
                                                 ompt_scope_endpoint_t Endpoint,
                                                 ompt_data_t *TargetData,
                                                 ompt_id_t *HostOpId) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::TargetSubmitEmi(Endpoint, TargetData,
                                                       HostOpId,
                                                       RequestedNumTeams));
}

OmptAssertEvent OmptAssertEvent::ControlTool(const std::string &Name,
                                             const std::string &Group,
                                             const ObserveState &Expected) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected, new internal::ControlTool());
}

OmptAssertEvent OmptAssertEvent::DeviceInitialize(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, int DeviceNum, const char *Type,
    ompt_device_t *Device, ompt_function_lookup_t LookupFn,
    const char *DocumentationStr) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::DeviceInitialize(DeviceNum, Type, Device,
                                                        LookupFn,
                                                        DocumentationStr));
}

OmptAssertEvent OmptAssertEvent::DeviceFinalize(const std::string &Name,
                                                const std::string &Group,
                                                const ObserveState &Expected,
                                                int DeviceNum) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::DeviceFinalize(DeviceNum));
}

OmptAssertEvent
OmptAssertEvent::DeviceLoad(const std::string &Name, const std::string &Group,
                            const ObserveState &Expected, int DeviceNum,
                            const char *Filename, int64_t OffsetInFile,
                            void *VmaInFile, size_t Bytes, void *HostAddr,
                            void *DeviceAddr, uint64_t ModuleId) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(
      EName, EGroup, Expected,
      new internal::DeviceLoad(DeviceNum, Filename, OffsetInFile, VmaInFile,
                               Bytes, HostAddr, DeviceAddr, ModuleId));
}

OmptAssertEvent OmptAssertEvent::DeviceUnload(const std::string &Name,
                                              const std::string &Group,
                                              const ObserveState &Expected) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected, new internal::DeviceUnload());
}

OmptAssertEvent OmptAssertEvent::BufferRequest(const std::string &Name,
                                               const std::string &Group,
                                               const ObserveState &Expected,
                                               int DeviceNum,
                                               ompt_buffer_t **Buffer,
                                               size_t *Bytes) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::BufferRequest(DeviceNum, Buffer, Bytes));
}

OmptAssertEvent OmptAssertEvent::BufferComplete(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, int DeviceNum, ompt_buffer_t *Buffer,
    size_t Bytes, ompt_buffer_cursor_t Begin, int BufferOwned) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::BufferComplete(DeviceNum, Buffer, Bytes,
                                                      Begin, BufferOwned));
}

OmptAssertEvent OmptAssertEvent::BufferRecord(const std::string &Name,
                                              const std::string &Group,
                                              const ObserveState &Expected,
                                              ompt_record_ompt_t *Record) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::BufferRecord(Record));
}

OmptAssertEvent OmptAssertEvent::BufferRecord(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_callbacks_t Type, ompt_target_t Kind,
    ompt_scope_endpoint_t Endpoint, int DeviceNum, ompt_id_t TaskId,
    ompt_id_t TargetId, const void *CodeptrRA) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);

  if (Type != ompt_callback_target)
    assert(false && "CTOR only suited for type: 'ompt_callback_target'");

  ompt_record_target_t Subrecord{Kind,   Endpoint, DeviceNum,
                                 TaskId, TargetId, CodeptrRA};

  ompt_record_ompt_t *RecordPtr =
      (ompt_record_ompt_t *)malloc(sizeof(ompt_record_ompt_t));
  memset(RecordPtr, 0, sizeof(ompt_record_ompt_t));
  RecordPtr->type = Type;
  RecordPtr->time = expectedDefault(ompt_device_time_t);
  RecordPtr->thread_id = expectedDefault(ompt_id_t);
  RecordPtr->target_id = TargetId;
  RecordPtr->record.target = Subrecord;

  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::BufferRecord(RecordPtr));
}

OmptAssertEvent OmptAssertEvent::BufferRecord(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_callbacks_t Type,
    ompt_target_data_op_t OpType, size_t Bytes,
    std::pair<ompt_device_time_t, ompt_device_time_t> Timeframe, void *SrcAddr,
    void *DstAddr, int SrcDeviceNum, int DstDeviceNum, ompt_id_t TargetId,
    ompt_id_t HostOpId, const void *CodeptrRA) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);

  if (Type != ompt_callback_target_data_op)
    assert(false &&
           "CTOR only suited for type: 'ompt_callback_target_data_op'");

  ompt_record_target_data_op_t Subrecord{
      HostOpId,     OpType, SrcAddr,          SrcDeviceNum, DstAddr,
      DstDeviceNum, Bytes,  Timeframe.second, CodeptrRA};

  ompt_record_ompt_t *RecordPtr =
      (ompt_record_ompt_t *)malloc(sizeof(ompt_record_ompt_t));
  memset(RecordPtr, 0, sizeof(ompt_record_ompt_t));
  RecordPtr->type = Type;
  RecordPtr->time = Timeframe.first;
  RecordPtr->thread_id = expectedDefault(ompt_id_t);
  RecordPtr->target_id = TargetId;
  RecordPtr->record.target_data_op = Subrecord;

  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::BufferRecord(RecordPtr));
}

OmptAssertEvent OmptAssertEvent::BufferRecord(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_callbacks_t Type,
    ompt_target_data_op_t OpType, size_t Bytes,
    ompt_device_time_t MinimumTimeDelta, void *SrcAddr, void *DstAddr,
    int SrcDeviceNum, int DstDeviceNum, ompt_id_t TargetId, ompt_id_t HostOpId,
    const void *CodeptrRA) {
  return BufferRecord(Name, Group, Expected, Type, OpType, Bytes,
                      {MinimumTimeDelta, expectedDefault(ompt_device_time_t)},
                      SrcAddr, DstAddr, SrcDeviceNum, DstDeviceNum, TargetId,
                      HostOpId, CodeptrRA);
}

OmptAssertEvent OmptAssertEvent::BufferRecord(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_callbacks_t Type,
    std::pair<ompt_device_time_t, ompt_device_time_t> Timeframe,
    unsigned int RequestedNumTeams, unsigned int GrantedNumTeams,
    ompt_id_t TargetId, ompt_id_t HostOpId) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);

  bool isDefault = (Timeframe.first == expectedDefault(ompt_device_time_t));
  isDefault &= (Timeframe.second == expectedDefault(ompt_device_time_t));
  isDefault &= (RequestedNumTeams == expectedDefault(unsigned int));
  isDefault &= (GrantedNumTeams == expectedDefault(unsigned int));
  isDefault &= (TargetId == expectedDefault(ompt_id_t));
  isDefault &= (HostOpId == expectedDefault(ompt_id_t));

  ompt_record_ompt_t *RecordPtr =
      (ompt_record_ompt_t *)malloc(sizeof(ompt_record_ompt_t));
  memset(RecordPtr, 0, sizeof(ompt_record_ompt_t));
  RecordPtr->type = Type;

  // This handles the simplest occurrence of a device tracing record
  // We can only check for Type -- since all other properties are set to default
  if (isDefault) {
    RecordPtr->time = expectedDefault(ompt_device_time_t);
    RecordPtr->thread_id = expectedDefault(ompt_id_t);
    RecordPtr->target_id = expectedDefault(ompt_id_t);
    if (Type == ompt_callback_target) {
      ompt_record_target_t Subrecord{expectedDefault(ompt_target_t),
                                     expectedDefault(ompt_scope_endpoint_t),
                                     expectedDefault(int),
                                     expectedDefault(ompt_id_t),
                                     expectedDefault(ompt_id_t),
                                     expectedDefault(void *)};
      RecordPtr->record.target = Subrecord;
    }

    if (Type == ompt_callback_target_data_op) {
      ompt_record_target_data_op_t Subrecord{
          expectedDefault(ompt_id_t), expectedDefault(ompt_target_data_op_t),
          expectedDefault(void *),    expectedDefault(int),
          expectedDefault(void *),    expectedDefault(int),
          expectedDefault(size_t),    expectedDefault(ompt_device_time_t),
          expectedDefault(void *)};
      RecordPtr->record.target_data_op = Subrecord;
    }

    if (Type == ompt_callback_target_submit) {
      ompt_record_target_kernel_t Subrecord{
          expectedDefault(ompt_id_t), expectedDefault(unsigned int),
          expectedDefault(unsigned int), expectedDefault(ompt_device_time_t)};
      RecordPtr->record.target_kernel = Subrecord;
    }

    return OmptAssertEvent(EName, EGroup, Expected,
                           new internal::BufferRecord(RecordPtr));
  }

  if (Type != ompt_callback_target_submit)
    assert(false && "CTOR only suited for type: 'ompt_callback_target_submit'");

  ompt_record_target_kernel_t Subrecord{HostOpId, RequestedNumTeams,
                                        GrantedNumTeams, Timeframe.second};

  RecordPtr->time = Timeframe.first;
  RecordPtr->thread_id = expectedDefault(ompt_id_t);
  RecordPtr->target_id = TargetId;
  RecordPtr->record.target_kernel = Subrecord;

  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::BufferRecord(RecordPtr));
}

OmptAssertEvent OmptAssertEvent::BufferRecord(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_callbacks_t Type,
    ompt_device_time_t MinimumTimeDelta, unsigned int RequestedNumTeams,
    unsigned int GrantedNumTeams, ompt_id_t TargetId, ompt_id_t HostOpId) {
  return BufferRecord(Name, Group, Expected, Type,
                      {MinimumTimeDelta, expectedDefault(ompt_device_time_t)},
                      RequestedNumTeams, GrantedNumTeams, TargetId, HostOpId);
}

OmptAssertEvent OmptAssertEvent::BufferRecordDeallocation(
    const std::string &Name, const std::string &Group,
    const ObserveState &Expected, ompt_buffer_t *Buffer) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected,
                         new internal::BufferRecordDeallocation(Buffer));
}

std::string OmptAssertEvent::getEventName() const { return Name; }

std::string OmptAssertEvent::getEventGroup() const { return Group; }

ObserveState OmptAssertEvent::getEventExpectedState() const {
  return ExpectedState;
}

internal::EventTy OmptAssertEvent::getEventType() const {
  return TheEvent->Type;
}

internal::InternalEvent *OmptAssertEvent::getEvent() const {
  return TheEvent.get();
}

std::string OmptAssertEvent::toString(bool PrefixEventName) const {
  std::string S;
  if (PrefixEventName)
    S.append(getEventName()).append(": ");
  S.append((TheEvent == nullptr) ? "OmptAssertEvent" : TheEvent->toString());
  return S;
}

bool omptest::operator==(const OmptAssertEvent &A, const OmptAssertEvent &B) {
  assert(A.TheEvent.get() != nullptr && "A is valid");
  assert(B.TheEvent.get() != nullptr && "B is valid");

  return A.TheEvent->Type == B.TheEvent->Type &&
         A.TheEvent->equals(B.TheEvent.get());
}
