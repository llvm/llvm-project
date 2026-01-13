//===- InternalEvent.h - Internal event representation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declares internal event representations along the default CTOR definition.
///
//===----------------------------------------------------------------------===//

#ifndef OPENMP_TOOLS_OMPTEST_INCLUDE_INTERNALEVENT_H
#define OPENMP_TOOLS_OMPTEST_INCLUDE_INTERNALEVENT_H

#include "InternalEventCommon.h"

#include <cstring>
#include <limits>
#include <omp-tools.h>

#define expectedDefault(TypeName) std::numeric_limits<TypeName>::min()

namespace omptest {

namespace util {

/// String manipulation helper function. Takes up to 8 bytes of data and returns
/// their hexadecimal representation as string. The data can be expanded to the
/// given size in bytes and will by default be prefixed with '0x'.
std::string makeHexString(uint64_t Data, bool IsPointer = true,
                          size_t DataBytes = 0, bool ShowHexBase = true);

} // namespace util

namespace internal {
struct AssertionSyncPoint : public EventBase<AssertionSyncPoint> {
  std::string toString() const override;
  AssertionSyncPoint(const std::string &Name) : Name(Name) {}
  const std::string Name;
};

struct AssertionSuspend : public EventBase<AssertionSuspend> {
  AssertionSuspend() = default;
};

struct ThreadBegin : public EventBase<ThreadBegin> {
  std::string toString() const override;
  ThreadBegin(ompt_thread_t ThreadType) : ThreadType(ThreadType) {}
  ompt_thread_t ThreadType;
};

struct ThreadEnd : public EventBase<ThreadEnd> {
  std::string toString() const override;
  ThreadEnd() = default;
};

struct ParallelBegin : public EventBase<ParallelBegin> {
  std::string toString() const override;
  ParallelBegin(int NumThreads) : NumThreads(NumThreads) {}
  unsigned int NumThreads;
};

struct ParallelEnd : public EventBase<ParallelEnd> {
  std::string toString() const override;
  ParallelEnd(ompt_data_t *ParallelData, ompt_data_t *EncounteringTaskData,
              int Flags, const void *CodeptrRA)
      : ParallelData(ParallelData), EncounteringTaskData(EncounteringTaskData),
        Flags(Flags), CodeptrRA(CodeptrRA) {}
  ompt_data_t *ParallelData;
  ompt_data_t *EncounteringTaskData;
  int Flags;
  const void *CodeptrRA;
};

struct Work : public EventBase<Work> {
  std::string toString() const override;
  Work(ompt_work_t WorkType, ompt_scope_endpoint_t Endpoint,
       ompt_data_t *ParallelData, ompt_data_t *TaskData, uint64_t Count,
       const void *CodeptrRA)
      : WorkType(WorkType), Endpoint(Endpoint), ParallelData(ParallelData),
        TaskData(TaskData), Count(Count), CodeptrRA(CodeptrRA) {}
  ompt_work_t WorkType;
  ompt_scope_endpoint_t Endpoint;
  ompt_data_t *ParallelData;
  ompt_data_t *TaskData;
  uint64_t Count;
  const void *CodeptrRA;
};

struct Dispatch : public EventBase<Dispatch> {
  std::string toString() const override;
  Dispatch(ompt_data_t *ParallelData, ompt_data_t *TaskData,
           ompt_dispatch_t Kind, ompt_data_t Instance)
      : ParallelData(ParallelData), TaskData(TaskData), Kind(Kind),
        Instance(Instance) {}
  ompt_data_t *ParallelData;
  ompt_data_t *TaskData;
  ompt_dispatch_t Kind;
  ompt_data_t Instance;
};

struct TaskCreate : public EventBase<TaskCreate> {
  std::string toString() const override;
  TaskCreate(ompt_data_t *EncounteringTaskData,
             const ompt_frame_t *EncounteringTaskFrame,
             ompt_data_t *NewTaskData, int Flags, int HasDependences,
             const void *CodeptrRA)
      : EncounteringTaskData(EncounteringTaskData),
        EncounteringTaskFrame(EncounteringTaskFrame), NewTaskData(NewTaskData),
        Flags(Flags), HasDependences(HasDependences), CodeptrRA(CodeptrRA) {}
  ompt_data_t *EncounteringTaskData;
  const ompt_frame_t *EncounteringTaskFrame;
  ompt_data_t *NewTaskData;
  int Flags;
  int HasDependences;
  const void *CodeptrRA;
};

struct Dependences : public EventBase<Dependences> {
  Dependences() = default;
};

struct TaskDependence : public EventBase<TaskDependence> {
  TaskDependence() = default;
};

struct TaskSchedule : public EventBase<TaskSchedule> {
  TaskSchedule() = default;
};

struct ImplicitTask : public EventBase<ImplicitTask> {
  std::string toString() const override;
  ImplicitTask(ompt_scope_endpoint_t Endpoint, ompt_data_t *ParallelData,
               ompt_data_t *TaskData, unsigned int ActualParallelism,
               unsigned int Index, int Flags)
      : Endpoint(Endpoint), ParallelData(ParallelData), TaskData(TaskData),
        ActualParallelism(ActualParallelism), Index(Index), Flags(Flags) {}
  ompt_scope_endpoint_t Endpoint;
  ompt_data_t *ParallelData;
  ompt_data_t *TaskData;
  unsigned int ActualParallelism;
  unsigned int Index;
  int Flags;
};

struct Masked : public EventBase<Masked> {
  Masked() = default;
};

struct SyncRegion : public EventBase<SyncRegion> {
  std::string toString() const override;
  SyncRegion(ompt_sync_region_t Kind, ompt_scope_endpoint_t Endpoint,
             ompt_data_t *ParallelData, ompt_data_t *TaskData,
             const void *CodeptrRA)
      : Kind(Kind), Endpoint(Endpoint), ParallelData(ParallelData),
        TaskData(TaskData), CodeptrRA(CodeptrRA) {}
  ompt_sync_region_t Kind;
  ompt_scope_endpoint_t Endpoint;
  ompt_data_t *ParallelData;
  ompt_data_t *TaskData;
  const void *CodeptrRA;
};

struct MutexAcquire : public EventBase<MutexAcquire> {
  MutexAcquire() = default;
};

struct Mutex : public EventBase<Mutex> {
  Mutex() = default;
};

struct NestLock : public EventBase<NestLock> {
  NestLock() = default;
};

struct Flush : public EventBase<Flush> {
  Flush() = default;
};

struct Cancel : public EventBase<Cancel> {
  Cancel() = default;
};

struct Target : public EventBase<Target> {
  std::string toString() const override;
  Target(ompt_target_t Kind, ompt_scope_endpoint_t Endpoint, int DeviceNum,
         ompt_data_t *TaskData, ompt_id_t TargetId, const void *CodeptrRA)
      : Kind(Kind), Endpoint(Endpoint), DeviceNum(DeviceNum),
        TaskData(TaskData), TargetId(TargetId), CodeptrRA(CodeptrRA) {}
  ompt_target_t Kind;
  ompt_scope_endpoint_t Endpoint;
  int DeviceNum;
  ompt_data_t *TaskData;
  ompt_id_t TargetId;
  const void *CodeptrRA;
};

struct TargetEmi : public EventBase<TargetEmi> {
  std::string toString() const override;
  TargetEmi(ompt_target_t Kind, ompt_scope_endpoint_t Endpoint, int DeviceNum,
            ompt_data_t *TaskData, ompt_data_t *TargetTaskData,
            ompt_data_t *TargetData, const void *CodeptrRA)
      : Kind(Kind), Endpoint(Endpoint), DeviceNum(DeviceNum),
        TaskData(TaskData), TargetTaskData(TargetTaskData),
        TargetData(TargetData), CodeptrRA(CodeptrRA) {}
  ompt_target_t Kind;
  ompt_scope_endpoint_t Endpoint;
  int DeviceNum;
  ompt_data_t *TaskData;
  ompt_data_t *TargetTaskData;
  ompt_data_t *TargetData;
  const void *CodeptrRA;
};

struct TargetDataOp : public EventBase<TargetDataOp> {
  std::string toString() const override;
  TargetDataOp(ompt_id_t TargetId, ompt_id_t HostOpId,
               ompt_target_data_op_t OpType, void *SrcAddr, int SrcDeviceNum,
               void *DstAddr, int DstDeviceNum, size_t Bytes,
               const void *CodeptrRA)
      : TargetId(TargetId), HostOpId(HostOpId), OpType(OpType),
        SrcAddr(SrcAddr), SrcDeviceNum(SrcDeviceNum), DstAddr(DstAddr),
        DstDeviceNum(DstDeviceNum), Bytes(Bytes), CodeptrRA(CodeptrRA) {}
  ompt_id_t TargetId;
  ompt_id_t HostOpId;
  ompt_target_data_op_t OpType;
  void *SrcAddr;
  int SrcDeviceNum;
  void *DstAddr;
  int DstDeviceNum;
  size_t Bytes;
  const void *CodeptrRA;
};

struct TargetDataOpEmi : public EventBase<TargetDataOpEmi> {
  std::string toString() const override;
  TargetDataOpEmi(ompt_scope_endpoint_t Endpoint, ompt_data_t *TargetTaskData,
                  ompt_data_t *TargetData, ompt_id_t *HostOpId,
                  ompt_target_data_op_t OpType, void *SrcAddr, int SrcDeviceNum,
                  void *DstAddr, int DstDeviceNum, size_t Bytes,
                  const void *CodeptrRA)
      : Endpoint(Endpoint), TargetTaskData(TargetTaskData),
        TargetData(TargetData), HostOpId(HostOpId), OpType(OpType),
        SrcAddr(SrcAddr), SrcDeviceNum(SrcDeviceNum), DstAddr(DstAddr),
        DstDeviceNum(DstDeviceNum), Bytes(Bytes), CodeptrRA(CodeptrRA) {}
  ompt_scope_endpoint_t Endpoint;
  ompt_data_t *TargetTaskData;
  ompt_data_t *TargetData;
  ompt_id_t *HostOpId;
  ompt_target_data_op_t OpType;
  void *SrcAddr;
  int SrcDeviceNum;
  void *DstAddr;
  int DstDeviceNum;
  size_t Bytes;
  const void *CodeptrRA;
};

struct TargetSubmit : public EventBase<TargetSubmit> {
  std::string toString() const override;
  TargetSubmit(ompt_id_t TargetId, ompt_id_t HostOpId,
               unsigned int RequestedNumTeams)
      : TargetId(TargetId), HostOpId(HostOpId),
        RequestedNumTeams(RequestedNumTeams) {}
  ompt_id_t TargetId;
  ompt_id_t HostOpId;
  unsigned int RequestedNumTeams;
};

struct TargetSubmitEmi : public EventBase<TargetSubmitEmi> {
  std::string toString() const override;
  TargetSubmitEmi(ompt_scope_endpoint_t Endpoint, ompt_data_t *TargetData,
                  ompt_id_t *HostOpId, unsigned int RequestedNumTeams)
      : Endpoint(Endpoint), TargetData(TargetData), HostOpId(HostOpId),
        RequestedNumTeams(RequestedNumTeams) {}
  ompt_scope_endpoint_t Endpoint;
  ompt_data_t *TargetData;
  ompt_id_t *HostOpId;
  unsigned int RequestedNumTeams;
};

struct ControlTool : public EventBase<ControlTool> {
  ControlTool() = default;
};

struct DeviceInitialize : public EventBase<DeviceInitialize> {
  std::string toString() const override;
  DeviceInitialize(int DeviceNum, const char *Type, ompt_device_t *Device,
                   ompt_function_lookup_t LookupFn, const char *DocStr)
      : DeviceNum(DeviceNum), Type(Type), Device(Device), LookupFn(LookupFn),
        DocStr(DocStr) {}
  int DeviceNum;
  const char *Type;
  ompt_device_t *Device;
  ompt_function_lookup_t LookupFn;
  const char *DocStr;
};

struct DeviceFinalize : public EventBase<DeviceFinalize> {
  std::string toString() const override;
  DeviceFinalize(int DeviceNum) : DeviceNum(DeviceNum) {}
  int DeviceNum;
};

struct DeviceLoad : public EventBase<DeviceLoad> {
  std::string toString() const override;
  DeviceLoad(int DeviceNum, const char *Filename, int64_t OffsetInFile,
             void *VmaInFile, size_t Bytes, void *HostAddr, void *DeviceAddr,
             uint64_t ModuleId)
      : DeviceNum(DeviceNum), Filename(Filename), OffsetInFile(OffsetInFile),
        VmaInFile(VmaInFile), Bytes(Bytes), HostAddr(HostAddr),
        DeviceAddr(DeviceAddr), ModuleId(ModuleId) {}
  int DeviceNum;
  const char *Filename;
  int64_t OffsetInFile;
  void *VmaInFile;
  size_t Bytes;
  void *HostAddr;
  void *DeviceAddr;
  uint64_t ModuleId;
};

struct DeviceUnload : public EventBase<DeviceUnload> {
  DeviceUnload() = default;
};

struct BufferRequest : public EventBase<BufferRequest> {
  std::string toString() const override;
  BufferRequest(int DeviceNum, ompt_buffer_t **Buffer, size_t *Bytes)
      : DeviceNum(DeviceNum), Buffer(Buffer), Bytes(Bytes) {}
  int DeviceNum;
  ompt_buffer_t **Buffer;
  size_t *Bytes;
};

struct BufferComplete : public EventBase<BufferComplete> {
  std::string toString() const override;
  BufferComplete(int DeviceNum, ompt_buffer_t *Buffer, size_t Bytes,
                 ompt_buffer_cursor_t Begin, int BufferOwned)
      : DeviceNum(DeviceNum), Buffer(Buffer), Bytes(Bytes), Begin(Begin),
        BufferOwned(BufferOwned) {}
  int DeviceNum;
  ompt_buffer_t *Buffer;
  size_t Bytes;
  ompt_buffer_cursor_t Begin;
  int BufferOwned;
};

struct BufferRecord : public EventBase<BufferRecord> {
  std::string toString() const override;
  BufferRecord(ompt_record_ompt_t *RecordPtr) : RecordPtr(RecordPtr) {
    if (RecordPtr != nullptr)
      Record = *RecordPtr;
    else
      memset(&Record, 0, sizeof(ompt_record_ompt_t));
  }
  ompt_record_ompt_t Record;
  ompt_record_ompt_t *RecordPtr;
};

struct BufferRecordDeallocation : public EventBase<BufferRecordDeallocation> {
  std::string toString() const override;
  BufferRecordDeallocation(ompt_buffer_t *Buffer) : Buffer(Buffer) {}
  ompt_buffer_t *Buffer;
};

// Add specialized event equality operators here.
// Note: Placement of these forward declarations is important as they need to
// take precedence over the following default equality operator definition.
bool operator==(const ParallelBegin &, const ParallelBegin &);
bool operator==(const Work &, const Work &);
bool operator==(const Dispatch &, const Dispatch &);
bool operator==(const ImplicitTask &, const ImplicitTask &);
bool operator==(const SyncRegion &, const SyncRegion &);
bool operator==(const Target &, const Target &);
bool operator==(const TargetEmi &, const TargetEmi &);
bool operator==(const TargetDataOp &, const TargetDataOp &);
bool operator==(const TargetDataOpEmi &, const TargetDataOpEmi &);
bool operator==(const TargetSubmit &, const TargetSubmit &);
bool operator==(const TargetSubmitEmi &, const TargetSubmitEmi &);
bool operator==(const DeviceInitialize &, const DeviceInitialize &);
bool operator==(const DeviceFinalize &, const DeviceFinalize &);
bool operator==(const DeviceLoad &, const DeviceLoad &);
bool operator==(const BufferRequest &, const BufferRequest &);
bool operator==(const BufferComplete &, const BufferComplete &);
bool operator==(const BufferRecord &, const BufferRecord &);

/// Default (fallback) event equality operator definition.
template <typename Event> bool operator==(const Event &, const Event &) {
  return true;
}

// clang-format off
event_type_trait(AssertionSyncPoint)
event_type_trait(AssertionSuspend)
event_type_trait(ThreadBegin)
event_type_trait(ThreadEnd)
event_type_trait(ParallelBegin)
event_type_trait(ParallelEnd)
event_type_trait(Work)
event_type_trait(Dispatch)
event_type_trait(TaskCreate)
event_type_trait(Dependences)
event_type_trait(TaskDependence)
event_type_trait(TaskSchedule)
event_type_trait(ImplicitTask)
event_type_trait(Masked)
event_type_trait(SyncRegion)
event_type_trait(MutexAcquire)
event_type_trait(Mutex)
event_type_trait(NestLock)
event_type_trait(Flush)
event_type_trait(Cancel)
event_type_trait(Target)
event_type_trait(TargetEmi)
event_type_trait(TargetDataOp)
event_type_trait(TargetDataOpEmi)
event_type_trait(TargetSubmit)
event_type_trait(TargetSubmitEmi)
event_type_trait(ControlTool)
event_type_trait(DeviceInitialize)
event_type_trait(DeviceFinalize)
event_type_trait(DeviceLoad)
event_type_trait(DeviceUnload)
event_type_trait(BufferRequest)
event_type_trait(BufferComplete)
event_type_trait(BufferRecord)
event_type_trait(BufferRecordDeallocation)
// clang-format on

} // namespace internal

} // namespace omptest

#endif
