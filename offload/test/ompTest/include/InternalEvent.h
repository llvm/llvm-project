#ifndef OFFLOAD_TEST_OMPTEST_INCLUDE_INTERNALEVENT_H
#define OFFLOAD_TEST_OMPTEST_INCLUDE_INTERNALEVENT_H

#include "InternalEventCommon.h"

#include <cstring>
#include <limits>
#include <omp-tools.h>

#define expectedDefault(TypeName) std::numeric_limits<TypeName>::min()

namespace omptest {

namespace internal {
// clang-format off
event_class_w_custom_body(AssertionSyncPoint,                                  \
  AssertionSyncPoint(const std::string &Name)                                  \
    : InternalEvent(EventTy::AssertionSyncPoint), Name(Name) {}                \
                                                                               \
  const std::string Name;                                                      \
)
event_class_stub(AssertionSuspend)
event_class_w_custom_body(ThreadBegin,                                         \
  ThreadBegin(ompt_thread_t ThreadType)                                        \
    : InternalEvent(EventTy::ThreadBegin), ThreadType(ThreadType) {}           \
                                                                               \
  ompt_thread_t ThreadType;                                                    \
)
event_class_w_custom_body(ThreadEnd,                                           \
  ThreadEnd() : InternalEvent(EventTy::ThreadEnd) {}                           \
)
event_class_w_custom_body(ParallelBegin,                                       \
  ParallelBegin(int NumThreads)                                                \
    : InternalEvent(EventTy::ParallelBegin), NumThreads(NumThreads) {}         \
                                                                               \
  unsigned int NumThreads;                                                              \
)
event_class_w_custom_body(ParallelEnd,                                         \
  ParallelEnd(ompt_data_t *ParallelData, ompt_data_t *EncounteringTaskData,    \
              int Flags, const void *CodeptrRA)                                \
  : InternalEvent(EventTy::ParallelEnd), ParallelData(ParallelData),           \
    EncounteringTaskData(EncounteringTaskData), Flags(Flags),                  \
    CodeptrRA(CodeptrRA) {}                                                    \
                                                                               \
ompt_data_t *ParallelData;                                                     \
ompt_data_t *EncounteringTaskData;                                             \
int Flags;                                                                     \
const void *CodeptrRA;                                                         \
)
event_class_w_custom_body(Work,                                                \
  Work(ompt_work_t WorkType, ompt_scope_endpoint_t Endpoint,                   \
       ompt_data_t *ParallelData, ompt_data_t *TaskData, uint64_t Count,       \
       const void *CodeptrRA)                                                  \
  : InternalEvent(EventTy::Work), WorkType(WorkType), Endpoint(Endpoint),      \
  ParallelData(ParallelData), TaskData(TaskData), Count(Count),                \
  CodeptrRA(CodeptrRA) {}                                                      \
                                                                               \
ompt_work_t WorkType;                                                          \
ompt_scope_endpoint_t Endpoint;                                                \
ompt_data_t *ParallelData;                                                     \
ompt_data_t *TaskData;                                                         \
uint64_t Count;                                                                \
const void *CodeptrRA;                                                         \
)
event_class_w_custom_body(Dispatch,                                            \
  Dispatch(ompt_data_t *ParallelData, ompt_data_t *TaskData,                   \
           ompt_dispatch_t Kind, ompt_data_t Instance)                         \
  : InternalEvent(EventTy::Dispatch), ParallelData(ParallelData),              \
  TaskData(TaskData), Kind(Kind), Instance(Instance) {}                        \
                                                                               \
ompt_data_t *ParallelData;                                                     \
ompt_data_t *TaskData;                                                         \
ompt_dispatch_t Kind;                                                          \
ompt_data_t Instance;                                                          \
)
event_class_w_custom_body(TaskCreate,                                          \
  TaskCreate(ompt_data_t *EncounteringTaskData,                                \
             const ompt_frame_t *EncounteringTaskFrame,                        \
             ompt_data_t *NewTaskData, int Flags, int HasDependences,          \
             const void *CodeptrRA)                                            \
  : InternalEvent(EventTy::TaskCreate),                                        \
  EncounteringTaskData(EncounteringTaskData),                                  \
  EncounteringTaskFrame(EncounteringTaskFrame), NewTaskData(NewTaskData),      \
  Flags(Flags), HasDependences(HasDependences), CodeptrRA(CodeptrRA) {}        \
                                                                               \
ompt_data_t *EncounteringTaskData;                                             \
const ompt_frame_t *EncounteringTaskFrame;                                     \
ompt_data_t *NewTaskData;                                                      \
int Flags;                                                                     \
int HasDependences;                                                            \
const void *CodeptrRA;                                                         \
)
event_class_stub(Dependences)
event_class_stub(TaskDependence)
event_class_stub(TaskSchedule)
event_class_w_custom_body(ImplicitTask,                                        \
  ImplicitTask(ompt_scope_endpoint_t Endpoint, ompt_data_t *ParallelData,        \
               ompt_data_t *TaskData, unsigned int ActualParallelism,            \
               unsigned int Index, int Flags)                                  \
  : InternalEvent(EventTy::ImplicitTask), Endpoint(Endpoint),                  \
  ParallelData(ParallelData), TaskData(TaskData), ActualParallelism(ActualParallelism),\
  Index(Index), Flags(Flags) {}                                                \
                                                                               \
ompt_scope_endpoint_t Endpoint;                                                \
ompt_data_t *ParallelData;                                                       \
ompt_data_t *TaskData;                                                           \
unsigned int ActualParallelism;                                                \
unsigned int Index;                                                            \
int Flags;                                                                     \
)
event_class_stub(Masked)
event_class_w_custom_body(SyncRegion,                                          \
  SyncRegion(ompt_sync_region_t Kind, ompt_scope_endpoint_t Endpoint,          \
            ompt_data_t *ParallelData, ompt_data_t *TaskData,                  \
            const void *CodeptrRA)                                             \
  : InternalEvent(EventTy::SyncRegion), Kind(Kind), Endpoint(Endpoint),        \
    ParallelData(ParallelData), TaskData(TaskData), CodeptrRA(CodeptrRA) {}    \
                                                                               \
ompt_sync_region_t Kind;                                                       \
ompt_scope_endpoint_t Endpoint;                                                \
ompt_data_t *ParallelData;                                                     \
ompt_data_t *TaskData;                                                         \
const void *CodeptrRA;                                                         \
)
event_class_stub(MutexAcquire)
event_class_stub(Mutex)
event_class_stub(NestLock)
event_class_stub(Flush)
event_class_stub(Cancel)
event_class_w_custom_body(Target,                                              \
  Target(ompt_target_t Kind, ompt_scope_endpoint_t Endpoint, int DeviceNum,    \
         ompt_data_t *TaskData, ompt_id_t TargetId, const void *CodeptrRA)     \
    : InternalEvent(EventTy::Target), Kind(Kind), Endpoint(Endpoint),          \
      DeviceNum(DeviceNum), TaskData(TaskData), TargetId(TargetId),            \
      CodeptrRA(CodeptrRA) {}                                                  \
                                                                               \
    ompt_target_t Kind;                                                        \
    ompt_scope_endpoint_t Endpoint;                                            \
    int DeviceNum;                                                             \
    ompt_data_t *TaskData;                                                     \
    ompt_id_t TargetId;                                                        \
    const void *CodeptrRA;                                                     \
)
event_class_w_custom_body(TargetEmi,                                           \
  TargetEmi(ompt_target_t Kind, ompt_scope_endpoint_t Endpoint, int DeviceNum, \
            ompt_data_t *TaskData, ompt_data_t *TargetTaskData,                \
            ompt_data_t *TargetData, const void *CodeptrRA)                    \
    : InternalEvent(EventTy::TargetEmi), Kind(Kind), Endpoint(Endpoint),       \
      DeviceNum(DeviceNum), TaskData(TaskData),                                \
      TargetTaskData(TargetTaskData), TargetData(TargetData),                  \
      CodeptrRA(CodeptrRA) {}                                                  \
                                                                               \
    ompt_target_t Kind;                                                        \
    ompt_scope_endpoint_t Endpoint;                                            \
    int DeviceNum;                                                             \
    ompt_data_t *TaskData;                                                     \
    ompt_data_t *TargetTaskData;                                               \
    ompt_data_t *TargetData;                                                   \
    const void *CodeptrRA;                                                     \
)
event_class_w_custom_body(TargetDataOp,                                        \
  TargetDataOp(ompt_id_t TargetId, ompt_id_t HostOpId,                         \
               ompt_target_data_op_t OpType, void *SrcAddr, int SrcDeviceNum,  \
               void *DstAddr, int DstDeviceNum, size_t Bytes,                  \
               const void *CodeptrRA)                                          \
    : InternalEvent(EventTy::TargetDataOp), TargetId(TargetId),                \
      HostOpId(HostOpId), OpType(OpType), SrcAddr(SrcAddr),                    \
      SrcDeviceNum(SrcDeviceNum), DstAddr(DstAddr),                            \
      DstDeviceNum(DstDeviceNum), Bytes(Bytes), CodeptrRA(CodeptrRA) {}        \
                                                                               \
  ompt_id_t TargetId;                                                          \
  ompt_id_t HostOpId;                                                          \
  ompt_target_data_op_t OpType;                                                \
  void *SrcAddr;                                                               \
  int SrcDeviceNum;                                                            \
  void *DstAddr;                                                               \
  int DstDeviceNum;                                                            \
  size_t Bytes;                                                                \
  const void *CodeptrRA;                                                       \
)
event_class_w_custom_body(TargetDataOpEmi,                                     \
  TargetDataOpEmi(ompt_scope_endpoint_t Endpoint, ompt_data_t *TargetTaskData, \
                  ompt_data_t *TargetData, ompt_id_t *HostOpId,                \
                  ompt_target_data_op_t OpType, void *SrcAddr,                 \
                  int SrcDeviceNum, void *DstAddr, int DstDeviceNum,           \
                  size_t Bytes, const void *CodeptrRA)                         \
    : InternalEvent(EventTy::TargetDataOpEmi), Endpoint(Endpoint),             \
      TargetTaskData(TargetTaskData), TargetData(TargetData),                  \
      HostOpId(HostOpId), OpType(OpType), SrcAddr(SrcAddr),                    \
      SrcDeviceNum(SrcDeviceNum), DstAddr(DstAddr),                            \
      DstDeviceNum(DstDeviceNum), Bytes(Bytes), CodeptrRA(CodeptrRA) {}        \
                                                                               \
  ompt_scope_endpoint_t Endpoint;                                              \
  ompt_data_t *TargetTaskData;                                                 \
  ompt_data_t *TargetData;                                                     \
  ompt_id_t *HostOpId;                                                         \
  ompt_target_data_op_t OpType;                                                \
  void *SrcAddr;                                                               \
  int SrcDeviceNum;                                                            \
  void *DstAddr;                                                               \
  int DstDeviceNum;                                                            \
  size_t Bytes;                                                                \
  const void *CodeptrRA;                                                       \
)
event_class_w_custom_body(TargetSubmit,                                        \
  TargetSubmit(ompt_id_t TargetId, ompt_id_t HostOpId,                         \
               unsigned int RequestedNumTeams)                                 \
    : InternalEvent(EventTy::TargetSubmit), TargetId(TargetId),                \
      HostOpId(HostOpId), RequestedNumTeams(RequestedNumTeams) {}              \
                                                                               \
  ompt_id_t TargetId;                                                          \
  ompt_id_t HostOpId;                                                          \
  unsigned int RequestedNumTeams;                                              \
)
event_class_w_custom_body(TargetSubmitEmi,                                     \
  TargetSubmitEmi(ompt_scope_endpoint_t Endpoint, ompt_data_t *TargetData,     \
                  ompt_id_t *HostOpId, unsigned int RequestedNumTeams)         \
    : InternalEvent(EventTy::TargetSubmitEmi), Endpoint(Endpoint),             \
      TargetData(TargetData), HostOpId(HostOpId),                              \
      RequestedNumTeams(RequestedNumTeams) {}                                  \
                                                                               \
  ompt_scope_endpoint_t Endpoint;                                              \
  ompt_data_t *TargetData;                                                     \
  ompt_id_t *HostOpId;                                                         \
  unsigned int RequestedNumTeams;                                              \
)
event_class_stub(ControlTool)
event_class_w_custom_body(DeviceInitialize,                                    \
  DeviceInitialize(int DeviceNum, const char *Type, ompt_device_t *Device,     \
                   ompt_function_lookup_t LookupFn, const char *DocStr)        \
    : InternalEvent(EventTy::DeviceInitialize), DeviceNum(DeviceNum),          \
      Type(Type), Device(Device), LookupFn(LookupFn), DocStr(DocStr) {}        \
                                                                               \
  int DeviceNum;                                                               \
  const char *Type;                                                            \
  ompt_device_t *Device;                                                       \
  ompt_function_lookup_t LookupFn;                                             \
  const char *DocStr;                                                          \
)
event_class_w_custom_body(DeviceFinalize,                                      \
  DeviceFinalize(int DeviceNum)                                                \
    : InternalEvent(EventTy::DeviceFinalize), DeviceNum(DeviceNum) {}          \
                                                                               \
  int DeviceNum;                                                               \
)
event_class_w_custom_body(DeviceLoad,                                          \
  DeviceLoad(int DeviceNum, const char *Filename, int64_t OffsetInFile,        \
             void *VmaInFile, size_t Bytes, void *HostAddr, void *DeviceAddr,  \
             uint64_t ModuleId)                                                \
    : InternalEvent(EventTy::DeviceLoad), DeviceNum(DeviceNum),                \
      Filename(Filename), OffsetInFile(OffsetInFile), VmaInFile(VmaInFile),    \
      Bytes(Bytes), HostAddr(HostAddr), DeviceAddr(DeviceAddr),                \
      ModuleId(ModuleId) {}                                                    \
                                                                               \
  int DeviceNum;                                                               \
  const char *Filename;                                                        \
  int64_t OffsetInFile;                                                        \
  void *VmaInFile;                                                             \
  size_t Bytes;                                                                \
  void *HostAddr;                                                              \
  void *DeviceAddr;                                                            \
  uint64_t ModuleId;                                                           \
)
event_class_stub(DeviceUnload)
event_class_w_custom_body(BufferRequest,                                       \
  BufferRequest(int DeviceNum, ompt_buffer_t **Buffer, size_t *Bytes)          \
    : InternalEvent(EventTy::BufferRequest), DeviceNum(DeviceNum),             \
      Buffer(Buffer), Bytes(Bytes) {}                                          \
                                                                               \
  int DeviceNum;                                                               \
  ompt_buffer_t **Buffer;                                                      \
  size_t *Bytes;                                                               \
)
event_class_w_custom_body(BufferComplete,                                      \
  BufferComplete(int DeviceNum, ompt_buffer_t *Buffer, size_t Bytes,           \
                 ompt_buffer_cursor_t Begin, int BufferOwned)                  \
    : InternalEvent(EventTy::BufferComplete), DeviceNum(DeviceNum),            \
      Buffer(Buffer), Bytes(Bytes), Begin(Begin), BufferOwned(BufferOwned) {}  \
                                                                               \
  int DeviceNum;                                                               \
  ompt_buffer_t *Buffer;                                                       \
  size_t Bytes;                                                                \
  ompt_buffer_cursor_t Begin;                                                  \
  int BufferOwned;                                                             \
)
event_class_w_custom_body(BufferRecord,                                        \
  BufferRecord(ompt_record_ompt_t *RecordPtr)                                  \
    : InternalEvent(EventTy::BufferRecord), RecordPtr(RecordPtr) {             \
      if (RecordPtr != nullptr) Record = *RecordPtr;                           \
      else memset(&Record, 0, sizeof(ompt_record_ompt_t));                     \
    }                                                                          \
                                                                               \
  ompt_record_ompt_t Record;                                                   \
  ompt_record_ompt_t *RecordPtr;                                               \
)
event_class_w_custom_body(BufferRecordDeallocation,                            \
  BufferRecordDeallocation(ompt_buffer_t *Buffer)                              \
    : InternalEvent(EventTy::BufferRecordDeallocation), Buffer(Buffer) {}      \
                                                                               \
  ompt_buffer_t *Buffer;                                                       \
)
// clang-format on

} // namespace internal

} // namespace omptest

#endif
