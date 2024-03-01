#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_INTERNALEVENT_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_INTERNALEVENT_H

#include "InternalEventCommon.h"

#include <cstring>

namespace omptest {

namespace internal {
// clang-format off
event_class_stub(Asserter)
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
  int NumThreads;                                                              \
)
event_class_w_custom_body(ParallelEnd,                                         \
  ParallelEnd() : InternalEvent(EventTy::ParallelEnd) {}                       \
)
event_class_stub(TaskCreate)
event_class_stub(TaskSchedule)
event_class_stub(ImplicitTask)
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
// clang-format on

} // namespace internal

} // namespace omptest

#endif
