#include "InternalEvent.h"

namespace omptest {

namespace internal {
// clang-format off
event_class_operator_stub(AssertionSyncPoint)
event_class_operator_stub(AssertionSuspend)
event_class_operator_stub(ThreadBegin)
event_class_operator_stub(ThreadEnd)
event_class_operator_w_body(ParallelBegin,                                     \
  return Expected.NumThreads == Observed.NumThreads;                           \
)
event_class_operator_stub(ParallelEnd)
event_class_operator_stub(TaskCreate)
event_class_operator_stub(TaskSchedule)
event_class_operator_stub(ImplicitTask)
event_class_operator_w_body(Target,                                            \
  bool isSameKind = (Expected.Kind == Observed.Kind);                          \
  bool isSameEndpoint = (Expected.Endpoint == Observed.Endpoint);              \
  bool isSameDeviceNum = (Expected.DeviceNum == expectedDefault(int)) ?        \
                            true : (Expected.DeviceNum == Observed.DeviceNum); \
  return isSameKind && isSameEndpoint && isSameDeviceNum;                      \
)
event_class_operator_w_body(TargetEmi,                                         \
  bool isSameKind = (Expected.Kind == Observed.Kind);                          \
  bool isSameEndpoint = (Expected.Endpoint == Observed.Endpoint);              \
  bool isSameDeviceNum = (Expected.DeviceNum == expectedDefault(int)) ?        \
                            true : (Expected.DeviceNum == Observed.DeviceNum); \
  return isSameKind && isSameEndpoint && isSameDeviceNum;                      \
)
event_class_operator_w_body(TargetDataOp,                                      \
  bool isSameOpType = (Expected.OpType == Observed.OpType);                    \
  bool isSameSize = (Expected.Bytes == expectedDefault(size_t)) ?              \
                       true : (Expected.Bytes == Observed.Bytes);              \
  bool isSameSrcAddr = (Expected.SrcAddr == expectedDefault(void *)) ?         \
                          true : (Expected.SrcAddr == Observed.SrcAddr);       \
  bool isSameDstAddr = (Expected.DstAddr == expectedDefault(void *)) ?         \
                          true :  (Expected.DstAddr == Observed.DstAddr);      \
  bool isSameSrcDeviceNum =                                                    \
    (Expected.SrcDeviceNum == expectedDefault(int)) ?                          \
       true : (Expected.SrcDeviceNum == Observed.SrcDeviceNum);                \
  bool isSameDstDeviceNum =                                                    \
    (Expected.DstDeviceNum == expectedDefault(int)) ?                          \
       true : (Expected.DstDeviceNum == Observed.DstDeviceNum);                \
  return isSameOpType && isSameSize && isSameSrcAddr && isSameDstAddr &&       \
         isSameSrcDeviceNum && isSameDstDeviceNum;                             \
)
event_class_operator_w_body(TargetDataOpEmi,                                   \
  bool isSameOpType = (Expected.OpType == Observed.OpType);                    \
  bool isSameEndpoint = (Expected.Endpoint == Observed.Endpoint);              \
  bool isSameSize = (Expected.Bytes == expectedDefault(size_t)) ?              \
                       true : (Expected.Bytes == Observed.Bytes);              \
  bool isSameSrcAddr = (Expected.SrcAddr == expectedDefault(void *)) ?         \
                          true : (Expected.SrcAddr == Observed.SrcAddr);       \
  bool isSameDstAddr = (Expected.DstAddr == expectedDefault(void *)) ?         \
                          true :  (Expected.DstAddr == Observed.DstAddr);      \
  bool isSameSrcDeviceNum =                                                    \
    (Expected.SrcDeviceNum == expectedDefault(int)) ?                          \
       true : (Expected.SrcDeviceNum == Observed.SrcDeviceNum);                \
  bool isSameDstDeviceNum =                                                    \
    (Expected.DstDeviceNum == expectedDefault(int)) ?                          \
       true : (Expected.DstDeviceNum == Observed.DstDeviceNum);                \
  return isSameOpType && isSameEndpoint && isSameSize && isSameSrcAddr &&      \
         isSameDstAddr && isSameSrcDeviceNum && isSameDstDeviceNum;            \
)
event_class_operator_w_body(TargetSubmit,                                      \
  bool isSameReqNumTeams =                                                     \
    (Expected.RequestedNumTeams == Observed.RequestedNumTeams);                \
  return isSameReqNumTeams;                                                    \
)
event_class_operator_w_body(TargetSubmitEmi,                                   \
  bool isSameReqNumTeams =                                                     \
    (Expected.RequestedNumTeams == Observed.RequestedNumTeams);                \
  bool isSameEndpoint = (Expected.Endpoint == Observed.Endpoint);              \
  return isSameReqNumTeams && isSameEndpoint;                                  \
)
event_class_operator_stub(ControlTool)
event_class_operator_w_body(DeviceInitialize,                                  \
  bool isSameDeviceNum = (Expected.DeviceNum == Observed.DeviceNum);           \
  bool isSameType = (Expected.Type == expectedDefault(const char *)) ?         \
                       true :                                                  \
                       ((Expected.Type == Observed.Type) ||                    \
                        (strcmp(Expected.Type, Observed.Type) == 0));          \
  bool isSameDevice =                                                          \
    (Expected.Device == expectedDefault(ompt_device_t *)) ?                    \
       true : (Expected.Device == Observed.Device);                            \
  return isSameDeviceNum && isSameType && isSameDevice;                        \
)
event_class_operator_stub(DeviceFinalize)
event_class_operator_w_body(DeviceLoad,                                        \
  bool isSameDeviceNum = (Expected.DeviceNum == expectedDefault(int)) ?        \
                            true : (Expected.DeviceNum == Observed.DeviceNum); \
  bool isSameSize = (Expected.Bytes == expectedDefault(size_t)) ?              \
                       true : (Expected.Bytes == Observed.Bytes);              \
  return isSameDeviceNum && isSameSize;                                        \
)
event_class_operator_stub(DeviceUnload)
event_class_operator_w_body(BufferRequest,                                     \
  bool isSameDeviceNum = (Expected.DeviceNum == expectedDefault(int)) ?        \
                            true : (Expected.DeviceNum == Observed.DeviceNum); \
  bool isSameSize = (Expected.Bytes == expectedDefault(size_t *)) ?            \
                       true : (Expected.Bytes == Observed.Bytes);              \
  return isSameDeviceNum && isSameSize;                                        \
)
event_class_operator_w_body(BufferComplete,                                    \
  bool isSameDeviceNum = (Expected.DeviceNum == expectedDefault(int)) ?        \
                            true : (Expected.DeviceNum == Observed.DeviceNum); \
  bool isSameSize = (Expected.Bytes == expectedDefault(size_t)) ?              \
                       true : (Expected.Bytes == Observed.Bytes);              \
  return isSameDeviceNum && isSameSize;                                        \
)
event_class_operator_w_body(BufferRecord,                                      \
  bool isSameType = (Expected.Record.type == Observed.Record.type);            \
  bool isSameTargetId =                                                        \
      (Expected.Record.target_id == expectedDefault(ompt_id_t))                \
       ? true                                                                  \
       : (Expected.Record.target_id == Observed.Record.target_id);             \
  if (!(isSameType && isSameTargetId)) return false;                           \
  bool isEqual = true;                                                         \
  ompt_device_time_t ObservedDurationNs =                                      \
      Observed.Record.record.target_data_op.end_time - Observed.Record.time;   \
  switch(Expected.Record.type) {                                               \
  case ompt_callback_target:                                                   \
    isEqual &=                                                                 \
      (Expected.Record.record.target.kind == expectedDefault(ompt_target_t))   \
        ? true                                                                 \
        : (Expected.Record.record.target.kind ==                               \
           Observed.Record.record.target.kind);                                \
    isEqual &=                                                                 \
      (Expected.Record.record.target.endpoint ==                               \
       expectedDefault(ompt_scope_endpoint_t))                                 \
        ? true                                                                 \
        : (Expected.Record.record.target.endpoint ==                           \
           Observed.Record.record.target.endpoint);                            \
    isEqual &=                                                                 \
      (Expected.Record.record.target.device_num == expectedDefault(int))       \
        ? true                                                                 \
        : (Expected.Record.record.target.device_num ==                         \
           Observed.Record.record.target.device_num);                          \
    break;                                                                     \
  case ompt_callback_target_data_op:                                           \
    isEqual &=                                                                 \
      (Expected.Record.record.target_data_op.optype ==                         \
       expectedDefault(ompt_target_data_op_t))                                 \
       ? true                                                                  \
       : (Expected.Record.record.target_data_op.optype ==                      \
          Observed.Record.record.target_data_op.optype);                       \
    isEqual &=                                                                 \
      (Expected.Record.record.target_data_op.bytes == expectedDefault(size_t)) \
       ? true                                                                  \
       : (Expected.Record.record.target_data_op.bytes ==                       \
          Observed.Record.record.target_data_op.bytes);                        \
    isEqual &=                                                                 \
      (Expected.Record.record.target_data_op.src_addr ==                       \
       expectedDefault(void *))                                                \
       ? true                                                                  \
       : (Expected.Record.record.target_data_op.src_addr ==                    \
          Observed.Record.record.target_data_op.src_addr);                     \
    isEqual &=                                                                 \
      (Expected.Record.record.target_data_op.dest_addr ==                      \
       expectedDefault(void *))                                                \
       ? true                                                                  \
       : (Expected.Record.record.target_data_op.dest_addr ==                   \
          Observed.Record.record.target_data_op.dest_addr);                    \
    isEqual &=                                                                 \
      (Expected.Record.record.target_data_op.src_device_num ==                 \
       expectedDefault(int))                                                   \
       ? true                                                                  \
       : (Expected.Record.record.target_data_op.src_device_num ==              \
          Observed.Record.record.target_data_op.src_device_num);               \
    isEqual &=                                                                 \
      (Expected.Record.record.target_data_op.dest_device_num ==                \
       expectedDefault(int))                                                   \
       ? true                                                                  \
       : (Expected.Record.record.target_data_op.dest_device_num ==             \
          Observed.Record.record.target_data_op.dest_device_num);              \
    isEqual &=                                                                 \
      (Expected.Record.record.target_data_op.host_op_id ==                     \
       expectedDefault(ompt_id_t))                                             \
       ? true                                                                  \
       : (Expected.Record.record.target_data_op.host_op_id ==                  \
          Observed.Record.record.target_data_op.host_op_id);                   \
    isEqual &=                                                                 \
      (Expected.Record.record.target_data_op.codeptr_ra ==                     \
       expectedDefault(void *))                                                \
       ? true                                                                  \
       : (Expected.Record.record.target_data_op.codeptr_ra ==                  \
          Observed.Record.record.target_data_op.codeptr_ra);                   \
    if (Expected.Record.record.target_data_op.end_time !=                      \
        expectedDefault(ompt_device_time_t)) {                                 \
      isEqual &=                                                               \
         ObservedDurationNs <= Expected.Record.record.target_data_op.end_time; \
    }                                                                          \
    isEqual &= ObservedDurationNs >= Expected.Record.time;                     \
    break;                                                                     \
  case ompt_callback_target_submit:                                            \
    isEqual &=                                                                 \
      (Expected.Record.record.target_kernel.requested_num_teams ==             \
       expectedDefault(unsigned int))                                          \
       ? true                                                                  \
       : (Expected.Record.record.target_kernel.requested_num_teams ==          \
          Observed.Record.record.target_kernel.requested_num_teams);           \
    isEqual &=                                                                 \
      (Expected.Record.record.target_kernel.granted_num_teams ==               \
       expectedDefault(unsigned int))                                          \
       ? true                                                                  \
       : (Expected.Record.record.target_kernel.granted_num_teams ==            \
          Observed.Record.record.target_kernel.granted_num_teams);             \
    isEqual &=                                                                 \
      (Expected.Record.record.target_kernel.host_op_id ==                      \
       expectedDefault(ompt_id_t))                                             \
       ? true                                                                  \
       : (Expected.Record.record.target_kernel.host_op_id ==                   \
          Observed.Record.record.target_kernel.host_op_id);                    \
    if (Expected.Record.record.target_kernel.end_time !=                       \
        expectedDefault(ompt_device_time_t)) {                                 \
      isEqual &=                                                               \
         ObservedDurationNs <= Expected.Record.record.target_kernel.end_time;  \
    }                                                                          \
    isEqual &= ObservedDurationNs >= Expected.Record.time;                     \
    break;                                                                     \
  default:                                                                     \
    assert(false && "Encountered invalid record type");                        \
  }                                                                            \
  return isEqual;                                                              \
)
event_class_operator_stub(BufferRecordDeallocation)

define_cast_func(AssertionSyncPoint)
define_cast_func(AssertionSuspend)
define_cast_func(ThreadBegin)
define_cast_func(ThreadEnd)
define_cast_func(ParallelBegin)
define_cast_func(ParallelEnd)
define_cast_func(TaskCreate)
define_cast_func(TaskSchedule)
define_cast_func(ImplicitTask)
define_cast_func(Target)
define_cast_func(TargetEmi)
define_cast_func(TargetDataOp)
define_cast_func(TargetDataOpEmi)
define_cast_func(TargetSubmit)
define_cast_func(TargetSubmitEmi)
define_cast_func(ControlTool)
define_cast_func(DeviceInitialize)
define_cast_func(DeviceFinalize)
define_cast_func(DeviceLoad)
define_cast_func(DeviceUnload)
define_cast_func(BufferRequest)
define_cast_func(BufferComplete)
define_cast_func(BufferRecord)
define_cast_func(BufferRecordDeallocation)

class_equals_op(AssertionSyncPoint)
class_equals_op(AssertionSuspend)
class_equals_op(ThreadBegin)
class_equals_op(ThreadEnd)
class_equals_op(ParallelBegin)
class_equals_op(ParallelEnd)
class_equals_op(TaskCreate)
class_equals_op(TaskSchedule)
class_equals_op(ImplicitTask)
class_equals_op(Target)
class_equals_op(TargetEmi)
class_equals_op(TargetDataOp)
class_equals_op(TargetDataOpEmi)
class_equals_op(TargetSubmit)
class_equals_op(TargetSubmitEmi)
class_equals_op(ControlTool)
class_equals_op(DeviceInitialize)
class_equals_op(DeviceFinalize)
class_equals_op(DeviceLoad)
class_equals_op(DeviceUnload)
class_equals_op(BufferRequest)
class_equals_op(BufferComplete)
class_equals_op(BufferRecord)
class_equals_op(BufferRecordDeallocation)
// clang-format on

} // namespace internal

} // namespace omptest
