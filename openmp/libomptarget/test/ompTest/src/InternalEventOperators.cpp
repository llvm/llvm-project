#include "../include/InternalEvent.h"

namespace omptest {

namespace internal {
// clang-format off
event_class_operator_stub(ThreadBegin)
event_class_operator_stub(ThreadEnd)
event_class_operator_w_body(ParallelBegin,                                     \
return a.NumThreads == b.NumThreads;                                           \
)
event_class_operator_stub(ParallelEnd)
event_class_operator_stub(TaskCreate)
event_class_operator_stub(TaskSchedule)
event_class_operator_stub(ImplicitTask)
event_class_operator_stub(Target)
event_class_operator_stub(TargetEmi)
event_class_operator_stub(TargetDataOp)
event_class_operator_stub(TargetDataOpEmi)
event_class_operator_stub(TargetSubmit)
event_class_operator_stub(TargetSubmitEmi)
event_class_operator_stub(ControlTool)
event_class_operator_stub(DeviceInitialize)
event_class_operator_stub(DeviceFinalize)
event_class_operator_stub(DeviceLoad)
event_class_operator_stub(DeviceUnload)
event_class_operator_stub(BufferRequest)
event_class_operator_stub(BufferComplete)
event_class_operator_stub(BufferRecord)

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
// clang-format on

} // namespace internal

} // namespace omptest
