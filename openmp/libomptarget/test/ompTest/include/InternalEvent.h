#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_INTERNALEVENT_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_INTERNALEVENT_H

#include "omp-tools.h"

#include <cassert>
#include <iomanip>
#include <sstream>
#include <string>

#include <iostream>

namespace omptest {

namespace internal {
/// Enum values are used for comparison of observed and asserted events
/// List is based on OpenMP 5.2 specification, table 19.2 (page 447)
enum class EventTy {
  None, // not part of OpenMP spec, used for implementation
  ThreadBegin,
  ThreadEnd,
  ParallelBegin,
  ParallelEnd,
  TaskCreate,
  TaskSchedule,
  ImplicitTask,
  Target,
  TargetEmi,
  TargetDataOp,
  TargetDataOpEmi,
  TargetSubmit,
  TargetSubmitEmi,
  ControlTool,
  DeviceInitialize,
  DeviceFinalize,
  DeviceLoad,
  DeviceUnload
};

/// String manipulation helper function. Takes up to 8 bytes of data and returns
/// their hexadecimal representation as string. The data can be truncated to a
/// certain size in bytes and will by default be prefixed with '0x'.
std::string makeHexString(uint64_t Data, size_t DataBytes = 0,
                          bool ShowHexBase = true) {
  if (Data == 0)
    return "(nil)";

  static std::ostringstream os;
  // Clear the content of the stream
  os.str(std::string());

  // Manually prefixing "0x" will make the use of std::setfill more easy
  if (ShowHexBase)
    os << "0x";

  // Default to 32bit (8 hex digits) width if exceeding 64bit or zero value
  size_t NumDigits = (DataBytes > 0 && DataBytes < 9) ? (DataBytes << 1) : 8;

  if (DataBytes > 0)
    os << std::setfill('0') << std::setw(NumDigits);

  os << std::hex << Data;
  return os.str();
}

struct InternalEvent {
  EventTy Type;
  EventTy getType() const { return Type; }

  InternalEvent() : Type(EventTy::None) {}
  InternalEvent(EventTy T) : Type(T) {}
  virtual ~InternalEvent() = default;

  virtual bool equals(const InternalEvent *o) const {
    assert(false && "Base class implementation");
    return false;
  };

  virtual std::string toString() const { return "InternalEvent"; }
};

#define event_class_stub(EvTy)                                                 \
  struct EvTy : public InternalEvent {                                         \
    virtual bool equals(const InternalEvent *o) const override;                \
    EvTy() : InternalEvent(EventTy::EvTy) {}                                   \
  };

#define event_class_w_custom_body(EvTy, ...)                                   \
  struct EvTy : public InternalEvent {                                         \
    virtual bool equals(const InternalEvent *o) const override;                \
    std::string toString() const override;                                     \
    __VA_ARGS__                                                                \
  };

// clang-format off
event_class_w_custom_body(ThreadBegin,                                         \
  ThreadBegin(ompt_thread_t ThreadType)                                        \
    : InternalEvent(EventTy::ThreadBegin), ThreadType(ThreadType) {}           \
                                                                               \
  ompt_thread_t ThreadType;                                                    \
)
event_class_stub(ThreadEnd)
event_class_w_custom_body(ParallelBegin,                                       \
  ParallelBegin(int NumThreads)                                                \
    : InternalEvent(EventTy::ParallelBegin), NumThreads(NumThreads) {}         \
                                                                               \
  int NumThreads;                                                              \
)
event_class_w_custom_body(ParallelEnd,
  ParallelEnd() : InternalEvent(EventTy::ParallelEnd) {}
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
event_class_stub(TargetEmi)
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
event_class_stub(TargetDataOpEmi)
event_class_w_custom_body(TargetSubmit,                                        \
  TargetSubmit(ompt_id_t TargetId, ompt_id_t HostOpId,                         \
               unsigned int ReqNumTeams)                                       \
    : InternalEvent(EventTy::TargetSubmit), TargetId(TargetId),                \
      HostOpId(HostOpId), ReqNumTeams(ReqNumTeams) {}                          \
                                                                               \
  ompt_id_t TargetId;                                                          \
  ompt_id_t HostOpId;                                                          \
  unsigned int ReqNumTeams;                                                    \
)
event_class_stub(TargetSubmitEmi)
event_class_stub(ControlTool)
event_class_w_custom_body(DeviceInitialize,                                    \
  DeviceInitialize(int DeviceNum, const char *Type, ompt_device_t *Device,     \
    ompt_function_lookup_t LookupFn, const char *DocStr)                       \
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
    void *VmaInFile, size_t Bytes, void *HostAddr, void *DeviceAddr,           \
    uint64_t ModuleId)                                                         \
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
//     clang-format on

std::string ThreadBegin::toString() const {
  std::string S{"OMPT Callback ThreadBegin: "};
  S.append("ThreadType=").append(std::to_string(ThreadType));
  return S;
}

std::string ParallelBegin::toString() const {
  std::string S{"OMPT Callback ParallelBegin: "};
  S.append("NumThreads=").append(std::to_string(NumThreads));
  return S;
}

std::string ParallelEnd::toString() const {
  std::string S{"OMPT Callback ParallelEnd"};
  return S;
}

std::string Target::toString() const {
  std::string S{"Callback Target: target_id="};
  S.append(std::to_string(TargetId));
  S.append(" kind=").append(std::to_string(Kind));
  S.append(" endpoint=").append(std::to_string(Endpoint));
  S.append(" device_num=").append(std::to_string(DeviceNum));
  S.append(" code=").append(makeHexString((uint64_t)CodeptrRA));
  return S;
}

std::string TargetDataOp::toString() const {
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

std::string TargetSubmit::toString() const {
  std::string S{"  Callback Submit: target_id="};
  S.append(std::to_string(TargetId));
  S.append(" host_op_id=").append(std::to_string(HostOpId));
  S.append(" req_num_teams=").append(std::to_string(ReqNumTeams));
  return S;
}

std::string DeviceInitialize::toString() const {
  std::string S{"Callback Init: device_num="};
  S.append(std::to_string(DeviceNum));
  S.append(" type=").append(Type);
  S.append(" device=").append(makeHexString((uint64_t)Device));
  S.append(" lookup=").append(makeHexString((uint64_t)LookupFn));
  S.append(" doc=").append(makeHexString((uint64_t)DocStr));
  return S;
}

std::string DeviceFinalize::toString() const {
  std::string S{"Callback Fini: device_num="};
  S.append(std::to_string(DeviceNum));
  return S;
}

std::string DeviceLoad::toString() const {
  std::string S{"Callback Load: device_num:"};
  S.append(std::to_string(DeviceNum));
  S.append(" module_id:").append(std::to_string(ModuleId));
  S.append(" filename:").append((Filename == nullptr) ? "(null)" : Filename);
  S.append(" host_adddr:").append(makeHexString((uint64_t)HostAddr));
  S.append(" device_addr:").append(makeHexString((uint64_t)DeviceAddr));
  S.append(" bytes:").append(std::to_string(Bytes));
  return S;
}

#define event_class_operator_stub(EvTy)                                        \
  bool operator==(const EvTy &a, const EvTy &b) { return true; }

#define event_class_operator_w_body(EvTy, ...)                                 \
  bool operator==(const EvTy &a, const EvTy &b) { __VA_ARGS__ }

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
// clang-format on

/// Template "base" for the cast functions generated in the define_cast_func
/// macro
template <typename To>
const To *cast(const InternalEvent *From) {
  return nullptr;
}

/// Generates template specialization of the cast operation for the specified
/// EvTy as the template parameter
#define define_cast_func(EvTy)                                                 \
  template <> const EvTy *cast(const InternalEvent *From) {                    \
    if (From->getType() == EventTy::EvTy)                                      \
      return static_cast<const EvTy *>(From);                                  \
    return nullptr;                                                            \
  }

// clang-format off
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
// clang-format on

/// Auto generate the equals override to cast and dispatch to the specific class
/// operator==
#define class_equals_op(EvTy)                                                  \
  bool EvTy::equals(const InternalEvent *o) const {                            \
    if (const auto O = cast<EvTy>(o))                                          \
      return *this == *O;                                                      \
    return false;                                                              \
  }

// clang-format off
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
// clang-format on

} // namespace internal

} // namespace omptest

#endif
