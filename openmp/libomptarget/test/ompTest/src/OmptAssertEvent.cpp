#include "OmptAssertEvent.h"

using namespace omptest;

OmptAssertEvent::OmptAssertEvent(const std::string &Name,
                                 const std::string &Group,
                                 const ObserveState &Expected,
                                 internal::InternalEvent *IE)
    : Name(Name), Group(Group), ExpectedState(Expected), TheEvent(IE) {}

OmptAssertEvent OmptAssertEvent::Asserter(const std::string &Name,
                                          const std::string &Group,
                                          const ObserveState &Expected) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected, new internal::Asserter());
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
                                             const ObserveState &Expected) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected, new internal::ParallelEnd());
}

OmptAssertEvent OmptAssertEvent::TaskCreate(const std::string &Name,
                                            const std::string &Group,
                                            const ObserveState &Expected) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected, new internal::TaskCreate());
}

OmptAssertEvent OmptAssertEvent::TaskSchedule(const std::string &Name,
                                              const std::string &Group,
                                              const ObserveState &Expected) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected, new internal::TaskSchedule());
}

OmptAssertEvent OmptAssertEvent::ImplicitTask(const std::string &Name,
                                              const std::string &Group,
                                              const ObserveState &Expected) {
  auto EName = getName(Name);
  auto EGroup = getGroup(Group);
  return OmptAssertEvent(EName, EGroup, Expected, new internal::ImplicitTask());
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

std::string OmptAssertEvent::getEventName() const { return Name; }

std::string OmptAssertEvent::getEventGroup() const { return Group; }

ObserveState OmptAssertEvent::getEventExpectedState() const {
  return ExpectedState;
}

internal::EventTy OmptAssertEvent::getEventType() const {
  return TheEvent->getType();
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

  return A.TheEvent->getType() == B.TheEvent->getType() &&
         A.TheEvent->equals(B.TheEvent.get());
}
