#include "../include/OmptAssertEvent.h"

using namespace omptest;

OmptAssertEvent::OmptAssertEvent(const std::string &Name,
                                 internal::InternalEvent *IE)
    : Name(Name), TheEvent(IE) {}

OmptAssertEvent OmptAssertEvent::ThreadBegin(ompt_thread_t ThreadType,
                                             const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName, new internal::ThreadBegin(ThreadType));
}

OmptAssertEvent OmptAssertEvent::ThreadEnd(const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName, new internal::ThreadEnd());
}

OmptAssertEvent OmptAssertEvent::ParallelBegin(int NumThreads,
                                               const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName, new internal::ParallelBegin(NumThreads));
}

OmptAssertEvent OmptAssertEvent::ParallelEnd(const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName, new internal::ParallelEnd());
}

OmptAssertEvent OmptAssertEvent::TaskCreate(const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName, new internal::TaskCreate());
}

OmptAssertEvent OmptAssertEvent::TaskSchedule(const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName, new internal::TaskSchedule());
}

OmptAssertEvent OmptAssertEvent::ImplicitTask(const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName, new internal::ImplicitTask());
}

OmptAssertEvent OmptAssertEvent::Target(ompt_target_t Kind,
                                        ompt_scope_endpoint_t Endpoint,
                                        int DeviceNum, ompt_data_t *TaskData,
                                        ompt_id_t TargetId,
                                        const void *CodeptrRA,
                                        const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName,
                         new internal::Target(Kind, Endpoint, DeviceNum,
                                              TaskData, TargetId, CodeptrRA));
}

OmptAssertEvent
OmptAssertEvent::TargetEmi(ompt_target_t Kind, ompt_scope_endpoint_t Endpoint,
                           int DeviceNum, ompt_data_t *TaskData,
                           ompt_data_t *TargetTaskData, ompt_data_t *TargetData,
                           const void *CodeptrRA, const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(
      EName, new internal::TargetEmi(Kind, Endpoint, DeviceNum, TaskData,
                                     TargetTaskData, TargetData, CodeptrRA));
}

/// Create a DataAlloc Event
OmptAssertEvent OmptAssertEvent::TargetDataOp(
    ompt_id_t TargetId, ompt_id_t HostOpId, ompt_target_data_op_t OpType,
    void *SrcAddr, int SrcDeviceNum, void *DstAddr, int DstDeviceNum,
    size_t Bytes, const void *CodeptrRA, const std::string &Name) {
  return OmptAssertEvent(
      Name, new internal::TargetDataOp(TargetId, HostOpId, OpType, SrcAddr,
                                       SrcDeviceNum, DstAddr, DstDeviceNum,
                                       Bytes, CodeptrRA));
}

OmptAssertEvent OmptAssertEvent::TargetDataOpEmi(
    ompt_scope_endpoint_t Endpoint, ompt_data_t *TargetTaskData,
    ompt_data_t *TargetData, ompt_id_t *HostOpId, ompt_target_data_op_t OpType,
    void *SrcAddr, int SrcDeviceNum, void *DstAddr, int DstDeviceNum,
    size_t Bytes, const void *CodeptrRA, const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName, new internal::TargetDataOpEmi(
                                    Endpoint, TargetTaskData, TargetData,
                                    HostOpId, OpType, SrcAddr, SrcDeviceNum,
                                    DstAddr, DstDeviceNum, Bytes, CodeptrRA));
}

OmptAssertEvent OmptAssertEvent::TargetSubmit(ompt_id_t TargetId,
                                              ompt_id_t HostOpId,
                                              unsigned int RequestedNumTeams,
                                              const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(
      EName, new internal::TargetSubmit(TargetId, HostOpId, RequestedNumTeams));
}

OmptAssertEvent OmptAssertEvent::TargetSubmitEmi(ompt_scope_endpoint_t Endpoint,
                                                 ompt_data_t *TargetData,
                                                 ompt_id_t *HostOpId,
                                                 unsigned int RequestedNumTeams,
                                                 const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(
      EName, new internal::TargetSubmitEmi(Endpoint, TargetData, HostOpId,
                                           RequestedNumTeams));
}

OmptAssertEvent OmptAssertEvent::ControlTool(std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName, new internal::ControlTool());
}

OmptAssertEvent OmptAssertEvent::DeviceInitialize(
    int DeviceNum, const char *Type, ompt_device_t *Device,
    ompt_function_lookup_t LookupFn, const char *DocumentationStr,
    const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(
      EName, new internal::DeviceInitialize(DeviceNum, Type, Device, LookupFn,
                                            DocumentationStr));
}

OmptAssertEvent OmptAssertEvent::DeviceFinalize(int DeviceNum,
                                                const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName, new internal::DeviceFinalize(DeviceNum));
}

OmptAssertEvent OmptAssertEvent::DeviceLoad(int DeviceNum, const char *Filename,
                                            int64_t OffsetInFile,
                                            void *VmaInFile, size_t Bytes,
                                            void *HostAddr, void *DeviceAddr,
                                            uint64_t ModuleId,
                                            const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(
      EName,
      new internal::DeviceLoad(DeviceNum, Filename, OffsetInFile, VmaInFile,
                               Bytes, HostAddr, DeviceAddr, ModuleId));
}

OmptAssertEvent OmptAssertEvent::DeviceUnload(const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName, new internal::DeviceUnload());
}

OmptAssertEvent OmptAssertEvent::BufferRequest(int DeviceNum,
                                               ompt_buffer_t **Buffer,
                                               size_t *Bytes,
                                               const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName,
                         new internal::BufferRequest(DeviceNum, Buffer, Bytes));
}

OmptAssertEvent
OmptAssertEvent::BufferComplete(int DeviceNum, ompt_buffer_t *Buffer,
                                size_t Bytes, ompt_buffer_cursor_t Begin,
                                int BufferOwned, const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName,
                         new internal::BufferComplete(DeviceNum, Buffer, Bytes,
                                                      Begin, BufferOwned));
}

OmptAssertEvent OmptAssertEvent::BufferRecord(ompt_record_ompt_t *Record,
                                              const std::string &Name) {
  auto EName = getName(Name);
  return OmptAssertEvent(EName, new internal::BufferRecord(Record));
}

std::string OmptAssertEvent::getEventName() const { return Name; }

internal::EventTy OmptAssertEvent::getEventType() const {
  return TheEvent->getType();
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
