#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTEVENT_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTEVENT_H

#include "InternalEvent.h"
#include "omp-tools.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>

namespace omptest{

enum class AssertState { pass, fail };

struct OmptAssertEvent {
  static OmptAssertEvent ThreadBegin(ompt_thread_t ThreadType,
                                     const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::ThreadBegin(ThreadType));
  }

  static OmptAssertEvent ThreadEnd(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::ThreadEnd());
  }

  static OmptAssertEvent ParallelBegin(int NumThreads,
                                       const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::ParallelBegin(NumThreads));
  }

  static OmptAssertEvent ParallelEnd(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::ParallelEnd());
  }

  static OmptAssertEvent TaskCreate(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::TaskCreate());
  }

  static OmptAssertEvent TaskSchedule(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::TaskSchedule());
  }

  static OmptAssertEvent ImplicitTask(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::ImplicitTask());
  }

  static OmptAssertEvent Target(ompt_target_t Kind,
                                ompt_scope_endpoint_t Endpoint, int DeviceNum,
                                ompt_data_t *TaskData, ompt_id_t TargetId,
                                const void *CodeptrRA,
                                const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName,
                           new internal::Target(Kind, Endpoint, DeviceNum,
                                                TaskData, TargetId, CodeptrRA));
  }

  static OmptAssertEvent
  TargetEmi(ompt_target_t Kind, ompt_scope_endpoint_t Endpoint, int DeviceNum,
            ompt_data_t *TaskData, ompt_data_t *TargetTaskData,
            ompt_data_t *TargetData, const void *CodeptrRA,
            const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(
        EName, new internal::TargetEmi(Kind, Endpoint, DeviceNum, TaskData,
                                       TargetTaskData, TargetData, CodeptrRA));
  }

  /// Create a DataAlloc Event
  static OmptAssertEvent TargetDataOp(ompt_id_t TargetId, ompt_id_t HostOpId,
                                      ompt_target_data_op_t OpType,
                                      void *SrcAddr, int SrcDeviceNum,
                                      void *DstAddr, int DstDeviceNum,
                                      size_t Bytes, const void *CodeptrRA,
                                      const std::string &Name) {
    return OmptAssertEvent(
        Name, new internal::TargetDataOp(TargetId, HostOpId, OpType, SrcAddr,
                                         SrcDeviceNum, DstAddr, DstDeviceNum,
                                         Bytes, CodeptrRA));
  }

  static OmptAssertEvent
  TargetDataOpEmi(ompt_scope_endpoint_t Endpoint, ompt_data_t *TargetTaskData,
                  ompt_data_t *TargetData, ompt_id_t *HostOpId,
                  ompt_target_data_op_t OpType, void *SrcAddr, int SrcDeviceNum,
                  void *DstAddr, int DstDeviceNum, size_t Bytes,
                  const void *CodeptrRA, const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::TargetDataOpEmi(
                                      Endpoint, TargetTaskData, TargetData,
                                      HostOpId, OpType, SrcAddr, SrcDeviceNum,
                                      DstAddr, DstDeviceNum, Bytes, CodeptrRA));
  }

  static OmptAssertEvent TargetSubmit(ompt_id_t TargetId, ompt_id_t HostOpId,
                                      unsigned int RequestedNumTeams,
                                      const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::TargetSubmit(
                                      TargetId, HostOpId, RequestedNumTeams));
  }

  static OmptAssertEvent TargetSubmitEmi(ompt_scope_endpoint_t Endpoint,
                                         ompt_data_t *TargetData,
                                         ompt_id_t *HostOpId,
                                         unsigned int RequestedNumTeams,
                                         const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(
        EName, new internal::TargetSubmitEmi(Endpoint, TargetData, HostOpId,
                                             RequestedNumTeams));
  }

  static OmptAssertEvent ControlTool(std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::ControlTool());
  }

  static OmptAssertEvent DeviceInitialize(int DeviceNum, const char *Type,
                                          ompt_device_t *Device,
                                          ompt_function_lookup_t LookupFn,
                                          const char *DocumentationStr,
                                          const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(
        EName, new internal::DeviceInitialize(DeviceNum, Type, Device, LookupFn,
                                              DocumentationStr));
  }

  static OmptAssertEvent DeviceFinalize(int DeviceNum,
                                        const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::DeviceFinalize(DeviceNum));
  }

  static OmptAssertEvent DeviceLoad(int DeviceNum, const char *Filename,
                                    int64_t OffsetInFile, void *VmaInFile,
                                    size_t Bytes, void *HostAddr,
                                    void *DeviceAddr, uint64_t ModuleId,
                                    const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(
        EName,
        new internal::DeviceLoad(DeviceNum, Filename, OffsetInFile, VmaInFile,
                                 Bytes, HostAddr, DeviceAddr, ModuleId));
  }

  static OmptAssertEvent DeviceUnload(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::DeviceUnload());
  }

  static OmptAssertEvent BufferRequest(int DeviceNum, ompt_buffer_t **Buffer,
                                       size_t *Bytes, const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(
        EName, new internal::BufferRequest(DeviceNum, Buffer, Bytes));
  }

  static OmptAssertEvent BufferComplete(int DeviceNum, ompt_buffer_t *Buffer,
                                        size_t Bytes,
                                        ompt_buffer_cursor_t Begin,
                                        int BufferOwned,
                                        const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(
        EName, new internal::BufferComplete(DeviceNum, Buffer, Bytes, Begin,
                                            BufferOwned));
  }

  static OmptAssertEvent BufferRecord(ompt_record_ompt_t *Record,
                                      const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::BufferRecord(Record));
  }

  /// Allow move construction (due to std::unique_ptr)
  OmptAssertEvent(OmptAssertEvent &&o) = default;
  OmptAssertEvent &operator=(OmptAssertEvent &&o) = default;

  std::string getEventName() const { return Name; }

  internal::EventTy getEventType() const { return TheEvent->getType(); }

  /// Make events comparable
  friend bool operator==(const OmptAssertEvent &A, const OmptAssertEvent &B);

  std::string toString(bool PrefixEventName = false) const {
    std::string S;
    if (PrefixEventName)
      S.append(getEventName()).append(": ");
    S.append((TheEvent == nullptr) ? "OmptAssertEvent" : TheEvent->toString());
    return S;
  }

private:
  OmptAssertEvent(const std::string &Name, internal::InternalEvent *IE)
      : Name(Name), TheEvent(IE) {}
  OmptAssertEvent(const OmptAssertEvent &o) = delete;

  static std::string getName(const std::string &Name,
                             const char *Caller = __builtin_FUNCTION()) {
    std::string EName = Name;
    if (EName.empty())
      EName.append(Caller).append(" (auto generated)");

    return EName;
  }

  std::string Name;
  std::unique_ptr<internal::InternalEvent> TheEvent;
};

bool operator==(const OmptAssertEvent &A, const OmptAssertEvent &B) {
  assert(A.TheEvent.get() != nullptr && "A is valid");
  assert(B.TheEvent.get() != nullptr && "B is valid");

  return A.TheEvent->getType() == B.TheEvent->getType() &&
         A.TheEvent->equals(B.TheEvent.get());
}

} // namespace omptest

#endif
