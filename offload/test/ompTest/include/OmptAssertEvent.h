#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTEVENT_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTEVENT_H

#include "InternalEvent.h"
#include "omp-tools.h"

#include <cassert>
#include <limits>
#include <memory>
#include <string>

namespace omptest{

enum class AssertState { pass, fail };
enum class ObserveState { generated, always, never };

struct OmptAssertEvent {

  static OmptAssertEvent Asserter(const std::string &Name,
                                  const std::string &Group,
                                  const ObserveState &Expected);

  static OmptAssertEvent ThreadBegin(const std::string &Name,
                                     const std::string &Group,
                                     const ObserveState &Expected,
                                     ompt_thread_t ThreadType);

  static OmptAssertEvent ThreadEnd(const std::string &Name,
                                   const std::string &Group,
                                   const ObserveState &Expected);

  static OmptAssertEvent ParallelBegin(const std::string &Name,
                                       const std::string &Group,
                                       const ObserveState &Expected,
                                       int NumThreads);

  static OmptAssertEvent ParallelEnd(const std::string &Name,
                                     const std::string &Group,
                                     const ObserveState &Expected);

  static OmptAssertEvent TaskCreate(const std::string &Name,
                                    const std::string &Group,
                                    const ObserveState &Expected);

  static OmptAssertEvent TaskSchedule(const std::string &Name,
                                      const std::string &Group,
                                      const ObserveState &Expected);

  static OmptAssertEvent ImplicitTask(const std::string &Name,
                                      const std::string &Group,
                                      const ObserveState &Expected);

  static OmptAssertEvent
  Target(const std::string &Name, const std::string &Group,
         const ObserveState &Expected, ompt_target_t Kind,
         ompt_scope_endpoint_t Endpoint,
         int DeviceNum = std::numeric_limits<int>::min(),
         ompt_data_t *TaskData = std::numeric_limits<ompt_data_t *>::min(),
         ompt_id_t TargetId = std::numeric_limits<ompt_id_t>::min(),
         const void *CodeptrRA = std::numeric_limits<void *>::min());

  static OmptAssertEvent TargetEmi(
      const std::string &Name, const std::string &Group,
      const ObserveState &Expected, ompt_target_t Kind,
      ompt_scope_endpoint_t Endpoint, int DeviceNum,
      ompt_data_t *TaskData = std::numeric_limits<ompt_data_t *>::min(),
      ompt_data_t *TargetTaskData = std::numeric_limits<ompt_data_t *>::min(),
      ompt_data_t *TargetData = std::numeric_limits<ompt_data_t *>::min(),
      const void *CodeptrRA = std::numeric_limits<void *>::min());

  static OmptAssertEvent
  TargetDataOp(const std::string &Name, const std::string &Group,
               const ObserveState &Expected, ompt_id_t TargetId,
               ompt_id_t HostOpId, ompt_target_data_op_t OpType, void *SrcAddr,
               int SrcDeviceNum, void *DstAddr, int DstDeviceNum, size_t Bytes,
               const void *CodeptrRA);

  static OmptAssertEvent
  TargetDataOp(const std::string &Name, const std::string &Group,
               const ObserveState &Expected, ompt_target_data_op_t OpType,
               size_t Bytes, void *SrcAddr = std::numeric_limits<void *>::min(),
               void *DstAddr = std::numeric_limits<void *>::min(),
               int SrcDeviceNum = std::numeric_limits<int>::min(),
               int DstDeviceNum = std::numeric_limits<int>::min(),
               ompt_id_t TargetId = std::numeric_limits<ompt_id_t>::min(),
               ompt_id_t HostOpId = std::numeric_limits<ompt_id_t>::min(),
               const void *CodeptrRA = std::numeric_limits<void *>::min());

  static OmptAssertEvent
  TargetDataOpEmi(const std::string &Name, const std::string &Group,
                  const ObserveState &Expected, ompt_scope_endpoint_t Endpoint,
                  ompt_data_t *TargetTaskData, ompt_data_t *TargetData,
                  ompt_id_t *HostOpId, ompt_target_data_op_t OpType,
                  void *SrcAddr, int SrcDeviceNum, void *DstAddr,
                  int DstDeviceNum, size_t Bytes, const void *CodeptrRA);

  static OmptAssertEvent TargetDataOpEmi(
      const std::string &Name, const std::string &Group,
      const ObserveState &Expected, ompt_target_data_op_t OpType,
      ompt_scope_endpoint_t Endpoint, size_t Bytes,
      void *SrcAddr = std::numeric_limits<void *>::min(),
      void *DstAddr = std::numeric_limits<void *>::min(),
      int SrcDeviceNum = std::numeric_limits<int>::min(),
      int DstDeviceNum = std::numeric_limits<int>::min(),
      ompt_data_t *TargetTaskData = std::numeric_limits<ompt_data_t *>::min(),
      ompt_data_t *TargetData = std::numeric_limits<ompt_data_t *>::min(),
      ompt_id_t *HostOpId = std::numeric_limits<ompt_id_t *>::min(),
      const void *CodeptrRA = std::numeric_limits<void *>::min());

  static OmptAssertEvent TargetSubmit(const std::string &Name,
                                      const std::string &Group,
                                      const ObserveState &Expected,
                                      ompt_id_t TargetId, ompt_id_t HostOpId,
                                      unsigned int RequestedNumTeams);

  static OmptAssertEvent
  TargetSubmit(const std::string &Name, const std::string &Group,
               const ObserveState &Expected, unsigned int RequestedNumTeams,
               ompt_id_t TargetId = std::numeric_limits<ompt_id_t>::min(),
               ompt_id_t HostOpId = std::numeric_limits<ompt_id_t>::min());

  static OmptAssertEvent
  TargetSubmitEmi(const std::string &Name, const std::string &Group,
                  const ObserveState &Expected, ompt_scope_endpoint_t Endpoint,
                  ompt_data_t *TargetData, ompt_id_t *HostOpId,
                  unsigned int RequestedNumTeams);

  static OmptAssertEvent TargetSubmitEmi(
      const std::string &Name, const std::string &Group,
      const ObserveState &Expected, unsigned int RequestedNumTeams,
      ompt_scope_endpoint_t Endpoint,
      ompt_data_t *TargetData = std::numeric_limits<ompt_data_t *>::min(),
      ompt_id_t *HostOpId = std::numeric_limits<ompt_id_t *>::min());

  static OmptAssertEvent ControlTool(const std::string &Name,
                                     const std::string &Group,
                                     const ObserveState &Expected);

  static OmptAssertEvent DeviceInitialize(
      const std::string &Name, const std::string &Group,
      const ObserveState &Expected, int DeviceNum,
      const char *Type = std::numeric_limits<const char *>::min(),
      ompt_device_t *Device = std::numeric_limits<ompt_device_t *>::min(),
      ompt_function_lookup_t LookupFn =
          std::numeric_limits<ompt_function_lookup_t>::min(),
      const char *DocumentationStr = std::numeric_limits<const char *>::min());

  static OmptAssertEvent DeviceFinalize(const std::string &Name,
                                        const std::string &Group,
                                        const ObserveState &Expected,
                                        int DeviceNum);

  static OmptAssertEvent
  DeviceLoad(const std::string &Name, const std::string &Group,
             const ObserveState &Expected, int DeviceNum,
             const char *Filename = std::numeric_limits<const char *>::min(),
             int64_t OffsetInFile = std::numeric_limits<int64_t>::min(),
             void *VmaInFile = std::numeric_limits<void *>::min(),
             size_t Bytes = std::numeric_limits<size_t>::min(),
             void *HostAddr = std::numeric_limits<void *>::min(),
             void *DeviceAddr = std::numeric_limits<void *>::min(),
             uint64_t ModuleId = std::numeric_limits<int64_t>::min());

  static OmptAssertEvent DeviceUnload(const std::string &Name,
                                      const std::string &Group,
                                      const ObserveState &Expected);

  static OmptAssertEvent BufferRequest(const std::string &Name,
                                       const std::string &Group,
                                       const ObserveState &Expected,
                                       int DeviceNum, ompt_buffer_t **Buffer,
                                       size_t *Bytes);

  static OmptAssertEvent
  BufferComplete(const std::string &Name, const std::string &Group,
                 const ObserveState &Expected, int DeviceNum,
                 ompt_buffer_t *Buffer, size_t Bytes,
                 ompt_buffer_cursor_t Begin, int BufferOwned);

  static OmptAssertEvent BufferRecord(const std::string &Name,
                                      const std::string &Group,
                                      const ObserveState &Expected,
                                      ompt_record_ompt_t *Record);

  /// Allow move construction (due to std::unique_ptr)
  OmptAssertEvent(OmptAssertEvent &&o) = default;
  OmptAssertEvent &operator=(OmptAssertEvent &&o) = default;

  std::string getEventName() const;

  std::string getEventGroup() const;

  ObserveState getEventExpectedState() const;

  internal::EventTy getEventType() const;

  internal::InternalEvent *getEvent() const;

  /// Make events comparable
  friend bool operator==(const OmptAssertEvent &A, const OmptAssertEvent &B);

  std::string toString(bool PrefixEventName = false) const;

private:
  OmptAssertEvent(const std::string &Name, const std::string &Group,
                  const ObserveState &Expected, internal::InternalEvent *IE);
  OmptAssertEvent(const OmptAssertEvent &o) = delete;

  static std::string getName(const std::string &Name,
                             const char *Caller = __builtin_FUNCTION()) {
    std::string EName = Name;
    if (EName.empty())
      EName.append(Caller).append(" (auto generated)");

    return EName;
  }

  static std::string getGroup(const std::string &Group) {
    if (Group.empty())
      return "default";

    return Group;
  }

  std::string Name;
  std::string Group;
  ObserveState ExpectedState;
  std::unique_ptr<internal::InternalEvent> TheEvent;
};

struct AssertEventGroup {
  AssertEventGroup(uint64_t TargetRegion) : TargetRegion(TargetRegion) {}
  uint64_t TargetRegion;
};

bool operator==(const OmptAssertEvent &A, const OmptAssertEvent &B);

} // namespace omptest

#endif
