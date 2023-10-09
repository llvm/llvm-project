#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTEVENT_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTEVENT_H

#include "InternalEvent.h"
#include "omp-tools.h"

#include <cassert>
#include <memory>
#include <string>

namespace omptest{

enum class AssertState { pass, fail };

struct OmptAssertEvent {
  static OmptAssertEvent ThreadBegin(ompt_thread_t ThreadType,
                                     const std::string &Name);

  static OmptAssertEvent ThreadEnd(const std::string &Name);

  static OmptAssertEvent ParallelBegin(int NumThreads, const std::string &Name);

  static OmptAssertEvent ParallelEnd(const std::string &Name);

  static OmptAssertEvent TaskCreate(const std::string &Name);

  static OmptAssertEvent TaskSchedule(const std::string &Name);

  static OmptAssertEvent ImplicitTask(const std::string &Name);

  static OmptAssertEvent Target(ompt_target_t Kind,
                                ompt_scope_endpoint_t Endpoint, int DeviceNum,
                                ompt_data_t *TaskData, ompt_id_t TargetId,
                                const void *CodeptrRA, const std::string &Name);

  static OmptAssertEvent
  TargetEmi(ompt_target_t Kind, ompt_scope_endpoint_t Endpoint, int DeviceNum,
            ompt_data_t *TaskData, ompt_data_t *TargetTaskData,
            ompt_data_t *TargetData, const void *CodeptrRA,
            const std::string &Name);

  /// Create a DataAlloc Event
  static OmptAssertEvent TargetDataOp(ompt_id_t TargetId, ompt_id_t HostOpId,
                                      ompt_target_data_op_t OpType,
                                      void *SrcAddr, int SrcDeviceNum,
                                      void *DstAddr, int DstDeviceNum,
                                      size_t Bytes, const void *CodeptrRA,
                                      const std::string &Name);

  static OmptAssertEvent
  TargetDataOpEmi(ompt_scope_endpoint_t Endpoint, ompt_data_t *TargetTaskData,
                  ompt_data_t *TargetData, ompt_id_t *HostOpId,
                  ompt_target_data_op_t OpType, void *SrcAddr, int SrcDeviceNum,
                  void *DstAddr, int DstDeviceNum, size_t Bytes,
                  const void *CodeptrRA, const std::string &Name);

  static OmptAssertEvent TargetSubmit(ompt_id_t TargetId, ompt_id_t HostOpId,
                                      unsigned int RequestedNumTeams,
                                      const std::string &Name);

  static OmptAssertEvent TargetSubmitEmi(ompt_scope_endpoint_t Endpoint,
                                         ompt_data_t *TargetData,
                                         ompt_id_t *HostOpId,
                                         unsigned int RequestedNumTeams,
                                         const std::string &Name);

  static OmptAssertEvent ControlTool(std::string &Name);

  static OmptAssertEvent DeviceInitialize(int DeviceNum, const char *Type,
                                          ompt_device_t *Device,
                                          ompt_function_lookup_t LookupFn,
                                          const char *DocumentationStr,
                                          const std::string &Name);

  static OmptAssertEvent DeviceFinalize(int DeviceNum, const std::string &Name);

  static OmptAssertEvent DeviceLoad(int DeviceNum, const char *Filename,
                                    int64_t OffsetInFile, void *VmaInFile,
                                    size_t Bytes, void *HostAddr,
                                    void *DeviceAddr, uint64_t ModuleId,
                                    const std::string &Name);

  static OmptAssertEvent DeviceUnload(const std::string &Name);

  static OmptAssertEvent BufferRequest(int DeviceNum, ompt_buffer_t **Buffer,
                                       size_t *Bytes, const std::string &Name);

  static OmptAssertEvent BufferComplete(int DeviceNum, ompt_buffer_t *Buffer,
                                        size_t Bytes,
                                        ompt_buffer_cursor_t Begin,
                                        int BufferOwned,
                                        const std::string &Name);

  static OmptAssertEvent BufferRecord(ompt_record_ompt_t *Record,
                                      const std::string &Name);

  /// Allow move construction (due to std::unique_ptr)
  OmptAssertEvent(OmptAssertEvent &&o) = default;
  OmptAssertEvent &operator=(OmptAssertEvent &&o) = default;

  std::string getEventName() const;

  internal::EventTy getEventType() const;

  /// Make events comparable
  friend bool operator==(const OmptAssertEvent &A, const OmptAssertEvent &B);

  std::string toString(bool PrefixEventName = false) const;

private:
  OmptAssertEvent(const std::string &Name, internal::InternalEvent *IE);
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

bool operator==(const OmptAssertEvent &A, const OmptAssertEvent &B);

} // namespace omptest

#endif
