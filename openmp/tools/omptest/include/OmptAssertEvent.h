//===- OmptAssertEvent.h - Assertion event declarations ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Contains assertion event constructors, for generally all observable events.
/// This includes user-generated events, like synchronization.
///
//===----------------------------------------------------------------------===//

#ifndef OPENMP_TOOLS_OMPTEST_INCLUDE_OMPTASSERTEVENT_H
#define OPENMP_TOOLS_OMPTEST_INCLUDE_OMPTASSERTEVENT_H

#include "InternalEvent.h"
#include "omp-tools.h"

#include <cassert>
#include <limits>
#include <memory>
#include <string>

namespace omptest {

enum class ObserveState { Generated, Always, Never };

/// Helper function, returning an ObserveState string representation
const char *to_string(ObserveState State);

/// Assertion event struct, provides statically callable CTORs.
struct OmptAssertEvent {
  static OmptAssertEvent AssertionSyncPoint(const std::string &Name,
                                            const std::string &Group,
                                            const ObserveState &Expected,
                                            const std::string &SyncPointName);

  static OmptAssertEvent AssertionSuspend(const std::string &Name,
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

  static OmptAssertEvent ParallelEnd(
      const std::string &Name, const std::string &Group,
      const ObserveState &Expected,
      ompt_data_t *ParallelData = expectedDefault(ompt_data_t *),
      ompt_data_t *EncounteringTaskData = expectedDefault(ompt_data_t *),
      int Flags = expectedDefault(int),
      const void *CodeptrRA = expectedDefault(const void *));

  static OmptAssertEvent
  Work(const std::string &Name, const std::string &Group,
       const ObserveState &Expected, ompt_work_t WorkType,
       ompt_scope_endpoint_t Endpoint,
       ompt_data_t *ParallelData = expectedDefault(ompt_data_t *),
       ompt_data_t *TaskData = expectedDefault(ompt_data_t *),
       uint64_t Count = expectedDefault(uint64_t),
       const void *CodeptrRA = expectedDefault(const void *));

  static OmptAssertEvent
  Dispatch(const std::string &Name, const std::string &Group,
           const ObserveState &Expected,
           ompt_data_t *ParallelData = expectedDefault(ompt_data_t *),
           ompt_data_t *TaskData = expectedDefault(ompt_data_t *),
           ompt_dispatch_t Kind = expectedDefault(ompt_dispatch_t),
           ompt_data_t Instance = expectedDefault(ompt_data_t));

  static OmptAssertEvent
  TaskCreate(const std::string &Name, const std::string &Group,
             const ObserveState &Expected,
             ompt_data_t *EncounteringTaskData = expectedDefault(ompt_data_t *),
             const ompt_frame_t *EncounteringTaskFrame =
                 expectedDefault(ompt_frame_t *),
             ompt_data_t *NewTaskData = expectedDefault(ompt_data_t *),
             int Flags = expectedDefault(int),
             int HasDependences = expectedDefault(int),
             const void *CodeptrRA = expectedDefault(const void *));

  static OmptAssertEvent TaskSchedule(const std::string &Name,
                                      const std::string &Group,
                                      const ObserveState &Expected);

  static OmptAssertEvent
  ImplicitTask(const std::string &Name, const std::string &Group,
               const ObserveState &Expected, ompt_scope_endpoint_t Endpoint,
               ompt_data_t *ParallelData = expectedDefault(ompt_data_t *),
               ompt_data_t *TaskData = expectedDefault(ompt_data_t *),
               unsigned int ActualParallelism = expectedDefault(unsigned int),
               unsigned int Index = expectedDefault(unsigned int),
               int Flags = expectedDefault(int));

  static OmptAssertEvent
  SyncRegion(const std::string &Name, const std::string &Group,
             const ObserveState &Expected, ompt_sync_region_t Kind,
             ompt_scope_endpoint_t Endpoint,
             ompt_data_t *ParallelData = expectedDefault(ompt_data_t *),
             ompt_data_t *TaskData = expectedDefault(ompt_data_t *),
             const void *CodeptrRA = expectedDefault(const void *));

  static OmptAssertEvent
  Target(const std::string &Name, const std::string &Group,
         const ObserveState &Expected, ompt_target_t Kind,
         ompt_scope_endpoint_t Endpoint, int DeviceNum = expectedDefault(int),
         ompt_data_t *TaskData = expectedDefault(ompt_data_t *),
         ompt_id_t TargetId = expectedDefault(ompt_id_t),
         const void *CodeptrRA = expectedDefault(void *));

  static OmptAssertEvent
  TargetEmi(const std::string &Name, const std::string &Group,
            const ObserveState &Expected, ompt_target_t Kind,
            ompt_scope_endpoint_t Endpoint,
            int DeviceNum = expectedDefault(int),
            ompt_data_t *TaskData = expectedDefault(ompt_data_t *),
            ompt_data_t *TargetTaskData = expectedDefault(ompt_data_t *),
            ompt_data_t *TargetData = expectedDefault(ompt_data_t *),
            const void *CodeptrRA = expectedDefault(void *));

  static OmptAssertEvent
  TargetDataOp(const std::string &Name, const std::string &Group,
               const ObserveState &Expected, ompt_id_t TargetId,
               ompt_id_t HostOpId, ompt_target_data_op_t OpType, void *SrcAddr,
               int SrcDeviceNum, void *DstAddr, int DstDeviceNum, size_t Bytes,
               const void *CodeptrRA);

  static OmptAssertEvent
  TargetDataOp(const std::string &Name, const std::string &Group,
               const ObserveState &Expected, ompt_target_data_op_t OpType,
               size_t Bytes = expectedDefault(size_t),
               void *SrcAddr = expectedDefault(void *),
               void *DstAddr = expectedDefault(void *),
               int SrcDeviceNum = expectedDefault(int),
               int DstDeviceNum = expectedDefault(int),
               ompt_id_t TargetId = expectedDefault(ompt_id_t),
               ompt_id_t HostOpId = expectedDefault(ompt_id_t),
               const void *CodeptrRA = expectedDefault(void *));

  static OmptAssertEvent
  TargetDataOpEmi(const std::string &Name, const std::string &Group,
                  const ObserveState &Expected, ompt_scope_endpoint_t Endpoint,
                  ompt_data_t *TargetTaskData, ompt_data_t *TargetData,
                  ompt_id_t *HostOpId, ompt_target_data_op_t OpType,
                  void *SrcAddr, int SrcDeviceNum, void *DstAddr,
                  int DstDeviceNum, size_t Bytes, const void *CodeptrRA);

  static OmptAssertEvent
  TargetDataOpEmi(const std::string &Name, const std::string &Group,
                  const ObserveState &Expected, ompt_target_data_op_t OpType,
                  ompt_scope_endpoint_t Endpoint,
                  size_t Bytes = expectedDefault(size_t),
                  void *SrcAddr = expectedDefault(void *),
                  void *DstAddr = expectedDefault(void *),
                  int SrcDeviceNum = expectedDefault(int),
                  int DstDeviceNum = expectedDefault(int),
                  ompt_data_t *TargetTaskData = expectedDefault(ompt_data_t *),
                  ompt_data_t *TargetData = expectedDefault(ompt_data_t *),
                  ompt_id_t *HostOpId = expectedDefault(ompt_id_t *),
                  const void *CodeptrRA = expectedDefault(void *));

  static OmptAssertEvent TargetSubmit(const std::string &Name,
                                      const std::string &Group,
                                      const ObserveState &Expected,
                                      ompt_id_t TargetId, ompt_id_t HostOpId,
                                      unsigned int RequestedNumTeams);

  static OmptAssertEvent
  TargetSubmit(const std::string &Name, const std::string &Group,
               const ObserveState &Expected, unsigned int RequestedNumTeams,
               ompt_id_t TargetId = expectedDefault(ompt_id_t),
               ompt_id_t HostOpId = expectedDefault(ompt_id_t));

  static OmptAssertEvent
  TargetSubmitEmi(const std::string &Name, const std::string &Group,
                  const ObserveState &Expected, ompt_scope_endpoint_t Endpoint,
                  ompt_data_t *TargetData, ompt_id_t *HostOpId,
                  unsigned int RequestedNumTeams);

  static OmptAssertEvent
  TargetSubmitEmi(const std::string &Name, const std::string &Group,
                  const ObserveState &Expected, unsigned int RequestedNumTeams,
                  ompt_scope_endpoint_t Endpoint,
                  ompt_data_t *TargetData = expectedDefault(ompt_data_t *),
                  ompt_id_t *HostOpId = expectedDefault(ompt_id_t *));

  static OmptAssertEvent ControlTool(const std::string &Name,
                                     const std::string &Group,
                                     const ObserveState &Expected);

  static OmptAssertEvent DeviceInitialize(
      const std::string &Name, const std::string &Group,
      const ObserveState &Expected, int DeviceNum,
      const char *Type = expectedDefault(const char *),
      ompt_device_t *Device = expectedDefault(ompt_device_t *),
      ompt_function_lookup_t LookupFn = expectedDefault(ompt_function_lookup_t),
      const char *DocumentationStr = expectedDefault(const char *));

  static OmptAssertEvent DeviceFinalize(const std::string &Name,
                                        const std::string &Group,
                                        const ObserveState &Expected,
                                        int DeviceNum);

  static OmptAssertEvent
  DeviceLoad(const std::string &Name, const std::string &Group,
             const ObserveState &Expected, int DeviceNum,
             const char *Filename = expectedDefault(const char *),
             int64_t OffsetInFile = expectedDefault(int64_t),
             void *VmaInFile = expectedDefault(void *),
             size_t Bytes = expectedDefault(size_t),
             void *HostAddr = expectedDefault(void *),
             void *DeviceAddr = expectedDefault(void *),
             uint64_t ModuleId = expectedDefault(int64_t));

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

  /// Handle type = ompt_record_target_t
  static OmptAssertEvent
  BufferRecord(const std::string &Name, const std::string &Group,
               const ObserveState &Expected, ompt_callbacks_t Type,
               ompt_target_t Kind, ompt_scope_endpoint_t Endpoint,
               int DeviceNum = expectedDefault(int),
               ompt_id_t TaskId = expectedDefault(ompt_id_t),
               ompt_id_t TargetId = expectedDefault(ompt_id_t),
               const void *CodeptrRA = expectedDefault(void *));

  /// Handle type = ompt_callback_target_data_op
  static OmptAssertEvent
  BufferRecord(const std::string &Name, const std::string &Group,
               const ObserveState &Expected, ompt_callbacks_t Type,
               ompt_target_data_op_t OpType, size_t Bytes,
               std::pair<ompt_device_time_t, ompt_device_time_t> Timeframe,
               void *SrcAddr = expectedDefault(void *),
               void *DstAddr = expectedDefault(void *),
               int SrcDeviceNum = expectedDefault(int),
               int DstDeviceNum = expectedDefault(int),
               ompt_id_t TargetId = expectedDefault(ompt_id_t),
               ompt_id_t HostOpId = expectedDefault(ompt_id_t),
               const void *CodeptrRA = expectedDefault(void *));

  /// Handle type = ompt_callback_target_data_op
  static OmptAssertEvent BufferRecord(
      const std::string &Name, const std::string &Group,
      const ObserveState &Expected, ompt_callbacks_t Type,
      ompt_target_data_op_t OpType, size_t Bytes = expectedDefault(size_t),
      ompt_device_time_t MinimumTimeDelta = expectedDefault(ompt_device_time_t),
      void *SrcAddr = expectedDefault(void *),
      void *DstAddr = expectedDefault(void *),
      int SrcDeviceNum = expectedDefault(int),
      int DstDeviceNum = expectedDefault(int),
      ompt_id_t TargetId = expectedDefault(ompt_id_t),
      ompt_id_t HostOpId = expectedDefault(ompt_id_t),
      const void *CodeptrRA = expectedDefault(void *));

  /// Handle type = ompt_callback_target_submit
  static OmptAssertEvent
  BufferRecord(const std::string &Name, const std::string &Group,
               const ObserveState &Expected, ompt_callbacks_t Type,
               std::pair<ompt_device_time_t, ompt_device_time_t> Timeframe,
               unsigned int RequestedNumTeams = expectedDefault(unsigned int),
               unsigned int GrantedNumTeams = expectedDefault(unsigned int),
               ompt_id_t TargetId = expectedDefault(ompt_id_t),
               ompt_id_t HostOpId = expectedDefault(ompt_id_t));

  /// Handle type = ompt_callback_target_submit
  /// Note: This will also act as the simplest default CTOR
  static OmptAssertEvent BufferRecord(
      const std::string &Name, const std::string &Group,
      const ObserveState &Expected, ompt_callbacks_t Type,
      ompt_device_time_t MinimumTimeDelta = expectedDefault(ompt_device_time_t),
      unsigned int RequestedNumTeams = expectedDefault(unsigned int),
      unsigned int GrantedNumTeams = expectedDefault(unsigned int),
      ompt_id_t TargetId = expectedDefault(ompt_id_t),
      ompt_id_t HostOpId = expectedDefault(ompt_id_t));

  static OmptAssertEvent BufferRecordDeallocation(const std::string &Name,
                                                  const std::string &Group,
                                                  const ObserveState &Expected,
                                                  ompt_buffer_t *Buffer);

  /// Allow move construction (due to std::unique_ptr)
  OmptAssertEvent(OmptAssertEvent &&o) = default;
  OmptAssertEvent &operator=(OmptAssertEvent &&o) = default;

  /// Get the event's name
  std::string getEventName() const;

  /// Get the event's group name
  std::string getEventGroup() const;

  /// Get the event's expected observation state
  ObserveState getEventExpectedState() const;

  /// Return the actual event type enum value
  internal::EventTy getEventType() const;

  /// Get a pointer to the internal event
  internal::InternalEvent *getEvent() const;

  /// Make events comparable
  friend bool operator==(const OmptAssertEvent &A, const OmptAssertEvent &B);

  /// Returns the string representation of the event
  std::string toString(bool PrefixEventName = false) const;

private:
  OmptAssertEvent(const std::string &Name, const std::string &Group,
                  const ObserveState &Expected, internal::InternalEvent *IE);
  OmptAssertEvent(const OmptAssertEvent &o) = delete;

  /// Determine the event name. Either it is provided directly or determined
  /// from the calling function's name.
  static std::string getName(const std::string &Name,
                             const char *Caller = __builtin_FUNCTION()) {
    std::string EName = Name;
    if (EName.empty())
      EName.append(Caller).append(" (auto generated)");

    return EName;
  }

  /// Determine the event name. Either it is provided directly or "default".
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

/// POD type, which holds the target region id, corresponding to an event group.
struct AssertEventGroup {
  AssertEventGroup(uint64_t TargetRegion) : TargetRegion(TargetRegion) {}
  uint64_t TargetRegion;
};

bool operator==(const OmptAssertEvent &A, const OmptAssertEvent &B);

} // namespace omptest

#endif
