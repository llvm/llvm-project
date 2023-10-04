#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTCALLBACKHANDLER_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTCALLBACKHANDLER_H

#include "OmptAssertEvent.h"
#include "OmptAsserter.h"

#include "omp-tools.h"

#include <utility>
#include <vector>

/// Handler class to do whatever is needed to be done when a callback is invoked
/// by the OMP runtime
/// Supports a RecordAndReplay mechanism in which all OMPT events are recorded
/// and then replayed. This is so that a test can assert on, e.g., a device
/// initialize event, even though this would occur before a unit test is
/// actually executed.
class OmptCallbackHandler {
public:
  /// Singleton handler
  static OmptCallbackHandler &get() {
    static OmptCallbackHandler Handler;
    return Handler;
  }

  /// Subscribe a listener to be notified for OMPT events
  void subscribe(OmptListener *Listener) { Subscribers.push_back(Listener); }

  /// Remove all subscribers
  void clearSubscribers() {
    replay();

    Subscribers.clear();
  }

  /// When the record and replay mechanism is enabled this replays all OMPT
  /// events
  void replay() {
    if (!RecordAndReplay)
      return;

    for (auto &E : RecordedEvents)
      for (const auto &S : Subscribers)
        S->notify(std::move(E));
  }

  void handleThreadBegin(ompt_thread_t ThreadType, ompt_data_t *ThreadData) {
    if (RecordAndReplay) {
      recordEvent(
          omptest::OmptAssertEvent::ThreadBegin(ThreadType, "Thread Begin"));
      return;
    }

    // Initial thread event likely to preceed assertion registration, so skip
    if (ThreadType == ompt_thread_initial)
      return;
    for (const auto &S : Subscribers)
      S->notify(
          omptest::OmptAssertEvent::ThreadBegin(ThreadType, "Thread Begin"));
  }

  void handleThreadEnd(ompt_data_t *ThreadData) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::ThreadEnd("Thread End"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::ThreadEnd("Thread End"));
  }

  void handleTaskCreate(ompt_data_t *EncounteringTaskData,
                        const ompt_frame_t *EncounteringTaskFrame,
                        ompt_data_t *NewTaskData, int Flags, int HasDependences,
                        const void *CodeptrRA) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::TaskCreate("Task Create"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::TaskCreate("Task Create"));
  }

  void handleTaskSchedule(ompt_data_t *PriorTaskData,
                          ompt_task_status_t PriorTaskStatus,
                          ompt_data_t *NextTaskData) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::TaskSchedule("Task Schedule"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::TaskSchedule("Task Schedule"));
  }

  void handleImplicitTask(ompt_scope_endpoint_t Endpoint,
                          ompt_data_t *ParallelData, ompt_data_t *TaskData,
                          unsigned int ActualParallelism, unsigned int Index,
                          int Flags) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::ImplicitTask("Implicit Task"));
      return;
    }

    return; // FIXME Is called for implicit task by main thread before test case
            // inserts asserts.
    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::ImplicitTask("Implicit Task"));
  }

  void handleParallelBegin(ompt_data_t *EncounteringTaskData,
                           const ompt_frame_t *EncounteringTaskFrame,
                           ompt_data_t *ParallelData,
                           unsigned int RequestedParallelism, int Flags,
                           const void *CodeptrRA) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::ParallelBegin(RequestedParallelism,
                                                          "Parallel Begin"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::ParallelBegin(RequestedParallelism,
                                                        "Parallel Begin"));
  }

  void handleParallelEnd(ompt_data_t *ParallelData,
                         ompt_data_t *EncounteringTaskData, int Flags,
                         const void *CodeptrRA) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::ParallelEnd("Parallel End"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::ParallelEnd("Parallel End"));
  }

  void handleDeviceInitialize(int DeviceNum, const char *Type,
                              ompt_device_t *Device,
                              ompt_function_lookup_t LookupFn,
                              const char *DocumentationStr) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::DeviceInitialize(
          DeviceNum, Type, Device, LookupFn, DocumentationStr,
          "Device Inititalize"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::DeviceInitialize(
          DeviceNum, Type, Device, LookupFn, DocumentationStr,
          "Device Initialize"));
  }

  void handleDeviceFinalize(int DeviceNum) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::DeviceFinalize(DeviceNum,
                                                           "Device Finalize"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::DeviceFinalize(DeviceNum,
                                                         "Device Finalize"));
  }

  void handleTarget(ompt_target_t Kind, ompt_scope_endpoint_t Endpoint,
                    int DeviceNum, ompt_data_t *TaskData, ompt_id_t TargetId,
                    const void *CodeptrRA) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::Target(
          Kind, Endpoint, DeviceNum, TaskData, TargetId, CodeptrRA, "Target"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::Target(
          Kind, Endpoint, DeviceNum, TaskData, TargetId, CodeptrRA, "Target"));
  }

  void handleTargetEmi(ompt_target_t Kind, ompt_scope_endpoint_t Endpoint,
                       int DeviceNum, ompt_data_t *TaskData,
                       ompt_data_t *TargetTaskData, ompt_data_t *TargetData,
                       const void *CodeptrRA) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::TargetEmi(
          Kind, Endpoint, DeviceNum, TaskData, TargetTaskData, TargetData,
          CodeptrRA, "Target EMI"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::TargetEmi(
          Kind, Endpoint, DeviceNum, TaskData, TargetTaskData, TargetData,
          CodeptrRA, "Target EMI"));
  }

  void handleTargetSubmit(ompt_id_t TargetId, ompt_id_t HostOpId,
                          unsigned int RequestedNumTeams) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::TargetSubmit(
          TargetId, HostOpId, RequestedNumTeams, "Target Submit"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::TargetSubmit(
          TargetId, HostOpId, RequestedNumTeams, "Target Submit"));
  }

  void handleTargetSubmitEmi(ompt_scope_endpoint_t Endpoint,
                             ompt_data_t *TargetData, ompt_id_t *HostOpId,
                             unsigned int RequestedNumTeams) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::TargetSubmitEmi(
          Endpoint, TargetData, HostOpId, RequestedNumTeams,
          "Target Submit EMI"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::TargetSubmitEmi(
          Endpoint, TargetData, HostOpId, RequestedNumTeams,
          "Target Submit EMI"));
  }

  void handleTargetDataOp(ompt_id_t TargetId, ompt_id_t HostOpId,
                          ompt_target_data_op_t OpType, void *SrcAddr,
                          int SrcDeviceNum, void *DstAddr, int DstDeviceNum,
                          size_t Bytes, const void *CodeptrRA) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::TargetDataOp(
          TargetId, HostOpId, OpType, SrcAddr, SrcDeviceNum, DstAddr,
          DstDeviceNum, Bytes, CodeptrRA, "Target Data Op"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::TargetDataOp(
          TargetId, HostOpId, OpType, SrcAddr, SrcDeviceNum, DstAddr,
          DstDeviceNum, Bytes, CodeptrRA, "Target Data Op"));
  }

  void handleTargetDataOpEmi(ompt_scope_endpoint_t Endpoint,
                             ompt_data_t *TargetTaskData,
                             ompt_data_t *TargetData, ompt_id_t *HostOpId,
                             ompt_target_data_op_t OpType, void *SrcAddr,
                             int SrcDeviceNum, void *DstAddr, int DstDeviceNum,
                             size_t Bytes, const void *CodeptrRA) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::TargetDataOpEmi(
          Endpoint, TargetTaskData, TargetData, HostOpId, OpType, SrcAddr,
          SrcDeviceNum, DstAddr, DstDeviceNum, Bytes, CodeptrRA,
          "Target Data Op EMI"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::TargetDataOpEmi(
          Endpoint, TargetTaskData, TargetData, HostOpId, OpType, SrcAddr,
          SrcDeviceNum, DstAddr, DstDeviceNum, Bytes, CodeptrRA,
          "Target Data Op EMI"));
  }

  void handleDeviceLoad(int DeviceNum, const char *Filename,
                        int64_t OffsetInFile, void *VmaInFile, size_t Bytes,
                        void *HostAddr, void *DeviceAddr, uint64_t ModuleId) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::DeviceLoad(
          DeviceNum, Filename, OffsetInFile, VmaInFile, Bytes, HostAddr,
          DeviceAddr, ModuleId, "Device Load"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::DeviceLoad(
          DeviceNum, Filename, OffsetInFile, VmaInFile, Bytes, HostAddr,
          DeviceAddr, ModuleId, "Device Load"));
  }

  void handleDeviceUnload(int DeviceNum, uint64_t ModuleId) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::DeviceUnload("Device Unload"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::DeviceUnload("Device Unload"));
  }

  void handleBufferRequest(int DeviceNum, ompt_buffer_t **Buffer,
                           size_t *Bytes) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::BufferRequest(
          DeviceNum, Buffer, Bytes, "Buffer Request"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::BufferRequest(
          DeviceNum, Buffer, Bytes, "Buffer Request"));
  }

  void handleBufferComplete(int DeviceNum, ompt_buffer_t *Buffer, size_t Bytes,
                            ompt_buffer_cursor_t Begin, int BufferOwned) {
    if (RecordAndReplay) {
      recordEvent(omptest::OmptAssertEvent::BufferComplete(
          DeviceNum, Buffer, Bytes, Begin, BufferOwned, "Buffer Complete"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::BufferComplete(
          DeviceNum, Buffer, Bytes, Begin, BufferOwned, "Buffer Complete"));
  }

  void handleBufferRecord(ompt_record_ompt_t *Record) {
    if (RecordAndReplay) {
      recordEvent(
          omptest::OmptAssertEvent::BufferRecord(Record, "Buffer Record"));
      return;
    }

    for (const auto &S : Subscribers)
      S->notify(
          omptest::OmptAssertEvent::BufferRecord(Record, "Buffer Record"));
  }

  /// Not needed for a conforming minimal OMPT implementation
  void handleWorkBegin(ompt_work_t work_type, ompt_scope_endpoint_t endpoint,
                       ompt_data_t *parallel_data, ompt_data_t *task_data,
                       uint64_t count, const void *codeptr_ra) {}

  void handleWorkEnd(ompt_work_t work_type, ompt_scope_endpoint_t endpoint,
                     ompt_data_t *parallel_data, ompt_data_t *task_data,
                     uint64_t count, const void *codeptr_ra) {}

private:
  /// Wrapper around emplace_back for potential additional logging / checking or
  /// so
  void recordEvent(omptest::OmptAssertEvent &&Event) {
    RecordedEvents.emplace_back(std::forward<omptest::OmptAssertEvent>(Event));
  }

  /// Listeners to be notified
  std::vector<OmptListener *> Subscribers;

  /// Toggle if OMPT events should notify subscribers immediately or not
  bool RecordAndReplay{false};

  /// Recorded events in Record and Replay mode
  std::vector<omptest::OmptAssertEvent> RecordedEvents;
};
#endif
