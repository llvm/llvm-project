#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTCALLBACKHANDLER_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTCALLBACKHANDLER_H

#include "OmptAssertEvent.h"
#include "OmptAsserter.h"

#include "omp-tools.h"

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
  static OmptCallbackHandler &get();

  /// Subscribe a listener to be notified for OMPT events
  void subscribe(OmptListener *Listener);

  /// Remove all subscribers
  void clearSubscribers();

  /// When the record and replay mechanism is enabled this replays all OMPT
  /// events
  void replay();

  void handleThreadBegin(ompt_thread_t ThreadType, ompt_data_t *ThreadData);

  void handleThreadEnd(ompt_data_t *ThreadData);

  void handleTaskCreate(ompt_data_t *EncounteringTaskData,
                        const ompt_frame_t *EncounteringTaskFrame,
                        ompt_data_t *NewTaskData, int Flags, int HasDependences,
                        const void *CodeptrRA);

  void handleTaskSchedule(ompt_data_t *PriorTaskData,
                          ompt_task_status_t PriorTaskStatus,
                          ompt_data_t *NextTaskData);

  void handleImplicitTask(ompt_scope_endpoint_t Endpoint,
                          ompt_data_t *ParallelData, ompt_data_t *TaskData,
                          unsigned int ActualParallelism, unsigned int Index,
                          int Flags);

  void handleParallelBegin(ompt_data_t *EncounteringTaskData,
                           const ompt_frame_t *EncounteringTaskFrame,
                           ompt_data_t *ParallelData,
                           unsigned int RequestedParallelism, int Flags,
                           const void *CodeptrRA);

  void handleParallelEnd(ompt_data_t *ParallelData,
                         ompt_data_t *EncounteringTaskData, int Flags,
                         const void *CodeptrRA);

  void handleDeviceInitialize(int DeviceNum, const char *Type,
                              ompt_device_t *Device,
                              ompt_function_lookup_t LookupFn,
                              const char *DocumentationStr);

  void handleDeviceFinalize(int DeviceNum);

  void handleTarget(ompt_target_t Kind, ompt_scope_endpoint_t Endpoint,
                    int DeviceNum, ompt_data_t *TaskData, ompt_id_t TargetId,
                    const void *CodeptrRA);

  void handleTargetEmi(ompt_target_t Kind, ompt_scope_endpoint_t Endpoint,
                       int DeviceNum, ompt_data_t *TaskData,
                       ompt_data_t *TargetTaskData, ompt_data_t *TargetData,
                       const void *CodeptrRA);

  void handleTargetSubmit(ompt_id_t TargetId, ompt_id_t HostOpId,
                          unsigned int RequestedNumTeams);

  void handleTargetSubmitEmi(ompt_scope_endpoint_t Endpoint,
                             ompt_data_t *TargetData, ompt_id_t *HostOpId,
                             unsigned int RequestedNumTeams);

  void handleTargetDataOp(ompt_id_t TargetId, ompt_id_t HostOpId,
                          ompt_target_data_op_t OpType, void *SrcAddr,
                          int SrcDeviceNum, void *DstAddr, int DstDeviceNum,
                          size_t Bytes, const void *CodeptrRA);

  void handleTargetDataOpEmi(ompt_scope_endpoint_t Endpoint,
                             ompt_data_t *TargetTaskData,
                             ompt_data_t *TargetData, ompt_id_t *HostOpId,
                             ompt_target_data_op_t OpType, void *SrcAddr,
                             int SrcDeviceNum, void *DstAddr, int DstDeviceNum,
                             size_t Bytes, const void *CodeptrRA);

  void handleDeviceLoad(int DeviceNum, const char *Filename,
                        int64_t OffsetInFile, void *VmaInFile, size_t Bytes,
                        void *HostAddr, void *DeviceAddr, uint64_t ModuleId);

  void handleDeviceUnload(int DeviceNum, uint64_t ModuleId);

  void handleBufferRequest(int DeviceNum, ompt_buffer_t **Buffer,
                           size_t *Bytes);

  void handleBufferComplete(int DeviceNum, ompt_buffer_t *Buffer, size_t Bytes,
                            ompt_buffer_cursor_t Begin, int BufferOwned);

  void handleBufferRecord(ompt_record_ompt_t *Record);

  /// Not needed for a conforming minimal OMPT implementation
  void handleWorkBegin(ompt_work_t work_type, ompt_scope_endpoint_t endpoint,
                       ompt_data_t *parallel_data, ompt_data_t *task_data,
                       uint64_t count, const void *codeptr_ra);

  void handleWorkEnd(ompt_work_t work_type, ompt_scope_endpoint_t endpoint,
                     ompt_data_t *parallel_data, ompt_data_t *task_data,
                     uint64_t count, const void *codeptr_ra);

private:
  /// Wrapper around emplace_back for potential additional logging / checking or
  /// so
  void recordEvent(omptest::OmptAssertEvent &&Event);

  /// Listeners to be notified
  std::vector<OmptListener *> Subscribers;

  /// Toggle if OMPT events should notify subscribers immediately or not
  bool RecordAndReplay{false};

  /// Recorded events in Record and Replay mode
  std::vector<omptest::OmptAssertEvent> RecordedEvents;
};
#endif
