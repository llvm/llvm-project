#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTCALLBACKHANDLER_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTCALLBACKHANDLER_H

#include "OmptAssertEvent.h"
#include "OmptAsserter.h"

#include "omp-tools.h"

#include <vector>

/// Handler class to do whatever is needed to be done when a callback is invoked
/// by the OMP runtime
struct OmptCallbackHandler {

  static OmptCallbackHandler &get() {
    static OmptCallbackHandler Handler;
    return Handler;
  }

  void subscribe(OmptAsserter *Asserter) { Subscribers.push_back(Asserter); }

  void clearSubscribers() { Subscribers.clear(); }

  void handleThreadBegin(ompt_thread_t ThreadType, ompt_data_t *ThreadData) {
    // Initial thread event likely to preceed assertion registration, so skip
    if (ThreadType == ompt_thread_initial)
      return;
    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::ThreadBegin("Thread Begin"));
  }

  void handleThreadEnd(ompt_data_t *ThreadData) {
    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::ThreadEnd("Thread End"));
  }

  void handleTaskCreate(ompt_data_t *EncounteringTaskData,
                        const ompt_frame_t *EncounteringTaskFrame,
                        ompt_data_t *NewTaskData, int Flags, int HasDependences,
                        const void *CodeptrRA) {
    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::TaskCreate("Task Create"));
  }

  void handleTaskSchedule(ompt_data_t *PriorTaskData,
                          ompt_task_status_t PriorTaskStatus,
                          ompt_data_t *NextTaskData) {
    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::TaskSchedule("Task Schedule"));
  }

  void handleImplicitTask(ompt_scope_endpoint_t Endpoint,
                          ompt_data_t *ParallelData, ompt_data_t *TaskData,
                          unsigned int ActualParallelism, unsigned int Index,
                          int Flags) {
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
    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::ParallelBegin(RequestedParallelism,
                                                        "Parallel Begin"));
  }

  void handleParallelEnd(ompt_data_t *ParallelData,
                         ompt_data_t *EncounteringTaskData, int Flags,
                         const void *CodeptrRA) {
    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::ParallelEnd("Parallel End"));
  }

  void handleDeviceInitialize(int DeviceNum, const char *Type,
                              ompt_device_t *Device,
                              ompt_function_lookup_t LookupFn,
                              const char *DocumentationStr) {
    std::cout << "Got Device Init Event. Ignoring" << std::endl;
    return;

    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::DeviceInitialize("Device Init"));
  }

  void handleDeviceFinalize(int DeviceNum) {
    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::DeviceFinalize("Device Finalize"));
  }

  void handleTarget(ompt_target_t Kind, ompt_scope_endpoint_t Endpoint,
                    int DeviceNum, ompt_data_t *TaskData, ompt_id_t TargetId,
                    const void *CodeptrRA) {
    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::Target("Target"));
  }

  void handleTargetSubmit(ompt_id_t TargetId, ompt_id_t HostOpId,
                          unsigned int RequestedNumTeams) {
    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::TargetSubmit("Target Submit"));
  }

  void handleTargetDataOp(ompt_id_t TargetId, ompt_id_t HostOpId,
                          ompt_target_data_op_t OpType, void *SrcAddr,
                          int SrcDeviceNum, void *DestAddr, int DestDeviceNum,
                          size_t Bytes, const void *CodeptrRA) {
    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::TargetDataOp("Data Target Op"));
  }

  void handleDeviceLoad(int DeviceNum, const char *Filename,
                        int64_t OffsetInFile, void *VmaInFile, size_t Bytes,
                        void *HostAddr, void *DeviceAddr, uint64_t ModuleId) {
    return; // FIXME: This would need to handled the same as DeviceInit
    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::DeviceLoad("Device Load"));
  }

  void handleDeviceUnload(int DeviceNum, uint64_t ModuleId) {
    for (const auto &S : Subscribers)
      S->notify(omptest::OmptAssertEvent::DeviceUnload("Device Unload"));
  }

  /// Not needed for a conforming minimal OMPT implementation
  void handleWorkBegin(ompt_work_t work_type, ompt_scope_endpoint_t endpoint,
                       ompt_data_t *parallel_data, ompt_data_t *task_data,
                       uint64_t count, const void *codeptr_ra) {}

  void handleWorkEnd(ompt_work_t work_type, ompt_scope_endpoint_t endpoint,
                     ompt_data_t *parallel_data, ompt_data_t *task_data,
                     uint64_t count, const void *codeptr_ra) {}

  std::vector<OmptAsserter *> Subscribers;
};
#endif
