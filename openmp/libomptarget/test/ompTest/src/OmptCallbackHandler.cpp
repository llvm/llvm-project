#include "../include/OmptCallbackHandler.h"

using namespace omptest;

OmptCallbackHandler &OmptCallbackHandler::get() {
  static OmptCallbackHandler Handler;
  return Handler;
}

void OmptCallbackHandler::subscribe(OmptListener *Listener) {
  Subscribers.push_back(Listener);
}

void OmptCallbackHandler::clearSubscribers() {
  replay();

  Subscribers.clear();
}

void OmptCallbackHandler::replay() {
  if (!RecordAndReplay)
    return;

  for (auto &E : RecordedEvents)
    for (const auto &S : Subscribers)
      S->notify(std::move(E));
}

void OmptCallbackHandler::handleThreadBegin(ompt_thread_t ThreadType,
                                            ompt_data_t *ThreadData) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::ThreadBegin(ThreadType, "Thread Begin"));
    return;
  }

  // Initial thread event likely to preceed assertion registration, so skip
  if (ThreadType == ompt_thread_initial)
    return;
  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::ThreadBegin(ThreadType, "Thread Begin"));
}

void OmptCallbackHandler::handleThreadEnd(ompt_data_t *ThreadData) {
  // FixMe: Return early, because of an ordering issue which causes SEGFAULTS.
  // Currently, the libomtest library will be evicted from memory before all
  // OMPT events occurred. This will cause calls to handleX() on a dangling
  // OmptCallbackHandler object.
  // Observed ordering looks like this:
  // __attribute__((destructor)) -> DTOR OmptCallbackHandler ->
  // handleThreadEnd
  return;

  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::ThreadEnd("Thread End"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::ThreadEnd("Thread End"));
}

void OmptCallbackHandler::handleTaskCreate(
    ompt_data_t *EncounteringTaskData,
    const ompt_frame_t *EncounteringTaskFrame, ompt_data_t *NewTaskData,
    int Flags, int HasDependences, const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TaskCreate("Task Create"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TaskCreate("Task Create"));
}

void OmptCallbackHandler::handleTaskSchedule(ompt_data_t *PriorTaskData,
                                             ompt_task_status_t PriorTaskStatus,
                                             ompt_data_t *NextTaskData) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TaskSchedule("Task Schedule"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TaskSchedule("Task Schedule"));
}

void OmptCallbackHandler::handleImplicitTask(ompt_scope_endpoint_t Endpoint,
                                             ompt_data_t *ParallelData,
                                             ompt_data_t *TaskData,
                                             unsigned int ActualParallelism,
                                             unsigned int Index, int Flags) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::ImplicitTask("Implicit Task"));
    return;
  }

  return; // FIXME Is called for implicit task by main thread before test case
          // inserts asserts.
  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::ImplicitTask("Implicit Task"));
}

void OmptCallbackHandler::handleParallelBegin(
    ompt_data_t *EncounteringTaskData,
    const ompt_frame_t *EncounteringTaskFrame, ompt_data_t *ParallelData,
    unsigned int RequestedParallelism, int Flags, const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(
        OmptAssertEvent::ParallelBegin(RequestedParallelism, "Parallel Begin"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(
        OmptAssertEvent::ParallelBegin(RequestedParallelism, "Parallel Begin"));
}

void OmptCallbackHandler::handleParallelEnd(ompt_data_t *ParallelData,
                                            ompt_data_t *EncounteringTaskData,
                                            int Flags, const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::ParallelEnd("Parallel End"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::ParallelEnd("Parallel End"));
}

void OmptCallbackHandler::handleDeviceInitialize(
    int DeviceNum, const char *Type, ompt_device_t *Device,
    ompt_function_lookup_t LookupFn, const char *DocumentationStr) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::DeviceInitialize(DeviceNum, Type, Device,
                                                  LookupFn, DocumentationStr,
                                                  "Device Inititalize"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::DeviceInitialize(DeviceNum, Type, Device,
                                                LookupFn, DocumentationStr,
                                                "Device Initialize"));
}

void OmptCallbackHandler::handleDeviceFinalize(int DeviceNum) {
  // FixMe: Return early, because of an ordering issue which causes SEGFAULTS.
  // Currently, the libomtest library will be evicted from memory before all
  // OMPT events occurred. This will cause calls to handleX() on a dangling
  // OmptCallbackHandler object. Note that the handler's DTOR is called even
  // before ompt_finalize. Device finalization will happen even after that.
  // Observed ordering looks like this:
  // __attribute__((destructor)) -> DTOR OmptCallbackHandler ->
  // handleThreadEnd -> ompt_finalize -> handleDeviceFinalize
  return;
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::DeviceFinalize(DeviceNum, "Device Finalize"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::DeviceFinalize(DeviceNum, "Device Finalize"));
}

void OmptCallbackHandler::handleTarget(ompt_target_t Kind,
                                       ompt_scope_endpoint_t Endpoint,
                                       int DeviceNum, ompt_data_t *TaskData,
                                       ompt_id_t TargetId,
                                       const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::Target(Kind, Endpoint, DeviceNum, TaskData,
                                        TargetId, CodeptrRA, "Target"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::Target(Kind, Endpoint, DeviceNum, TaskData,
                                      TargetId, CodeptrRA, "Target"));
}

void OmptCallbackHandler::handleTargetEmi(ompt_target_t Kind,
                                          ompt_scope_endpoint_t Endpoint,
                                          int DeviceNum, ompt_data_t *TaskData,
                                          ompt_data_t *TargetTaskData,
                                          ompt_data_t *TargetData,
                                          const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TargetEmi(Kind, Endpoint, DeviceNum, TaskData,
                                           TargetTaskData, TargetData,
                                           CodeptrRA, "Target EMI"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TargetEmi(Kind, Endpoint, DeviceNum, TaskData,
                                         TargetTaskData, TargetData, CodeptrRA,
                                         "Target EMI"));
}

void OmptCallbackHandler::handleTargetSubmit(ompt_id_t TargetId,
                                             ompt_id_t HostOpId,
                                             unsigned int RequestedNumTeams) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TargetSubmit(
        TargetId, HostOpId, RequestedNumTeams, "Target Submit"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TargetSubmit(
        TargetId, HostOpId, RequestedNumTeams, "Target Submit"));
}

void OmptCallbackHandler::handleTargetSubmitEmi(
    ompt_scope_endpoint_t Endpoint, ompt_data_t *TargetData,
    ompt_id_t *HostOpId, unsigned int RequestedNumTeams) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TargetSubmitEmi(Endpoint, TargetData, HostOpId,
                                                 RequestedNumTeams,
                                                 "Target Submit EMI"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TargetSubmitEmi(Endpoint, TargetData, HostOpId,
                                               RequestedNumTeams,
                                               "Target Submit EMI"));
}

void OmptCallbackHandler::handleTargetDataOp(
    ompt_id_t TargetId, ompt_id_t HostOpId, ompt_target_data_op_t OpType,
    void *SrcAddr, int SrcDeviceNum, void *DstAddr, int DstDeviceNum,
    size_t Bytes, const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TargetDataOp(
        TargetId, HostOpId, OpType, SrcAddr, SrcDeviceNum, DstAddr,
        DstDeviceNum, Bytes, CodeptrRA, "Target Data Op"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TargetDataOp(
        TargetId, HostOpId, OpType, SrcAddr, SrcDeviceNum, DstAddr,
        DstDeviceNum, Bytes, CodeptrRA, "Target Data Op"));
}

void OmptCallbackHandler::handleTargetDataOpEmi(
    ompt_scope_endpoint_t Endpoint, ompt_data_t *TargetTaskData,
    ompt_data_t *TargetData, ompt_id_t *HostOpId, ompt_target_data_op_t OpType,
    void *SrcAddr, int SrcDeviceNum, void *DstAddr, int DstDeviceNum,
    size_t Bytes, const void *CodeptrRA) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::TargetDataOpEmi(
        Endpoint, TargetTaskData, TargetData, HostOpId, OpType, SrcAddr,
        SrcDeviceNum, DstAddr, DstDeviceNum, Bytes, CodeptrRA,
        "Target Data Op EMI"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::TargetDataOpEmi(
        Endpoint, TargetTaskData, TargetData, HostOpId, OpType, SrcAddr,
        SrcDeviceNum, DstAddr, DstDeviceNum, Bytes, CodeptrRA,
        "Target Data Op EMI"));
}

void OmptCallbackHandler::handleDeviceLoad(int DeviceNum, const char *Filename,
                                           int64_t OffsetInFile,
                                           void *VmaInFile, size_t Bytes,
                                           void *HostAddr, void *DeviceAddr,
                                           uint64_t ModuleId) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::DeviceLoad(
        DeviceNum, Filename, OffsetInFile, VmaInFile, Bytes, HostAddr,
        DeviceAddr, ModuleId, "Device Load"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::DeviceLoad(DeviceNum, Filename, OffsetInFile,
                                          VmaInFile, Bytes, HostAddr,
                                          DeviceAddr, ModuleId, "Device Load"));
}

void OmptCallbackHandler::handleDeviceUnload(int DeviceNum, uint64_t ModuleId) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::DeviceUnload("Device Unload"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::DeviceUnload("Device Unload"));
}

void OmptCallbackHandler::handleBufferRequest(int DeviceNum,
                                              ompt_buffer_t **Buffer,
                                              size_t *Bytes) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::BufferRequest(DeviceNum, Buffer, Bytes,
                                               "Buffer Request"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::BufferRequest(DeviceNum, Buffer, Bytes,
                                             "Buffer Request"));
}

void OmptCallbackHandler::handleBufferComplete(int DeviceNum,
                                               ompt_buffer_t *Buffer,
                                               size_t Bytes,
                                               ompt_buffer_cursor_t Begin,
                                               int BufferOwned) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::BufferComplete(
        DeviceNum, Buffer, Bytes, Begin, BufferOwned, "Buffer Complete"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::BufferComplete(DeviceNum, Buffer, Bytes, Begin,
                                              BufferOwned, "Buffer Complete"));
}

void OmptCallbackHandler::handleBufferRecord(ompt_record_ompt_t *Record) {
  if (RecordAndReplay) {
    recordEvent(OmptAssertEvent::BufferRecord(Record, "Buffer Record"));
    return;
  }

  for (const auto &S : Subscribers)
    S->notify(OmptAssertEvent::BufferRecord(Record, "Buffer Record"));
}

void OmptCallbackHandler::handleWorkBegin(ompt_work_t work_type,
                                          ompt_scope_endpoint_t endpoint,
                                          ompt_data_t *parallel_data,
                                          ompt_data_t *task_data,
                                          uint64_t count,
                                          const void *codeptr_ra) {}

void OmptCallbackHandler::handleWorkEnd(ompt_work_t work_type,
                                        ompt_scope_endpoint_t endpoint,
                                        ompt_data_t *parallel_data,
                                        ompt_data_t *task_data, uint64_t count,
                                        const void *codeptr_ra) {}

void OmptCallbackHandler::recordEvent(OmptAssertEvent &&Event) {
  RecordedEvents.emplace_back(std::forward<OmptAssertEvent>(Event));
}
