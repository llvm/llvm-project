#include "../include/OmptTester.h"

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>

// From openmp/runtime/test/ompt/callback.h
#define register_ompt_callback_t(name, type)                                   \
  do {                                                                         \
    type f_##name = &on_##name;                                                \
    if (ompt_set_callback(name, (ompt_callback_t)f_##name) == ompt_set_never)  \
      printf("0: Could not register callback '" #name "'\n");                  \
  } while (0)

#define register_ompt_callback(name) register_ompt_callback_t(name, name##_t)

#define OMPT_BUFFER_REQUEST_SIZE 256

std::unordered_map<std::string, TestSuite> TestRegistrar::Tests;
static std::atomic<ompt_id_t> NextOpId{0x8000000000000001};
static bool UseEMICallbacks = false;
static bool UseTracing = false;

// EventListener which will print OMPT events
static OmptEventReporter EventReporter;

// OMPT entry point handles
static ompt_set_trace_ompt_t ompt_set_trace_ompt = 0;
static ompt_start_trace_t ompt_start_trace = 0;
static ompt_flush_trace_t ompt_flush_trace = 0;
static ompt_stop_trace_t ompt_stop_trace = 0;
static ompt_get_record_ompt_t ompt_get_record_ompt = 0;
static ompt_advance_buffer_cursor_t ompt_advance_buffer_cursor = 0;
static ompt_get_record_type_t ompt_get_record_type_fn = 0;

// ToDo: Currently using only 1 device
static ompt_device_t *Device = nullptr;

// Tracing buffer helper function
static void delete_buffer_ompt(ompt_buffer_t *buffer) {
  free(buffer);
  printf("Deallocated %p\n", buffer);
}

// OMPT callbacks

// Trace record callbacks
static void on_ompt_callback_buffer_request(int device_num,
                                            ompt_buffer_t **buffer,
                                            size_t *bytes) {
  *bytes = OMPT_BUFFER_REQUEST_SIZE;
  *buffer = malloc(*bytes);
  printf("Allocated %lu bytes at %p in buffer request callback\n", *bytes,
         *buffer);
  OmptCallbackHandler::get().handleBufferRequest(device_num, buffer, bytes);
}

// Note: This callback must handle a null begin cursor. Currently,
// ompt_get_record_ompt, print_record_ompt, and
// ompt_advance_buffer_cursor handle a null cursor.
static void on_ompt_callback_buffer_complete(
    int device_num, ompt_buffer_t *buffer,
    size_t bytes, /* bytes returned in this callback */
    ompt_buffer_cursor_t begin, int buffer_owned) {
  OmptCallbackHandler::get().handleBufferComplete(device_num, buffer, bytes,
                                                  begin, buffer_owned);

  int Status = 1;
  ompt_buffer_cursor_t CurrentPos = begin;
  while (Status) {
    ompt_record_ompt_t *Record = ompt_get_record_ompt(buffer, CurrentPos);
    if (ompt_get_record_type_fn(buffer, CurrentPos) != ompt_record_ompt) {
      printf("WARNING: received non-ompt type buffer object\n");
    }
    // ToDo: Sometimes it may happen that the retrieved record may be null?!
    // Only handle non-null records
    if (Record != nullptr)
      OmptCallbackHandler::get().handleBufferRecord(Record);
    Status = ompt_advance_buffer_cursor(/*device=*/NULL, buffer, bytes,
                                        CurrentPos, &CurrentPos);
  }
  if (buffer_owned)
    delete_buffer_ompt(buffer);
}

static ompt_set_result_t set_trace_ompt() {
  if (!ompt_set_trace_ompt)
    return ompt_set_error;

  if (UseEMICallbacks) {
    ompt_set_trace_ompt(/*device=*/0, /*enable=*/1,
                        /*etype=*/ompt_callback_target_emi);
    ompt_set_trace_ompt(/*device=*/0, /*enable=*/1,
                        /*etype=*/ompt_callback_target_data_op_emi);
    ompt_set_trace_ompt(/*device=*/0, /*enable=*/1,
                        /*etype=*/ompt_callback_target_submit_emi);
  } else {
    ompt_set_trace_ompt(/*device=*/0, /*enable=*/1,
                        /*etype=*/ompt_callback_target);
    ompt_set_trace_ompt(/*device=*/0, /*enable=*/1,
                        /*etype=*/ompt_callback_target_data_op);
    ompt_set_trace_ompt(/*device=*/0, /*enable=*/1,
                        /*etype=*/ompt_callback_target_submit);
  }

  return ompt_set_always;
}

/////// HOST-RELATED //////

static void on_ompt_callback_thread_begin(ompt_thread_t thread_type,
                                          ompt_data_t *thread_data) {
  OmptCallbackHandler::get().handleThreadBegin(thread_type, thread_data);
}

static void on_ompt_callback_thread_end(ompt_data_t *thread_data) {
  OmptCallbackHandler::get().handleThreadEnd(thread_data);
}

static void on_ompt_callback_parallel_begin(
    ompt_data_t *encountering_task_data,
    const ompt_frame_t *encountering_task_frame, ompt_data_t *parallel_data,
    unsigned int requested_parallelism, int flags, const void *codeptr_ra) {
  OmptCallbackHandler::get().handleParallelBegin(
      encountering_task_data, encountering_task_frame, parallel_data,
      requested_parallelism, flags, codeptr_ra);
}

static void on_ompt_callback_parallel_end(ompt_data_t *parallel_data,
                                          ompt_data_t *encountering_task_data,
                                          int flags, const void *codeptr_ra) {
  OmptCallbackHandler::get().handleParallelEnd(
      parallel_data, encountering_task_data, flags, codeptr_ra);
}

static void
on_ompt_callback_task_create(ompt_data_t *encountering_task_data,
                             const ompt_frame_t *encountering_task_frame,
                             ompt_data_t *new_task_data, int flags,
                             int has_dependences, const void *codeptr_ra) {
  OmptCallbackHandler::get().handleTaskCreate(
      encountering_task_data, encountering_task_frame, new_task_data, flags,
      has_dependences, codeptr_ra);
}

static void on_ompt_callback_task_schedule(ompt_data_t *prior_task_data,
                                           ompt_task_status_t prior_task_status,
                                           ompt_data_t *next_task_data) {
  OmptCallbackHandler::get().handleTaskSchedule(
      prior_task_data, prior_task_status, next_task_data);
}

static void on_ompt_callback_implicit_task(ompt_scope_endpoint_t endpoint,
                                           ompt_data_t *parallel_data,
                                           ompt_data_t *task_data,
                                           unsigned int actual_parallelism,
                                           unsigned int index, int flags) {
  OmptCallbackHandler::get().handleImplicitTask(
      endpoint, parallel_data, task_data, actual_parallelism, index, flags);
}

// Callbacks as of Table 19.4, which are not considered required for a minimal
// conforming OMPT implementation.
static void on_ompt_callback_work(ompt_work_t work_type,
                                  ompt_scope_endpoint_t endpoint,
                                  ompt_data_t *parallel_data,
                                  ompt_data_t *task_data, uint64_t count,
                                  const void *codeptr_ra) {
  if (endpoint == ompt_scope_begin || endpoint == ompt_scope_beginend)
    OmptCallbackHandler::get().handleWorkBegin(
        work_type, endpoint, parallel_data, task_data, count, codeptr_ra);

  if (endpoint == ompt_scope_end || endpoint == ompt_scope_beginend)
    OmptCallbackHandler::get().handleWorkEnd(work_type, endpoint, parallel_data,
                                             task_data, count, codeptr_ra);
}

/////// DEVICE-RELATED //////

// Synchronous callbacks
static void on_ompt_callback_device_initialize(int device_num, const char *type,
                                               ompt_device_t *device,
                                               ompt_function_lookup_t lookup,
                                               const char *documentation) {
  OmptCallbackHandler::get().handleDeviceInitialize(device_num, type, device,
                                                    lookup, documentation);
  if (!UseTracing)
    return;

  if (!lookup) {
    printf("Trace collection disabled on device %d\n", device_num);
    return;
  }

  ompt_set_trace_ompt = (ompt_set_trace_ompt_t)lookup("ompt_set_trace_ompt");
  ompt_start_trace = (ompt_start_trace_t)lookup("ompt_start_trace");
  ompt_flush_trace = (ompt_flush_trace_t)lookup("ompt_flush_trace");
  ompt_stop_trace = (ompt_stop_trace_t)lookup("ompt_stop_trace");
  ompt_get_record_ompt = (ompt_get_record_ompt_t)lookup("ompt_get_record_ompt");
  ompt_advance_buffer_cursor =
      (ompt_advance_buffer_cursor_t)lookup("ompt_advance_buffer_cursor");

  ompt_get_record_type_fn =
      (ompt_get_record_type_t)lookup("ompt_get_record_type");
  if (!ompt_get_record_type_fn) {
    printf("WARNING: No function ompt_get_record_type found in device "
           "callbacks\n");
  }

  Device = device;

  set_trace_ompt();

  // In many scenarios, this will be a good place to start the
  // trace. If start_trace is called from the main program before this
  // callback is dispatched, the start_trace handle will be null. This
  // is because this device_init callback is invoked during the first
  // target construct implementation.

  start_trace();
}

static void on_ompt_callback_device_finalize(int device_num) {
  OmptCallbackHandler::get().handleDeviceFinalize(device_num);
}

static void on_ompt_callback_device_load(int device_num, const char *filename,
                                         int64_t offset_in_file,
                                         void *vma_in_file, size_t bytes,
                                         void *host_addr, void *device_addr,
                                         uint64_t module_id) {
  OmptCallbackHandler::get().handleDeviceLoad(
      device_num, filename, offset_in_file, vma_in_file, bytes, host_addr,
      device_addr, module_id);
}

static void on_ompt_callback_device_unload(int device_num, uint64_t module_id) {
  OmptCallbackHandler::get().handleDeviceUnload(device_num, module_id);
}

static void on_ompt_callback_target_data_op(
    ompt_id_t target_id, ompt_id_t host_op_id, ompt_target_data_op_t optype,
    void *src_addr, int src_device_num, void *dest_addr, int dest_device_num,
    size_t bytes, const void *codeptr_ra) {
  OmptCallbackHandler::get().handleTargetDataOp(
      target_id, host_op_id, optype, src_addr, src_device_num, dest_addr,
      dest_device_num, bytes, codeptr_ra);
}

static void on_ompt_callback_target(ompt_target_t kind,
                                    ompt_scope_endpoint_t endpoint,
                                    int device_num, ompt_data_t *task_data,
                                    ompt_id_t target_id,
                                    const void *codeptr_ra) {
  OmptCallbackHandler::get().handleTarget(kind, endpoint, device_num, task_data,
                                          target_id, codeptr_ra);
}

static void on_ompt_callback_target_submit(ompt_id_t target_id,
                                           ompt_id_t host_op_id,
                                           unsigned int requested_num_teams) {
  OmptCallbackHandler::get().handleTargetSubmit(target_id, host_op_id,
                                                requested_num_teams);
}

static void on_ompt_callback_target_data_op_emi(
    ompt_scope_endpoint_t endpoint, ompt_data_t *target_task_data,
    ompt_data_t *target_data, ompt_id_t *host_op_id,
    ompt_target_data_op_t optype, void *src_addr, int src_device_num,
    void *dest_addr, int dest_device_num, size_t bytes,
    const void *codeptr_ra) {
  assert(codeptr_ra != 0 && "Unexpected null codeptr");
  // Both src and dest must not be null
  assert((src_addr != 0 || dest_addr != 0) && "Both src and dest addr null");
  if (endpoint == ompt_scope_begin)
    *host_op_id = NextOpId.fetch_add(1, std::memory_order_relaxed);
  OmptCallbackHandler::get().handleTargetDataOpEmi(
      endpoint, target_task_data, target_data, host_op_id, optype, src_addr,
      src_device_num, dest_addr, dest_device_num, bytes, codeptr_ra);
}

static void on_ompt_callback_target_emi(ompt_target_t kind,
                                        ompt_scope_endpoint_t endpoint,
                                        int device_num, ompt_data_t *task_data,
                                        ompt_data_t *target_task_data,
                                        ompt_data_t *target_data,
                                        const void *codeptr_ra) {
  assert(codeptr_ra != 0 && "Unexpected null codeptr");
  if (endpoint == ompt_scope_begin)
    target_data->value = NextOpId.fetch_add(1, std::memory_order_relaxed);
  OmptCallbackHandler::get().handleTargetEmi(kind, endpoint, device_num,
                                             task_data, target_task_data,
                                             target_data, codeptr_ra);
}

static void on_ompt_callback_target_submit_emi(
    ompt_scope_endpoint_t endpoint, ompt_data_t *target_data,
    ompt_id_t *host_op_id, unsigned int requested_num_teams) {
  OmptCallbackHandler::get().handleTargetSubmitEmi(
      endpoint, target_data, host_op_id, requested_num_teams);
}

static void on_ompt_callback_target_map(ompt_id_t target_id,
                                        unsigned int nitems, void **host_addr,
                                        void **device_addr, size_t *bytes,
                                        unsigned int *mapping_flags,
                                        const void *codeptr_ra) {
  assert(0 && "Target map callback is unimplemented");
}

static void on_ompt_callback_target_map_emi(ompt_data_t *target_data,
                                            unsigned int nitems,
                                            void **host_addr,
                                            void **device_addr, size_t *bytes,
                                            unsigned int *mapping_flags,
                                            const void *codeptr_ra) {
  assert(0 && "Target map emi callback is unimplemented");
}

/// Load the value of a given boolean environmental variable.
bool getBoolEnvironmentVariable(const char *VariableName) {
  if (VariableName == nullptr)
    return false;
  if (const char *EnvValue = std::getenv(VariableName)) {
    std::string S{EnvValue};
    for (auto &C : S)
      C = (char)std::tolower(C);
    if (S == "1" || S == "on" || S == "true" || S == "yes")
      return true;
  }
  return false;
}

/// Called by the OMP runtime to initialize the OMPT
int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                    ompt_data_t *tool_data) {
  ompt_set_callback_t ompt_set_callback = nullptr;
  ompt_set_callback = (ompt_set_callback_t)lookup("ompt_set_callback");
  if (!ompt_set_callback)
    return 0; // failure

  UseEMICallbacks = getBoolEnvironmentVariable("OMPTEST_USE_OMPT_EMI");
  UseTracing = getBoolEnvironmentVariable("OMPTEST_USE_OMPT_TRACING");

  register_ompt_callback(ompt_callback_thread_begin);
  register_ompt_callback(ompt_callback_thread_end);
  register_ompt_callback(ompt_callback_parallel_begin);
  register_ompt_callback(ompt_callback_parallel_end);
  register_ompt_callback(ompt_callback_task_create);
  register_ompt_callback(ompt_callback_task_schedule);
  register_ompt_callback(ompt_callback_implicit_task);
  register_ompt_callback(ompt_callback_work);
  register_ompt_callback(ompt_callback_device_initialize);
  register_ompt_callback(ompt_callback_device_finalize);
  register_ompt_callback(ompt_callback_device_load);
  register_ompt_callback(ompt_callback_device_unload);

  if (UseEMICallbacks) {
    register_ompt_callback(ompt_callback_target_emi);
    register_ompt_callback(ompt_callback_target_submit_emi);
    register_ompt_callback(ompt_callback_target_data_op_emi);
    register_ompt_callback(ompt_callback_target_map_emi);
  } else {
    register_ompt_callback(ompt_callback_target);
    register_ompt_callback(ompt_callback_target_submit);
    register_ompt_callback(ompt_callback_target_data_op);
    register_ompt_callback(ompt_callback_target_map);
  }

  // Subscribe the reporter, so it will be notified of events
  OmptCallbackHandler::get().subscribe(&EventReporter);

  return 1; // success
}

void ompt_finalize(ompt_data_t *tool_data) {}

#ifdef __cplusplus
extern "C" {
#endif
/// Called from the OMP Runtime to start / initialize the tool
ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t ompt_start_tool_result = {
      &ompt_initialize, &ompt_finalize, {0}};
  return &ompt_start_tool_result;
}

int start_trace() {
  if (!ompt_start_trace)
    return 0;
  return ompt_start_trace(Device, &on_ompt_callback_buffer_request,
                          &on_ompt_callback_buffer_complete);
}

int flush_trace() {
  if (!ompt_flush_trace)
    return 0;
  return ompt_flush_trace(Device);
}

int stop_trace() {
  if (!ompt_stop_trace)
    return 0;
  return ompt_stop_trace(Device);
}
#ifdef __cplusplus
}
#endif
