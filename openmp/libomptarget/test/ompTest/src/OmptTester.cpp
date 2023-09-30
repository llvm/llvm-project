#include "../include/OmptTester.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

std::unordered_map<std::string, TestSuite> TestRegistrar::Tests;
std::atomic<ompt_id_t> NextOpId{0x8000000000000001};

// From openmp/runtime/test/ompt/callback.h
#define register_ompt_callback_t(name, type)                                   \
  do {                                                                         \
    type f_##name = &on_##name;                                                \
    if (ompt_set_callback(name, (ompt_callback_t)f_##name) == ompt_set_never)  \
      printf("0: Could not register callback '" #name "'\n");                  \
  } while (0)

#define register_ompt_callback(name) register_ompt_callback_t(name, name##_t)

// OMPT callbacks

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

void on_ompt_callback_parallel_end(ompt_data_t *parallel_data,
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
void on_ompt_callback_work(ompt_work_t work_type,
                           ompt_scope_endpoint_t endpoint,
                           ompt_data_t *parallel_data, ompt_data_t *task_data,
                           uint64_t count, const void *codeptr_ra) {
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

/// Called by the OMP runtime to initialize the OMPT
int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                    ompt_data_t *tool_data) {
  ompt_set_callback_t ompt_set_callback = nullptr;
  ompt_set_callback = (ompt_set_callback_t)lookup("ompt_set_callback");
  if (!ompt_set_callback)
    return 0; // failure

  bool RegisterEMICallbacks = false;
  if (const char *EnvUseEMI = std::getenv("OMPTEST_USE_OMPT_EMI")) {
    std::string UseEMI{EnvUseEMI};
    for (auto &C : UseEMI)
      C = (char)std::tolower(C);
    if (UseEMI == "1" || UseEMI == "on" || UseEMI == "true" || UseEMI == "yes")
      RegisterEMICallbacks = true;
  }

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

  if (RegisterEMICallbacks) {
    register_ompt_callback(ompt_callback_target_emi);
    register_ompt_callback(ompt_callback_target_submit_emi);
    register_ompt_callback(ompt_callback_target_data_op_emi);
  } else {
    register_ompt_callback(ompt_callback_target);
    register_ompt_callback(ompt_callback_target_submit);
    register_ompt_callback(ompt_callback_target_data_op);
  }

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
#ifdef __cplusplus
}
#endif
