// clang-format off
// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// REQUIRES: gpu
// clang-format on

/*
 * Example OpenMP program that checks if get_target_info entry point is
 * correctly implemented.
 */

#include <assert.h>
#include <stdio.h>

#include <omp-tools.h>
#include <omp.h>

// From openmp/runtime/test/ompt/callback.h
#define register_ompt_callback_t(name, type)                                   \
  do {                                                                         \
    type f_##name = &on_##name;                                                \
    if (ompt_set_callback(name, (ompt_callback_t)f_##name) == ompt_set_never)  \
      printf("0: Could not register callback '" #name "'\n");                  \
  } while (0)

#define register_ompt_callback(name) register_ompt_callback_t(name, name##_t)

// OMPT entry point handles
static ompt_set_callback_t ompt_set_callback = 0;
static ompt_get_target_info_t ompt_get_target_info = 0;
static ompt_get_unique_id_t ompt_get_unique_id = 0;

// Callback implementation
static void on_ompt_callback_device_initialize(int device_num, const char *type,
                                               ompt_device_t *device,
                                               ompt_function_lookup_t lookup,
                                               const char *documentation) {
  printf("Callback Device Init\n");
}

static void on_ompt_callback_target_emi(ompt_target_t kind,
                                        ompt_scope_endpoint_t endpoint,
                                        int device_num, ompt_data_t *task_data,
                                        ompt_data_t *target_task_data,
                                        ompt_data_t *target_data,
                                        const void *codeptr_ra) {
  printf("Callback Target EMI\n");

  if (endpoint == ompt_scope_begin)
    target_data->value = ompt_get_unique_id();
  else if (endpoint == ompt_scope_end) {
    uint64_t retrieved_device_num;
    ompt_id_t retrieved_target_id;
    ompt_id_t retrieved_host_op_id;
    int in_target_region = ompt_get_target_info(
        &retrieved_device_num, &retrieved_target_id, &retrieved_host_op_id);

    assert(in_target_region && "ompt_get_target_info did not return 1");
    assert(retrieved_device_num == device_num && "device_num does not match!");
    assert(retrieved_target_id == target_data->value &&
           "target_id does not match!");
    assert(retrieved_host_op_id == ompt_id_none &&
           "host_op_id should not be set!");
  }
}

static void on_ompt_callback_target_data_op_emi(
    ompt_scope_endpoint_t endpoint, ompt_data_t *target_task_data,
    ompt_data_t *target_data, ompt_id_t *host_op_id,
    ompt_target_data_op_t optype, void *src_addr, int src_device_num,
    void *dest_addr, int dest_device_num, size_t bytes,
    const void *codeptr_ra) {
  printf("Callback DataOp EMI\n");

  if (endpoint == ompt_scope_begin)
    *host_op_id = ompt_get_unique_id();
  else if (endpoint == ompt_scope_end) {
    ompt_id_t retrieved_target_id;
    ompt_id_t retrieved_host_op_id;
    int in_target_region =
        ompt_get_target_info(NULL, &retrieved_target_id, &retrieved_host_op_id);

    // Values are only valid if thread is inside of a target region.
    // This is not always the case for data_op_emi callbacks though, e.g.
    // during omp_target_memcpy or while transferring the device image on
    // runtime startup.
    if (in_target_region) {
      assert(retrieved_target_id == target_data->value &&
             "target_id does not match!");
      assert(retrieved_host_op_id == *host_op_id &&
             "host_op_id does not match!");
    }
  }
}

static void on_ompt_callback_target_submit_emi(
    ompt_scope_endpoint_t endpoint, ompt_data_t *target_data,
    ompt_id_t *host_op_id, unsigned int requested_num_teams) {
  printf("Callback Submit EMI\n");

  if (endpoint == ompt_scope_begin)
    *host_op_id = ompt_get_unique_id();
  else if (endpoint == ompt_scope_end) {
    ompt_id_t retrieved_target_id;
    ompt_id_t retrieved_host_op_id;
    int in_target_region =
        ompt_get_target_info(NULL, &retrieved_target_id, &retrieved_host_op_id);

    assert(in_target_region && "ompt_get_target_info did not return 1");
    assert(retrieved_target_id == target_data->value &&
           "target_id does not match!");
    assert(retrieved_host_op_id == *host_op_id && "host_op_id does not match!");
  }
}

// Init functions
int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                    ompt_data_t *tool_data) {
  ompt_set_callback = (ompt_set_callback_t)lookup("ompt_set_callback");
  if (!ompt_set_callback)
    return 0; // failed

  ompt_get_unique_id = (ompt_get_unique_id_t)lookup("ompt_get_unique_id");
  if (!ompt_get_unique_id)
    return 0; // failed

  ompt_get_target_info = (ompt_get_target_info_t)lookup("ompt_get_target_info");
  if (!ompt_get_target_info)
    return 0; // failed

  register_ompt_callback(ompt_callback_device_initialize);
  register_ompt_callback(ompt_callback_target_data_op_emi);
  register_ompt_callback(ompt_callback_target_emi);
  register_ompt_callback(ompt_callback_target_submit_emi);

  return 1; // success
}

void ompt_finalize(ompt_data_t *tool_data) {}

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,
                                                            &ompt_finalize, 0};
  return &ompt_start_tool_result;
}

int main(void) {
  int numDevices = omp_get_num_devices();
  int N = 100000;

  int a[N];
  int b[N];

  int i;

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

  for (int dev = 0; dev < numDevices; ++dev) {
#pragma omp target parallel for device(dev)
    {
      for (int j = 0; j < N; j++)
        a[j] = b[j];
    }
  }
}

// clang-format off

/// CHECK: Callback Device Init
/// CHECK: Callback Target EMI
/// CHECK: Callback DataOp EMI
/// CHECK: Callback DataOp EMI
/// CHECK: Callback Submit EMI
/// CHECK: Callback Submit EMI
/// CHECK: Callback DataOp EMI
/// CHECK: Callback DataOp EMI
/// CHECK: Callback Target EMI

// clang-format on
