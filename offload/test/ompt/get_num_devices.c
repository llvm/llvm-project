// clang-format off
// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// REQUIRES: gpu
// clang-format on

/*
 * Test to ensure that a tool receives the same number of devices
 * via the OMPT entry point that an application receives via the
 * omp_get_num_devices runtime function.
 *
 * Note, that the get_num_devices() entry point will _not_ provide
 * the correct amount of devices before the PluginManager has been
 * fully initialized, i.e. until after `ompt_initialize`.
 */

#include <assert.h>
#include <stdio.h>

#include <omp-tools.h>
#include <omp.h>

// OMPT entry point handles
static ompt_get_num_devices_t ompt_get_num_devices = 0;

// Init functions
int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                    ompt_data_t *tool_data) {
  ompt_get_num_devices = (ompt_get_num_devices_t)lookup("ompt_get_num_devices");

  if (!ompt_get_num_devices)
    return 0; // failed

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
  const int NumDevices = omp_get_num_devices();
  assert(ompt_get_num_devices != NULL &&
         "ompt_get_num_devices() should not be NULL\n");
  const int NumDevicesOmpt = ompt_get_num_devices();

  printf("Num Devices: %d\n", NumDevices);
  printf("Num Devices via OMPT: %d\n", NumDevicesOmpt);
}

/// CHECK: Num Devices: [[NUM:[0-9]+]]
/// CHECK: Num Devices via OMPT: [[NUM]]
