#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTTESTERGLOBALS_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTTESTERGLOBALS_H

#include <omp-tools.h>

#ifdef __cplusplus
extern "C" {
#endif
ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version);
int start_trace(ompt_device_t *Device);
int flush_trace(ompt_device_t *Device);
// Function which calls flush_trace(ompt_device_t *) on all traced devices.
int flush_traced_devices();
int stop_trace(ompt_device_t *Device);
// Function which calls stop_trace(ompt_device_t *) on all traced devices.
int stop_trace_devices();
void libomptest_global_eventreporter_set_active(bool State);
#ifdef __cplusplus
}
#endif

#endif
