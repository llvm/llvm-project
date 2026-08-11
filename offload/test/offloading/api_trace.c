// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_INFO=128 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=TRACE
// RUN: %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -allow-empty -check-prefix=NOTRACE

#include <stdio.h>

int main() {
  int Data = 0;
#pragma omp target map(tofrom : Data)
  Data = 1;

  printf("Data = %d\n", Data);
  return Data != 1;
}

// Argument values other than the device number vary between runs.
// TRACE-DAG: ---> init_device(.DeviceId = 0)-> OFFLOAD_SUCCESS ({{[0-9]+}} us)
// TRACE-DAG: ---> load_binary(.DeviceId = 0, {{.*}})-> OFFLOAD_SUCCESS ({{[0-9]+}} us)
// TRACE-DAG: ---> data_alloc(.DeviceId = 0, {{.*}}, .Kind = TARGET_ALLOC_{{[A-Z]+}})-> 0x{{[0-9a-f]+}} ({{[0-9]+}} us)
// TRACE-DAG: ---> data_submit_async(.DeviceId = 0, {{.*}})-> OFFLOAD_SUCCESS ({{[0-9]+}} us)
// TRACE-DAG: ---> launch_kernel(.DeviceId = 0, {{.*}})-> OFFLOAD_SUCCESS ({{[0-9]+}} us)
// TRACE-DAG: ---> data_retrieve_async(.DeviceId = 0, {{.*}})-> OFFLOAD_SUCCESS ({{[0-9]+}} us)
// TRACE-DAG: ---> data_delete(.DeviceId = 0, {{.*}}, .Kind = TARGET_ALLOC_{{[A-Z]+}})-> OFFLOAD_SUCCESS ({{[0-9]+}} us)
// TRACE-DAG: ---> number_of_devices()-> {{[0-9]+}} ({{[0-9]+}} us)

// NOTRACE-NOT: --->
