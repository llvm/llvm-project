// clang-format off
// RUN: %libomptarget-compilexx-generic && env LIBOMPTARGET_AMDGPU_MAX_ASYNC_COPY_BYTES=1 LIBOMPTARGET_OMPT_FLUSH_ON_SHUTDOWN=false LIBOMPTARGET_OMPT_FLUSH_ON_BUFFER_FULL=false %libomptarget-run-generic | %fcheck-generic
// REQUIRES: ompt
// REQUIRES: amdgpu
// clang-format on

// Exercises OMPT device tracing for AMDGPU's synchronous large-copy path.
// The transfer size is intentionally above the plugin's default async-copy
// threshold, while the RUN line also lowers the threshold to keep the test
// independent of future default changes.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "../veccopy-ompt-target-data-tracing-emi/callbacks.h"

#define COPY_BYTES (50 * 1024 * 1024)

DeviceMapPtr_t DeviceMapPtr;

int main() {
  if (omp_get_num_devices() == 0) {
    printf("Success\n");
    return 0;
  }

  const size_t N = COPY_BYTES / sizeof(int);
  int *Input = (int *)malloc(COPY_BYTES);
  int *Output = (int *)malloc(COPY_BYTES);

  if (!Input || !Output) {
    free(Input);
    free(Output);
    printf("Failure\n");
    return 1;
  }

  for (size_t I = 0; I < N; ++I) {
    Input[I] = (int)(I & 0x7fff);
    Output[I] = 0;
  }

#pragma omp target teams distribute parallel for map(to : Input[0 : N])        \
    map(from : Output[0 : N])
  for (size_t I = 0; I < N; ++I)
    Output[I] = Input[I] + 1;

  for (auto Dev : *DeviceMapPtr)
    flush_trace(Dev);

  int Failed = 0;
  for (size_t I = 0; I < N; ++I) {
    if (Output[I] != Input[I] + 1) {
      Failed = 1;
      break;
    }
  }

  free(Input);
  free(Output);

  printf("%s\n", Failed ? "Failure" : "Success");
  return Failed;
}

// clang-format off

/// CHECK: Callback Init:
/// CHECK: Callback Load:
/// CHECK-DAG: Callback DataOp EMI: endpoint=1 optype=1
/// CHECK-DAG: Callback DataOp EMI: endpoint=2 optype=1
/// CHECK-DAG: Callback DataOp EMI: endpoint=1 optype=2 {{.*}} bytes=52428800
/// CHECK-DAG: Callback DataOp EMI: endpoint=2 optype=2 {{.*}} bytes=52428800
/// CHECK-DAG: Callback DataOp EMI: endpoint=1 optype=3 {{.*}} bytes=52428800
/// CHECK-DAG: Callback DataOp EMI: endpoint=2 optype=3 {{.*}} bytes=52428800
/// CHECK-DAG: Callback DataOp EMI: endpoint=1 optype=4
/// CHECK-DAG: Callback DataOp EMI: endpoint=2 optype=4

/// CHECK-DAG: type=9 (Target data op) {{.*}} optype=1 {{.*}} bytes=52428800
/// CHECK: Success
