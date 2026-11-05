// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.testdir
// RUN: mkdir -p %t.testdir
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.testdir %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --relocate --verify {} | FileCheck --check-prefix=FALLBACK %s
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --relocate {} | FileCheck --check-prefix=RELOCATE %s
// clang-format on

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

// The kernel stores a device pointer in device memory, so a relocated replay
// can run but not be verified.
// FALLBACK: Replaying at the recorded address: 1 unknown pointers in the recorded output
// FALLBACK: Replay done, device memory verified
// RELOCATE: Relocated the replay memory from 0x{{[0-9a-f]+}} to 0x{{[0-9a-f]+}} (0 pointers)
// RELOCATE: Replay done, verification skipped

#include <cstddef>
#include <cstdio>
#include <omp.h>

int main() {
  int Device = omp_get_default_device();
  int Host = omp_get_initial_device();

  double *Data =
      static_cast<double *>(omp_target_alloc(sizeof(double), Device));
  double **Table =
      static_cast<double **>(omp_target_alloc(sizeof(double *), Device));

#pragma omp target is_device_ptr(Table, Data)
  *Table = Data;

  double *Result = nullptr;
  omp_target_memcpy(&Result, Table, sizeof(double *), 0, 0, Host, Device);

  // CHECK: PASS
  if (Result == Data)
    printf("PASS\n");

  omp_target_free(Table, Device);
  omp_target_free(Data, Device);
}
