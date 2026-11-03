// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.testdir
// RUN: mkdir -p %t.testdir
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.testdir %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --relocate --verify {} | FileCheck --check-prefix=FALLBACK %s
// clang-format on

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

// The program copies a device pointer into device memory itself, so the
// runtime does not know about it and the replay memory cannot be relocated.
// FALLBACK: Replaying at the recorded address: 1 unknown pointers in the recorded memory
// FALLBACK-NOT: Relocated
// FALLBACK: Replay done, device memory verified

#include <cstddef>
#include <cstdio>
#include <omp.h>

int main() {
  constexpr size_t Size = 1000;
  int Device = omp_get_default_device();
  int Host = omp_get_initial_device();

  double *Data =
      static_cast<double *>(omp_target_alloc(Size * sizeof(double), Device));
  double **Table =
      static_cast<double **>(omp_target_alloc(sizeof(double *), Device));
  omp_target_memcpy(Table, &Data, sizeof(double *), 0, 0, Device, Host);

#pragma omp target teams distribute parallel for is_device_ptr(Table)
  for (size_t I = 0; I < Size; ++I)
    (*Table)[I] = I;

  double Result[Size];
  omp_target_memcpy(Result, Data, sizeof(Result), 0, 0, Host, Device);

  // CHECK: PASS
  if (Result[Size - 1] == Size - 1)
    printf("PASS\n");

  omp_target_free(Table, Device);
  omp_target_free(Data, Device);
}
