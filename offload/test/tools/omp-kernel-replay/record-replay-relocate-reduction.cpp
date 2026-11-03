// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.testdir
// RUN: mkdir -p %t.testdir
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.testdir %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --relocate --verify {} | FileCheck --check-prefix=RELOCATE %s
// clang-format on

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

// The kernel launch environment of a cross-team reduction points to the
// reduction buffer, which the relocated replay has to adjust.
// RELOCATE: Relocated the replay memory from 0x{{[0-9a-f]+}} to 0x{{[0-9a-f]+}} ({{[1-9][0-9]*}} pointers)
// RELOCATE: Replay done, device memory verified

#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
  constexpr size_t Size = 1 << 20;
  uint64_t *Data = new uint64_t[Size];
  for (size_t I = 0; I < Size; ++I)
    Data[I] = I;

  uint64_t Sum = 0;
#pragma omp target teams distribute parallel for reduction(+ : Sum)            \
    map(to : Data[0 : Size])
  for (size_t I = 0; I < Size; ++I)
    Sum += Data[I];

  // CHECK: PASS
  if (Sum == Size * (Size - 1) / 2)
    printf("PASS\n");

  delete[] Data;
}
