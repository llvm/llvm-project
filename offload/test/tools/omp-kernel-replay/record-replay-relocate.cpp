// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.testdir
// RUN: mkdir -p %t.testdir
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.testdir %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --relocate --verify {} | FileCheck --check-prefix=RELOCATE %s
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --relocate --verify --num-threads=32 --repetitions=2 {} | FileCheck --check-prefix=RELOCATE %s
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --verify {} | FileCheck --check-prefix=FIXED %s
// clang-format on

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

// The device copy of the struct holds a pointer attached by the runtime, and
// the kernel argument points to the struct.
// RELOCATE: Relocated the replay memory from 0x{{[0-9a-f]+}} to 0x{{[0-9a-f]+}} ({{[1-9][0-9]*}} pointers)
// RELOCATE: Replay done, device memory verified
// FIXED-NOT: Relocated
// FIXED: Replay done, device memory verified

#include <cstddef>
#include <cstdio>

struct VectorTy {
  size_t Size;
  double *Data;
};

int main() {
  constexpr size_t Size = 1000;
  VectorTy Vector{Size, new double[Size]};
  for (size_t I = 0; I < Size; ++I)
    Vector.Data[I] = 1.0;

#pragma omp target teams distribute parallel for map(to : Vector)              \
    map(tofrom : Vector.Data[0 : Size])
  for (size_t I = 0; I < Vector.Size; ++I)
    Vector.Data[I] = 2.0 * Vector.Data[I] + I;

  double Sum = 0.0;
  for (size_t I = 0; I < Size; ++I)
    Sum += Vector.Data[I];

  // CHECK: PASS
  if (Sum == 2.0 * Size + Size * (Size - 1) / 2)
    printf("PASS\n");

  delete[] Vector.Data;
}
