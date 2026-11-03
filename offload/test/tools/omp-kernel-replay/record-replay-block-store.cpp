// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.blocks %t.full
// RUN: mkdir -p %t.blocks %t.full
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.blocks %libomptarget-run-generic 2>&1 | %fcheck-generic
// The 16 MiB of repeated values are stored as a few blocks, and each input
// snapshot only lists its blocks.
// RUN: test -f %t.blocks/record_blocks.store
// RUN: sh -c 'test $(wc -c < %t.blocks/record_blocks.store) -lt 1048576'
// RUN: ls %t.blocks/*.record_input | xargs -I {} sh -c 'test $(wc -c < {}) -lt 16384'
// RUN: ls %t.blocks/*.json | xargs -I {} %omp-kernel-replay --verify {} | FileCheck --check-prefix=VERIFY %s
// RUN: ls %t.blocks/*.json | xargs -I {} %omp-kernel-replay --verify --relocate {} | FileCheck --check-prefix=VERIFY %s
// Full copies when the block store is disabled.
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_BLOCK_STORE=0 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.full %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: not test -e %t.full/record_blocks.store
// RUN: ls %t.full/*.record_input | xargs -I {} sh -c 'test $(wc -c < {}) -ge 16777216'
// RUN: ls %t.full/*.json | xargs -I {} %omp-kernel-replay --verify {} | FileCheck --check-prefix=VERIFY %s
// A missing block store is reported.
// RUN: rm %t.blocks/record_blocks.store
// RUN: ls %t.blocks/*.json | head -n 1 | xargs -I {} not %omp-kernel-replay {} 2>&1 | FileCheck --check-prefix=NOSTORE %s
// clang-format on

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

// VERIFY: Replay done, device memory verified
// NOSTORE: [llvm-omp-kernel-replay] Error: failed to read the block store file

#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
  constexpr size_t Size = 2 * 1024 * 1024;
  uint64_t *A = new uint64_t[Size];
  for (size_t I = 0; I < Size; ++I)
    A[I] = 7;

  // Two kernels that see mostly the same memory.
#pragma omp target data map(tofrom : A[0 : Size])
  {
#pragma omp target teams distribute parallel for
    for (size_t I = 0; I < Size; ++I)
      A[I] += 1;
#pragma omp target teams distribute parallel for
    for (size_t I = 0; I < Size / 2; ++I)
      A[I] *= 2;
  }

  uint64_t Sum = 0;
  for (size_t I = 0; I < Size; ++I)
    Sum += A[I];

  // CHECK: PASS
  if (Sum == Size / 2 * 16 + Size / 2 * 8)
    printf("PASS\n");

  delete[] A;
}
