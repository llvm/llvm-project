// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.compressed %t.raw
// RUN: mkdir -p %t.compressed %t.raw
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.compressed %libomptarget-run-generic 2>&1 | %fcheck-generic
// The 16 MiB of changed values are stored compressed.
// RUN: ls %t.compressed/*.record_output | xargs -I {} sh -c 'head -c 8 {} | grep -q OMPRRDZ1'
// RUN: ls %t.compressed/*.record_output | xargs -I {} sh -c 'test $(wc -c < {}) -lt 1048576'
// RUN: ls %t.compressed/*.json | xargs -I {} %omp-kernel-replay --verify {} | FileCheck --check-prefix=VERIFY %s
// RUN: ls %t.compressed/*.json | xargs -I {} %omp-kernel-replay --verify --relocate {} | FileCheck --check-prefix=VERIFY %s
// RUN: ls %t.compressed/*.json | xargs -I {} %omp-kernel-replay --verify --repetitions=3 {} | FileCheck --check-prefix=VERIFY %s
// Verifying does not store the replay output, saving it does.
// RUN: not sh -c 'ls %t.compressed/*.replay_output 2>/dev/null'
// RUN: ls %t.compressed/*.json | xargs -I {} %omp-kernel-replay --verify --save-output {} | FileCheck --check-prefix=VERIFY %s
// RUN: ls %t.compressed/*.replay_output | xargs -I {} sh -c 'head -c 8 {} | grep -q OMPRRDF1'
// A truncated compressed record output is reported.
// RUN: ls %t.compressed/*.record_output | xargs -I {} truncate -s 100 {}
// RUN: ls %t.compressed/*.json | xargs -I {} not %omp-kernel-replay --verify {} 2>&1 | FileCheck --check-prefix=CORRUPTED %s
// Uncompressed when disabled. A mismatch reports where the memory differs.
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_COMPRESS_OUTPUT=0 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.raw %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: ls %t.raw/*.record_output | xargs -I {} sh -c 'test $(wc -c < {}) -ge 16777216'
// RUN: ls %t.raw/*.json | xargs -I {} %omp-kernel-replay --verify {} | FileCheck --check-prefix=VERIFY %s
// RUN: ls %t.raw/*.record_output | xargs -I {} sh -c "printf 'X' | dd of={} bs=1 seek=40 conv=notrunc 2>/dev/null"
// RUN: ls %t.raw/*.json | xargs -I {} not %omp-kernel-replay --verify {} 2>&1 | FileCheck --check-prefix=MISMATCH %s
// clang-format on

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

// VERIFY: Replay done, device memory verified
// CORRUPTED: [llvm-omp-kernel-replay] Error: the kernel record output file is corrupted
// MISMATCH: [llvm-omp-kernel-replay] Error: replay device memory failed to verify (first difference at offset {{[0-9]+}})

#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
  constexpr size_t Size = 2 * 1024 * 1024;
  uint64_t *A = new uint64_t[Size];
  for (size_t I = 0; I < Size; ++I)
    A[I] = 7;

  // Change every element of a large buffer.
#pragma omp target teams distribute parallel for map(tofrom : A[0 : Size])
  for (size_t I = 0; I < Size; ++I)
    A[I] += 1;

  uint64_t Sum = 0;
  for (size_t I = 0; I < Size; ++I)
    Sum += A[I];

  // CHECK: PASS
  if (Sum == Size * 8)
    printf("PASS\n");

  delete[] A;
}
