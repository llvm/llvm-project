// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.testdir
// RUN: mkdir -p %t.testdir
// Record full input snapshots and uncompressed output diffs, to compare sizes
// and edit the diff.
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_BLOCK_STORE=0 LIBOMPTARGET_RECORD_COMPRESS_OUTPUT=0 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.testdir %libomptarget-run-generic 2>&1 | %fcheck-generic
// The input snapshot holds the whole buffer, the output only the changes.
// RUN: ls %t.testdir/*.record_input | xargs -I {} sh -c 'test $(wc -c < {}) -ge 8388608'
// RUN: ls %t.testdir/*.record_output | xargs -I {} sh -c 'test $(wc -c < {}) -lt 4096'
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --verify {} | FileCheck --check-prefix=VERIFY %s
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --verify --num-threads=32 {} | FileCheck --check-prefix=VERIFY %s
// Corrupt the first changed byte stored in the record output.
// RUN: ls %t.testdir/*.record_output | xargs -I {} sh -c "printf 'X' | dd of={} bs=1 seek=40 conv=notrunc 2>/dev/null"
// RUN: ls %t.testdir/*.json | xargs -I {} not %omp-kernel-replay --verify {} 2>&1 | FileCheck --check-prefix=MISMATCH %s
// RUN: ls %t.testdir/*.record_output | xargs -I {} sh -c 'printf "" > {}'
// RUN: ls %t.testdir/*.json | xargs -I {} not %omp-kernel-replay --verify {} 2>&1 | FileCheck --check-prefix=TRUNCATED %s
// clang-format on

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

// VERIFY: Replay done, device memory verified
// MISMATCH: [llvm-omp-kernel-replay] Error: replay device memory failed to verify
// TRUNCATED: [llvm-omp-kernel-replay] Error: the kernel record output file is truncated

#include <cstdint>
#include <cstdio>

int main() {
  constexpr size_t Size = 1024 * 1024;
  constexpr size_t NumUpdates = 16;
  constexpr size_t Stride = Size / NumUpdates;
  uint64_t *Data = new uint64_t[Size];

  for (size_t I = 0; I < Size; ++I)
    Data[I] = 1;

  // Change a few scattered elements of a large buffer.
#pragma omp target teams distribute parallel for map(tofrom : Data[0 : Size])
  for (size_t I = 0; I < NumUpdates; ++I)
    Data[I * Stride] += I + 1;

  uint64_t Sum = 0;
  for (size_t I = 0; I < Size; ++I)
    Sum += Data[I];

  // CHECK: PASS
  if (Sum == Size + NumUpdates * (NumUpdates + 1) / 2)
    printf("PASS\n");

  delete[] Data;
}
