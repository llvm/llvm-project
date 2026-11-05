// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.dir %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: sh -c 'wc -c < %t.dir/record_blocks.0.store > %t.size'
// Recording again into the directory does not store the blocks again.
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.dir %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: sh -c 'test $(wc -c < %t.dir/record_blocks.0.store) -eq $(cat %t.size)'
// Remove the first kernel, whose input blocks no other snapshot uses, and
// compact the store into a new generation.
// RUN: ls -tr %t.dir/*.json | head -n 1 | sed 's/\.json$//' | xargs -I {} sh -c 'rm {}.*'
// RUN: %omp-kernel-replay --compact-store %t.dir | FileCheck --check-prefix=COMPACT %s
// RUN: not test -e %t.dir/record_blocks.0.store
// RUN: sh -c 'test $(wc -c < %t.dir/record_blocks.1.store) -lt $(( $(cat %t.size) * 3 / 4 ))'
// RUN: ls %t.dir/*.json | xargs -I {} %omp-kernel-replay --verify {} | FileCheck --check-prefix=VERIFY %s
// Compacting again keeps the same blocks, and recording appends to the latest
// generation.
// RUN: %omp-kernel-replay --compact-store %t.dir | FileCheck --check-prefix=COMPACT %s
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.dir %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: test -f %t.dir/record_blocks.2.store
// RUN: not test -e %t.dir/record_blocks.1.store
// RUN: ls %t.dir/*.json | xargs -I {} %omp-kernel-replay --verify {} | FileCheck --check-prefix=VERIFY %s
// A snapshot whose block store is missing stops the compaction.
// RUN: rm %t.dir/record_blocks.2.store
// RUN: not %omp-kernel-replay --compact-store %t.dir 2>&1 | FileCheck --check-prefix=MISSING %s
// clang-format on

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

// COMPACT: Compacted the block store of {{[1-9][0-9]*}} snapshots from {{[0-9]+}} to {{[0-9]+}} bytes
// VERIFY: Replay done, device memory verified
// MISSING: [llvm-omp-kernel-replay] Error: {{.*}}.record_input refers to a missing block store

#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
  constexpr size_t Size = 256 * 1024;
  uint64_t *X = new uint64_t[Size];
  for (size_t I = 0; I < Size; ++I)
    X[I] = (I + 1) * 0x9E3779B97F4A7C15ULL;

  // The first kernel changes all of X, so its input blocks only appear in its
  // own snapshot.
  uint64_t Sum = 0;
#pragma omp target data map(tofrom : X[0 : Size])
  {
#pragma omp target teams distribute parallel for
    for (size_t I = 0; I < Size; ++I)
      X[I] ^= I;
#pragma omp target teams distribute parallel for reduction(+ : Sum)
    for (size_t I = 0; I < Size; ++I)
      Sum += X[I] & 1;
  }

  // CHECK: PASS
  if (Sum <= Size)
    printf("PASS\n");

  delete[] X;
}
