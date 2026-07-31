// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// A `declare target` variable is in the device data environment for its whole
// extent, so omp_target_is_present must report it as present for every byte of
// it, not just for its first few bytes.
//
// Under unified shared memory such a variable is represented on the device by a
// reference pointer to the host storage, and the offload entry for it describes
// that *pointer*: it gives neither the variable's address nor its extent.
//
// Under unified shared memory such a variable is represented on the device by a
// reference pointer to the host storage, and code generation communicates the
// variable's own extent so the runtime can register its storage.

#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

#pragma omp begin declare target
int scalar = 111;
int arr[64] = {0};
#pragma omp end declare target

static int present(void *P) {
  return omp_target_is_present(P, omp_get_default_device());
}

int main() {
  // Make sure the device image (and with it the declare-target registration)
  // has been loaded before querying presence.
#pragma omp target
  {
  }

  // CHECK: scalar present: 1
  printf("scalar present: %d\n", present(&scalar));

  // Every element of a declare-target array is present, including the last one.
  // CHECK: arr present first/mid/last: 1 1 1
  printf("arr present first/mid/last: %d %d %d\n", present(&arr[0]),
         present(&arr[32]), present(&arr[63]));

  int fails = 0;
  for (int i = 0; i < 64; ++i)
    if (!present(&arr[i]))
      fails++;
  // CHECK: arr present for all elements: Succeeded
  printf("arr present for all elements: %s\n",
         (fails == 0) ? "Succeeded" : "Failed");

  // CHECK: Done!
  printf("Done!\n");
  return 0;
}
