// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory

// amdgcn does not have printf definition
// XFAIL: amdgcn-amd-amdhsa

#include <stdio.h>
#include <omp.h>

// ---------------------------------------------------------------------------
// Various definitions copied from OpenMP RTL

extern void __tgt_register_requires(int64_t);

// End of definitions copied from OpenMP RTL.
// ---------------------------------------------------------------------------

#pragma omp requires unified_shared_memory

#define N 1024

int main(int argc, char *argv[]) {
  int fails;
  void *host_alloc, *device_alloc;
  void *host_data, *device_data;
  int *alloc = (int *)malloc(N * sizeof(int));
  int data[N];

  // Manual registration of requires flags for Clang versions
  // that do not support requires.
  __tgt_register_requires(8);

  for (int i = 0; i < N; ++i) {
    alloc[i] = 10;
    data[i] = 1;
  }

  host_data = &data[0];
  host_alloc = &alloc[0];

// implicit mapping of data
#pragma omp target map(tofrom : device_data, device_alloc)
  {
    device_data = &data[0];
    device_alloc = &alloc[0];

    for (int i = 0; i < N; i++) {
      alloc[i] += 1;
      data[i] += 1;
    }
  }

  // CHECK: Address of alloc on device matches host address.
  if (device_alloc == host_alloc)
    printf("Address of alloc on device matches host address.\n");

  // CHECK: Address of data on device matches host address.
  if (device_data == host_data)
    printf("Address of data on device matches host address.\n");

  // On the host, check that the arrays have been updated.
  // CHECK: Alloc device values updated: Succeeded
  fails = 0;
  for (int i = 0; i < N; i++) {
    if (alloc[i] != 11)
      fails++;
  }
  printf("Alloc device values updated: %s\n",
         (fails == 0) ? "Succeeded" : "Failed");

  // CHECK: Data device values updated: Succeeded
  fails = 0;
  for (int i = 0; i < N; i++) {
    if (data[i] != 2)
      fails++;
  }
  printf("Data device values updated: %s\n",
         (fails == 0) ? "Succeeded" : "Failed");

  //
  // Test that updates on the host snd on the device are both visible.
  //

  // Update on the host.
  for (int i = 0; i < N; ++i) {
    alloc[i] += 1;
    data[i] += 1;
  }

#pragma omp target
  {
    // CHECK: Alloc host values updated: Succeeded
    fails = 0;
    for (int i = 0; i < N; i++) {
      if (alloc[i] != 12)
        fails++;
    }
    printf("Alloc host values updated: %s\n",
           (fails == 0) ? "Succeeded" : "Failed");
    // CHECK: Data host values updated: Succeeded
    fails = 0;
    for (int i = 0; i < N; i++) {
      if (data[i] != 3)
        fails++;
    }
    printf("Data host values updated: %s\n",
           (fails == 0) ? "Succeeded" : "Failed");
  }

  free(alloc);

  printf("Done!\n");

  return 0;
}
