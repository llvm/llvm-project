// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// UNSUPPORTED: amdgcn-amd-amdhsa


#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

#define N 1024

int main(int argc, char *argv[]) {
  int fails;
  void *host_alloc, *device_alloc;
  void *host_data, *device_data;
  int *alloc = (int *)malloc(N * sizeof(int));
  int data[N];

  for (int i = 0; i < N; ++i) {
    alloc[i] = 10;
    data[i] = 1;
  }

  host_data = &data[0];
  host_alloc = &alloc[0];

//
// Test that updates on the device are not visible to host
// when only a TO mapping is used.
//
#pragma omp target map(tofrom                                                  \
                       : device_data, device_alloc) map(close, to              \
                                                        : alloc[:N], data      \
                                                        [:N])
  {
    device_data = &data[0];
    device_alloc = &alloc[0];

    for (int i = 0; i < N; i++) {
      alloc[i] += 1;
      data[i] += 1;
    }
  }

  // CHECK: Address of alloc on device different from host address.
  if (device_alloc != host_alloc)
    printf("Address of alloc on device different from host address.\n");

  // CHECK: Address of data on device different from host address.
  if (device_data != host_data)
    printf("Address of data on device different from host address.\n");

  // On the host, check that the arrays have been updated.
  // CHECK: Alloc host values not updated: Succeeded
  fails = 0;
  for (int i = 0; i < N; i++) {
    if (alloc[i] != 10)
      fails++;
  }
  printf("Alloc host values not updated: %s\n",
         (fails == 0) ? "Succeeded" : "Failed");

  // CHECK: Data host values not updated: Succeeded
  fails = 0;
  for (int i = 0; i < N; i++) {
    if (data[i] != 1)
      fails++;
  }
  printf("Data host values not updated: %s\n",
         (fails == 0) ? "Succeeded" : "Failed");

  //
  // Test that updates on the device are visible on host
  // when a from is used.
  //

  for (int i = 0; i < N; i++) {
    alloc[i] += 1;
    data[i] += 1;
  }

#pragma omp target map(close, tofrom : alloc[:N], data[:N])
  {
    // CHECK: Alloc device values are correct: Succeeded
    fails = 0;
    for (int i = 0; i < N; i++) {
      if (alloc[i] != 11)
        fails++;
    }
    printf("Alloc device values are correct: %s\n",
           (fails == 0) ? "Succeeded" : "Failed");
    // CHECK: Data device values are correct: Succeeded
    fails = 0;
    for (int i = 0; i < N; i++) {
      if (data[i] != 2)
        fails++;
    }
    printf("Data device values are correct: %s\n",
           (fails == 0) ? "Succeeded" : "Failed");

    // Update values on the device
    for (int i = 0; i < N; i++) {
      alloc[i] += 1;
      data[i] += 1;
    }
  }

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

  free(alloc);

  // CHECK: Done!
  printf("Done!\n");

  return 0;
}
