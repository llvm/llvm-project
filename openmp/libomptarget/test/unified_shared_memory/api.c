// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: nvptx64-nvidia-cuda
// XFAIL: nvptx64-nvidia-cuda-newRTL

// Fails on amdgcn with error: GPU Memory Error
// XFAIL: amdgcn-amd-amdhsa
// XFAIL: amdgcn-amd-amdhsa-newRTL

#include <stdio.h>
#include <omp.h>

// ---------------------------------------------------------------------------
// Various definitions copied from OpenMP RTL

extern void __tgt_register_requires(int64_t);

// End of definitions copied from OpenMP RTL.
// ---------------------------------------------------------------------------

#pragma omp requires unified_shared_memory

#define N 1024

void init(int A[], int B[], int C[]) {
  for (int i = 0; i < N; ++i) {
    A[i] = 0;
    B[i] = 1;
    C[i] = i;
  }
}

int main(int argc, char *argv[]) {
  const int device = omp_get_default_device();

  // Manual registration of requires flags for Clang versions
  // that do not support requires.
  __tgt_register_requires(8);

  // CHECK: Initial device: [[INITIAL_DEVICE:[0-9]+]]
  printf("Initial device: %d\n", omp_get_initial_device());
  // CHECK: Num devices: [[INITIAL_DEVICE]]
  printf("Num devices: %d\n", omp_get_num_devices());

  //
  // Target alloc & target memcpy
  //
  int A[N], B[N], C[N];

  // Init
  init(A, B, C);

  int *pA, *pB, *pC;

  // map ptrs
  pA = &A[0];
  pB = &B[0];
  pC = &C[0];

  int *d_A = (int *)omp_target_alloc(N * sizeof(int), device);
  int *d_B = (int *)omp_target_alloc(N * sizeof(int), device);
  int *d_C = (int *)omp_target_alloc(N * sizeof(int), device);

  // CHECK: omp_target_alloc succeeded
  printf("omp_target_alloc %s\n", d_A && d_B && d_C ? "succeeded" : "failed");

  omp_target_memcpy(d_B, pB, N * sizeof(int), 0, 0, device,
                    omp_get_initial_device());
  omp_target_memcpy(d_C, pC, N * sizeof(int), 0, 0, device,
                    omp_get_initial_device());

#pragma omp target is_device_ptr(d_A, d_B, d_C) device(device)
  {
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < N; i++) {
      d_A[i] = d_B[i] + d_C[i] + 1;
    }
  }

  omp_target_memcpy(pA, d_A, N * sizeof(int), 0, 0, omp_get_initial_device(),
                    device);

  // CHECK: Test omp_target_memcpy: Succeeded
  int fail = 0;
  for (int i = 0; i < N; ++i) {
    if (A[i] != i + 2)
      fail++;
  }
  if (fail) {
    printf("Test omp_target_memcpy: Failed\n");
  } else {
    printf("Test omp_target_memcpy: Succeeded\n");
  }

  //
  // target_is_present and target_associate/disassociate_ptr
  //
  init(A, B, C);

  // CHECK: B is not present, associating it...
  // CHECK: omp_target_associate_ptr B succeeded
  if (!omp_target_is_present(B, device)) {
    printf("B is not present, associating it...\n");
    int rc = omp_target_associate_ptr(B, d_B, N * sizeof(int), 0, device);
    printf("omp_target_associate_ptr B %s\n", !rc ? "succeeded" : "failed");
  }

  // CHECK: C is not present, associating it...
  // CHECK: omp_target_associate_ptr C succeeded
  if (!omp_target_is_present(C, device)) {
    printf("C is not present, associating it...\n");
    int rc = omp_target_associate_ptr(C, d_C, N * sizeof(int), 0, device);
    printf("omp_target_associate_ptr C %s\n", !rc ? "succeeded" : "failed");
  }

// CHECK: Inside target data: A is not present
// CHECK: Inside target data: B is present
// CHECK: Inside target data: C is present
#pragma omp target data map(from : B, C) device(device)
  {
    printf("Inside target data: A is%s present\n",
           omp_target_is_present(A, device) ? "" : " not");
    printf("Inside target data: B is%s present\n",
           omp_target_is_present(B, device) ? "" : " not");
    printf("Inside target data: C is%s present\n",
           omp_target_is_present(C, device) ? "" : " not");

#pragma omp target map(from : A) device(device)
    {
#pragma omp parallel for schedule(static, 1)
      for (int i = 0; i < N; i++)
        A[i] = B[i] + C[i] + 1;
    }
  }

  // CHECK: B is present, disassociating it...
  // CHECK: omp_target_disassociate_ptr B succeeded
  // CHECK: C is present, disassociating it...
  // CHECK: omp_target_disassociate_ptr C succeeded
  if (omp_target_is_present(B, device)) {
    printf("B is present, disassociating it...\n");
    int rc = omp_target_disassociate_ptr(B, device);
    printf("omp_target_disassociate_ptr B %s\n", !rc ? "succeeded" : "failed");
  }
  if (omp_target_is_present(C, device)) {
    printf("C is present, disassociating it...\n");
    int rc = omp_target_disassociate_ptr(C, device);
    printf("omp_target_disassociate_ptr C %s\n", !rc ? "succeeded" : "failed");
  }

  // CHECK: Test omp_target_associate_ptr: Succeeded
  fail = 0;
  for (int i = 0; i < N; ++i) {
    if (A[i] != i + 2)
      fail++;
  }
  if (fail) {
    printf("Test omp_target_associate_ptr: Failed\n");
  } else {
    printf("Test omp_target_associate_ptr: Succeeded\n");
  }

  omp_target_free(d_A, device);
  omp_target_free(d_B, device);
  omp_target_free(d_C, device);

  printf("Done!\n");

  return 0;
}
