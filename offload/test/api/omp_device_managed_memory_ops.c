// RUN: %libomptarget-compile-run-and-check-generic

// Check the data movement API on device managed memory.

#include <omp.h>
#include <stdio.h>

void *llvm_omp_target_alloc_shared(size_t, int);
void llvm_omp_target_free_shared(void *, int);

int main() {
  const int N = 128;
  const int Device = omp_get_default_device();
  const int Host = omp_get_initial_device();

  int *Shared = llvm_omp_target_alloc_shared(N * sizeof(int), Device);
  int Buffer[N];
  int Failures = 0;

  if (!Shared) {
    printf("FAIL: allocation\n");
    return 1;
  }

  // The host can access the allocation directly.
  for (int I = 0; I < N; ++I)
    Shared[I] = I;

  // The device can access the allocation directly.
#pragma omp target teams distribute parallel for device(Device)                \
    is_device_ptr(Shared)
  for (int I = 0; I < N; ++I)
    Shared[I] += 1;

  for (int I = 0; I < N; ++I)
    Failures += (Shared[I] != I + 1);

  // Filling the whole allocation.
  if (omp_target_memset(Shared, 0, N * sizeof(int), Device) != Shared)
    ++Failures;
  for (int I = 0; I < N; ++I)
    Failures += (Shared[I] != 0);

  // Filling a subrange of the allocation.
  if (omp_target_memset(&Shared[1], 0xFF, sizeof(int), Device) != &Shared[1])
    ++Failures;
  Failures += (Shared[0] != 0) + (Shared[1] != -1) + (Shared[2] != 0);

  // Copying into and out of the allocation.
  for (int I = 0; I < N; ++I)
    Buffer[I] = 2 * I;
  Failures += (omp_target_memcpy(Shared, Buffer, N * sizeof(int), 0, 0, Device,
                                 Host) != 0);
  for (int I = 0; I < N; ++I)
    Buffer[I] = 0;
  Failures += (omp_target_memcpy(Buffer, Shared, N * sizeof(int), 0, 0, Host,
                                 Device) != 0);
  for (int I = 0; I < N; ++I)
    Failures += (Buffer[I] != 2 * I);

  llvm_omp_target_free_shared(Shared, Device);

  // CHECK: PASS
  if (!Failures)
    printf("PASS\n");
  else
    printf("FAIL: %d\n", Failures);
}
