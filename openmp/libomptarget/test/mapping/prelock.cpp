// RUN: %libomptarget-compilexx-run-and-check-generic

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

#include <cstdio>

#include <omp.h>

extern "C" {
void *llvm_omp_target_lock_mem(void *ptr, size_t size, int device_num);
void llvm_omp_target_unlock_mem(void *ptr, int device_num);
}

int main() {
  int n = 100;
  int *unlocked = new int[n];

  for (int i = 0; i < n; i++)
    unlocked[i] = i;

  int *locked = (int *)llvm_omp_target_lock_mem(unlocked, n * sizeof(int),
                                                omp_get_default_device());
  if (!locked)
    return 0;

#pragma omp target teams distribute parallel for map(tofrom : unlocked[ : n])
  for (int i = 0; i < n; i++)
    unlocked[i] += 1;

#pragma omp target teams distribute parallel for map(tofrom : unlocked[10 : 10])
  for (int i = 10; i < 20; i++)
    unlocked[i] += 1;

#pragma omp target teams distribute parallel for map(tofrom : locked[ : n])
  for (int i = 0; i < n; i++)
    locked[i] += 1;

#pragma omp target teams distribute parallel for map(tofrom : locked[10 : 10])
  for (int i = 10; i < 20; i++)
    locked[i] += 1;

  llvm_omp_target_unlock_mem(unlocked, omp_get_default_device());

  int err = 0;
  for (int i = 0; i < n; i++) {
    if (i < 10 || i > 19) {
      if (unlocked[i] != i + 2) {
        printf("Err at %d, got %d, expected %d\n", i, unlocked[i], i + 1);
        err++;
      }
    } else if (unlocked[i] != i + 4) {
      printf("Err at %d, got %d, expected %d\n", i, unlocked[i], i + 2);
      err++;
    }
  }

  // CHECK: PASS
  if (err == 0)
    printf("PASS\n");

  return err;
}
