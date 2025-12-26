// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=disabled %libomptarget-run-generic 2>&1 |
// %fcheck-generic
//
// Integration test: omp_set_default_device() should not cause crashes
// and target regions should work when offload is disabled

#include <omp.h>
#include <stdio.h>

int main() {
  // Force runtime initialization
#pragma omp parallel
  {
  }

  int initial_device = omp_get_initial_device();

  // CHECK: initial_device: [[INITIAL:[0-9]+]]
  printf("initial_device: %d\n", initial_device);

  // Explicitly set high device number that would be invalid
  omp_set_default_device(10);

  int default_device = omp_get_default_device();

  // Should return initial_device (not 10)
  // CHECK: default_device: [[INITIAL]]
  printf("default_device: %d\n", default_device);

  // Verify target region works with device clause using default device
  int executed = 0;
#pragma omp target device(omp_get_default_device()) map(tofrom : executed)
  {
    executed = 1;
  }

  // CHECK: Target with device clause: PASS
  printf("Target with device clause: %s\n", executed ? "PASS" : "FAIL");

  // Try different device numbers
  for (int i = 0; i < 5; i++) {
    omp_set_default_device(i * 10);
    int dev = omp_get_default_device();
    if (dev != initial_device) {
      printf("FAIL at iteration %d\n", i);
      return 1;
    }
  }

  // CHECK: Multiple sets: PASS
  printf("Multiple sets: PASS\n");

  // Target region should execute on host (initial device)
  int on_host = 0;
#pragma omp target map(from : on_host)
  {
    on_host = omp_is_initial_device();
  }

  // CHECK: Executes on host: YES
  printf("Executes on host: %s\n", on_host ? "YES" : "NO");

  return 0;
}
