// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=disabled OMP_DEFAULT_DEVICE=5 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic
//
// When offload is disabled, an invalid OMP_DEFAULT_DEVICE must not remain in
// the default-device-var ICV. omp_set_default_device() must clamp the same way,
// and a target region must still run on the host/initial device.

#include <omp.h>
#include <stdio.h>

int main(void) {
  int initial = omp_get_initial_device();
  int default_device = omp_get_default_device();

  // CHECK-NOT: out of range
  // CHECK-NOT: fatal error
  // CHECK: default_matches_initial: YES
  printf("default_matches_initial: %s\n",
         (default_device == initial) ? "YES" : "NO");

  omp_set_default_device(10);
  default_device = omp_get_default_device();
  // CHECK: after_set_matches_initial: YES
  printf("after_set_matches_initial: %s\n",
         (default_device == initial) ? "YES" : "NO");

  int on_host = 0;
#pragma omp target map(from : on_host)
  {
    on_host = omp_is_initial_device();
  }

  // CHECK: target_on_host: YES
  printf("target_on_host: %s\n", on_host ? "YES" : "NO");
  return on_host ? 0 : 1;
}
