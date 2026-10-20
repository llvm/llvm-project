// RUN: %libomp-compile
// RUN: env OMP_TARGET_OFFLOAD=DISABLED OMP_DEFAULT_DEVICE=5 %libomp-run
//
// When offload is disabled, keep the default-device-var ICV valid:
// OMP_DEFAULT_DEVICE and omp_set_default_device() must both resolve to the
// initial device.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  int initial = omp_get_initial_device();
  int default_device = omp_get_default_device();

  if (default_device != initial) {
    fprintf(stderr,
            "FAIL: OMP_DEFAULT_DEVICE=5 left in ICV while offload disabled\n"
            "      initial=%d default=%d\n",
            initial, default_device);
    return EXIT_FAILURE;
  }

  omp_set_default_device(10);
  default_device = omp_get_default_device();
  if (default_device != initial) {
    fprintf(stderr,
            "FAIL: omp_set_default_device(10) did not store initial device\n"
            "      initial=%d default=%d\n",
            initial, default_device);
    return EXIT_FAILURE;
  }

  printf("PASS\n");
  return EXIT_SUCCESS;
}
