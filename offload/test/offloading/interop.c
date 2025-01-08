// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: nvptx64-nvidia-cuda

#include <assert.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef void *cudaStream_t;

int main() {

  int device_id = omp_get_default_device();

#pragma omp parallel master
  {

    double D0, D2;
    omp_interop_t interop;

#pragma omp interop init(targetsync : interop) device(device_id) nowait
    assert(interop != NULL);

    int err;
    for (int i = omp_ipr_first; i < 0; i++) {
      const char *n =
          omp_get_interop_name(interop, (omp_interop_property_t)(i));
      long int li =
          omp_get_interop_int(interop, (omp_interop_property_t)(i), &err);
      const void *p =
          omp_get_interop_ptr(interop, (omp_interop_property_t)(i), &err);
      const char *s =
          omp_get_interop_str(interop, (omp_interop_property_t)(i), &err);
      const char *n1 =
          omp_get_interop_type_desc(interop, (omp_interop_property_t)(i));
    }
#pragma omp interop use(interop) depend(in : D0, D2)

    cudaStream_t stream =
        (omp_get_interop_ptr(interop, omp_ipr_targetsync, NULL));
    assert(stream != NULL);

#pragma omp interop destroy(interop) depend(in : D0, D2) device(device_id)
  }
  printf("PASS\n");
}
// CHECK: PASS
