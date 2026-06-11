// RUN: %libomp-compile
// RUN: env OMP_NUM_THREADS_ALL=8 OMP_NUM_THREADS_DEV_0=128 %libomp-run \
// RUN:   | FileCheck %s
//
// Demo: host=8 (_ALL), device 0=128 (_DEV_0), other devices=8 (_ALL fallback).

#include <omp.h>
#include <stdio.h>

extern const char *__kmpc_get_resolved_device_env(const char *name,
                                                  int device_id);

int main(void) {
  int host_max = omp_get_max_threads();
  printf("host omp_get_max_threads() = %d\n", host_max);
  for (int d = 0; d < 3; ++d) {
    const char *v = __kmpc_get_resolved_device_env("OMP_NUM_THREADS", d);
    printf("device %d resolved OMP_NUM_THREADS = %s\n", d, v ? v : "(default)");
  }
  return 0;
}

// CHECK: host omp_get_max_threads() = 8
// CHECK: device 0 resolved OMP_NUM_THREADS = 128
// CHECK: device 1 resolved OMP_NUM_THREADS = 8
// CHECK: device 2 resolved OMP_NUM_THREADS = 8
