// RUN: %libomptarget-compilexx-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-x86_64-pc-linux-gnu

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 128

int main(void) {
  int num_d = omp_get_num_devices();
  int d[num_d];
  int h = omp_get_initial_device();
  double *mem_dev_src;
  double *mem_dev_dst;

  int rc = -1;

  if (num_d < 1) {
    printf("no device in system\n");
  } else if (num_d == 1) {
    printf("only one device in system\n");
  } else {

// access all device number and offset
#pragma omp target parallel for
    for (int i = 0; i < num_d; i++) {
      d[i] = omp_get_device_num();
    }

    // memory allocation and initialization
    mem_dev_src = (double *)omp_target_alloc(sizeof(double) * N, d[0]);
    if (mem_dev_src == NULL) {
      printf("mem allocation in src device failed\n");
      return -1;
    }
    for (int i = 0; i < N; i++) {
      mem_dev_src[i] = (double)rand();
    }
    mem_dev_dst = (double *)omp_target_alloc(sizeof(double) * N, d[1]);
    if (mem_dev_dst == NULL) {
      printf("mem allocation in src device failed\n");
      return -1;
    }

    rc = omp_target_memcpy(mem_dev_dst, mem_dev_src, N, 0, 0, d[1], d[0]);
  }
  return rc;
}