// RUN: %libomptarget-compile-and-run-generic

#include <assert.h>
#include <omp.h>
#include <stdlib.h>

#define N 1024
#define OFFSET 16

int main(int argc, char *argv[]) {
  int *host_data = (int *)malloc(sizeof(int) * N);
  void *device_ptr = omp_get_mapped_ptr(host_data, 0);

  assert(device_ptr == NULL && "the pointer should not be mapped right now");

#pragma omp target enter data map(to: host_data[:N])

  device_ptr = omp_get_mapped_ptr(host_data, 0);

  assert(device_ptr && "the pointer should be mapped now");

  void *ptr = NULL;

#pragma omp target map(from: ptr)
  { ptr = host_data; }

  assert(ptr == device_ptr && "wrong pointer mapping");

  device_ptr = omp_get_mapped_ptr(host_data + OFFSET, 0);

  assert(device_ptr && "the pointer with offset should be mapped");

#pragma omp target map(from: ptr)
  { ptr = host_data + OFFSET; }

  assert(ptr == device_ptr && "wrong pointer mapping");

  return 0;
}
