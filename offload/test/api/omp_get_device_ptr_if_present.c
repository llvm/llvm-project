// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: unified_shared_memory

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define OFFSET 16

int main(int argc, char *argv[]) {
  // Check if we have any devices
  if (omp_get_num_devices() == 0) {
    printf("SKIP: No devices available\n");
    return 0;
  }

  int *host_data = (int *)malloc(sizeof(int) * N);
  if (!host_data) {
    printf("FAIL: Memory allocation failed\n");
    return 1;
  }

  int device_num = omp_get_default_device();

  // Initialize data
  for (int i = 0; i < N; i++) {
    host_data[i] = i;
  }

  // Test 1: NULL pointer should return NULL
  void *result = __omp_get_device_ptr_if_present(NULL, device_num);
  if (result != NULL) {
    printf("FAIL: NULL input should return NULL\n");
    free(host_data);
    return 1;
  }

  // Test 2: Invalid device number should return original pointer
  result = __omp_get_device_ptr_if_present(host_data, -1);
  if (result != host_data) {
    printf("FAIL: Invalid device should return original pointer\n");
    free(host_data);
    return 1;
  }

  result =
      __omp_get_device_ptr_if_present(host_data, omp_get_num_devices() + 1);
  if (result != host_data) {
    printf("FAIL: Out of range device should return original pointer\n");
    free(host_data);
    return 1;
  }

  // Test 3: Unmapped pointer should return original pointer
  result = __omp_get_device_ptr_if_present(host_data, device_num);
  if (result != host_data) {
    printf("FAIL: Unmapped pointer should return original\n");
    free(host_data);
    return 1;
  }

  // Map data to device
#pragma omp target enter data map(to : host_data[:N])

  // Test 4: Mapped pointer should return device pointer
  result = __omp_get_device_ptr_if_present(host_data, device_num);
  void *expected_device_ptr = omp_get_mapped_ptr(host_data, device_num);
  if (result != expected_device_ptr) {
    printf("FAIL: Should return device pointer for mapped data\n");
#pragma omp target exit data map(delete : host_data[:N])
    free(host_data);
    return 1;
  }

  // Test 5: Pointer with offset should also work
  result = __omp_get_device_ptr_if_present(host_data + OFFSET, device_num);
  expected_device_ptr = omp_get_mapped_ptr(host_data + OFFSET, device_num);
  if (result != expected_device_ptr) {
    printf("FAIL: Should handle offset correctly\n");
#pragma omp target exit data map(delete : host_data[:N])
    free(host_data);
    return 1;
  }

  // Test 6: Verify device pointer works in target region
  void *ptr_from_device = NULL;
  void *ptr_from_api = __omp_get_device_ptr_if_present(host_data, device_num);

#pragma omp target map(from : ptr_from_device)
  { ptr_from_device = host_data; }

  if (ptr_from_device != ptr_from_api) {
    printf("FAIL: Device pointer should match in target region\n");
#pragma omp target exit data map(delete : host_data[:N])
    free(host_data);
    return 1;
  }

  // Unmap data
#pragma omp target exit data map(delete : host_data[:N])

  // Test 7: After unmapping, should return original pointer again
  result = __omp_get_device_ptr_if_present(host_data, device_num);
  if (result != host_data) {
    printf("FAIL: After unmapping should return original pointer\n");
    free(host_data);
    return 1;
  }

  // Test 8: Already device pointer scenario
  void *device_alloc = omp_target_alloc(sizeof(int) * N, device_num);
  if (device_alloc) {
    result = __omp_get_device_ptr_if_present(device_alloc, device_num);
    if (result != device_alloc) {
      printf("FAIL: Device-allocated pointer should return as-is\n");
      omp_target_free(device_alloc, device_num);
      free(host_data);
      return 1;
    }
    omp_target_free(device_alloc, device_num);
  }

  free(host_data);
  printf("PASS\n");
  return 0;
}
