// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: unified_shared_memory

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define OFFSET 16

int main(int argc, char *argv[]) {
  int *host_data = (int *)malloc(sizeof(int) * N);
  int device_num = omp_get_default_device();

  // Initialize data
  for (int i = 0; i < N; i++) {
    host_data[i] = i;
  }

  // Test 1: NULL pointer should return NULL
  void *result = omp_get_device_ptr_if_present(NULL, device_num);
  assert(result == NULL && "NULL input should return NULL");

  // Test 2: Invalid device number should return original pointer
  result = omp_get_device_ptr_if_present(host_data, -1);
  assert(result == host_data &&
         "Invalid device should return original pointer");

  result = omp_get_device_ptr_if_present(host_data, omp_get_num_devices() + 1);
  assert(result == host_data &&
         "Out of range device should return original pointer");

  // Test 3: Unmapped pointer should return original pointer
  result = omp_get_device_ptr_if_present(host_data, device_num);
  assert(result == host_data && "Unmapped pointer should return original");

  // Map data to device
#pragma omp target enter data map(to : host_data[:N])

  // Test 4: Mapped pointer should return device pointer
  result = omp_get_device_ptr_if_present(host_data, device_num);
  void *expected_device_ptr = omp_get_mapped_ptr(host_data, device_num);
  assert(result == expected_device_ptr &&
         "Should return device pointer for mapped data");
  assert(result != host_data &&
         "Device pointer should differ from host pointer");

  // Test 5: Pointer with offset should also work
  result = omp_get_device_ptr_if_present(host_data + OFFSET, device_num);
  expected_device_ptr = omp_get_mapped_ptr(host_data + OFFSET, device_num);
  assert(result == expected_device_ptr && "Should handle offset correctly");

  // Test 6: Verify device pointer works in target region
  void *ptr_from_device = NULL;
  void *ptr_from_api = omp_get_device_ptr_if_present(host_data, device_num);

#pragma omp target map(from : ptr_from_device)
  { ptr_from_device = host_data; }

  assert(ptr_from_device == ptr_from_api &&
         "Device pointer should match in target region");

  // Unmap data
#pragma omp target exit data map(delete : host_data[:N])

  // Test 7: After unmapping, should return original pointer again
  result = omp_get_device_ptr_if_present(host_data, device_num);
  assert(result == host_data &&
         "After unmapping should return original pointer");

  // Test 8: Already device pointer scenario
  // Allocate directly on device
  void *device_alloc = omp_target_alloc(sizeof(int) * N, device_num);
  if (device_alloc) {
    // This pointer is not mapped, so should return as-is
    result = omp_get_device_ptr_if_present(device_alloc, device_num);
    assert(result == device_alloc &&
           "Device-allocated pointer should return as-is");
    omp_target_free(device_alloc, device_num);
  }

  free(host_data);
  printf("PASS\n");
  return 0;
}
