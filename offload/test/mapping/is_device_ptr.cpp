// RUN: %libomptarget-compilexx-run-and-check-generic

#include <assert.h>
#include <iostream>
#include <omp.h>

struct view {
  const int size = 10;
  int *data_host;
  int *data_device;
  void foo() {
    std::size_t bytes = size * sizeof(int);
    const int host_id = omp_get_initial_device();
    const int device_id = omp_get_default_device();
    data_host = (int *)malloc(bytes);
    data_device = (int *)omp_target_alloc(bytes, device_id);
#pragma omp target teams distribute parallel for is_device_ptr(data_device)
    for (int i = 0; i < size; ++i)
      data_device[i] = i;
    omp_target_memcpy(data_host, data_device, bytes, 0, 0, host_id, device_id);
    for (int i = 0; i < size; ++i)
      assert(data_host[i] == i);
  }
};

int main() {
  view a;
  a.foo();
  // CHECK: PASSED
  printf("PASSED\n");
}
