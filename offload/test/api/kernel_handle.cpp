// RUN: %libomptarget-compilexx-run-and-check-generic

#include <omp.h>
#include <omptarget.h>
#include <stdio.h>

int main() {
  int Device = omp_get_num_devices() ? omp_get_default_device()
                                     : omp_get_initial_device();
  if (omp_get_num_devices()) {
    int X = 0;
#pragma omp target map(tofrom : X) device(Device)
    {
      X = 1;
    }
  }

  __tgt_kernel_handle Kernel = reinterpret_cast<void *>(1);
  // CHECK: null name failed
  if (__tgt_get_kernel_handle(Device, nullptr, &Kernel) == OFFLOAD_FAIL)
    printf("null name failed\n");

  // CHECK: null output failed
  if (__tgt_get_kernel_handle(Device, "missing_kernel", nullptr) ==
      OFFLOAD_FAIL)
    printf("null output failed\n");

  // CHECK: missing kernel failed
  if (__tgt_get_kernel_handle(Device, "missing_kernel", &Kernel) ==
          OFFLOAD_FAIL &&
      Kernel == nullptr)
    printf("missing kernel failed\n");

  __tgt_kernel_properties Properties;
  // CHECK: null properties failed
  if (__tgt_get_kernel_properties(nullptr, &Properties) == OFFLOAD_FAIL)
    printf("null properties failed\n");
}
