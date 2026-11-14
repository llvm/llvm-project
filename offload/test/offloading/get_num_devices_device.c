// Check that omp_get_num_devices() and omp_get_device_num() called from a
// target region report the host's view (device count and OpenMP device id).

// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: gpu

#include <omp.h>
#include <stdio.h>

int main() {
  int HostNum = omp_get_num_devices();
  int Fail = 0;

  for (int Dev = 0; Dev < HostNum; ++Dev) {
    int DeviceNum = -1;
    int DeviceID = -1;

#pragma omp target device(Dev) map(from : DeviceNum, DeviceID)
    {
      DeviceNum = omp_get_num_devices();
      DeviceID = omp_get_device_num();
    }

    if (DeviceNum != HostNum || DeviceID != Dev) {
      printf("FAIL: dev=%d host_num=%d device_num=%d device_id=%d\n", Dev,
             HostNum, DeviceNum, DeviceID);
      Fail = 1;
    }
  }

  // CHECK: PASS
  if (!Fail)
    printf("PASS\n");
  return 0;
}
