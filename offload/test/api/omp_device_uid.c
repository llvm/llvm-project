// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>
#include <string.h>

int test_omp_device_uid(int device_num) {
  const char *device_uid = omp_get_uid_from_device(device_num);
  if (device_uid == NULL) {
    printf("FAIL for device %d: omp_get_uid_from_device returned NULL\n",
           device_num);
    return 0;
  }

  int device_num_from_uid = omp_get_device_from_uid(device_uid);
  if (device_num_from_uid != device_num) {
    printf(
        "FAIL for device %d: omp_get_device_from_uid returned %d (UID: %s)\n",
        device_num, device_num_from_uid, device_uid);
    return 0;
  }

  if (device_num == omp_get_initial_device())
    return 1;

  int success = 1;

// Note that the following code may be executed on the host if the host is the
// device
#pragma omp target map(tofrom : success) device(device_num)
  {
    int device_num = omp_get_device_num();

    // omp_get_uid_from_device() in the device runtime is a dummy function
    // returning NULL
    const char *device_uid = omp_get_uid_from_device(device_num);

    // omp_get_device_from_uid() in the device runtime is a dummy function
    // returning omp_invalid_device.
    int device_num_from_uid = omp_get_device_from_uid(device_uid);

    // Depending on whether we're executing on the device or the host, we either
    // got NULL as the device UID or the correct device UID.  Consequently,
    // omp_get_device_from_uid() either returned omp_invalid_device or the
    // correct device number (aka omp_get_initial_device()).
    if (device_uid ? device_num_from_uid != device_num
                   : device_num_from_uid != omp_invalid_device) {
      printf("FAIL for device %d (target): omp_get_device_from_uid returned %d "
             "(UID: %s)\n",
             device_num, device_num_from_uid, device_uid);
      success = 0;
    }
  }

  return success;
}

int main() {
  int num_devices = omp_get_num_devices();
  int num_failed = 0;
  // (also test initial device aka num_devices)
  for (int i = 0; i < num_devices + 1; i++) {
    if (!test_omp_device_uid(i)) {
      printf("FAIL for device %d\n", i);
      num_failed++;
    }
  }
  if (num_failed) {
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

// CHECK: PASS
