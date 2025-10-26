// RUN: %libomptarget-compile-amdgcn-amd-amdhsa
// RUN:   %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-amdgcn-amd-amdhsa -check-prefixes=AMD

// RUN: %libomptarget-compile-nvptx64-nvidia-cuda
// RUN:   %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-nvptx64-nvidia-cuda -check-prefixes=NVIDIA

// REQUIRES: gpu
// XFAIL: nvptx64-nvidia-cuda

#include <omp.h>
#include <stdio.h>

const char *interop_int_to_string(const int interop_int) {
  switch (interop_int) {
  case 1:
    return "cuda";
  case 2:
    return "cuda_driver";
  case 3:
    return "opencl";
  case 4:
    return "sycl";
  case 5:
    return "hip";
  case 6:
    return "level_zero";
  case 7:
    return "hsa";
  default:
    return "unknown";
  }
}

int main(int argc, char **argv) {

  // Loop over all available devices
  for (int id = 0; id < omp_get_num_devices(); ++id) {
    omp_interop_t iobj = omp_interop_none;

    // TODO: Change targetsync to target when AMD toolchain supports it.
#pragma omp interop init(target : iobj) device(id)

    int err;
    int interop_int = omp_get_interop_int(iobj, omp_ipr_fr_id, &err);

    if (err) {
      fprintf(stderr, "omp_get_interop_int failed: %d\n", err);
      return -1;
    }

    // AMD: {{.*}} hsa
    // NVIDIA: {{.*}} cuda
    printf("omp_get_interop_int returned %s\n",
           interop_int_to_string(interop_int));

    const char *interop_vendor =
        omp_get_interop_str(iobj, omp_ipr_vendor_name, &err);
    if (err) {
      fprintf(stderr, "omp_get_interop_str failed: %d\n", err);
      return -1;
    }

    // AMD: {{.*}} amd
    // NVIDIA: {{.*}} nvidia
    printf("omp_get_interop_str returned %s\n", interop_vendor);

    const char *interop_fr_name =
        omp_get_interop_str(iobj, omp_ipr_fr_name, &err);
    if (err) {
      fprintf(stderr, "omp_get_interop_str failed: %d\n", err);
      return -1;
    }

    // AMD: {{.*}} hsa
    // NVIDIA: {{.*}} cuda
    printf("omp_get_interop_str returned %s\n", interop_fr_name);

#pragma omp interop destroy(iobj)
  }
  return 0;
}
