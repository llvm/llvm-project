// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory

#include <omp.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Various definitions copied from OpenMP RTL

extern void __tgt_target_data_begin(int64_t device_id, int32_t arg_num,
                                    void **args_base, void **args,
                                    int64_t *arg_sizes, int64_t *arg_types);

extern void __tgt_target_data_end(int64_t device_id, int32_t arg_num,
                                  void **args_base, void **args,
                                  int64_t *arg_sizes, int64_t *arg_types);

// End of definitions copied from OpenMP RTL.
// ---------------------------------------------------------------------------

#pragma omp requires unified_shared_memory

#define N 1024

int main(int argc, char *argv[]) {
  int fails;
  void *host_alloc = 0, *device_alloc = 0;
  int *a = (int *)malloc(N * sizeof(int));

  // Init
  for (int i = 0; i < N; ++i) {
    a[i] = 10;
  }
  host_alloc = &a[0];

// Dummy target region that ensures the runtime library is loaded when
// the target data begin/end functions are manually called below.
#pragma omp target
  {}

  // Manual calls
  int device_id = omp_get_default_device();
  int arg_num = 1;
  void **args_base = (void **)&a;
  void **args = (void **)&a;
  int64_t arg_sizes[arg_num];

  arg_sizes[0] = sizeof(int) * N;

  int64_t arg_types[arg_num];

  // Ox400 enables the CLOSE map type in the runtime:
  // OMP_TGT_MAPTYPE_CLOSE = 0x400
  // OMP_TGT_MAPTYPE_TO    = 0x001
  arg_types[0] = 0x400 | 0x001;

  device_alloc = host_alloc;

  __tgt_target_data_begin(device_id, arg_num, args_base, args, arg_sizes,
                          arg_types);

#pragma omp target data use_device_ptr(a)
  { device_alloc = a; }

  __tgt_target_data_end(device_id, arg_num, args_base, args, arg_sizes,
                        arg_types);

  // CHECK: a was copied to the device
  if (device_alloc != host_alloc)
    printf("a was copied to the device\n");

  free(a);

  // CHECK: Done!
  printf("Done!\n");

  return 0;
}
