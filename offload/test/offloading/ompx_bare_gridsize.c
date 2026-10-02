// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_INFO=63 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic
//
// REQUIRES: gpu

#include <ompx.h>
#include <stdio.h>

void get_gridsizes(int *nblocks, int *nthreads) {
  if (ompx_block_id_x() == 0 && ompx_thread_id_x() == 0 &&
      ompx_block_id_y() == 0 && ompx_thread_id_y() == 0 &&
      ompx_block_id_z() == 0 && ompx_thread_id_z() == 0) {
    nblocks[0] = ompx_grid_dim_x();
    nblocks[1] = ompx_grid_dim_y();
    nblocks[2] = ompx_grid_dim_z();
    nthreads[0] = ompx_block_dim_x();
    nthreads[1] = ompx_block_dim_y();
    nthreads[2] = ompx_block_dim_z();
  }
}

int main(int argc, char *argv[]) {
  int nblocks[3], nthreads[3];

  // CHECK: PluginInterface device 0 info: Launching kernel
  // CHECK-SAME: __omp_offloading_{{.*}} with [64,1,1] blocks and [32,1,1]
  // CHECK-SAME: threads in BARE mode
  nblocks[0] = nblocks[1] = nblocks[2] = nthreads[0] = nthreads[1] =
      nthreads[2] = 0;
#pragma omp target teams ompx_bare num_teams(64) thread_limit(32)              \
    map(tofrom : nblocks, nthreads)
  {
    get_gridsizes(nblocks, nthreads);
  }
  // CHECK: nblocks: 64 1 1, nthreads: 32 1 1
  fprintf(stderr, "nblocks: %d %d %d, nthreads: %d %d %d\n", nblocks[0],
          nblocks[1], nblocks[2], nthreads[0], nthreads[1], nthreads[2]);

  // CHECK: PluginInterface device 0 info: Launching kernel
  // CHECK-SAME: __omp_offloading_{{.*}} with [64,1,1] blocks and [32,4,1]
  // CHECK-SAME: threads in BARE mode
  nblocks[0] = nblocks[1] = nblocks[2] = nthreads[0] = nthreads[1] =
      nthreads[2] = 0;
#pragma omp target teams ompx_bare num_teams(64) thread_limit(32, 4)           \
    map(tofrom : nblocks, nthreads)
  {
    get_gridsizes(nblocks, nthreads);
  }
  // CHECK: nblocks: 64 1 1, nthreads: 32 4 1
  fprintf(stderr, "nblocks: %d %d %d, nthreads: %d %d %d\n", nblocks[0],
          nblocks[1], nblocks[2], nthreads[0], nthreads[1], nthreads[2]);

  // CHECK: PluginInterface device 0 info: Launching kernel
  // CHECK-SAME: __omp_offloading_{{.*}} with [64,1,1] blocks and [32,4,2]
  // CHECK-SAME: threads in BARE mode
  nblocks[0] = nblocks[1] = nblocks[2] = nthreads[0] = nthreads[1] =
      nthreads[2] = 0;
#pragma omp target teams ompx_bare num_teams(64) thread_limit(32, 4, 2)        \
    map(tofrom : nblocks, nthreads)
  {
    get_gridsizes(nblocks, nthreads);
  }
  // CHECK: nblocks: 64 1 1, nthreads: 32 4 2
  fprintf(stderr, "nblocks: %d %d %d, nthreads: %d %d %d\n", nblocks[0],
          nblocks[1], nblocks[2], nthreads[0], nthreads[1], nthreads[2]);

  // CHECK: PluginInterface device 0 info: Launching kernel
  // CHECK-SAME: __omp_offloading_{{.*}} with [64,16,1] blocks and [32,1,1]
  // CHECK-SAME: threads in BARE mode
  nblocks[0] = nblocks[1] = nblocks[2] = nthreads[0] = nthreads[1] =
      nthreads[2] = 0;
#pragma omp target teams ompx_bare num_teams(64, 16, 1) thread_limit(32)       \
    map(tofrom : nblocks, nthreads)
  {
    get_gridsizes(nblocks, nthreads);
  }
  // CHECK: nblocks: 64 16 1, nthreads: 32 1 1
  fprintf(stderr, "nblocks: %d %d %d, nthreads: %d %d %d\n", nblocks[0],
          nblocks[1], nblocks[2], nthreads[0], nthreads[1], nthreads[2]);

  // CHECK: PluginInterface device 0 info: Launching kernel
  // CHECK-SAME: __omp_offloading_{{.*}} with [64,16,1] blocks and [32,4,1]
  // CHECK-SAME: threads in BARE mode
  nblocks[0] = nblocks[1] = nblocks[2] = nthreads[0] = nthreads[1] =
      nthreads[2] = 0;
#pragma omp target teams ompx_bare num_teams(64, 16, 1) thread_limit(32, 4)    \
    map(tofrom : nblocks, nthreads)
  {
    get_gridsizes(nblocks, nthreads);
  }
  // CHECK: nblocks: 64 16 1, nthreads: 32 4 1
  fprintf(stderr, "nblocks: %d %d %d, nthreads: %d %d %d\n", nblocks[0],
          nblocks[1], nblocks[2], nthreads[0], nthreads[1], nthreads[2]);

  // CHECK: PluginInterface device 0 info: Launching kernel
  // CHECK-SAME: __omp_offloading_{{.*}} with [64,16,1] blocks and [32,4,2]
  // CHECK-SAME: threads in BARE mode
  nblocks[0] = nblocks[1] = nblocks[2] = nthreads[0] = nthreads[1] =
      nthreads[2] = 0;
#pragma omp target teams ompx_bare num_teams(64, 16, 1) thread_limit(32, 4, 2) \
    map(tofrom : nblocks, nthreads)
  {
    get_gridsizes(nblocks, nthreads);
  }
  // CHECK: nblocks: 64 16 1, nthreads: 32 4 2
  fprintf(stderr, "nblocks: %d %d %d, nthreads: %d %d %d\n", nblocks[0],
          nblocks[1], nblocks[2], nthreads[0], nthreads[1], nthreads[2]);

  // CHECK: PluginInterface device 0 info: Launching kernel
  // CHECK-SAME: __omp_offloading_{{.*}} with [64,16,8] blocks and [32,1,1]
  // CHECK-SAME: threads in BARE mode
  nblocks[0] = nblocks[1] = nblocks[2] = nthreads[0] = nthreads[1] =
      nthreads[2] = 0;
#pragma omp target teams ompx_bare num_teams(64, 16, 8) thread_limit(32)       \
    map(tofrom : nblocks, nthreads)
  {
    get_gridsizes(nblocks, nthreads);
  }
  // CHECK: nblocks: 64 16 8, nthreads: 32 1 1
  fprintf(stderr, "nblocks: %d %d %d, nthreads: %d %d %d\n", nblocks[0],
          nblocks[1], nblocks[2], nthreads[0], nthreads[1], nthreads[2]);

  // CHECK: PluginInterface device 0 info: Launching kernel
  // CHECK-SAME: __omp_offloading_{{.*}} with [64,16,8] blocks and [32,4,1]
  // CHECK-SAME: threads in BARE mode
  nblocks[0] = nblocks[1] = nblocks[2] = nthreads[0] = nthreads[1] =
      nthreads[2] = 0;
#pragma omp target teams ompx_bare num_teams(64, 16, 8) thread_limit(32, 4)    \
    map(tofrom : nblocks, nthreads)
  {
    get_gridsizes(nblocks, nthreads);
  }
  // CHECK: nblocks: 64 16 8, nthreads: 32 4 1
  fprintf(stderr, "nblocks: %d %d %d, nthreads: %d %d %d\n", nblocks[0],
          nblocks[1], nblocks[2], nthreads[0], nthreads[1], nthreads[2]);

  // CHECK: PluginInterface device 0 info: Launching kernel
  // CHECK-SAME: __omp_offloading_{{.*}} with [64,16,8] blocks and [32,4,2]
  // CHECK-SAME: threads in BARE mode
  nblocks[0] = nblocks[1] = nblocks[2] = nthreads[0] = nthreads[1] =
      nthreads[2] = 0;
#pragma omp target teams ompx_bare num_teams(64, 16, 8) thread_limit(32, 4, 2) \
    map(tofrom : nblocks, nthreads)
  {
    get_gridsizes(nblocks, nthreads);
  }
  // CHECK: nblocks: 64 16 8, nthreads: 32 4 2
  fprintf(stderr, "nblocks: %d %d %d, nthreads: %d %d %d\n", nblocks[0],
          nblocks[1], nblocks[2], nthreads[0], nthreads[1], nthreads[2]);

  return 0;
}
