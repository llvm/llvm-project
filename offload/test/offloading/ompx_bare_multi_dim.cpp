// RUN: %libomptarget-compilexx-generic
// RUN: env LIBOMPTARGET_INFO=63 %libomptarget-run-generic 2>&1 | %fcheck-generic
// REQUIRES: gpu

#include <ompx.h>

#include <cassert>
#include <vector>

// CHECK: "PluginInterface" device 0 info: Launching kernel __omp_offloading_{{.*}} with [2,4,6] blocks and [32,4,2] threads in SPMD mode

int main(int argc, char *argv[]) {
  int bs[3] = {32u, 4u, 2u};
  int gs[3] = {2u, 4u, 6u};
  int n = bs[0] * bs[1] * bs[2] * gs[0] * gs[1] * gs[2];
  std::vector<int> x_buf(n);
  std::vector<int> y_buf(n);
  std::vector<int> z_buf(n);

  auto x = x_buf.data();
  auto y = y_buf.data();
  auto z = z_buf.data();
  for (int i = 0; i < n; ++i) {
    x[i] = i;
    y[i] = i + 1;
  }

#pragma omp target teams ompx_bare num_teams(gs[0], gs[1], gs[2])              \
    thread_limit(bs[0], bs[1], bs[2]) map(to : x[ : n], y[ : n])               \
    map(from : z[ : n])
  {
    int tid_x = ompx_thread_id_x();
    int tid_y = ompx_thread_id_y();
    int tid_z = ompx_thread_id_z();
    int gid_x = ompx_block_id_x();
    int gid_y = ompx_block_id_y();
    int gid_z = ompx_block_id_z();
    int bs_x = ompx_block_dim_x();
    int bs_y = ompx_block_dim_y();
    int bs_z = ompx_block_dim_z();
    int bs = bs_x * bs_y * bs_z;
    int gs_x = ompx_grid_dim_x();
    int gs_y = ompx_grid_dim_y();
    int gid = gid_z * gs_y * gs_x + gid_y * gs_x + gid_x;
    int tid = tid_z * bs_x * bs_y + tid_y * bs_x + tid_x;
    int i = gid * bs + tid;
    z[i] = x[i] + y[i];
  }

  for (int i = 0; i < n; ++i) {
    if (z[i] != (2 * i + 1))
      return 1;
  }

  return 0;
}
