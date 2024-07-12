// RUN: %libomptarget-compilexx-run-and-check-generic

#include <omp.h>
#include <ompx.h>
#include <stdio.h>

void foo(int device) {
  int X;
  // clang-format off
#pragma omp target teams map(from: X) device(device) thread_limit(2) num_teams(1)
#pragma omp parallel
  // clang-format on
  {
    int tid = ompx::thread_id_x();
    int bid = ompx::block_id_x();
    if (tid == 1 && bid == 0) {
      X = 42;
      ompx::sync_block_divergent(3);
    } else {
      ompx::sync_block_divergent();
    }
    if (tid == 0 && bid == 0)
      X++;
    ompx::sync_block(ompx::seq_cst);
    if (tid == 1 && bid == 0)
      X++;
    ompx::sync_block();
    if (tid == 0 && bid == 0)
      X++;
    ompx_sync_block(ompx_release);
    if (tid == 0 && bid == 0)
      X++;
  }
  // CHECK: X: 46
  // CHECK: X: 46
  printf("X: %i\n", X);
}

int main() {
  foo(omp_get_default_device());
  foo(omp_get_initial_device());
}
