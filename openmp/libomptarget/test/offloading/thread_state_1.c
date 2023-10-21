// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic

// These are supported and work, but we compute bogus results on the GPU. For
// now we disable the CPU and enable it once the GPU is fixed.
//
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

#include <omp.h>
#include <stdio.h>

int main() {
  // TODO: Test all ICVs
  int lvl = 333, tid = 666, nt = 999;
#pragma omp target teams map(tofrom : lvl, tid, nt) num_teams(2)
  {
    if (omp_get_team_num() == 0) {
#pragma omp parallel num_threads(128)
      if (omp_get_thread_num() == 17) {
#pragma omp parallel num_threads(64)
        if (omp_get_thread_num() == omp_get_num_threads() - 1) {
          lvl = omp_get_level();
          tid = omp_get_thread_num();
          nt = omp_get_num_threads();
        }
      }
    }
  }
  // TODO: This is wrong, but at least it doesn't crash
  // CHECK: lvl: 333, tid: 666, nt: 999
  printf("lvl: %i, tid: %i, nt: %i\n", lvl, tid, nt);
  return 0;
}
