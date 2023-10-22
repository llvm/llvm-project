// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic

// UNSUPPORTED: amdgcn-amd-amdhsa

#include <omp.h>
#include <stdio.h>

int main() {
  // TODO: Test all ICVs on all levels
  int o_lvl = 111, i_lvl = 222, o_tid = 333, i_tid = 333, o_nt = 444,
      i_nt = 555;
#pragma omp target teams map(tofrom : o_lvl, i_lvl, o_tid, i_tid, o_nt, i_nt)  \
    num_teams(2) thread_limit(2)
  {
    if (omp_get_team_num() == 0) {
#pragma omp parallel num_threads(128)
      if (omp_get_thread_num() == omp_get_num_threads() - 1) {
        o_lvl = omp_get_level();
        o_tid = omp_get_thread_num();
        o_nt = omp_get_num_threads();
#pragma omp parallel num_threads(64)
        if (omp_get_thread_num() == omp_get_num_threads() - 1) {
          i_lvl = omp_get_level();
          i_tid = omp_get_thread_num();
          i_nt = omp_get_num_threads();
        }
      }
    }
  }
  // CHECK: outer: lvl: 1, tid: 1, nt: 2
  // CHECK: inner: lvl: 2, tid: 0, nt: 1
  printf("outer: lvl: %i, tid: %i, nt: %i\n", o_lvl, o_tid, o_nt);
  printf("inner: lvl: %i, tid: %i, nt: %i\n", i_lvl, i_tid, i_nt);
  return 0;
}
