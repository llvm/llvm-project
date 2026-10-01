// RUN: %libomptarget-compile-generic -fopenmp-version=61 && \
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic --implicit-check-not='declined taskgraph'

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental
// REQUIRES: libomptarget-debug

// A kernel launch outside a graph gets a launch environment of its own when it
// needs one: for a cross-team reduction's buffer, and for dynamic group memory
// and its fallback.  A device graph has to provide the same, once, for every
// replay; without it the reduction dereferences a null buffer and the dynamic
// group memory comes back empty.

#include <omp.h>
#include <stdio.h>

#define N 256
#define TEAMS 64
#define ITERS 4

int main() {
  int Errors = 0;

  // A cross-team reduction, which needs a reduction buffer.
  for (int It = 0; It < ITERS; ++It) {
    long Sum = 0;
#pragma omp taskgraph
    {
#pragma omp target teams distribute parallel for num_teams(TEAMS)              \
    reduction(+ : Sum) map(tofrom : Sum)
      for (int I = 0; I < TEAMS * N; ++I)
        Sum += I;
    }
    if (Sum != (long)TEAMS * N * (TEAMS * N - 1) / 2) {
      printf("reduction replay %d: sum %ld\n", It, Sum);
      ++Errors;
    }
  }

  // Dynamic group memory the device provides natively, which the kernel finds
  // through its launch environment.
  for (int It = 0; It < ITERS; ++It) {
    int Result[N];
    size_t Size = 0;
#pragma omp taskgraph
    {
#pragma omp target teams num_teams(1) thread_limit(N)                          \
    dyn_groupprivate(fallback(abort) : N * sizeof(int))                        \
    map(from : Result, Size)
      {
#pragma omp parallel
        {
          int *Dyn = (int *)omp_get_dyn_gprivate_nofb_ptr(0, omp_access_cgroup);
          int T = omp_get_thread_num();
          int NT = omp_get_num_threads();
          if (T == 0)
            Size = omp_get_dyn_gprivate_size(omp_access_cgroup);
          if (Dyn)
            Dyn[T] = T + It;
#pragma omp barrier
          Result[T] = Dyn ? Dyn[(T + 1) % NT] : -1;
        }
      }
    }
    if (Size != N * sizeof(int) || Result[0] != 1 + It) {
      printf("native replay %d: size %zu, result %d\n", It, Size, Result[0]);
      ++Errors;
    }
  }

  // More dynamic group memory than the device has, which falls back to global
  // memory, one slice per team.
  size_t Big = omp_get_gprivate_limit(0, omp_access_cgroup) + sizeof(int);
  for (int It = 0; It < ITERS; ++It) {
    int Result[TEAMS];
#pragma omp taskgraph
    {
#pragma omp target teams num_teams(TEAMS) thread_limit(1)                      \
    dyn_groupprivate(fallback(default_mem) : Big) map(from : Result)
      {
        int *Dyn = (int *)omp_get_dyn_gprivate_ptr(0, omp_access_cgroup);
        int T = omp_get_team_num();
        if (Dyn) {
          Dyn[0] = T + It;
          Dyn[Big / sizeof(int) - 1] = T;
        }
        Result[T] = Dyn ? Dyn[0] + Dyn[Big / sizeof(int) - 1] : -1;
      }
    }
    for (int T = 0; T < TEAMS; ++T)
      if (Result[T] != 2 * T + It) {
        printf("fallback replay %d: team %d result %d\n", It, T, Result[T]);
        ++Errors;
        break;
      }
  }

  // CHECK: taskgraph launch environment errors: 0
  printf("taskgraph launch environment errors: %d\n", Errors);
  return Errors != 0;
}
