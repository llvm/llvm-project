// Run lines are in the target specific versions.

#include <omp.h>
#include <stdio.h>

#define N 512

int main() {
  int Result[N], NumThreads;

#pragma omp target teams num_teams(1) thread_limit(N)                          \
                         ompx_dyn_cgroup_mem(N * sizeof(Result[0]))            \
                         map(from : Result, NumThreads)
  {
    int Buffer[N];
#pragma omp parallel
    {
      int *DynBuffer = (int *)llvm_omp_target_dynamic_shared_alloc();
      int TId = omp_get_thread_num();
      if (TId == 0)
        NumThreads = omp_get_num_threads();
      Buffer[TId] = 7;
      DynBuffer[TId] = 3;
#pragma omp barrier
      int WrappedTId = (TId + 37) % NumThreads;
      Result[TId] = Buffer[WrappedTId] + DynBuffer[WrappedTId];
    }
  }

  if (llvm_omp_target_dynamic_shared_alloc())
    return -1;

  if (NumThreads < N / 2 || NumThreads > N) {
    printf("Expected number of threads to be in [%i:%i], but got: %i", N / 2, N,
           NumThreads);
    return -1;
  }

  int Failed = 0;
  for (int i = 0; i < NumThreads; ++i) {
    if (Result[i] != 7 + 3) {
      printf("Result[%i] is %i, expected %i\n", i, Result[i], 7 + 3);
      ++Failed;
    }
  }

  if (!Failed)
    printf("PASS\n");
}
