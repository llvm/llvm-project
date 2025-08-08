// RUN: %libomptarget-compilexx-run-and-check-generic
// REQUIRES: gpu

#include <omp.h>
#include <stdio.h>

#define N 512

int main() {
  int Result[N], NumThreads;

#pragma omp target teams num_teams(1) thread_limit(N)                          \
    dyn_groupprivate(strict : N * sizeof(Result[0]))                           \
    map(from : Result, NumThreads)
  {
    int Buffer[N];
#pragma omp parallel
    {
      int *DynBuffer = (int *)omp_get_dyn_groupprivate_ptr();
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

  // Verify that the routines in the host returns NULL and zero.
  if (omp_get_dyn_groupprivate_ptr())
    ++Failed;
  if (omp_get_dyn_groupprivate_size())
    ++Failed;

  size_t MaxSize = omp_get_groupprivate_limit(0, omp_access_cgroup);
  size_t ExceededSize = MaxSize + 10;

// Verify that the fallback modifier works.
#pragma omp target dyn_groupprivate(fallback : ExceededSize)                   \
    map(tofrom : Failed)
  {
    int IsFallback;
    if (!omp_get_dyn_groupprivate_ptr(0, &IsFallback))
      ++Failed;
    if (!omp_get_dyn_groupprivate_size())
      ++Failed;
    if (omp_get_dyn_groupprivate_size() != ExceededSize)
      ++Failed;
    if (!IsFallback)
      ++Failed;
  }

// Verify that the default modifier is fallback.
#pragma omp target dyn_groupprivate(ExceededSize)
  {
  }

// Verify that the strict modifier works.
#pragma omp target dyn_groupprivate(strict : N) map(tofrom : Failed)
  {
    int IsFallback;
    if (!omp_get_dyn_groupprivate_ptr(0, &IsFallback))
      ++Failed;
    if (!omp_get_dyn_groupprivate_size())
      ++Failed;
    if (omp_get_dyn_groupprivate_size() != N)
      ++Failed;
    if (IsFallback)
      ++Failed;
  }

// Verify that the fallback does not trigger when not needed.
#pragma omp target dyn_groupprivate(fallback : N) map(tofrom : Failed)
  {
    int IsFallback;
    if (!omp_get_dyn_groupprivate_ptr(0, &IsFallback))
      ++Failed;
    if (!omp_get_dyn_groupprivate_size())
      ++Failed;
    if (omp_get_dyn_groupprivate_size() != N)
      ++Failed;
    if (IsFallback)
      ++Failed;
  }

// Verify that the clause works when passing a zero size.
#pragma omp target dyn_groupprivate(strict : 0) map(tofrom : Failed)
  {
    int IsFallback;
    if (omp_get_dyn_groupprivate_ptr(0, &IsFallback))
      ++Failed;
    if (omp_get_dyn_groupprivate_size())
      ++Failed;
    if (IsFallback)
      ++Failed;
  }

// Verify that the clause works when passing a zero size.
#pragma omp target dyn_groupprivate(fallback : 0) map(tofrom : Failed)
  {
    int IsFallback;
    if (omp_get_dyn_groupprivate_ptr(0, &IsFallback))
      ++Failed;
    if (omp_get_dyn_groupprivate_size())
      ++Failed;
    if (IsFallback)
      ++Failed;
  }

// Verify that omitting the clause is the same as setting zero size.
#pragma omp target map(tofrom : Failed)
  {
    int IsFallback;
    if (omp_get_dyn_groupprivate_ptr(0, &IsFallback))
      ++Failed;
    if (omp_get_dyn_groupprivate_size())
      ++Failed;
    if (IsFallback)
      ++Failed;
  }

  // CHECK: PASS
  if (!Failed)
    printf("PASS\n");
}
