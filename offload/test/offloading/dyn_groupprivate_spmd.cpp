// RUN: %libomptarget-compilexx-generic -fopenmp-version=61
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileoptxx-generic -fopenmp-version=61
// RUN: %libomptarget-run-generic | %fcheck-generic
// REQUIRES: gpu

#include <omp.h>
#include <stdio.h>

int main() {
  constexpr int NumElements = 512;
  int Result[NumElements] = {};
  int NumThreads = 0;
  int IsInitialDevice = 1;

#pragma omp target teams num_teams(1) thread_limit(NumElements)                \
    dyn_groupprivate(fallback(abort) : sizeof(Result))                         \
    map(tofrom : Result, NumThreads, IsInitialDevice)
  {
#pragma omp parallel
    {
      int *Buffer = static_cast<int *>(omp_get_dyn_gprivate_nofb_ptr());
      int Thread = omp_get_thread_num();
      if (Thread == 0) {
        NumThreads = omp_get_num_threads();
        IsInitialDevice = omp_is_initial_device();
      }
      Buffer[Thread] = Thread + 1;
#pragma omp barrier
      int Neighbor = (Thread + 1) % omp_get_num_threads();
      Result[Thread] = Buffer[Neighbor];
    }
  }

  if (IsInitialDevice || NumThreads < 2 || NumThreads > NumElements) {
    printf("Unexpected execution: initial device=%d, threads=%d\n",
           IsInitialDevice, NumThreads);
    return 1;
  }

  for (int Index = 0; Index < NumThreads; ++Index) {
    int Expected = (Index + 1) % NumThreads + 1;
    if (Result[Index] != Expected) {
      printf("Result[%d]=%d, expected %d\n", Index, Result[Index], Expected);
      return 1;
    }
  }

  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
