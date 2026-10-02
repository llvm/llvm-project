// RUN: %libomptarget-compile-generic -fopenmp-version=61 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=61 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// A `use_device_ptr` list item that nothing maps has no device address to be
// told, and the clause says what to hand back instead: the host address by
// default, or null under `fb_nullify`.  Inside a recorded `target data` the
// body is a task that asks for the answer afresh whenever it starts, so the
// fallback has to be taken there too, on every replay.  An absent entry is
// found by asking the mapping table, which answers with the host address
// rather than with a null, so a fallback keyed on a null answer is never
// taken.

#include <stdio.h>
#include <stdlib.h>

#define N 64
#define ITERS 3

int main() {
  int *A = (int *)malloc(N * sizeof(int));
  int *Kept = (int *)malloc(N * sizeof(int));
  int *Nulled = (int *)malloc(N * sizeof(int));
  int WrongKept = 0, WrongNulled = 0;

  for (int It = 0; It < ITERS; ++It) {
    int *SeenKept = NULL, *SeenNulled = Nulled;

#pragma omp taskgraph
    {
#pragma omp target data map(tofrom : A[0 : N]) use_device_ptr(Kept)            \
    use_device_ptr(fb_nullify : Nulled)
      {
#pragma omp task shared(Kept, Nulled, SeenKept, SeenNulled)
        {
          SeenKept = Kept;
          SeenNulled = Nulled;
        }
      }
    }

    if (SeenKept != Kept)
      ++WrongKept;
    if (SeenNulled != NULL)
      ++WrongNulled;
  }

  // CHECK: replays not handed the host address: 0
  printf("replays not handed the host address: %d\n", WrongKept);
  // CHECK: replays not handed null: 0
  printf("replays not handed null: %d\n", WrongNulled);

  free(Nulled);
  free(Kept);
  free(A);
  return (WrongKept + WrongNulled) != 0;
}
