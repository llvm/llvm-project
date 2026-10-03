// SPMD no-loop distribute: the index is BId * NumThreads + TId, so a NumThreads
// larger than the real block size drops iterations.
//
// The entry is called directly because no C/C++ construct reaches it: clang
// emits __kmpc_for_static_init_4, only flang emits this one.
//
// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: gpu

#include <omp.h>
#include <stdio.h>

#define N 128
#define NUM_THREADS 256
#define THREAD_LIMIT 32

struct Args {
  double *Out;
};

#pragma omp begin declare target
extern void __kmpc_distribute_for_static_loop_4u(
    void *Loc, void (*Fn)(unsigned, void *), void *Arg, unsigned NumIters,
    unsigned NumThreads, unsigned BlockChunk, unsigned ThreadChunk,
    unsigned char OneIterationPerThread);

__attribute__((noinline)) static void body(unsigned I, void *A) {
  ((struct Args *)A)->Out[I] = (double)(I + 1);
}
#pragma omp end declare target

// For the host fallback copy only; the device uses the runtime's definition.
#ifndef __AMDGCN__
void __kmpc_distribute_for_static_loop_4u(
    void *Loc, void (*Fn)(unsigned, void *), void *Arg, unsigned NumIters,
    unsigned NumThreads, unsigned BlockChunk, unsigned ThreadChunk,
    unsigned char OneIterationPerThread) {
  for (unsigned I = 0; I < NumIters; ++I)
    Fn(I, Arg);
}
#endif

int main(void) {
  static double Out[N];
  for (int I = 0; I < N; ++I)
    Out[I] = -1.0;

  // NUM_THREADS deliberately exceeds the block's actual thread count.
#pragma omp target teams map(tofrom : Out[0 : N]) num_teams(4)                 \
    thread_limit(THREAD_LIMIT)
  {
#pragma omp parallel
    {
      struct Args A;
      A.Out = Out;
      __kmpc_distribute_for_static_loop_4u(0, body, &A, N, NUM_THREADS,
                                           /*BlockChunk=*/0,
                                           /*ThreadChunk=*/0,
                                           /*OneIterationPerThread=*/1);
    }
  }

  int Unwritten = 0;
  for (int I = 0; I < N; ++I)
    if (Out[I] == -1.0)
      ++Unwritten;

  // CHECK: unwritten: 0
  printf("unwritten: %d\n", Unwritten);
  return 0;
}
