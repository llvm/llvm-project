// Chunked static distribution: thread T owns iterations [T*chunk, T*chunk+chunk)
// within each block chunk. The index used to start each thread has to account
// for the chunk, and the block chunk has to cover one chunk per thread.
//
// The entry is called directly because no C/C++ construct reaches it: clang
// emits __kmpc_for_static_init_4, only flang emits this one.
//
// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: gpu

#include <omp.h>
#include <stdio.h>

#define N 32
#define NT 8
#define CHUNK 4

struct Args {
  int *Tid;
};

#pragma omp begin declare target
extern void __kmpc_distribute_for_static_loop_4u(
    void *Loc, void (*Fn)(unsigned, void *), void *Arg, unsigned NumIters,
    unsigned NumThreads, unsigned BlockChunk, unsigned ThreadChunk,
    unsigned char OneIterationPerThread);

__attribute__((noinline)) static void body(unsigned I, void *A) {
  ((struct Args *)A)->Tid[I] = omp_get_thread_num();
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
  static int Tid[N];
  for (int I = 0; I < N; ++I)
    Tid[I] = -1;

#pragma omp target teams map(tofrom : Tid[0 : N]) num_teams(1) thread_limit(NT)
  {
#pragma omp parallel num_threads(NT)
    {
      struct Args A;
      A.Tid = Tid;
      __kmpc_distribute_for_static_loop_4u(0, body, &A, N, NT,
                                           /*BlockChunk=*/0,
                                           /*ThreadChunk=*/CHUNK,
                                           /*OneIterationPerThread=*/0);
    }
  }

  int Unwritten = 0, Misplaced = 0;
  for (int I = 0; I < N; ++I) {
    if (Tid[I] == -1)
      ++Unwritten;
    else if (Tid[I] != (I / CHUNK) % NT)
      ++Misplaced;
  }

  // CHECK: unwritten: 0
  // CHECK: misplaced: 0
  printf("unwritten: %d\n", Unwritten);
  printf("misplaced: %d\n", Misplaced);
  return 0;
}
