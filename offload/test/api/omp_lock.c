// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: amdgpu

// Device-side omp_set_lock and omp_unset_lock. omp_test_lock is covered by
// omp_test_lock.c.

#include <omp.h>
#include <stdio.h>

#pragma omp declare target
omp_lock_t SeqLock;
omp_lock_t CountLock;
#pragma omp end declare target

#define THREADS 256
#define ITERS 200

int main() {
  int Freed = 0;
  int Count = 0;
  int Contenders = 0;

  // Locks may only be shared within a contention group, so each lock is
  // created, used and destroyed by the threads of a single team.
#pragma omp target parallel num_threads(THREADS)                               \
    map(tofrom : Freed, Count, Contenders)
  {
#pragma omp single
    {
      omp_init_lock(&SeqLock);
      omp_init_lock(&CountLock);

      // A lock that is set and then unset is free again.
      omp_set_lock(&SeqLock);
      omp_unset_lock(&SeqLock);
      Freed = omp_test_lock(&SeqLock);
      omp_unset_lock(&SeqLock);
    }

    // Mutual exclusion between threads in different warps. Only one thread of
    // each warp takes part, so no two contenders share a wavefront.
    if (omp_get_thread_num() % 64 == 0) {
#pragma omp atomic
      Contenders += 1;
      for (int I = 0; I < ITERS; ++I) {
        omp_set_lock(&CountLock);
        volatile int *P = &Count;
        int V = *P;
        *P = V + 1;
        omp_unset_lock(&CountLock);
      }
    }

#pragma omp barrier
#pragma omp single
    {
      omp_destroy_lock(&SeqLock);
      omp_destroy_lock(&CountLock);
    }
  }

  // CHECK: Freed = 1
  printf("Freed = %d\n", Freed != 0);
  // CHECK: Count OK
  printf("Count %s\n",
         Count == Contenders * ITERS && Contenders > 1 ? "OK" : "BAD");
  return 0;
}
