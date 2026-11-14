// RUN: %libomptarget-compile-run-and-check-generic

// omp_test_lock in device code tries to take the lock and reports whether it
// succeeded.

// The other AMDGPU device lock routines are not implemented yet.
// XFAIL: amdgpu

#include <omp.h>
#include <stdio.h>

#pragma omp declare target
omp_lock_t RaceLock;
omp_lock_t SeqLock;
#pragma omp end declare target

int main() {
  int Winners = 0;
  int Seq[2] = {0, 0};

  // Locks may only be shared within a contention group, so each lock is
  // created, used and destroyed by the threads of a single team.
#pragma omp target parallel num_threads(256) map(tofrom : Winners, Seq)
  {
#pragma omp single
    {
      omp_init_lock(&RaceLock);
      omp_init_lock(&SeqLock);
    }

    // Of many threads trying to take a free lock, exactly one succeeds.
    int Won = omp_test_lock(&RaceLock);
    if (Won) {
#pragma omp atomic
      Winners += 1;
    }
#pragma omp barrier
    if (Won)
      omp_unset_lock(&RaceLock);

    // A free lock is taken, and is free again once released.
#pragma omp single
    {
      Seq[0] = omp_test_lock(&SeqLock);
      omp_unset_lock(&SeqLock);
      Seq[1] = omp_test_lock(&SeqLock);
      omp_unset_lock(&SeqLock);
    }

#pragma omp single
    {
      omp_destroy_lock(&RaceLock);
      omp_destroy_lock(&SeqLock);
    }
  }

  // CHECK: Winners = 1
  printf("Winners = %d\n", Winners);
  // CHECK: Seq = 1 1
  printf("Seq = %d %d\n", Seq[0] != 0, Seq[1] != 0);
  return 0;
}
