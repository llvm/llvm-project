// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic

// REQUIRES: amdgpu

// Every thread of a team takes the same lock, including threads that share a
// wavefront. The compiler serializes the lanes of a wavefront across each
// omp_set_lock ... omp_unset_lock sequence below, since a lane that holds the
// lock could not otherwise release it while another lane of the same wavefront
// spins for it.
//
// Locks may only be shared within a contention group, so each lock is created,
// used and destroyed by the threads of a single team.

#include <omp.h>
#include <stdio.h>

#define TEAMS 4

#pragma omp declare target
omp_lock_t Lock;
omp_lock_t Locks[4];
omp_lock_t TeamLocks[TEAMS];

void bump(volatile int *P) { *P = *P + 1; }
#pragma omp end declare target

int main() {
  int Count = 0, Calls = 0, Shared = 0, Expected = 0, Even = 0;
  int Hist[4] = {0, 0, 0, 0};

#pragma omp target parallel num_threads(256)                                   \
    map(tofrom : Count, Calls, Shared, Expected, Even, Hist)
  {
#pragma omp single
    {
      omp_init_lock(&Lock);
      for (int I = 0; I < 4; ++I)
        omp_init_lock(&Locks[I]);
    }

    int Id = omp_get_thread_num();
    volatile int *C = &Count;

    // Straight-line code.
    omp_set_lock(&Lock);
    *C = *C + 1;
    omp_unset_lock(&Lock);

    // A function call and nested control flow inside the region.
    omp_set_lock(&Lock);
    if (Id % 2 == 0)
      bump(&Calls);
    for (int I = 0; I < 2; ++I)
      bump(&Shared);
    omp_unset_lock(&Lock);

    // A lock picked per thread; lanes of a wavefront want different locks.
    omp_set_lock(&Locks[Id % 4]);
    bump(&Hist[Id % 4]);
    omp_unset_lock(&Locks[Id % 4]);

#pragma omp atomic
    Expected += 1;
    if (Id % 2 == 0) {
#pragma omp atomic
      Even += 1;
    }

#pragma omp barrier
#pragma omp single
    {
      omp_destroy_lock(&Lock);
      for (int I = 0; I < 4; ++I)
        omp_destroy_lock(&Locks[I]);
    }
  }

  // Several teams at once, each with a lock of its own.
  int TeamCount[TEAMS] = {0}, TeamThreads[TEAMS] = {0};
#pragma omp target teams num_teams(TEAMS) map(tofrom : TeamCount, TeamThreads)
#pragma omp parallel num_threads(128)
  {
    int T = omp_get_team_num();
#pragma omp single
    omp_init_lock(&TeamLocks[T]);

    omp_set_lock(&TeamLocks[T]);
    bump(&TeamCount[T]);
    omp_unset_lock(&TeamLocks[T]);

#pragma omp atomic
    TeamThreads[T] += 1;

#pragma omp barrier
#pragma omp single
    omp_destroy_lock(&TeamLocks[T]);
  }
  int TeamsOK = 1;
  for (int T = 0; T < TEAMS; ++T)
    TeamsOK &= TeamCount[T] == TeamThreads[T];

  // CHECK: Count OK
  printf("Count %s\n", Count == Expected && Count > 0 ? "OK" : "BAD");
  // CHECK: Calls OK
  printf("Calls %s\n", Calls == Even ? "OK" : "BAD");
  // CHECK: Shared OK
  printf("Shared %s\n", Shared == 2 * Expected ? "OK" : "BAD");
  // CHECK: Hist OK
  printf("Hist %s\n",
         Hist[0] + Hist[1] + Hist[2] + Hist[3] == Expected ? "OK" : "BAD");
  // CHECK: Teams OK
  printf("Teams %s\n", TeamsOK ? "OK" : "BAD");
  return 0;
}
