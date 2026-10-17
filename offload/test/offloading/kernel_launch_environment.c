// Stress the kernel launch environment (KLE) of concurrent launches.
//
// A cross-team reduction gives the launch a KLE, which carries that launch's
// own reduction buffer. The plugin may either copy the KLE to the device
// asynchronously or, where device memory is host-writable, store it in place.
// Both have to hold up when launches overlap: every launch needs its own KLE,
// and a launch's KLE has to be visible to the device before its kernel runs
// and stay intact until it finishes. Otherwise a launch reduces into another
// launch's buffer, or reads a KLE that has not landed yet. Every launch here
// accumulates a distinct value, so either shows up as a wrong sum.
//
// RUN: %libomptarget-compile-generic -fopenmp-offload-mandatory
// RUN: %libomptarget-run-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-offload-mandatory
// RUN: %libomptarget-run-generic
// RUN: rm -rf %t.rr && mkdir -p %t.rr
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_DIR=%t.rr \
// RUN:   %libomptarget-run-generic
//
// REQUIRES: gpu

#include <omp.h>
#include <stdio.h>

#define NUM_LAUNCHES 32
#define N 4096

// Launch k accumulates k + 1 per element.
static long expected(int k) { return (long)(k + 1) * N; }

static int check(const char *Phase, long *Results) {
  int Errors = 0;
  for (int k = 0; k < NUM_LAUNCHES; k++)
    if (Results[k] != expected(k)) {
      fprintf(stderr, "%s: launch %d reduced to %ld, expected %ld\n", Phase, k,
              Results[k], expected(k));
      Errors++;
    }
  return Errors;
}

int main(void) {
  static long Results[NUM_LAUNCHES];
  int Errors = 0;

  // Launches issued back to back without an intervening synchronization, so
  // that several launches are in flight at once.
  for (int k = 0; k < NUM_LAUNCHES; k++) {
    Results[k] = 0;
#pragma omp target teams distribute parallel for map(tofrom : Results[k : 1])  \
    reduction(+ : Results[k]) firstprivate(k) nowait
    for (int i = 0; i < N; i++)
      Results[k] += k + 1;
  }
#pragma omp taskwait
  Errors += check("nowait", Results);

  // Same, but with the launches and the synchronizations spread over several
  // host threads: a thread finalizing its queue must not disturb the launch
  // environment of a launch another thread still has in flight.
#pragma omp parallel for num_threads(8)
  for (int k = 0; k < NUM_LAUNCHES; k++) {
    long Sum = 0;
#pragma omp target teams distribute parallel for map(tofrom : Sum)             \
    reduction(+ : Sum) firstprivate(k)
    for (int i = 0; i < N; i++)
      Sum += k + 1;
    Results[k] = Sum;
  }
  Errors += check("threaded", Results);

  if (Errors)
    return 1;
  printf("PASS\n");
  return 0;
}
