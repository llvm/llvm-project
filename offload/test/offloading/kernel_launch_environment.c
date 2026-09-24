// Stress the kernel launch environment (KLE) transfer.
//
// A cross-team reduction gives the launch a KLE, which the plugin stages in a
// host buffer and copies to the device asynchronously. The KLE carries that
// launch's own reduction buffer, so overlapping launches must neither share a
// staging buffer nor release one while its transfer is still in flight: either
// makes a launch reduce into another launch's buffer. Every launch here
// accumulates a distinct value, so that shows up as a wrong sum.
//
// RUN: %libomptarget-compile-generic -fopenmp-offload-mandatory
// RUN: %libomptarget-run-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-offload-mandatory
// RUN: %libomptarget-run-generic
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
  // that several KLE transfers are outstanding at once.
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
  // host threads: a thread finalizing its queue must not release a staging
  // buffer that another thread's launch is still using.
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
