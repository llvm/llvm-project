// Stress the kernel launch environment (KLE) of concurrent launches.
//
// Same as kernel_launch_environment_reduction.c, but triggering the use of a
// KLE through dynamic group private memory instead of reductions.
//
// RUN: %libomptarget-compile-generic -fopenmp-version=61 \
// RUN:   -fopenmp-offload-mandatory
// RUN: %libomptarget-run-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=61 \
// RUN:   -fopenmp-offload-mandatory
// RUN: %libomptarget-run-generic
// RUN: rm -rf %t.rr && mkdir -p %t.rr
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_DIR=%t.rr \
// RUN:   %libomptarget-run-generic
//
// REQUIRES: gpu
// UNSUPPORTED: intelgpu

#include <omp.h>
#include <stdio.h>

#define NUM_LAUNCHES 32
#define N 4096
#define THREADS 256

// Launch k requests a distinct dynamic groupprivate size, large enough for one
// element per thread.
size_t expected(int k) { return (THREADS + k) * sizeof(size_t); }

// Return the dynamic groupprivate size seen by iteration i of launch k, or 0 if
// the buffer is not usable.
#pragma omp begin declare target
size_t probe(int k, int i) {
  size_t *Buf = (size_t *)omp_get_dyn_gprivate_nofb_ptr(0, omp_access_cgroup);
  if (!Buf)
    return 0;
  int T = omp_get_thread_num();
  Buf[T] = k * N + i;
  if (Buf[T] != (size_t)(k * N + i))
    return 0;
  return omp_get_dyn_gprivate_size(omp_access_cgroup);
}
#pragma omp end declare target

int check(const char *Phase, const size_t *Results) {
  int Errors = 0;
  for (int k = 0; k < NUM_LAUNCHES; k++) {
    for (int i = 0; i < N; i++) {
      if (Results[k * N + i] != expected(k)) {
        fprintf(stderr, "%s: launch %d iteration %d saw %zu, expected %zu\n",
                Phase, k, i, Results[k * N + i], expected(k));
        Errors++;
        break;
      }
    }
  }
  return Errors;
}

int main(void) {
  size_t Results[NUM_LAUNCHES * N];
  int Errors = 0;

  // Launches issued back to back without an intervening synchronization, so
  // that several launches are in flight at once.
  for (int k = 0; k < NUM_LAUNCHES; k++) {
#pragma omp target teams distribute parallel for thread_limit(THREADS)         \
    dyn_groupprivate(fallback(abort) : expected(k))                            \
    map(from : Results[k * N : N]) firstprivate(k) nowait
    for (int i = 0; i < N; i++)
      Results[k * N + i] = probe(k, i);
  }
#pragma omp taskwait
  Errors += check("nowait", Results);

  for (int j = 0; j < NUM_LAUNCHES * N; j++)
    Results[j] = 0;

  // Same, but with the launches and the synchronizations spread over several
  // host threads: a thread finalizing its queue must not disturb the launch
  // environment of a launch another thread still has in flight.
#pragma omp parallel for num_threads(8)
  for (int k = 0; k < NUM_LAUNCHES; k++) {
#pragma omp target teams distribute parallel for thread_limit(THREADS)         \
    dyn_groupprivate(fallback(abort) : expected(k))                            \
    map(from : Results[k * N : N]) firstprivate(k)
    for (int i = 0; i < N; i++)
      Results[k * N + i] = probe(k, i);
  }
  Errors += check("threaded", Results);

  if (Errors)
    return 1;
  printf("PASS\n");
  return 0;
}
