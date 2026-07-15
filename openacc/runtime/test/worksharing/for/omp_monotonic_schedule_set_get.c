// RUN: %libomp-compile-and-run

// The test checks OMP 5.0 monotonic/nonmonotonic scheduling API
//   1. initial schedule should be (static,0)
//   2. omp_get_schedule() should return the schedule set by omp_set_schedule()
//   3. schedules set inside parallel should not impact outer tasks' schedules

#include <stdio.h>
#ifndef __INTEL_COMPILER
#define _OMPIMP
#endif

#define NO_MODIFIERS ((omp_sched_t)0)

#include "omp.h"

int global = 0;
int err = 0;

omp_sched_t sched_append_modifiers(omp_sched_t sched, omp_sched_t modifiers) {
  return (omp_sched_t)((int)sched | (int)modifiers);
}

omp_sched_t sched_without_modifiers(omp_sched_t sched) {
  return (omp_sched_t)((int)sched & ~((int)omp_sched_monotonic));
}

int sched_has_modifiers(omp_sched_t sched, omp_sched_t modifiers) {
  return (((int)sched & ((int)omp_sched_monotonic)) > 0);
}

// check that sched = hope | modifiers
void check_schedule(const char *extra, const omp_sched_t sched, int chunk,
                    omp_sched_t hope_sched, int hope_chunk) {

  if (sched != hope_sched || chunk != hope_chunk) {
#pragma omp atomic
    ++err;
    printf("Error: %s: schedule: (%d, %d) is not equal to (%d, %d)\n", extra,
           (int)hope_sched, hope_chunk, (int)sched, chunk);
  }
}

int main() {
  int i;
  int chunk;
  omp_sched_t sched0;

  omp_set_dynamic(0);
  omp_set_nested(1);

  // check serial region
  omp_get_schedule(&sched0, &chunk);
#ifdef DEBUG
  printf("initial: (%d, %d)\n", sched0, chunk);
#endif
  check_schedule("initial", omp_sched_static, 0, sched0, chunk);
  // set schedule before the parallel, check it after the parallel
  omp_set_schedule(
      sched_append_modifiers(omp_sched_dynamic, omp_sched_monotonic), 3);

#pragma omp parallel num_threads(3) private(i)
  {
    omp_sched_t n_outer_set, n_outer_get;
    int c_outer;
    int tid = omp_get_thread_num();

    n_outer_set = sched_append_modifiers((omp_sched_t)(tid + 1),
                                         omp_sched_monotonic); // 1, 2, 3

    // check outer parallel region
    // master sets (static, unchunked), others - (dynamic, 1), (guided, 2)
    // set schedule before inner parallel, check it after the parallel
    omp_set_schedule(n_outer_set, tid);

// Make sure this schedule doesn't crash the runtime
#pragma omp for
    for (i = 0; i < 100; ++i) {
#pragma omp atomic
      global++;
    }

#pragma omp parallel num_threads(3) private(i) shared(n_outer_set)
    {
      omp_sched_t n_inner_set, n_inner_get;
      int c_inner_set, c_inner_get;
      int tid = omp_get_thread_num();

      n_inner_set = (omp_sched_t)(tid + 1); // 1, 2, 3
      c_inner_set = (int)(n_outer_set)*10 +
                    (int)n_inner_set; // 11, 12, 13, 21, 22, 23, 31, 32, 33
      n_inner_set = sched_append_modifiers(n_inner_set, omp_sched_monotonic);
      // schedules set inside parallel should not impact outer schedules
      omp_set_schedule(n_inner_set, c_inner_set);

// Make sure this schedule doesn't crash the runtime
#pragma omp for
      for (i = 0; i < 100; ++i) {
#pragma omp atomic
        global++;
      }

#pragma omp barrier
      omp_get_schedule(&n_inner_get, &c_inner_get);
#ifdef DEBUG
      printf("inner parallel: o_th %d, i_th %d, (%d, %d)\n", n_outer_set - 1,
             tid, n_inner_get, c_inner_get);
#endif
      check_schedule("inner", n_inner_set, c_inner_set, n_inner_get,
                     c_inner_get);
    }

    omp_get_schedule(&n_outer_get, &c_outer);
#ifdef DEBUG
    printf("outer parallel: thread %d, (%d, %d)\n", tid, n_outer_get, c_outer);
#endif
    check_schedule("outer", n_outer_set, tid, n_outer_get, c_outer);
  }

  omp_get_schedule(&sched0, &chunk);
#ifdef DEBUG
  printf("after parallels: (%d, %d)\n", sched0, chunk);
#endif
  check_schedule("after parallels",
                 sched_append_modifiers(omp_sched_dynamic, omp_sched_monotonic),
                 3, sched0, chunk);

  if (err > 0) {
    printf("Failed\n");
    return 1;
  }
  printf("Passed\n");
  return 0;
}
