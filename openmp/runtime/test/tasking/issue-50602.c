// RUN: %libomp-compile-and-run
// RUN: env OMP_NUM_THREADS=1 %libomp-run
// RUN: %libomp-compile -DUSE_HIDDEN_HELPERS=1
// RUN: %libomp-run
// RUN: env OMP_NUM_THREADS=1 %libomp-run
#include <omp.h>

int main(int argc, char *argv[]) {
  int i;

  omp_set_max_active_levels(1);
  omp_set_dynamic(0);

  for (i = 0; i < 10; ++i) {
#pragma omp parallel
    {
#ifndef USE_HIDDEN_HELPERS
      omp_event_handle_t event;
#endif
      int a = 0;

#ifdef USE_HIDDEN_HELPERS
#pragma omp target map(tofrom : a) nowait
#else
#pragma omp task shared(a) detach(event)
#endif
      { a = 1; }

#pragma omp parallel
      { a = 2; }

#ifndef USE_HIDDEN_HELPERS
      omp_fulfill_event(event);
#endif

#pragma omp taskwait
    }
  }
  return 0;
}
