// RUN: %libomp-compile-and-run
// RUN: env OMP_NUM_THREADS=1 %libomp-run
#include <omp.h>

int main(int argc, char *argv[]) {
  int i;

  omp_set_max_active_levels(1);
  omp_set_dynamic(0);

  for (i = 0; i < 10; ++i) {
#pragma omp parallel
    {
      omp_event_handle_t event;
      int a = 0;

#pragma omp task shared(a) detach(event)
      { a = 1; }

#pragma omp parallel
      { a = 2; }

      omp_fulfill_event(event);
#pragma omp taskwait
    }
  }
  return 0;
}
