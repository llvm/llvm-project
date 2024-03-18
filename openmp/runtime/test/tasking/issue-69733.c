// RUN: %libomp-compile-and-run
#include <omp.h>

void nested_parallel(int nth1, int nth2) {
#pragma omp parallel num_threads(nth1)
  {
#pragma omp parallel num_threads(nth2)
    {
      omp_event_handle_t ev;
#pragma omp task detach(ev)
      {}
      omp_fulfill_event(ev);
    }
  }
}

int main() {
  int i;

  omp_set_max_active_levels(2);
  omp_set_dynamic(0);

  for (i = 0; i < 10; ++i)
    nested_parallel(1, 1);
  for (i = 0; i < 10; ++i)
    nested_parallel(1, 2);
  for (i = 0; i < 10; ++i)
    nested_parallel(2, 1);
  for (i = 0; i < 10; ++i)
    nested_parallel(2, 2);

  return 0;
}
