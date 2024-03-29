// RUN: %libomp-compile-and-run
#include <omp.h>

void nested_parallel_detached(int nth1, int nth2) {
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

void nested_parallel_hidden_helpers(int nth1, int nth2) {
#pragma omp parallel num_threads(nth1)
  {
#pragma omp parallel num_threads(nth2)
    {
#pragma omp target nowait
      {}
    }
  }
}
int main() {
  int i;

  omp_set_max_active_levels(2);
  omp_set_dynamic(0);

  for (i = 0; i < 10; ++i)
    nested_parallel_detached(1, 1);
  for (i = 0; i < 10; ++i)
    nested_parallel_detached(1, 2);
  for (i = 0; i < 10; ++i)
    nested_parallel_detached(2, 1);
  for (i = 0; i < 10; ++i)
    nested_parallel_detached(2, 2);

  for (i = 0; i < 10; ++i)
    nested_parallel_hidden_helpers(1, 1);
  for (i = 0; i < 10; ++i)
    nested_parallel_hidden_helpers(1, 2);
  for (i = 0; i < 10; ++i)
    nested_parallel_hidden_helpers(2, 1);
  for (i = 0; i < 10; ++i)
    nested_parallel_hidden_helpers(2, 2);

  return 0;
}
