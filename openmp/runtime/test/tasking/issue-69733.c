// RUN: %libomp-compile-and-run

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int a;

void inc_a() {
#pragma omp atomic
  a++;
}

void root_team_detached() {
  a = 0;
  omp_event_handle_t ev;
#pragma omp task detach(ev)
  inc_a();
  omp_fulfill_event(ev);
  if (a != 1) {
    fprintf(stderr, "error: root_team_detached(): a != 1\n");
    exit(EXIT_FAILURE);
  }
}

void root_team_hidden_helpers() {
  a = 0;
#pragma omp target nowait
  inc_a();

#pragma omp taskwait

  if (a != 1) {
    fprintf(stderr, "error: root_team_hidden_helpers(): a != 1\n");
    exit(EXIT_FAILURE);
  }
}

void parallel_detached(int nth1) {
  a = 0;
  omp_event_handle_t *evs =
      (omp_event_handle_t *)malloc(sizeof(omp_event_handle_t) * nth1);
#pragma omp parallel num_threads(nth1)
  {
    int tid = omp_get_thread_num();
    omp_event_handle_t e = evs[tid];
#pragma omp task detach(e)
    inc_a();
    omp_fulfill_event(e);
  }
  free(evs);
  if (a != nth1) {
    fprintf(stderr, "error: parallel_detached(): a (%d) != %d\n", a, nth1);
    exit(EXIT_FAILURE);
  }
}

void parallel_hidden_helpers(int nth1) {
  a = 0;
#pragma omp parallel num_threads(nth1)
  {
#pragma omp target nowait
    inc_a();
  }
  if (a != nth1) {
    fprintf(stderr, "error: parallel_hidden_helpers(): a (%d) != %d\n", a,
            nth1);
    exit(EXIT_FAILURE);
  }
}

void nested_parallel_detached(int nth1, int nth2) {
  a = 0;
  omp_event_handle_t **evs =
      (omp_event_handle_t **)malloc(sizeof(omp_event_handle_t *) * nth1);
#pragma omp parallel num_threads(nth1)
  {
    int tid = omp_get_thread_num();
    evs[tid] = (omp_event_handle_t *)malloc(sizeof(omp_event_handle_t) * nth2);
#pragma omp parallel num_threads(nth2) shared(tid)
    {
      int tid2 = omp_get_thread_num();
      omp_event_handle_t e = evs[tid][tid2];
#pragma omp task detach(e)
      inc_a();
      omp_fulfill_event(e);
    }
    free(evs[tid]);
  }
  free(evs);
  if (a != nth1 * nth2) {
    fprintf(stderr, "error: nested_parallel_detached(): a (%d) != %d * %d\n", a,
            nth1, nth2);
    exit(EXIT_FAILURE);
  }
}

void nested_parallel_hidden_helpers(int nth1, int nth2) {
  a = 0;
#pragma omp parallel num_threads(nth1)
  {
#pragma omp parallel num_threads(nth2)
    {
#pragma omp target nowait
      inc_a();
    }
  }
  if (a != nth1 * nth2) {
    fprintf(stderr,
            "error: nested_parallel_hidden_helpers(): a (%d) != %d * %d\n", a,
            nth1, nth2);
    exit(EXIT_FAILURE);
  }
}

int main() {
  int i, nth1, nth2;

  omp_set_max_active_levels(2);
  omp_set_dynamic(0);

  for (i = 0; i < 10; ++i)
    root_team_detached();

  for (i = 0; i < 10; ++i)
    root_team_hidden_helpers();

  for (i = 0; i < 10; ++i)
    for (nth1 = 1; nth1 <= 4; ++nth1)
      parallel_detached(nth1);

  for (i = 0; i < 10; ++i)
    for (nth1 = 1; nth1 <= 4; ++nth1)
      parallel_hidden_helpers(nth1);

  for (i = 0; i < 10; ++i)
    for (nth1 = 1; nth1 <= 4; ++nth1)
      for (nth2 = 1; nth2 <= 4; ++nth2)
        nested_parallel_detached(nth1, nth2);

  for (i = 0; i < 10; ++i)
    for (nth1 = 1; nth1 <= 4; ++nth1)
      for (nth2 = 1; nth2 <= 4; ++nth2)
        nested_parallel_hidden_helpers(nth1, nth2);

  return 0;
}
