// RUN: %libomp-compile -fopenmp-version=52 && %libomp-run

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef struct {
  int team_num;
  int thread_num;
} omp_id_t;

/// Test if each worker threads can retrieve correct icv values.
void test_api(int nteams, int nthreads, int par_if) {
  int expected_nteams = nteams;
  int expected_nthreads = par_if ? nthreads : 1;
  int expected_size = expected_nteams * expected_nthreads;
  omp_id_t *expected = (omp_id_t *)malloc(expected_size * sizeof(omp_id_t));
  omp_id_t *observed = (omp_id_t *)malloc(expected_size * sizeof(omp_id_t));

  for (int i = 0; i < expected_size; i++) {
    expected[i].team_num = i / expected_nthreads;
    expected[i].thread_num = i % expected_nthreads;
    observed[i].team_num = -1;
    observed[i].thread_num = -1;
  }

#pragma omp teams num_teams(nteams)
#pragma omp parallel num_threads(nthreads) if (par_if)
  {
    omp_id_t id = {omp_get_team_num(), omp_get_thread_num()};
    if (omp_get_num_teams() == expected_nteams &&
        omp_get_num_threads() == expected_nthreads && id.team_num >= 0 &&
        id.team_num < expected_nteams && id.thread_num >= 0 &&
        id.thread_num < expected_nthreads) {
      int flat_id = id.thread_num + id.team_num * expected_nthreads;
      observed[flat_id] = id;
    }
  }

  for (int i = 0; i < expected_size; i++) {
    if (expected[i].team_num != observed[i].team_num ||
        expected[i].thread_num != observed[i].thread_num) {
      printf("failed at nteams=%d, nthreads=%d, par_if=%d\n", nteams, nthreads,
             par_if);
      exit(EXIT_FAILURE);
    }
  }
}

/// Test if __kmpc_dist_for_static_init works correctly.
void test_dist(int nteams, int nthreads, int par_if) {
  int ub = 1000;
  int index_sum_expected = ub * (ub + 1) / 2;
  int index_sum = 0;
#pragma omp teams distribute parallel for num_teams(nteams)                    \
    num_threads(nthreads) if (par_if)
  for (int i = 1; i <= ub; i++)
#pragma omp atomic update
    index_sum += i;

  if (index_sum != index_sum_expected) {
    printf("failed at nteams=%d, nthreads=%d, par_if=%d\n", nteams, nthreads,
           par_if);
    exit(EXIT_FAILURE);
  }
}

int main() {
  for (int par_if = 0; par_if < 2; par_if++) {
    for (int nteams = 1; nteams <= 16; nteams++) {
      for (int nthreads = 1; nthreads <= 16; nthreads++) {
        if (omp_get_max_threads() < nteams * nthreads)
          continue; // make sure requested resources are granted
        test_api(nteams, nthreads, par_if);
        test_dist(nteams, nthreads, par_if);
      }
    }
  }
  printf("passed\n");
  return EXIT_SUCCESS;
}
