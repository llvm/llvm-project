// RUN: %libomp-compile-and-run
// UNSUPPORTED: gcc

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NT 8

#ifdef __cplusplus
extern "C" {
#endif
typedef int kmp_int32;
typedef struct ident {
  kmp_int32 reserved_1;
  kmp_int32 flags;
  kmp_int32 reserved_2;
  kmp_int32 reserved_3;
  char const *psource;
} ident_t;
extern int __kmpc_global_thread_num(ident_t *);
extern void __kmpc_push_num_teams_51(ident_t *, kmp_int32, kmp_int32, kmp_int32,
                                     kmp_int32);
#ifdef __cplusplus
}
#endif

void check_num_teams(int num_teams_lb, int num_teams_ub, int thread_limit) {
  int nteams, nthreads;
  int a = 0;

  int gtid = __kmpc_global_thread_num(NULL);
  __kmpc_push_num_teams_51(NULL, gtid, num_teams_lb, num_teams_ub,
                           thread_limit);

#pragma omp target teams map(tofrom: a) map(from: nteams, nthreads)
  {
    int priv_nteams;
    int team_num = omp_get_team_num();
    if (team_num == 0)
      nteams = omp_get_num_teams();
    priv_nteams = omp_get_num_teams();
#pragma omp parallel
    {
      int priv_nthreads;
      int thread_num = omp_get_thread_num();
      int teams_ub, teams_lb, thr_limit;
      if (team_num == 0 && thread_num == 0)
        nthreads = omp_get_num_threads();
      priv_nthreads = omp_get_num_threads();

      teams_ub = (num_teams_ub ? num_teams_ub : priv_nteams);
      teams_lb = (num_teams_lb ? num_teams_lb : teams_ub);
      thr_limit = (thread_limit ? thread_limit : priv_nthreads);

      if (priv_nteams < teams_lb || priv_nteams > teams_ub) {
        fprintf(stderr, "error: invalid number of teams=%d\n", priv_nteams);
        exit(1);
      }
      if (priv_nthreads > thr_limit) {
        fprintf(stderr, "error: invalid number of threads=%d\n", priv_nthreads);
        exit(1);
      }
#pragma omp atomic
      a++;
    }
  }
  if (a != nteams * nthreads) {
    fprintf(stderr, "error: a (%d) != nteams * nthreads (%d)\n", a,
            nteams * nthreads);
    exit(1);
  } else {
    printf("#teams %d, #threads %d: Hello!\n", nteams, nthreads);
  }
}

int main(int argc, char *argv[]) {
  omp_set_num_threads(NT);

  check_num_teams(1, 8, 2);
  check_num_teams(2, 2, 2);
  check_num_teams(2, 2, 0);
  check_num_teams(8, 16, 2);
  check_num_teams(9, 16, 0);
  check_num_teams(9, 16, 2);
  check_num_teams(2, 3, 0);
  check_num_teams(0, 0, 2);
  check_num_teams(0, 4, 0);
  check_num_teams(0, 2, 2);

  printf("Test Passed\n");
  return 0;
}
