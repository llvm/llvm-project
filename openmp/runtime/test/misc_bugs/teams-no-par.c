// RUN: %libomp-compile-and-run
//
// The test checks the teams construct pseudocode executed on host
//

#include <stdio.h>
#include <omp.h>

#ifndef N_TEAMS
#define N_TEAMS 4
#endif
#ifndef N_THR
#define N_THR 3
#endif

static int err = 0;

// Internal library staff to emulate compiler's code generation:
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int reserved_1;
  int flags;
  int reserved_2;
  int reserved_3;
  char *psource;
} ident_t;

static ident_t dummy_loc = {0, 2, 0, 0, ";dummyFile;dummyFunc;0;0;;"};

int __kmpc_global_thread_num(void*);
void __kmpc_push_num_teams(ident_t const*, int, int, int);
void __kmpc_fork_teams(ident_t const*, int argc, void *microtask, ...);

#ifdef __cplusplus
}
#endif

// Outlined entry point:
void foo(int *gtid, int *tid, int *nt)
{ // start "serial" execution by master threads of each team
  if ( nt ) {
    printf(" team %d, param %d\n", omp_get_team_num(), *nt);
  } else {
    printf("ERROR: teams before parallel: gtid, tid: %d %d, bad pointer: %p\n", *gtid, *tid, nt);
    err++;
    return;
  }
}

int main()
{
  int nt = 4;
  int th = __kmpc_global_thread_num(NULL); // registers initial thread
  __kmpc_push_num_teams(&dummy_loc, th, N_TEAMS, N_THR);
  __kmpc_fork_teams(&dummy_loc, 1, &foo, &nt); // pass 1 shared parameter "nt"
  if (err)
    printf("failed with %d errors\n",err);
  else
    printf("passed\n");
  return err;
}
