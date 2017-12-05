// RUN: %libomp-compile-and-run
//
// The test checks the teams construct with reduction executed on the host.
//

#include <stdio.h>
#include <omp.h>

#include <stdint.h>

#ifndef N_TEAMS
#define N_TEAMS 4
#endif
#ifndef N_THR
#define N_THR 3
#endif

// Internal library stuff to emulate compiler's code generation:
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t reserved_1;
  int32_t flags;
  int32_t reserved_2;
  int32_t reserved_3;
  char const *psource;
} ident_t;

static ident_t dummy_loc = {0, 2, 0, 0, ";dummyFile;dummyFunc;0;0;;"};

typedef int32_t kmp_critical_name[8];
kmp_critical_name crit;

int32_t __kmpc_global_thread_num(ident_t *);
void __kmpc_push_num_teams(ident_t *, int32_t global_tid, int32_t num_teams,
                           int32_t num_threads);
void __kmpc_fork_teams(ident_t *, int32_t argc, void *microtask, ...);
int32_t __kmpc_reduce(ident_t *, int32_t global_tid, int32_t num_vars,
                      size_t reduce_size, void *reduce_data, void *reduce_func,
                      kmp_critical_name *lck);
void __kmpc_end_reduce(ident_t *, int32_t global_tid, kmp_critical_name *lck);

#ifdef __cplusplus
}
#endif

// Outlined entry point:
void outlined(int32_t *gtid, int32_t *tid) {
  int32_t ret = __kmpc_reduce(&dummy_loc, *gtid, 0, 0, NULL, NULL, &crit);
  __kmpc_end_reduce(&dummy_loc, *gtid, &crit);
}

int main() {
  int32_t th = __kmpc_global_thread_num(NULL); // registers initial thread
  __kmpc_push_num_teams(&dummy_loc, th, N_TEAMS, N_THR);
  __kmpc_fork_teams(&dummy_loc, 0, &outlined);

  // Test did not hang -> passed!
  printf("passed\n");
  return 0;
}
