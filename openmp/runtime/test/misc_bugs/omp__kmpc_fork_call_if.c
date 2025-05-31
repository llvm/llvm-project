// RUN: %libomp-compile && %t | FileCheck %s

#include <stdio.h>
#include <omp.h>

typedef int32_t kmp_int32;
typedef void *ident_t;
typedef void *kmpc_micro;

#ifdef __cplusplus
extern "C" {
#endif
extern void __kmpc_fork_call_if(ident_t *loc, kmp_int32 argc,
                                kmpc_micro microtask, kmp_int32 cond,
                                void *args);
#ifdef __cplusplus
}
#endif

// Microtask function for parallel region
void microtask(int *global_tid, int *bound_tid) {
  // CHECK: PASS
  if (omp_in_parallel()) {
    printf("FAIL\n");
  } else {
    printf("PASS\n");
  }
}

int main() {
  // Condition for parallelization (false in this case)
  int cond = 0;
  // Call __kmpc_fork_call_if
  __kmpc_fork_call_if(NULL, 0, microtask, cond, NULL);
  return 0;
}
