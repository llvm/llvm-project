// REQUIRES: linux
// RUN: %libomp-compile && env OMP_NUM_THREADS='2' %libomp-run

#include <assert.h>
#include <omp.h>

#include "kmp_task_deps.h"

// the test
int main(void) {
  volatile int done = 0;

#pragma omp parallel num_threads(2)
  {
    while (omp_get_thread_num() != 0 && !done)
      ;

#pragma omp single
    {
      kmp_task_t *A, *B;
      kmp_depnode_list_t *A_succ;
      kmp_base_depnode_t *B_node;
      dep deps[2];
      int gtid;
      int x, y;

      gtid = __kmpc_global_thread_num(&loc);

      // A - out(x, y)
      A = __kmpc_omp_task_alloc(&loc, gtid, TIED, sizeof(kmp_task_t), 0, NULL);
      deps[0].addr = (size_t)&x;
      deps[0].len = 0;
      deps[0].flags = 2; // OUT

      deps[1].addr = (size_t)&y;
      deps[1].len = 0;
      deps[1].flags = 2; // OUT

      __kmpc_omp_task_with_deps(&loc, gtid, A, 2, deps, 0, 0);

      // B - in(x, y)
      B = __kmpc_omp_task_alloc(&loc, gtid, TIED, sizeof(kmp_task_t), 0, NULL);
      deps[0].addr = (size_t)&x;
      deps[0].len = 0;
      deps[0].flags = 1; // IN

      deps[1].addr = (size_t)&y;
      deps[1].len = 0;
      deps[1].flags = 1; // IN

      __kmpc_omp_task_with_deps(&loc, gtid, B, 2, deps, 0, 0);

      // Retrieve TDG nodes
      A_succ = __kmpc_task_get_successors(A);
      B_node = __kmpc_task_get_depnode(B);

      // 'B' should only be added once to 'A' successors list
      assert(A_succ->node == B_node);
      assert(A_succ->next == NULL);

#pragma omp taskwait

      done = 1;
    }
  }
  return 0;
}
