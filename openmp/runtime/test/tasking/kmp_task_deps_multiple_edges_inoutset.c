// REQUIRES: linux
// RUN: %libomp-compile && env OMP_NUM_THREADS='2' %libomp-run

#include <assert.h>
#include <omp.h>

#include "kmp_task_deps.h"

//  Expected dependency graph (directed from top to bottom)
//
//          A   B   C       // inoutset(x), inoutset(x, y), inoutset(y)
//          | \ | / |
//          D   E   F       // in(x), in(x, y), in(y)
//               \ /
//                G         // out(y)

// the test
int main(void) {
  volatile int done = 0;

#pragma omp parallel num_threads(2)
  {
    while (omp_get_thread_num() != 0 && !done)
      ;

#pragma omp single
    {
      kmp_task_t *A, *B, *C, *D, *E, *F, *G;
      kmp_depnode_list_t *A_succ, *B_succ, *C_succ, *E_succ, *F_succ;
      kmp_base_depnode_t *D_node, *E_node, *F_node, *G_node;
      dep deps[2];
      int gtid;
      int x, y;

      gtid = __kmpc_global_thread_num(&loc);

      deps[0].addr = (size_t)&x;
      deps[0].len = 0;
      deps[0].flags = 8; // INOUTSET

      deps[1].addr = (size_t)&y;
      deps[1].len = 0;
      deps[1].flags = 8; // INOUTSET

      // A inoutset(x)
      A = __kmpc_omp_task_alloc(&loc, gtid, TIED, sizeof(kmp_task_t), 0, NULL);
      __kmpc_omp_task_with_deps(&loc, gtid, A, 1, deps + 0, 0, 0);

      // B inoutset(x, y)
      B = __kmpc_omp_task_alloc(&loc, gtid, TIED, sizeof(kmp_task_t), 0, NULL);
      __kmpc_omp_task_with_deps(&loc, gtid, B, 2, deps + 0, 0, 0);

      // C inoutset(y)
      C = __kmpc_omp_task_alloc(&loc, gtid, TIED, sizeof(kmp_task_t), 0, NULL);
      __kmpc_omp_task_with_deps(&loc, gtid, C, 1, deps + 1, 0, 0);

      deps[0].flags = 1; // IN
      deps[1].flags = 1; // IN

      // D in(x)
      D = __kmpc_omp_task_alloc(&loc, gtid, TIED, sizeof(kmp_task_t), 0, NULL);
      __kmpc_omp_task_with_deps(&loc, gtid, D, 1, deps + 0, 0, 0);

      // E in(x, y)
      E = __kmpc_omp_task_alloc(&loc, gtid, TIED, sizeof(kmp_task_t), 0, NULL);
      __kmpc_omp_task_with_deps(&loc, gtid, E, 2, deps + 0, 0, 0);

      // F in(y)
      F = __kmpc_omp_task_alloc(&loc, gtid, TIED, sizeof(kmp_task_t), 0, NULL);
      __kmpc_omp_task_with_deps(&loc, gtid, F, 1, deps + 1, 0, 0);

      deps[1].flags = 2; // OUT

      // G out(y)
      G = __kmpc_omp_task_alloc(&loc, gtid, TIED, sizeof(kmp_task_t), 0, NULL);
      __kmpc_omp_task_with_deps(&loc, gtid, G, 1, deps + 1, 0, 0);

      // Retrieve TDG nodes and check edges
      A_succ = __kmpc_task_get_successors(A);
      B_succ = __kmpc_task_get_successors(B);
      C_succ = __kmpc_task_get_successors(C);
      E_succ = __kmpc_task_get_successors(E);
      F_succ = __kmpc_task_get_successors(F);

      D_node = __kmpc_task_get_depnode(D);
      E_node = __kmpc_task_get_depnode(E);
      F_node = __kmpc_task_get_depnode(F);

      G_node = __kmpc_task_get_depnode(G);

      // A -> D and A -> E
      assert(A_succ && A_succ->next && !A_succ->next->next);
      assert((A_succ->node == D_node && A_succ->next->node == E_node) ||
             (A_succ->node == E_node && A_succ->next->node == D_node));

      // B -> D and B -> E and B -> F
      // valid lists are
      //  (D, E, F)
      //  (D, F, E)
      //  (E, D, F)
      //  (E, F, D)
      //  (F, D, E)
      //  (F, E, D)
      assert(B_succ && B_succ->next && B_succ->next->next &&
             !B_succ->next->next->next);
      assert((B_succ->node == D_node && B_succ->next->node == E_node &&
              B_succ->next->next->node == F_node) ||
             (B_succ->node == D_node && B_succ->next->node == F_node &&
              B_succ->next->next->node == E_node) ||
             (B_succ->node == E_node && B_succ->next->node == D_node &&
              B_succ->next->next->node == F_node) ||
             (B_succ->node == E_node && B_succ->next->node == F_node &&
              B_succ->next->next->node == D_node) ||
             (B_succ->node == F_node && B_succ->next->node == D_node &&
              B_succ->next->next->node == E_node) ||
             (B_succ->node == F_node && B_succ->next->node == E_node &&
              B_succ->next->next->node == D_node));

      // C -> E and C -> F
      assert(C_succ && C_succ->next && !C_succ->next->next);
      assert((C_succ->node == E_node && C_succ->next->node == F_node) ||
             (C_succ->node == F_node && C_succ->next->node == E_node));

      // E -> G and F -> G
      assert(E_succ && !E_succ->next);
      assert(E_succ->node == G_node);

      assert(F_succ && !F_succ->next);
      assert(F_succ->node == G_node);

#pragma omp taskwait

      done = 1;
    }
  }
  return 0;
}
