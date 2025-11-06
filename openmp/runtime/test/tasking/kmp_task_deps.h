#ifndef KMP_TASK_DEPS_H
#define KMP_TASK_DEPS_H

#include <stddef.h> /* size_t */

// ---------------------------------------------------------------------------
// internal data to emulate compiler codegen
typedef struct DEP {
  size_t addr;
  size_t len;
  unsigned char flags;
} dep;

typedef struct task {
  void **shareds;
  void *entry;
  int part_id;
  void *destr_thunk;
  int priority;
  long long device_id;
  int f_priv;
} kmp_task_t;
typedef int (*entry_t)(int, kmp_task_t *);
typedef struct ID {
  int reserved_1;
  int flags;
  int reserved_2;
  int reserved_3;
  char *psource;
} id;

#define TIED 1

struct kmp_depnode_list;

typedef struct kmp_base_depnode {
  struct kmp_depnode_list *successors;
  /* [...] more stuff down here */
} kmp_base_depnode_t;

typedef struct kmp_depnode_list {
  struct kmp_base_depnode *node;
  struct kmp_depnode_list *next;
} kmp_depnode_list_t;

static id loc = {0, 2, 0, 0, ";file;func;0;0;;"};
kmp_task_t *__kmpc_omp_task_alloc(id *loc, int gtid, int flags, size_t sz,
                                  size_t shar, entry_t rtn);
int __kmpc_omp_task_with_deps(id *loc, int gtid, kmp_task_t *task, int nd,
                              dep *dep_lst, int nd_noalias,
                              dep *noalias_dep_lst);
kmp_depnode_list_t *__kmpc_task_get_successors(kmp_task_t *task);
kmp_base_depnode_t *__kmpc_task_get_depnode(kmp_task_t *task);
int __kmpc_global_thread_num(id *);

#endif /* KMP_TASK_DEPS_H */
