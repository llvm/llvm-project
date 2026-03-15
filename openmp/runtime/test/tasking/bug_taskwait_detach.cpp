// RUN: %libomp-cxx-compile-and-run

#include <omp.h>

#include <chrono>
#include <iostream>
#include <thread>

// detached
#define PTASK_FLAG_DETACHABLE 0x40

// OpenMP RTL interfaces
typedef long long kmp_int64;

typedef struct ID {
  int reserved_1;
  int flags;
  int reserved_2;
  int reserved_3;
  char *psource;
} id;

// Compiler-generated code (emulation)
typedef struct ident {
  void *dummy; // not used in the library
} ident_t;

typedef enum kmp_event_type_t {
  KMP_EVENT_UNINITIALIZED = 0,
  KMP_EVENT_ALLOW_COMPLETION = 1
} kmp_event_type_t;

typedef struct {
  kmp_event_type_t type;
  union {
    void *task;
  } ed;
} kmp_event_t;

typedef struct shar { // shareds used in the task
} * pshareds;

typedef struct task {
  pshareds shareds;
  int (*routine)(int, struct task *);
  int part_id;
  // void *destructor_thunk; // optional, needs flag setting if provided
  // int priority; // optional, needs flag setting if provided
  // ------------------------------
  // privates used in the task:
  omp_event_handle_t evt;
} * ptask, kmp_task_t;

typedef int (*task_entry_t)(int, ptask);

#ifdef __cplusplus
extern "C" {
#endif
extern int __kmpc_global_thread_num(void *id_ref);
extern int **__kmpc_omp_task_alloc(id *loc, int gtid, int flags, size_t sz,
                                   size_t shar, task_entry_t rtn);
extern int __kmpc_omp_task(id *loc, kmp_int64 gtid, kmp_task_t *task);
extern omp_event_handle_t __kmpc_task_allow_completion_event(ident_t *loc_ref,
                                                             int gtid,
                                                             kmp_task_t *task);
#ifdef __cplusplus
}
#endif

int volatile checker;

void target(ptask task) {
  std::this_thread::sleep_for(std::chrono::seconds(3));
  checker = 1;
  omp_fulfill_event(task->evt);
}

// User's code
int task_entry(int gtid, ptask task) {
  std::thread t(target, task);
  t.detach();
  return 0;
}

int main(int argc, char *argv[]) {
  int gtid = __kmpc_global_thread_num(nullptr);
  checker = 0;

  /*
    #pragma omp task detach(evt)
    {}
  */
  std::cout << "detaching...\n";
  ptask task = (ptask)__kmpc_omp_task_alloc(
      nullptr, gtid, PTASK_FLAG_DETACHABLE, sizeof(struct task),
      sizeof(struct shar), &task_entry);
  omp_event_handle_t evt =
      (omp_event_handle_t)__kmpc_task_allow_completion_event(nullptr, gtid,
                                                             task);
  task->evt = evt;

  __kmpc_omp_task(nullptr, gtid, task);

#pragma omp taskwait

  // check results
  if (checker == 1) {
    std::cout << "PASS\n";
    return 0;
  }

  return 1;
}

// CHECK: PASS
