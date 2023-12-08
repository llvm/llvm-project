// REQUIRES: hidden-helper
// RUN: %libomp-cxx-compile
// RUN: env LIBOMP_USE_HIDDEN_HELPER_TASK=1 LIBOMP_NUM_HIDDEN_HELPER_THREADS=8 \
// RUN:     KMP_HIDDEN_HELPER_AFFINITY=verbose,granularity=socket,compact %libomp-run 2>&1 | FileCheck --check-prefix=SOCKET %s
// RUN: env LIBOMP_USE_HIDDEN_HELPER_TASK=1 LIBOMP_NUM_HIDDEN_HELPER_THREADS=8 \
// RUN:     KMP_HIDDEN_HELPER_AFFINITY=verbose,granularity=core,scatter %libomp-run 2>&1 | FileCheck --check-prefix=CORE %s
// RUN: env LIBOMP_USE_HIDDEN_HELPER_TASK=1 LIBOMP_NUM_HIDDEN_HELPER_THREADS=8 \
// RUN:     KMP_HIDDEN_HELPER_AFFINITY='verbose,granularity=fine,explicit,proclist=[0,1]' %libomp-run 2>&1 | FileCheck --check-prefix=FINE %s
// RUN: env LIBOMP_USE_HIDDEN_HELPER_TASK=1 LIBOMP_NUM_HIDDEN_HELPER_THREADS=8 \
// RUN:     KMP_HIDDEN_HELPER_AFFINITY=verbose,granularity=socket,compact KMP_AFFINITY=compact,granularity=fine %libomp-run 2>&1 | FileCheck --check-prefix=SOCKET %s
// RUN: env LIBOMP_USE_HIDDEN_HELPER_TASK=1 LIBOMP_NUM_HIDDEN_HELPER_THREADS=8 \
// RUN:     KMP_HIDDEN_HELPER_AFFINITY=verbose,granularity=core,scatter KMP_AFFINITY=compact,granularity=socket %libomp-run 2>&1 | FileCheck --check-prefix=CORE %s
// RUN: env LIBOMP_USE_HIDDEN_HELPER_TASK=1 LIBOMP_NUM_HIDDEN_HELPER_THREADS=8 \
// RUN:     KMP_HIDDEN_HELPER_AFFINITY='verbose,granularity=fine,explicit,proclist=[0,1]' KMP_AFFINITY=compact,granularity=core %libomp-run 2>&1 | FileCheck --check-prefix=FINE %s
// RUN: env LIBOMP_USE_HIDDEN_HELPER_TASK=1 LIBOMP_NUM_HIDDEN_HELPER_THREADS=8 \
// RUN:     KMP_HIDDEN_HELPER_AFFINITY=verbose,granularity=socket,compact OMP_PROC_BIND=close OMP_PLACES=threads %libomp-run 2>&1 | FileCheck --check-prefix=SOCKET %s
// RUN: env LIBOMP_USE_HIDDEN_HELPER_TASK=1 LIBOMP_NUM_HIDDEN_HELPER_THREADS=8 \
// RUN:     KMP_HIDDEN_HELPER_AFFINITY=verbose,granularity=core,scatter OMP_PROC_BIND=close OMP_PLACES=sockets %libomp-run 2>&1 | FileCheck --check-prefix=CORE %s
// RUN: env LIBOMP_USE_HIDDEN_HELPER_TASK=1 LIBOMP_NUM_HIDDEN_HELPER_THREADS=8 \
// RUN:     KMP_HIDDEN_HELPER_AFFINITY='verbose,granularity=fine,explicit,proclist=[0,1]' OMP_PROC_BIND=cores OMP_PLACES=cores %libomp-run 2>&1 | FileCheck --check-prefix=FINE %s

/*
 * This test aims to check hidden helper affinity
 *
 * #pragma omp parallel for
 * for (int i = 0; i < N; ++i) {
 *   int data1 = 0, data2 = 0;
 * #pragma omp taskgroup
 *   {
 * #pragma omp hidden helper task shared(data1)
 *    {
 *      data1 = 1;
 *    }
 * #pragma omp hidden helper task shared(data2)
 *    {
 *      data2 = 2;
 *    }
 *   }
 *   assert(data1 == 1);
 *   assert(data2 == 2);
 * }
 */

#include "common.h"

extern "C" {
struct kmp_task_t_with_privates {
  kmp_task_t task;
};

struct anon {
  int32_t *data;
};
}

template <int I>
kmp_int32 omp_task_entry(kmp_int32 gtid, kmp_task_t_with_privates *task) {
  auto shareds = reinterpret_cast<anon *>(task->task.shareds);
  auto p = shareds->data;
  *p = I;
  return 0;
}

int main(int argc, char *argv[]) {
  constexpr const int N = 16;
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    int32_t gtid = __kmpc_global_thread_num(nullptr);
    int32_t data1 = 0;
    __kmpc_taskgroup(nullptr, gtid);

    auto task1 = __kmpc_omp_target_task_alloc(
        nullptr, gtid, 1, sizeof(kmp_task_t_with_privates), sizeof(anon),
        reinterpret_cast<kmp_routine_entry_t>(omp_task_entry<1>), -1);
    auto shareds = reinterpret_cast<anon *>(task1->shareds);
    shareds->data = &data1;
    __kmpc_omp_task(nullptr, gtid, task1);

    __kmpc_end_taskgroup(nullptr, gtid);

    assert(data1 == 1);
  }

  std::cout << "PASS\n";
  return 0;
}

// SOCKET: OMP: Info #{{[0-9]+}}: KMP_HIDDEN_HELPER_AFFINITY: Threads may migrate across
// SOCKET-NOT: OMP: Info #{{[0-9]+}}: KMP_HIDDEN_HELPER_AFFINITY: pid {{[0-9]+}} tid {{[0-9]+}} thread 1 bound to OS proc set
// SOCKET-DAG: OMP: Info #[[NUM:[0-9]+]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID:[0-9]+]] tid {{[0-9]+}} thread 2 bound to OS proc set
// SOCKET-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 3 bound to OS proc set
// SOCKET-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 4 bound to OS proc set
// SOCKET-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 5 bound to OS proc set
// SOCKET-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 6 bound to OS proc set
// SOCKET-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 7 bound to OS proc set
// SOCKET-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 8 bound to OS proc set
// SOCKET-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 9 bound to OS proc set

// CORE: OMP: Info #{{[0-9]+}}: KMP_HIDDEN_HELPER_AFFINITY: Threads may migrate across
// CORE-NOT: OMP: Info #{{[0-9]+}}: KMP_HIDDEN_HELPER_AFFINITY: pid {{[0-9]+}} tid {{[0-9]+}} thread 1 bound to OS proc set
// CORE-DAG: OMP: Info #[[NUM:[0-9]+]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID:[0-9]+]] tid {{[0-9]+}} thread 2 bound to OS proc set
// CORE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 3 bound to OS proc set
// CORE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 4 bound to OS proc set
// CORE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 5 bound to OS proc set
// CORE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 6 bound to OS proc set
// CORE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 7 bound to OS proc set
// CORE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 8 bound to OS proc set
// CORE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 9 bound to OS proc set

// FINE-NOT: OMP: Info #{{[0-9]+}}: KMP_HIDDEN_HELPER_AFFINITY: Threads may migrate across
// FINE-NOT: OMP: Info #{{[0-9]+}}: KMP_HIDDEN_HELPER_AFFINITY: pid {{[0-9]+}} tid {{[0-9]+}} thread 1 bound to OS proc set
// FINE-DAG: OMP: Info #[[NUM:[0-9]+]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID:[0-9]+]] tid {{[0-9]+}} thread 2 bound to OS proc set
// FINE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 3 bound to OS proc set
// FINE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 4 bound to OS proc set
// FINE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 5 bound to OS proc set
// FINE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 6 bound to OS proc set
// FINE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 7 bound to OS proc set
// FINE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 8 bound to OS proc set
// FINE-DAG: OMP: Info #[[NUM]]: KMP_HIDDEN_HELPER_AFFINITY: pid [[PID]] tid {{[0-9]+}} thread 9 bound to OS proc set

// End of file
