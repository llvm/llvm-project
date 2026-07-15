// RUN: %libomp-cxx-compile-and-run

/*
 * This test aims to check whether hidden helper task can work with regular task
 * in terms of dependences. It is equivalent to the following code:
 *
 * #pragma omp parallel
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
  constexpr const int N = 1024;
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    int32_t gtid = __kmpc_global_thread_num(nullptr);
    int32_t data1 = 0, data2 = 0;
    __kmpc_taskgroup(nullptr, gtid);

    auto task1 = __kmpc_omp_target_task_alloc(
        nullptr, gtid, 1, sizeof(kmp_task_t_with_privates), sizeof(anon),
        reinterpret_cast<kmp_routine_entry_t>(omp_task_entry<1>), -1);
    auto shareds = reinterpret_cast<anon *>(task1->shareds);
    shareds->data = &data1;
    __kmpc_omp_task(nullptr, gtid, task1);

    auto task2 = __kmpc_omp_target_task_alloc(
        nullptr, gtid, 1, sizeof(kmp_task_t_with_privates), sizeof(anon),
        reinterpret_cast<kmp_routine_entry_t>(omp_task_entry<2>), -1);
    shareds = reinterpret_cast<anon *>(task2->shareds);
    shareds->data = &data2;
    __kmpc_omp_task(nullptr, gtid, task2);

    __kmpc_end_taskgroup(nullptr, gtid);

    assert(data1 == 1);
    assert(data2 == 2);
  }

  std::cout << "PASS\n";
  return 0;
}

// CHECK: PASS
