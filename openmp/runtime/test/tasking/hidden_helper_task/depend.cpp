// RUN: %libomp-cxx-compile-and-run

/*
 * This test aims to check whether hidden helper task can work with regular task
 * in terms of dependences. It is equivalent to the following code:
 *
 * #pragma omp parallel
 * for (int i = 0; i < N; ++i) {
 *   int data = -1;
 * #pragma omp task shared(data) depend(out: data)
 *   {
 *     data = 1;
 *   }
 * #pragma omp hidden helper task shared(data) depend(inout: data)
 *   {
 *     data += 2;
 *   }
 * #pragma omp hidden helper task shared(data) depend(inout: data)
 *   {
 *     data += 4;
 *   }
 * #pragma omp task shared(data) depend(inout: data)
 *   {
 *     data += 8;
 *   }
 * #pragma omp taskwait
 *   assert(data == 15);
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
  *p += I;
  return 0;
}

int main(int argc, char *argv[]) {
  constexpr const int N = 1024;
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    int32_t gtid = __kmpc_global_thread_num(nullptr);
    int32_t data = 0;

    // Task 1
    auto task1 = __kmpc_omp_task_alloc(
        nullptr, gtid, 1, sizeof(kmp_task_t_with_privates), sizeof(anon),
        reinterpret_cast<kmp_routine_entry_t>(omp_task_entry<1>));

    auto shareds = reinterpret_cast<anon *>(task1->shareds);
    shareds->data = &data;

    kmp_depend_info_t depinfo1;
    depinfo1.base_addr = reinterpret_cast<intptr_t>(&data);
    depinfo1.flag = 2; // OUT
    depinfo1.len = 4;

    __kmpc_omp_task_with_deps(nullptr, gtid, task1, 1, &depinfo1, 0, nullptr);

    // Task 2
    auto task2 = __kmpc_omp_target_task_alloc(
        nullptr, gtid, 1, sizeof(kmp_task_t_with_privates), sizeof(anon),
        reinterpret_cast<kmp_routine_entry_t>(omp_task_entry<2>), -1);

    shareds = reinterpret_cast<anon *>(task2->shareds);
    shareds->data = &data;

    kmp_depend_info_t depinfo2;
    depinfo2.base_addr = reinterpret_cast<intptr_t>(&data);
    depinfo2.flag = 3; // INOUT
    depinfo2.len = 4;

    __kmpc_omp_task_with_deps(nullptr, gtid, task2, 1, &depinfo2, 0, nullptr);

    // Task 3
    auto task3 = __kmpc_omp_target_task_alloc(
        nullptr, gtid, 1, sizeof(kmp_task_t_with_privates), sizeof(anon),
        reinterpret_cast<kmp_routine_entry_t>(omp_task_entry<4>), -1);

    shareds = reinterpret_cast<anon *>(task3->shareds);
    shareds->data = &data;

    kmp_depend_info_t depinfo3;
    depinfo3.base_addr = reinterpret_cast<intptr_t>(&data);
    depinfo3.flag = 3; // INOUT
    depinfo3.len = 4;

    __kmpc_omp_task_with_deps(nullptr, gtid, task3, 1, &depinfo3, 0, nullptr);

    // Task 4
    auto task4 = __kmpc_omp_task_alloc(
        nullptr, gtid, 1, sizeof(kmp_task_t_with_privates), sizeof(anon),
        reinterpret_cast<kmp_routine_entry_t>(omp_task_entry<8>));

    shareds = reinterpret_cast<anon *>(task4->shareds);
    shareds->data = &data;

    kmp_depend_info_t depinfo4;
    depinfo4.base_addr = reinterpret_cast<intptr_t>(&data);
    depinfo4.flag = 3; // INOUT
    depinfo4.len = 4;

    __kmpc_omp_task_with_deps(nullptr, gtid, task4, 1, &depinfo4, 0, nullptr);

    // Wait for all tasks
    __kmpc_omp_taskwait(nullptr, gtid);

    assert(data == 15);
  }

  std::cout << "PASS\n";
  return 0;
}

// CHECK: PASS
