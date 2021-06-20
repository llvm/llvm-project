// RUN: %libomp-cxx-compile-and-run

/*
 * This test aims to check whether hidden helper thread has right gtid. We also
 * test if there is mixed dependences between regular tasks and hidden helper
 * tasks, the tasks are executed by right set of threads. It is equivalent to
 * the following code:
 *
 * #pragma omp parallel for
 * for (int i = 0; i < N; ++i) {
 *   int data1 = -1, data2 = -1, data3 = -1;
 *   int depvar;
 * #pragma omp task shared(data1) depend(inout: depvar)
 *   {
 *     data1 = omp_get_global_thread_id();
 *   }
 * #pragma omp task hidden helper shared(data2) depend(inout: depvar)
 *   {
 *     data2 = omp_get_global_thread_id();
 *   }
 * #pragma omp task shared(data3) depend(inout: depvar)
 *   {
 *     data3 = omp_get_global_thread_id();
 *   }
 * #pragma omp taskwait
 *   assert(data1 == 0 || data1 > __kmp_num_hidden_helper_threads);
 *   assert(data2 > 0 && data2 <= __kmp_num_hidden_helper_threads);
 *   assert(data3 == 0 || data3 > __kmp_num_hidden_helper_threads);
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

kmp_int32 __kmp_hidden_helper_threads_num;

kmp_int32 omp_task_entry(kmp_int32 gtid, kmp_task_t_with_privates *task) {
  auto shareds = reinterpret_cast<anon *>(task->task.shareds);
  auto p = shareds->data;
  *p = __kmpc_global_thread_num(nullptr);
  return 0;
}

template <bool hidden_helper_task> void assert_gtid(int v) {
  if (__kmp_hidden_helper_threads_num) {
    if (hidden_helper_task) {
      assert(v > 0 && v <= __kmp_hidden_helper_threads_num);
    } else {
      assert(v == 0 || v > __kmp_hidden_helper_threads_num);
    }
  } else {
    assert(v >= 0);
  }
}

int main(int argc, char *argv[]) {
  __kmp_hidden_helper_threads_num = get_num_hidden_helper_threads();

  constexpr const int N = 1024;
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    int32_t data1 = -1, data2 = -1, data3 = -1;
    int depvar;
    int32_t gtid = __kmpc_global_thread_num(nullptr);

    // Task 1, regular task
    auto task1 = __kmpc_omp_task_alloc(
        nullptr, gtid, 1, sizeof(kmp_task_t_with_privates), sizeof(anon),
        reinterpret_cast<kmp_routine_entry_t>(omp_task_entry));
    auto shareds = reinterpret_cast<anon *>(task1->shareds);
    shareds->data = &data1;

    kmp_depend_info_t depinfo1;
    depinfo1.base_addr = reinterpret_cast<intptr_t>(&depvar);
    depinfo1.flag = 3; // INOUT
    depinfo1.len = 4;

    __kmpc_omp_task_with_deps(nullptr, gtid, task1, 1, &depinfo1, 0, nullptr);

    // Task 2, hidden helper task
    auto task2 = __kmpc_omp_target_task_alloc(
        nullptr, gtid, 1, sizeof(kmp_task_t_with_privates), sizeof(anon),
        reinterpret_cast<kmp_routine_entry_t>(omp_task_entry), -1);
    shareds = reinterpret_cast<anon *>(task2->shareds);
    shareds->data = &data2;

    kmp_depend_info_t depinfo2;
    depinfo2.base_addr = reinterpret_cast<intptr_t>(&depvar);
    depinfo2.flag = 3; // INOUT
    depinfo2.len = 4;

    __kmpc_omp_task_with_deps(nullptr, gtid, task2, 1, &depinfo2, 0, nullptr);

    // Task 3, regular task
    auto task3 = __kmpc_omp_task_alloc(
        nullptr, gtid, 1, sizeof(kmp_task_t_with_privates), sizeof(anon),
        reinterpret_cast<kmp_routine_entry_t>(omp_task_entry));
    shareds = reinterpret_cast<anon *>(task3->shareds);
    shareds->data = &data3;

    kmp_depend_info_t depinfo3;
    depinfo3.base_addr = reinterpret_cast<intptr_t>(&depvar);
    depinfo3.flag = 3; // INOUT
    depinfo3.len = 4;

    __kmpc_omp_task_with_deps(nullptr, gtid, task3, 1, &depinfo3, 0, nullptr);

    __kmpc_omp_taskwait(nullptr, gtid);

    // FIXME: 8 here is not accurate
    assert_gtid<false>(data1);
    assert_gtid<true>(data2);
    assert_gtid<false>(data3);
  }

  std::cout << "PASS\n";
  return 0;
}

// CHECK: PASS
