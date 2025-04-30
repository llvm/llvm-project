// RUN: %libomp-cxx-compile -g -l nsapi_core && %libomp-run  | FileCheck %s
// UNSUPPORTED: !linux
#include <unistd.h>
#include <random>
#include <cstring>
#include <atomic>
#include <thread>
#include <algorithm>
#include <array>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <omp.h>

constexpr int MAX_N = 1000;
constexpr int MAX_THREADS = 1000;

int N = 1000;
constexpr int M = 100000;

#if defined(__cplusplus)
#define C_EXTERN extern "C"
#else
#define C_EXTERN
#endif

#define TEST_LOG_OK std::cout << __FILE__ ":" << __LINE__ << ": OK: "
#define TEST_LOG std::cout << __FILE__ ":" << __LINE__ << "  : "
#define TEST_LOG_ERROR                                                         \
  std::cout << __FILE__ ":" << __LINE__ << "[***ERROR***]: "
#define TEST_ABORT()                                                           \
  {                                                                            \
    std::cout << __FILE__ ":" << __LINE__ << "**ABORT**\n";                    \
    std::cout.flush();                                                         \
    std::cerr.flush();                                                         \
    exit(-1);                                                                  \
  }

#define TEST_ASSERT(x)                                                         \
  {                                                                            \
    if (!(x)) {                                                                \
      TEST_LOG_ERROR << "[" << #x << "] assertion failed.";                    \
      TEST_ABORT();                                                            \
    }                                                                          \
  }

#define KMP_NS_EXTERN C_EXTERN __attribute__((weak))

#define NUM_ELEMENTS(v) (sizeof(v) / sizeof(v[0]))

struct col_s {
  int64_t a[M];
};

struct mat_s {
  col_s cols[MAX_N];
};

struct row_s {
  int64_t v[MAX_N];
};

bool is_equal_within_tolerance(int64_t a, int64_t b) { return a == b; }

int64_t sum_col(struct col_s *col) {
  int64_t s = 0;
  for (size_t i = 0; i < NUM_ELEMENTS(col->a); i++) {
    s += col->a[i];
  }
  return s;
}

struct work_s {
  work_s(bool for_separate_from_parallel, int N, int chunk_size,
         bool ns_expansion_expected)
      : N(N), chunk_size(chunk_size),
        ns_expansion_expected(ns_expansion_expected),
        for_separate_from_parallel(for_separate_from_parallel),
        is_omp_dynamic(omp_get_dynamic()){};
  bool for_separate_from_parallel;
  bool ns_expansion_expected;
  const bool is_omp_dynamic;
  int chunk_size;
  int global_min_team_size = INT32_MAX;
  int global_max_team_size = INT32_MIN;
  int team_size_mismatch_count = 0;

  int N;
  struct mat_s mat {
    0
  };
  struct col_s expected_col_sum {
    0
  };
  struct col_s actual_col_sum {
    0
  };
  struct row_s expected_trace_execution {
    0
  };
  struct row_s trace_execution {
    0
  };
  struct row_s loop_iter_executed {
    0
  };
  std::array<std::thread::id, MAX_N> thread_ids;
  std::array<std::atomic<int32_t>, MAX_N> tid_count;
  std::array<int, MAX_N> team_size{0};

  std::array<int64_t, MAX_N> tid{-1};
  std::array<int64_t, MAX_THREADS> first_iteration_by_thread_id{-1};

  std::array<volatile int64_t, MAX_N> iteration_by_tid{0};

  int64_t actual_total_sum;
  int64_t expected_total_sum;

  void prepare_for_loop();
  void loop_body(int i, int thread_num, std::thread::id thread_id,
                 int tid_setting, int team_size, bool &first_time,
                 int64_t &actual_total_sum);

  void prepare_work_area(int N, int seed);

  void check_test_result(int64_t actual_result);
};

int first_iteration_check[MAX_THREADS];

void work_s::loop_body(int i, int thread_num, std::thread::id thread_id,
                       int tid_setting, int team_size, bool &first_time,
                       int64_t &actual_total_sum) {
  auto w = this;
  struct mat_s &mat = w->mat;
  struct col_s &expected_col_sum = w->expected_col_sum;
  struct col_s &actual_col_sum = w->actual_col_sum;
  struct row_s &ete = w->expected_trace_execution;
  struct row_s &te = w->trace_execution;

  const auto current_sum = sum_col(&mat.cols[i]);
  actual_total_sum += current_sum;
  te.v[i] = current_sum;
  w->tid[i] = tid_setting;

  w->loop_iter_executed.v[i] = 1;
  w->thread_ids[i] = thread_id;
  w->tid_count[i]++;

  w->team_size[i] = team_size;

  if (first_time) {
    if (w->first_iteration_by_thread_id[thread_num] != -1) {
      TEST_LOG_ERROR << "tid[" << thread_num
                     << "] already has 1st iteration set at["
                     << w->first_iteration_by_thread_id[thread_num] << "], i=["
                     << i << "]\n";
      TEST_ABORT();
    }
    w->first_iteration_by_thread_id[thread_num] = i;
    first_time = false;
  }

  const bool ok =
      __sync_bool_compare_and_swap(&w->iteration_by_tid[i], -1, thread_num);
  if (!ok) {
    TEST_LOG_ERROR << "tid[" << thread_num << "], thread id[0x" << std::hex
                   << (thread_id) << std::dec << "], iteration[" << i
                   << "], found the iteration already been taken by ["
                   << w->iteration_by_tid[i] << "]\n";
    std::cout.flush();

    TEST_ABORT();
  }
}

void work_s::prepare_for_loop() {
  // clear the indication of which thread started at what iteration
  for (int l = 0; l < this->N; l++) {
    this->iteration_by_tid[l] = -1;
  }

  for (auto &first_iteration : this->first_iteration_by_thread_id) {
    first_iteration = -1;
  }
}

void work_s::prepare_work_area(int N, int seed) {
  auto w = this;
  w->N = N;
  struct mat_s &mat = w->mat;
  struct col_s &expected_col_sum = w->expected_col_sum;
  struct col_s &actual_col_sum = w->actual_col_sum;
  struct row_s &ete = w->expected_trace_execution;
  struct row_s &te = w->trace_execution;

  std::mt19937 gen(
      seed /* rd() */); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<int64_t> dis(-(1 << 16), (1 << 16));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      mat.cols[i].a[j] = dis(gen);
    } // for j loop. //
  }

  for (int i = 0; i < N; i++) {
    expected_col_sum.a[i] = sum_col(&mat.cols[i]);
  }

  int64_t expected_total_sum = 0;
  for (int i = 0; i < N; i++) {
    const auto expected_sum = expected_col_sum.a[i];
    expected_total_sum += expected_sum;
    ete.v[i] = expected_sum;
  }
  w->expected_total_sum = expected_total_sum;
}

void work_s::check_test_result(int64_t actual_result) {
  auto w = this;
  if (!is_equal_within_tolerance(w->expected_total_sum, actual_result)) {
    TEST_LOG_ERROR << "mismatch! diff=["
                   << w->expected_total_sum - actual_result << "]\n";

    for (int i = 0; i < N; i++) {
      TEST_LOG << "tid[" << i << "]=[" << w->tid[i] << "], te[" << i << "]=["
               << w->trace_execution.v[i] << "], ete[" << i << "] = ["
               << w->expected_trace_execution.v[i] << "], loop exec?[" << i
               << "]=[" << w->loop_iter_executed.v[i] << "], tid[" << i
               << "].count=[" << w->tid_count[i] << "], thread id["
               << (w->thread_ids[i]) << "], team size[" << w->team_size[i]
               << "]\n";
    }
    TEST_ABORT();
  } else {
    TEST_LOG_OK << "MATCH!: diff=[" << w->expected_total_sum - actual_result
                << "]\n";
    int min_team_size = INT32_MAX;
    int max_team_size = INT32_MIN;
    for (int i = 0; i < N; i++) {
      // verify that ech iteration got a consistent thread num and team size.
      if (min_team_size > w->team_size[i]) {
        min_team_size = w->team_size[i];
      }
      if (max_team_size < w->team_size[i]) {
        max_team_size = w->team_size[i];
      }

      // verify all iterations were executed.
      if (w->iteration_by_tid[i] < 0) {
        TEST_LOG_ERROR << "iteration not done? i=[" << i << "]\n";
        TEST_ABORT();
      }
    }

    // if we check thread by thread scheduling, we keep for each thread the
    // initial iteration. (there are no duplicates since this is checked in
    // runtime).
    for (int i = 0; i < MAX_THREADS; i++) {
      first_iteration_check[i] = -1;
    }

    int num_out_of_order_tid_schedules = 0;

    // only if #pragma omp for is run separately, we can control whether
    // threads may arrive out of order
    const bool expected_out_of_order = w->is_omp_dynamic &&
                                       w->ns_expansion_expected &&
                                       w->for_separate_from_parallel;

    int assigned_chunks = w->N / w->chunk_size;
    if (w->N % w->chunk_size) {
      assigned_chunks++;
    }

    int max_assigned_thread = max_team_size / assigned_chunks;
    if (max_team_size % assigned_chunks) {
      max_assigned_thread++;
    }
    //
    // it is meaningful and easy to find (and verify!) out of order execution
    // (namely, when scheduled to run thread by thread, the next iteration is
    // not assigned by thread number but by thread arrival order).
    //
    for (int i = 0; i < max_assigned_thread; i++) {
      // tid = 1st iteration divided by chunk size
      // this is the expected behavior for ns expanded team in by thread
      // schedule.
      const auto first_iteration = w->first_iteration_by_thread_id[i];
      const auto tid = first_iteration / w->chunk_size;

      if (first_iteration == -1) {
        TEST_LOG_ERROR << "i[" << i
                       << "], has not been assigned an interation. estimated "
                          "max assigned thread#["
                       << max_assigned_thread << "], N[" << w->N
                       << "], max_team_size[" << max_team_size
                       << "], chunk size[" << w->chunk_size
                       << "], separate for?[" << w->for_separate_from_parallel
                       << "], ns expansion?[" << w->ns_expansion_expected
                       << "]\n";
        TEST_ABORT();
      }

      if (first_iteration_check[tid] != -1) {
        TEST_LOG_ERROR << "tid[" << tid << "] already scheduled iteration ["
                       << first_iteration << "], i=[" << i << "], N=[" << w->N
                       << "], chunk size=[" << w->chunk_size
                       << "], separate for?[" << w->for_separate_from_parallel
                       << "], ns expansion?[" << w->ns_expansion_expected
                       << "], max assigned thread[" << max_assigned_thread
                       << "]\n";
        TEST_ABORT();
      }
      if (tid != i) {
        TEST_LOG << "interesting, thread id[" << tid << "] started iteration ["
                 << i << "]\n";
        num_out_of_order_tid_schedules++;
        if (!w->ns_expansion_expected) {
          TEST_LOG_ERROR << "***ERROR: arrival order should have been in "
                            "thread number order, chunk_size=["
                         << chunk_size << "]\n";
          TEST_ABORT();
        }
      }
      first_iteration_check[tid] = i;
    }
    if (expected_out_of_order && (num_out_of_order_tid_schedules == 0)) {
      TEST_LOG_ERROR
          << "expected at least one thread to arrive out of order, chunk_size=["
          << w->chunk_size << "]\n";
    }

    if (min_team_size != max_team_size) {
      TEST_LOG << "STRANGE! min/max team size mismatch!!:  min team size=["
               << min_team_size << "], max team size=[" << max_team_size
               << "]\n";
      team_size_mismatch_count++;
      TEST_ABORT();
    } else {
      TEST_LOG_OK << "team sizes match at[" << min_team_size << "]\n";
    }

    if (global_min_team_size > min_team_size) {
      global_min_team_size = min_team_size;
    }
    if (global_max_team_size < max_team_size) {
      global_max_team_size = max_team_size;
    }
  }
}

int test_parallel_then_for(int N = 100, int seed = 1, int test_count = 1000,
                           int chunk_size = 1,
                           bool ns_expansion_expected = false) {
  TEST_LOG << "parallel then for starting...\n";

  struct work_s *w =
      new struct work_s(true /* #pragma omp for separate from parallel */, N,
                        chunk_size, ns_expansion_expected);

  memset(w, 0, sizeof(*w));

  TEST_LOG << "N=[" << N << "], test_count=[" << test_count << "], chunk_size=["
           << chunk_size << "], ns expansion expected?=["
           << ns_expansion_expected << "]\n";
  w->prepare_work_area(N, seed);
  w->ns_expansion_expected = ns_expansion_expected;
  w->chunk_size = chunk_size;

  for (int k = 0; k < test_count; k++) {
    int64_t actual_total_sum = 0;
    TEST_LOG << "trying openmp team for the " << k << "'th time\n";

    // clear the indication of which thread started at what iteration
    w->prepare_for_loop();

#pragma omp parallel
    {
      //
      // 1st - parallel region begins here.
      // Each thread in the team executes this prior to the loop.
      // Every variable defined now is local to the thread..
      //
      const int thread_num = omp_get_thread_num();
      const auto thread_id = std::this_thread::get_id();
      const int tid_setting = (thread_num << 4) | 1;
      const int team_size = omp_get_num_threads();
      bool first_time_for_thread = true;

      //
      // Try and force reorder of threads.
      // This is possible in separate #pragma omp parallel/for,
      // since the assignment of thread to iteration in an ns expanded team is
      // done at the for-static init point.
      //
      const int thread_sleep_usec = 10000 - thread_num * 100;
      if (thread_sleep_usec > 0) {
        usleep(thread_sleep_usec); // force threads to sleep longer by their
                                   // thread num
      }

      if (thread_num >= MAX_THREADS) {
        TEST_LOG_ERROR << "thread id[" << thread_num
                       << "] greater than the max #defined[" << MAX_THREADS
                       << "]\n";
        TEST_ABORT();
      }
#pragma omp for reduction(+ : actual_total_sum) schedule(static, chunk_size)
      for (int i = 0; i < N; i++) {
        w->loop_body(i, thread_num, thread_id, tid_setting, team_size,
                     first_time_for_thread, actual_total_sum);
      }
    }

    w->check_test_result(actual_total_sum);
    TEST_LOG_OK << "finished the openmp team for the " << k << "'th time\n";
  }

  TEST_LOG_OK << "global min team size=[" << w->global_min_team_size
              << "], global max team size=[" << w->global_max_team_size
              << ", mismatch count[" << w->team_size_mismatch_count << "]\n";

  TEST_LOG_OK << "parallel then for ending\n";

  delete w;

  return 0;
}

int test_parallel_for(int N = 100, int seed = 1, int test_count = 1000,
                      int chunk_size = 1, bool ns_expansion_expected = false) {
  TEST_LOG << "parallel for: starting...\n";

  auto w = std::make_unique<struct work_s>(
      false /* #pragma omp for separate from parallel */, N, chunk_size,
      ns_expansion_expected);

  TEST_LOG << "N=[" << N << "], test_count=[" << test_count << "], chunk_size=["
           << chunk_size << "], ns expansion expected?=["
           << ns_expansion_expected << "]\n";
  w->prepare_work_area(N, seed);
  w->ns_expansion_expected = ns_expansion_expected;
  w->chunk_size = chunk_size;

  for (int k = 0; k < test_count; k++) {
    int64_t actual_total_sum = 0;
    TEST_LOG << "trying openmp team for the " << k << "'th time\n";

    // clear the indication of which thread started at what iteration
    w->prepare_for_loop();

    bool first_time_for_thread = true;

#pragma omp parallel for firstprivate(first_time_for_thread)                   \
    reduction(+ : actual_total_sum) schedule(static, chunk_size)
    for (int i = 0; i < N; i++) {
      //
      // 1st - parallel region begins here.
      // Each thread in the team executes this prior to the loop.
      // Every variable defined now is local to the thread..
      //
      const int thread_num = omp_get_thread_num();
      const auto thread_id = std::this_thread::get_id();
      const int tid_setting = (thread_num << 4) | 1;
      const int team_size = omp_get_num_threads();

      if (thread_num >= MAX_THREADS) {
        TEST_LOG_ERROR << "thread id[" << thread_num
                       << "] greater than the max #defined[" << MAX_THREADS
                       << "]\n";
        TEST_ABORT();
      }
      w->loop_body(i, thread_num, thread_id, tid_setting, team_size,
                   first_time_for_thread, actual_total_sum);
    }

    w->check_test_result(actual_total_sum);
    TEST_LOG_OK << "finished the openmp team for the " << k << "'th time\n";
  }

  TEST_LOG_OK << "global min team size=[" << w->global_min_team_size
              << "], global max team size=[" << w->global_max_team_size
              << ", mismatch count[" << w->team_size_mismatch_count << "]\n";

  TEST_LOG_OK << "parallel for ending\n";

  return 0;
}

int run_with_mockup(int thread_capacity, bool is_handed_off, int N = 400,
                    int chunk_size = 1, bool ns_expansion_expected = false,
                    int test_size = 1) {
  return test_parallel_then_for(N, 1, test_size, chunk_size,
                                ns_expansion_expected) |
         test_parallel_for(N, 1, test_size, chunk_size, ns_expansion_expected);
}

int main(int argc, char *argv[]) {
  const int num_cpus = sysconf(_SC_NPROCESSORS_CONF);
  // force openmp initialization before setting the debug level.
  const int max_threads = omp_get_max_threads();
  TEST_LOG_OK << "max threads[" << max_threads << "]\n";

  const int num_ns_threads = num_cpus * 4 - 1;
  const int test_n_size = num_ns_threads * 2;
  const int standard_chunk_size = test_n_size / num_cpus + 1;
  const int ns_chunk_size = test_n_size / test_n_size + 1;

  TEST_ASSERT(num_cpus != 0);
  TEST_ASSERT(num_ns_threads != 0);
  TEST_ASSERT(test_n_size != 0);
  TEST_ASSERT(standard_chunk_size != 0);
  TEST_ASSERT(ns_chunk_size != 0);

  //
  // 1st batch - not dynamic - no ns expansion
  //
  omp_set_dynamic(false);
  TEST_LOG << "no ns expansion expected\n";
  //             # thread capacity   handed   loop         chunk size         ns
  //                                 off?     size expansion?
  run_with_mockup(-1, false, test_n_size, standard_chunk_size, false);
  run_with_mockup(num_cpus - 1, true, test_n_size, ns_chunk_size, false);
  run_with_mockup(num_ns_threads, true, test_n_size, 1, false);
  //
  // 2nd batch - ns expansion only when capacity > num cpu cores.
  //
  omp_set_dynamic(true);
  // omp_dynamic=1 - ns expansion when mockup replies non-small # of threads.
  run_with_mockup(-1, false, test_n_size, standard_chunk_size, false);
  run_with_mockup(num_cpus - 1, true, test_n_size, ns_chunk_size, false);
  TEST_LOG << "expected ns expansion hence forward\n";
  run_with_mockup(num_ns_threads, true, test_n_size, 2, true);
  run_with_mockup(num_ns_threads, true, test_n_size, 1, true);

  TEST_LOG_OK << "main: all is well\n";
  return 0;
}
// CHECK:       {{(OK:|  :|NS-DEBUG:)}}
// CHECK-NOT:  [***ERROR***]
