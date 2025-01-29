#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "benchmark/benchmark.h"
#include "output_test.h"

namespace {

#define ADD_COMPLEXITY_CASES(...) \
  int CONCAT(dummy, __LINE__) = AddComplexityTest(__VA_ARGS__)

int AddComplexityTest(const std::string &test_name,
                      const std::string &big_o_test_name,
                      const std::string &rms_test_name,
                      const std::string &big_o, int family_index) {
  SetSubstitutions({{"%name", test_name},
                    {"%bigo_name", big_o_test_name},
                    {"%rms_name", rms_test_name},
                    {"%bigo_str", "[ ]* %float " + big_o},
                    {"%bigo", big_o},
                    {"%rms", "[ ]*[0-9]+ %"}});
  AddCases(
      TC_ConsoleOut,
      {{"^%bigo_name %bigo_str %bigo_str[ ]*$"},
       {"^%bigo_name", MR_Not},  // Assert we we didn't only matched a name.
       {"^%rms_name %rms %rms[ ]*$", MR_Next}});
  AddCases(
      TC_JSONOut,
      {{"\"name\": \"%bigo_name\",$"},
       {"\"family_index\": " + std::to_string(family_index) + ",$", MR_Next},
       {"\"per_family_instance_index\": 0,$", MR_Next},
       {"\"run_name\": \"%name\",$", MR_Next},
       {"\"run_type\": \"aggregate\",$", MR_Next},
       {"\"repetitions\": %int,$", MR_Next},
       {"\"threads\": 1,$", MR_Next},
       {"\"aggregate_name\": \"BigO\",$", MR_Next},
       {"\"aggregate_unit\": \"time\",$", MR_Next},
       {"\"cpu_coefficient\": %float,$", MR_Next},
       {"\"real_coefficient\": %float,$", MR_Next},
       {"\"big_o\": \"%bigo\",$", MR_Next},
       {"\"time_unit\": \"ns\"$", MR_Next},
       {"}", MR_Next},
       {"\"name\": \"%rms_name\",$"},
       {"\"family_index\": " + std::to_string(family_index) + ",$", MR_Next},
       {"\"per_family_instance_index\": 0,$", MR_Next},
       {"\"run_name\": \"%name\",$", MR_Next},
       {"\"run_type\": \"aggregate\",$", MR_Next},
       {"\"repetitions\": %int,$", MR_Next},
       {"\"threads\": 1,$", MR_Next},
       {"\"aggregate_name\": \"RMS\",$", MR_Next},
       {"\"aggregate_unit\": \"percentage\",$", MR_Next},
       {"\"rms\": %float$", MR_Next},
       {"}", MR_Next}});
  AddCases(TC_CSVOut, {{"^\"%bigo_name\",,%float,%float,%bigo,,,,,$"},
                       {"^\"%bigo_name\"", MR_Not},
                       {"^\"%rms_name\",,%float,%float,,,,,,$", MR_Next}});
  return 0;
}

}  // end namespace

// ========================================================================= //
// --------------------------- Testing BigO O(1) --------------------------- //
// ========================================================================= //

void BM_Complexity_O1(benchmark::State &state) {
  for (auto _ : state) {
    // This test requires a non-zero CPU time to avoid divide-by-zero
    benchmark::DoNotOptimize(state.iterations());
    long tmp = state.iterations();
    benchmark::DoNotOptimize(tmp);
    for (benchmark::IterationCount i = 0; i < state.iterations(); ++i) {
      benchmark::DoNotOptimize(state.iterations());
      tmp *= state.iterations();
      benchmark::DoNotOptimize(tmp);
    }

    // always 1ns per iteration
    state.SetIterationTime(42 * 1e-9);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Complexity_O1)
    ->Range(1, 1 << 18)
    ->UseManualTime()
    ->Complexity(benchmark::o1);
BENCHMARK(BM_Complexity_O1)->Range(1, 1 << 18)->UseManualTime()->Complexity();
BENCHMARK(BM_Complexity_O1)
    ->Range(1, 1 << 18)
    ->UseManualTime()
    ->Complexity([](benchmark::IterationCount) { return 1.0; });

const char *one_test_name = "BM_Complexity_O1/manual_time";
const char *big_o_1_test_name = "BM_Complexity_O1/manual_time_BigO";
const char *rms_o_1_test_name = "BM_Complexity_O1/manual_time_RMS";
const char *enum_auto_big_o_1 = "\\([0-9]+\\)";
const char *lambda_big_o_1 = "f\\(N\\)";

// Add enum tests
ADD_COMPLEXITY_CASES(one_test_name, big_o_1_test_name, rms_o_1_test_name,
                     enum_auto_big_o_1, /*family_index=*/0);

// Add auto tests
ADD_COMPLEXITY_CASES(one_test_name, big_o_1_test_name, rms_o_1_test_name,
                     enum_auto_big_o_1, /*family_index=*/1);

// Add lambda tests
ADD_COMPLEXITY_CASES(one_test_name, big_o_1_test_name, rms_o_1_test_name,
                     lambda_big_o_1, /*family_index=*/2);

// ========================================================================= //
// --------------------------- Testing BigO O(N) --------------------------- //
// ========================================================================= //

void BM_Complexity_O_N(benchmark::State &state) {
  for (auto _ : state) {
    // This test requires a non-zero CPU time to avoid divide-by-zero
    benchmark::DoNotOptimize(state.iterations());
    long tmp = state.iterations();
    benchmark::DoNotOptimize(tmp);
    for (benchmark::IterationCount i = 0; i < state.iterations(); ++i) {
      benchmark::DoNotOptimize(state.iterations());
      tmp *= state.iterations();
      benchmark::DoNotOptimize(tmp);
    }

    // 1ns per iteration per entry
    state.SetIterationTime(static_cast<double>(state.range(0)) * 42.0 * 1e-9);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Complexity_O_N)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 20)
    ->UseManualTime()
    ->Complexity(benchmark::oN);
BENCHMARK(BM_Complexity_O_N)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 20)
    ->UseManualTime()
    ->Complexity();
BENCHMARK(BM_Complexity_O_N)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 20)
    ->UseManualTime()
    ->Complexity([](benchmark::IterationCount n) -> double {
      return static_cast<double>(n);
    });

const char *n_test_name = "BM_Complexity_O_N/manual_time";
const char *big_o_n_test_name = "BM_Complexity_O_N/manual_time_BigO";
const char *rms_o_n_test_name = "BM_Complexity_O_N/manual_time_RMS";
const char *enum_auto_big_o_n = "N";
const char *lambda_big_o_n = "f\\(N\\)";

// Add enum tests
ADD_COMPLEXITY_CASES(n_test_name, big_o_n_test_name, rms_o_n_test_name,
                     enum_auto_big_o_n, /*family_index=*/3);

// Add auto tests
ADD_COMPLEXITY_CASES(n_test_name, big_o_n_test_name, rms_o_n_test_name,
                     enum_auto_big_o_n, /*family_index=*/4);

// Add lambda tests
ADD_COMPLEXITY_CASES(n_test_name, big_o_n_test_name, rms_o_n_test_name,
                     lambda_big_o_n, /*family_index=*/5);

// ========================================================================= //
// ------------------------- Testing BigO O(NlgN) ------------------------- //
// ========================================================================= //

static const double kLog2E = 1.44269504088896340736;
static void BM_Complexity_O_N_log_N(benchmark::State &state) {
  for (auto _ : state) {
    // This test requires a non-zero CPU time to avoid divide-by-zero
    benchmark::DoNotOptimize(state.iterations());
    long tmp = state.iterations();
    benchmark::DoNotOptimize(tmp);
    for (benchmark::IterationCount i = 0; i < state.iterations(); ++i) {
      benchmark::DoNotOptimize(state.iterations());
      tmp *= state.iterations();
      benchmark::DoNotOptimize(tmp);
    }

    state.SetIterationTime(static_cast<double>(state.range(0)) * kLog2E *
                           std::log(state.range(0)) * 42.0 * 1e-9);
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Complexity_O_N_log_N)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1U << 24)
    ->UseManualTime()
    ->Complexity(benchmark::oNLogN);
BENCHMARK(BM_Complexity_O_N_log_N)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1U << 24)
    ->UseManualTime()
    ->Complexity();
BENCHMARK(BM_Complexity_O_N_log_N)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1U << 24)
    ->UseManualTime()
    ->Complexity([](benchmark::IterationCount n) {
      return kLog2E * static_cast<double>(n) * std::log(static_cast<double>(n));
    });

const char *n_lg_n_test_name = "BM_Complexity_O_N_log_N/manual_time";
const char *big_o_n_lg_n_test_name = "BM_Complexity_O_N_log_N/manual_time_BigO";
const char *rms_o_n_lg_n_test_name = "BM_Complexity_O_N_log_N/manual_time_RMS";
const char *enum_auto_big_o_n_lg_n = "NlgN";
const char *lambda_big_o_n_lg_n = "f\\(N\\)";

// Add enum tests
ADD_COMPLEXITY_CASES(n_lg_n_test_name, big_o_n_lg_n_test_name,
                     rms_o_n_lg_n_test_name, enum_auto_big_o_n_lg_n,
                     /*family_index=*/6);

// NOTE: auto big-o is wron.g
ADD_COMPLEXITY_CASES(n_lg_n_test_name, big_o_n_lg_n_test_name,
                     rms_o_n_lg_n_test_name, enum_auto_big_o_n_lg_n,
                     /*family_index=*/7);

//// Add lambda tests
ADD_COMPLEXITY_CASES(n_lg_n_test_name, big_o_n_lg_n_test_name,
                     rms_o_n_lg_n_test_name, lambda_big_o_n_lg_n,
                     /*family_index=*/8);

// ========================================================================= //
// -------- Testing formatting of Complexity with captured args ------------ //
// ========================================================================= //

void BM_ComplexityCaptureArgs(benchmark::State &state, int n) {
  for (auto _ : state) {
    // This test requires a non-zero CPU time to avoid divide-by-zero
    benchmark::DoNotOptimize(state.iterations());
    long tmp = state.iterations();
    benchmark::DoNotOptimize(tmp);
    for (benchmark::IterationCount i = 0; i < state.iterations(); ++i) {
      benchmark::DoNotOptimize(state.iterations());
      tmp *= state.iterations();
      benchmark::DoNotOptimize(tmp);
    }

    state.SetIterationTime(static_cast<double>(state.range(0)) * 42.0 * 1e-9);
  }
  state.SetComplexityN(n);
}

BENCHMARK_CAPTURE(BM_ComplexityCaptureArgs, capture_test, 100)
    ->UseManualTime()
    ->Complexity(benchmark::oN)
    ->Ranges({{1, 2}, {3, 4}});

const std::string complexity_capture_name =
    "BM_ComplexityCaptureArgs/capture_test/manual_time";

ADD_COMPLEXITY_CASES(complexity_capture_name, complexity_capture_name + "_BigO",
                     complexity_capture_name + "_RMS", "N",
                     /*family_index=*/9);

// ========================================================================= //
// --------------------------- TEST CASES END ------------------------------ //
// ========================================================================= //

int main(int argc, char *argv[]) { RunOutputTests(argc, argv); }
