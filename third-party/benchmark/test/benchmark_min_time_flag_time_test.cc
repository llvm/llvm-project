#include <cassert>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"

// Tests that we can specify the min time with
// --benchmark_min_time=<NUM> (no suffix needed) OR
// --benchmark_min_time=<NUM>s
namespace {

// This is from benchmark.h
typedef int64_t IterationCount;

class TestReporter : public benchmark::ConsoleReporter {
 public:
  bool ReportContext(const Context& context) override {
    return ConsoleReporter::ReportContext(context);
  };

  void ReportRuns(const std::vector<Run>& report) override {
    assert(report.size() == 1);
    ConsoleReporter::ReportRuns(report);
  };

  void ReportRunsConfig(double min_time, bool /* has_explicit_iters */,
                        IterationCount /* iters */) override {
    min_times_.push_back(min_time);
  }

  TestReporter() {}

  ~TestReporter() override {}

  const std::vector<double>& GetMinTimes() const { return min_times_; }

 private:
  std::vector<double> min_times_;
};

bool AlmostEqual(double a, double b) {
  return std::fabs(a - b) < std::numeric_limits<double>::epsilon();
}

void DoTestHelper(int* argc, const char** argv, double expected) {
  benchmark::Initialize(argc, const_cast<char**>(argv));

  TestReporter test_reporter;
  const size_t returned_count =
      benchmark::RunSpecifiedBenchmarks(&test_reporter, "BM_MyBench");
  assert(returned_count == 1);

  // Check the min_time
  const std::vector<double>& min_times = test_reporter.GetMinTimes();
  assert(!min_times.empty() && AlmostEqual(min_times[0], expected));
}

void BM_MyBench(benchmark::State& state) {
  for (auto s : state) {
  }
}
BENCHMARK(BM_MyBench);

}  // end namespace

int main(int argc, char** argv) {
  benchmark::MaybeReenterWithoutASLR(argc, argv);

  // Make a fake argv and append the new --benchmark_min_time=<foo> to it.
  int fake_argc = argc + 1;
  std::vector<const char*> fake_argv(static_cast<size_t>(fake_argc));

  for (size_t i = 0; i < static_cast<size_t>(argc); ++i) {
    fake_argv[i] = argv[i];
  }

  const char* no_suffix = "--benchmark_min_time=4";
  const char* with_suffix = "--benchmark_min_time=4.0s";
  double expected = 4.0;

  fake_argv[static_cast<size_t>(argc)] = no_suffix;
  DoTestHelper(&fake_argc, fake_argv.data(), expected);

  fake_argv[static_cast<size_t>(argc)] = with_suffix;
  DoTestHelper(&fake_argc, fake_argv.data(), expected);

  return 0;
}
