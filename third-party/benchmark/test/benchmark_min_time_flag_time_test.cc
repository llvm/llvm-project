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
  virtual bool ReportContext(const Context& context) BENCHMARK_OVERRIDE {
    return ConsoleReporter::ReportContext(context);
  };

  virtual void ReportRuns(const std::vector<Run>& report) BENCHMARK_OVERRIDE {
    assert(report.size() == 1);
    ConsoleReporter::ReportRuns(report);
  };

  virtual void ReportRunsConfig(double min_time, bool /* has_explicit_iters */,
                                IterationCount /* iters */) BENCHMARK_OVERRIDE {
    min_times_.push_back(min_time);
  }

  TestReporter() {}

  virtual ~TestReporter() {}

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

}  // end namespace

static void BM_MyBench(benchmark::State& state) {
  for (auto s : state) {
  }
}
BENCHMARK(BM_MyBench);

int main(int argc, char** argv) {
  // Make a fake argv and append the new --benchmark_min_time=<foo> to it.
  int fake_argc = argc + 1;
  const char** fake_argv = new const char*[static_cast<size_t>(fake_argc)];

  for (int i = 0; i < argc; ++i) fake_argv[i] = argv[i];

  const char* no_suffix = "--benchmark_min_time=4";
  const char* with_suffix = "--benchmark_min_time=4.0s";
  double expected = 4.0;

  fake_argv[argc] = no_suffix;
  DoTestHelper(&fake_argc, fake_argv, expected);

  fake_argv[argc] = with_suffix;
  DoTestHelper(&fake_argc, fake_argv, expected);

  delete[] fake_argv;
  return 0;
}
