#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"

// Tests that we can specify the number of iterations with
// --benchmark_min_time=<NUM>x.
namespace {

class TestReporter : public benchmark::ConsoleReporter {
 public:
  virtual bool ReportContext(const Context& context) BENCHMARK_OVERRIDE {
    return ConsoleReporter::ReportContext(context);
  };

  virtual void ReportRuns(const std::vector<Run>& report) BENCHMARK_OVERRIDE {
    assert(report.size() == 1);
    iter_nums_.push_back(report[0].iterations);
    ConsoleReporter::ReportRuns(report);
  };

  TestReporter() {}

  virtual ~TestReporter() {}

  const std::vector<benchmark::IterationCount>& GetIters() const {
    return iter_nums_;
  }

 private:
  std::vector<benchmark::IterationCount> iter_nums_;
};

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
  fake_argv[argc] = "--benchmark_min_time=4x";

  benchmark::Initialize(&fake_argc, const_cast<char**>(fake_argv));

  TestReporter test_reporter;
  const size_t returned_count =
      benchmark::RunSpecifiedBenchmarks(&test_reporter, "BM_MyBench");
  assert(returned_count == 1);

  // Check the executed iters.
  const std::vector<benchmark::IterationCount> iters = test_reporter.GetIters();
  assert(!iters.empty() && iters[0] == 4);

  delete[] fake_argv;
  return 0;
}
