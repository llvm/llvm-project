#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"

// Tests that we can specify the number of iterations with
// --benchmark_min_time=<NUM>x.
namespace {

class TestReporter : public benchmark::ConsoleReporter {
 public:
  bool ReportContext(const Context& context) override {
    return ConsoleReporter::ReportContext(context);
  };

  void ReportRuns(const std::vector<Run>& report) override {
    assert(report.size() == 1);
    iter_nums_.push_back(report[0].iterations);
    ConsoleReporter::ReportRuns(report);
  };

  TestReporter() {}

  ~TestReporter() override {}

  const std::vector<benchmark::IterationCount>& GetIters() const {
    return iter_nums_;
  }

 private:
  std::vector<benchmark::IterationCount> iter_nums_;
};

void BM_MyBench(benchmark::State& state) {
  for (auto s : state) {
  }
}
}  // end namespace

BENCHMARK(BM_MyBench);

int main(int argc, char** argv) {
  benchmark::MaybeReenterWithoutASLR(argc, argv);

  // Make a fake argv and append the new --benchmark_min_time=<foo> to it.
  int fake_argc = argc + 1;
  std::vector<const char*> fake_argv(static_cast<size_t>(fake_argc));
  for (size_t i = 0; i < static_cast<size_t>(argc); ++i) {
    fake_argv[i] = argv[i];
  }
  fake_argv[static_cast<size_t>(argc)] = "--benchmark_min_time=4x";

  benchmark::Initialize(&fake_argc, const_cast<char**>(fake_argv.data()));

  TestReporter test_reporter;
  const size_t returned_count =
      benchmark::RunSpecifiedBenchmarks(&test_reporter, "BM_MyBench");
  assert(returned_count == 1);

  // Check the executed iters.
  const std::vector<benchmark::IterationCount> iters = test_reporter.GetIters();
  assert(!iters.empty() && iters[0] == 4);

  return 0;
}
