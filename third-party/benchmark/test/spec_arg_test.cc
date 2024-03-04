#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"

// Tests that we can override benchmark-spec value from FLAGS_benchmark_filter
// with argument to RunSpecifiedBenchmarks(...).

namespace {

class TestReporter : public benchmark::ConsoleReporter {
 public:
  virtual bool ReportContext(const Context& context) BENCHMARK_OVERRIDE {
    return ConsoleReporter::ReportContext(context);
  };

  virtual void ReportRuns(const std::vector<Run>& report) BENCHMARK_OVERRIDE {
    assert(report.size() == 1);
    matched_functions.push_back(report[0].run_name.function_name);
    ConsoleReporter::ReportRuns(report);
  };

  TestReporter() {}

  virtual ~TestReporter() {}

  const std::vector<std::string>& GetMatchedFunctions() const {
    return matched_functions;
  }

 private:
  std::vector<std::string> matched_functions;
};

}  // end namespace

static void BM_NotChosen(benchmark::State& state) {
  assert(false && "SHOULD NOT BE CALLED");
  for (auto _ : state) {
  }
}
BENCHMARK(BM_NotChosen);

static void BM_Chosen(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_Chosen);

int main(int argc, char** argv) {
  const std::string flag = "BM_NotChosen";

  // Verify that argv specify --benchmark_filter=BM_NotChosen.
  bool found = false;
  for (int i = 0; i < argc; ++i) {
    if (strcmp("--benchmark_filter=BM_NotChosen", argv[i]) == 0) {
      found = true;
      break;
    }
  }
  assert(found);

  benchmark::Initialize(&argc, argv);

  // Check that the current flag value is reported accurately via the
  // GetBenchmarkFilter() function.
  if (flag != benchmark::GetBenchmarkFilter()) {
    std::cerr
        << "Seeing different value for flags. GetBenchmarkFilter() returns ["
        << benchmark::GetBenchmarkFilter() << "] expected flag=[" << flag
        << "]\n";
    return 1;
  }
  TestReporter test_reporter;
  const char* const spec = "BM_Chosen";
  const size_t returned_count =
      benchmark::RunSpecifiedBenchmarks(&test_reporter, spec);
  assert(returned_count == 1);
  const std::vector<std::string> matched_functions =
      test_reporter.GetMatchedFunctions();
  assert(matched_functions.size() == 1);
  if (strcmp(spec, matched_functions.front().c_str()) != 0) {
    std::cerr << "Expected benchmark [" << spec << "] to run, but got ["
              << matched_functions.front() << "]\n";
    return 2;
  }
  return 0;
}
