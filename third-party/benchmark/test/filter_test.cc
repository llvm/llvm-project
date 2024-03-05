#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "benchmark/benchmark.h"

namespace {

class TestReporter : public benchmark::ConsoleReporter {
 public:
  bool ReportContext(const Context& context) override {
    return ConsoleReporter::ReportContext(context);
  };

  void ReportRuns(const std::vector<Run>& report) override {
    ++count_;
    max_family_index_ = std::max(max_family_index_, report[0].family_index);
    ConsoleReporter::ReportRuns(report);
  };

  TestReporter() : count_(0), max_family_index_(0) {}

  ~TestReporter() override {}

  int GetCount() const { return count_; }

  int64_t GetMaxFamilyIndex() const { return max_family_index_; }

 private:
  mutable int count_;
  mutable int64_t max_family_index_;
};

}  // end namespace

static void NoPrefix(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(NoPrefix);

static void BM_Foo(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_Foo);

static void BM_Bar(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_Bar);

static void BM_FooBar(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_FooBar);

static void BM_FooBa(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_FooBa);

int main(int argc, char** argv) {
  bool list_only = false;
  for (int i = 0; i < argc; ++i)
    list_only |= std::string(argv[i]).find("--benchmark_list_tests") !=
                 std::string::npos;

  benchmark::Initialize(&argc, argv);

  TestReporter test_reporter;
  const int64_t returned_count =
      static_cast<int64_t>(benchmark::RunSpecifiedBenchmarks(&test_reporter));

  if (argc == 2) {
    // Make sure we ran all of the tests
    std::stringstream ss(argv[1]);
    int64_t expected_return;
    ss >> expected_return;

    if (returned_count != expected_return) {
      std::cerr << "ERROR: Expected " << expected_return
                << " tests to match the filter but returned_count = "
                << returned_count << std::endl;
      return -1;
    }

    const int64_t expected_reports = list_only ? 0 : expected_return;
    const int64_t reports_count = test_reporter.GetCount();
    if (reports_count != expected_reports) {
      std::cerr << "ERROR: Expected " << expected_reports
                << " tests to be run but reported_count = " << reports_count
                << std::endl;
      return -1;
    }

    const int64_t max_family_index = test_reporter.GetMaxFamilyIndex();
    const int64_t num_families = reports_count == 0 ? 0 : 1 + max_family_index;
    if (num_families != expected_reports) {
      std::cerr << "ERROR: Expected " << expected_reports
                << " test families to be run but num_families = "
                << num_families << std::endl;
      return -1;
    }
  }

  return 0;
}
