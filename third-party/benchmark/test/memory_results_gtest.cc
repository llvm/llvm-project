#include <vector>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

namespace {

using benchmark::Benchmark;
using benchmark::ClearRegisteredBenchmarks;
using benchmark::ConsoleReporter;
using benchmark::MemoryManager;
using benchmark::RegisterBenchmark;
using benchmark::RunSpecifiedBenchmarks;
using benchmark::State;

constexpr int N_REPETITIONS = 100;
constexpr int N_ITERATIONS = 1;

int num_allocs = 0;
int max_bytes_used = 0;
int total_allocated_bytes = 0;
int net_heap_growth = 0;

void reset() {
  num_allocs = 0;
  max_bytes_used = 0;
  total_allocated_bytes = 0;
  net_heap_growth = 0;
}
class TestMemoryManager : public MemoryManager {
  void Start() override {}
  void Stop(Result& result) override {
    result.num_allocs = num_allocs;
    result.net_heap_growth = net_heap_growth;
    result.max_bytes_used = max_bytes_used;
    result.total_allocated_bytes = total_allocated_bytes;

    num_allocs += 1;
    max_bytes_used += 2;
    net_heap_growth += 4;
    total_allocated_bytes += 10;
  }
};

class TestReporter : public ConsoleReporter {
 public:
  TestReporter() = default;
  virtual ~TestReporter() = default;

  bool ReportContext(const Context& /*unused*/) override { return true; }

  void PrintHeader(const Run&) override {}
  void PrintRunData(const Run& run) override {
    if (run.repetition_index == -1) return;
    if (!run.memory_result.memory_iterations) return;

    store.push_back(run.memory_result);
  }

  std::vector<MemoryManager::Result> store;
};

class MemoryResultsTest : public testing::Test {
 public:
  Benchmark* bm;
  TestReporter reporter;

  void SetUp() override {
    bm = RegisterBenchmark("BM", [](State& st) {
      for (auto _ : st) {
      }
    });
    bm->Repetitions(N_REPETITIONS);
    bm->Iterations(N_ITERATIONS);
    reset();
  }
  void TearDown() override { ClearRegisteredBenchmarks(); }
};

TEST_F(MemoryResultsTest, NoMMTest) {
  RunSpecifiedBenchmarks(&reporter);
  EXPECT_EQ(reporter.store.size(), 0);
}

TEST_F(MemoryResultsTest, ResultsTest) {
  auto mm = std::make_unique<TestMemoryManager>();
  RegisterMemoryManager(mm.get());

  RunSpecifiedBenchmarks(&reporter);
  EXPECT_EQ(reporter.store.size(), N_REPETITIONS);

  for (size_t i = 0; i < reporter.store.size(); i++) {
    EXPECT_EQ(reporter.store[i].num_allocs, static_cast<int64_t>(i));
    EXPECT_EQ(reporter.store[i].max_bytes_used, static_cast<int64_t>(i) * 2);
    EXPECT_EQ(reporter.store[i].net_heap_growth, static_cast<int64_t>(i) * 4);
    EXPECT_EQ(reporter.store[i].total_allocated_bytes,
              static_cast<int64_t>(i) * 10);
  }
}

}  // namespace
