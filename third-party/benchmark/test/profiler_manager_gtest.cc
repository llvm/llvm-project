#include <memory>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

namespace {

class TestProfilerManager : public benchmark::ProfilerManager {
 public:
  void AfterSetupStart() override { ++start_called; }
  void BeforeTeardownStop() override { ++stop_called; }

  int start_called = 0;
  int stop_called = 0;
};

void BM_empty(benchmark::State& state) {
  for (auto _ : state) {
    auto iterations = state.iterations();
    benchmark::DoNotOptimize(iterations);
  }
}
BENCHMARK(BM_empty);

TEST(ProfilerManager, ReregisterManager) {
#if GTEST_HAS_DEATH_TEST
  // Tests only runnable in debug mode (when BM_CHECK is enabled).
#ifndef NDEBUG
#ifndef TEST_BENCHMARK_LIBRARY_HAS_NO_ASSERTIONS
  ASSERT_DEATH_IF_SUPPORTED(
      {
        std::unique_ptr<TestProfilerManager> pm(new TestProfilerManager());
        benchmark::RegisterProfilerManager(pm.get());
        benchmark::RegisterProfilerManager(pm.get());
      },
      "RegisterProfilerManager");
#endif
#endif
#endif
}

}  // namespace
