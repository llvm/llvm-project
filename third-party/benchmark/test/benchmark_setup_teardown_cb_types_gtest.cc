#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

using benchmark::Benchmark;
using benchmark::BenchmarkReporter;
using benchmark::callback_function;
using benchmark::ClearRegisteredBenchmarks;
using benchmark::RegisterBenchmark;
using benchmark::RunSpecifiedBenchmarks;
using benchmark::State;

static int functor_called = 0;
struct Functor {
  void operator()(const benchmark::State& /*unused*/) { functor_called++; }
};

class NullReporter : public BenchmarkReporter {
 public:
  bool ReportContext(const Context& /*context*/) override { return true; }
  void ReportRuns(const std::vector<Run>& /* report */) override {}
};

class BenchmarkTest : public testing::Test {
 public:
  Benchmark* bm;
  NullReporter null_reporter;

  int setup_calls;
  int teardown_calls;

  void SetUp() override {
    setup_calls = 0;
    teardown_calls = 0;
    functor_called = 0;

    bm = RegisterBenchmark("BM", [](State& st) {
      for (auto _ : st) {
      }
    });
    bm->Iterations(1);
  }

  void TearDown() override { ClearRegisteredBenchmarks(); }
};

// Test that Setup/Teardown can correctly take a lambda expressions
TEST_F(BenchmarkTest, LambdaTestCopy) {
  auto setup_lambda = [this](const State&) { setup_calls++; };
  auto teardown_lambda = [this](const State&) { teardown_calls++; };
  bm->Setup(setup_lambda);
  bm->Teardown(teardown_lambda);
  RunSpecifiedBenchmarks(&null_reporter);
  EXPECT_EQ(setup_calls, 1);
  EXPECT_EQ(teardown_calls, 1);
}

// Test that Setup/Teardown can correctly take a lambda expressions
TEST_F(BenchmarkTest, LambdaTestMove) {
  auto setup_lambda = [this](const State&) { setup_calls++; };
  auto teardown_lambda = [this](const State&) { teardown_calls++; };
  bm->Setup(std::move(setup_lambda));
  bm->Teardown(std::move(teardown_lambda));
  RunSpecifiedBenchmarks(&null_reporter);
  EXPECT_EQ(setup_calls, 1);
  EXPECT_EQ(teardown_calls, 1);
}

// Test that Setup/Teardown can correctly take std::function
TEST_F(BenchmarkTest, CallbackFunctionCopy) {
  callback_function setup_lambda = [this](const State&) { setup_calls++; };
  callback_function teardown_lambda = [this](const State&) {
    teardown_calls++;
  };
  bm->Setup(setup_lambda);
  bm->Teardown(teardown_lambda);
  RunSpecifiedBenchmarks(&null_reporter);
  EXPECT_EQ(setup_calls, 1);
  EXPECT_EQ(teardown_calls, 1);
}

// Test that Setup/Teardown can correctly take std::function
TEST_F(BenchmarkTest, CallbackFunctionMove) {
  callback_function setup_lambda = [this](const State&) { setup_calls++; };
  callback_function teardown_lambda = [this](const State&) {
    teardown_calls++;
  };
  bm->Setup(std::move(setup_lambda));
  bm->Teardown(std::move(teardown_lambda));
  RunSpecifiedBenchmarks(&null_reporter);
  EXPECT_EQ(setup_calls, 1);
  EXPECT_EQ(teardown_calls, 1);
}

// Test that Setup/Teardown can correctly take functors
TEST_F(BenchmarkTest, FunctorCopy) {
  Functor func;
  bm->Setup(func);
  bm->Teardown(func);
  RunSpecifiedBenchmarks(&null_reporter);
  EXPECT_EQ(functor_called, 2);
}

// Test that Setup/Teardown can correctly take functors
TEST_F(BenchmarkTest, FunctorMove) {
  Functor func1;
  Functor func2;
  bm->Setup(std::move(func1));
  bm->Teardown(std::move(func2));
  RunSpecifiedBenchmarks(&null_reporter);
  EXPECT_EQ(functor_called, 2);
}

// Test that Setup/Teardown can not take nullptr
TEST_F(BenchmarkTest, NullptrTest) {
#if GTEST_HAS_DEATH_TEST
  // Tests only runnable in debug mode (when BM_CHECK is enabled).
#ifndef NDEBUG
#ifndef TEST_BENCHMARK_LIBRARY_HAS_NO_ASSERTIONS
  EXPECT_DEATH(bm->Setup(nullptr), "setup != nullptr");
  EXPECT_DEATH(bm->Teardown(nullptr), "teardown != nullptr");
#else
  GTEST_SKIP() << "Test skipped because BM_CHECK is disabled";
#endif
#endif
#endif
}
