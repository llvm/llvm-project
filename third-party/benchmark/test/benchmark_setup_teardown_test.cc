#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

#include "benchmark/benchmark.h"

// Test that Setup() and Teardown() are called exactly once
// for each benchmark run (single-threaded).
namespace singlethreaded {
static int setup_call = 0;
static int teardown_call = 0;
}  // namespace singlethreaded
static void DoSetup1(const benchmark::State& state) {
  ++singlethreaded::setup_call;

  // Setup/Teardown should never be called with any thread_idx != 0.
  assert(state.thread_index() == 0);
}

static void DoTeardown1(const benchmark::State& state) {
  ++singlethreaded::teardown_call;
  assert(state.thread_index() == 0);
}

static void BM_with_setup(benchmark::State& state) {
  for (auto s : state) {
  }
}
BENCHMARK(BM_with_setup)
    ->Arg(1)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7)
    ->Iterations(100)
    ->Setup(DoSetup1)
    ->Teardown(DoTeardown1);

// Test that Setup() and Teardown() are called once for each group of threads.
namespace concurrent {
static std::atomic<int> setup_call(0);
static std::atomic<int> teardown_call(0);
static std::atomic<int> func_call(0);
}  // namespace concurrent

static void DoSetup2(const benchmark::State& state) {
  concurrent::setup_call.fetch_add(1, std::memory_order_acquire);
  assert(state.thread_index() == 0);
}

static void DoTeardown2(const benchmark::State& state) {
  concurrent::teardown_call.fetch_add(1, std::memory_order_acquire);
  assert(state.thread_index() == 0);
}

static void BM_concurrent(benchmark::State& state) {
  for (auto s : state) {
  }
  concurrent::func_call.fetch_add(1, std::memory_order_acquire);
}

BENCHMARK(BM_concurrent)
    ->Setup(DoSetup2)
    ->Teardown(DoTeardown2)
    ->Iterations(100)
    ->Threads(5)
    ->Threads(10)
    ->Threads(15);

// Testing interaction with Fixture::Setup/Teardown
namespace fixture_interaction {
int setup = 0;
int fixture_setup = 0;
}  // namespace fixture_interaction

#define FIXTURE_BECHMARK_NAME MyFixture

class FIXTURE_BECHMARK_NAME : public ::benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State&) override {
    fixture_interaction::fixture_setup++;
  }

  ~FIXTURE_BECHMARK_NAME() override {}
};

BENCHMARK_F(FIXTURE_BECHMARK_NAME, BM_WithFixture)(benchmark::State& st) {
  for (auto _ : st) {
  }
}

static void DoSetupWithFixture(const benchmark::State&) {
  fixture_interaction::setup++;
}

BENCHMARK_REGISTER_F(FIXTURE_BECHMARK_NAME, BM_WithFixture)
    ->Arg(1)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7)
    ->Setup(DoSetupWithFixture)
    ->Repetitions(1)
    ->Iterations(100);

// Testing repetitions.
namespace repetitions {
int setup = 0;
}

static void DoSetupWithRepetitions(const benchmark::State&) {
  repetitions::setup++;
}
static void BM_WithRep(benchmark::State& state) {
  for (auto _ : state) {
  }
}

BENCHMARK(BM_WithRep)
    ->Arg(1)
    ->Arg(3)
    ->Arg(5)
    ->Arg(7)
    ->Setup(DoSetupWithRepetitions)
    ->Iterations(100)
    ->Repetitions(4);

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);

  size_t ret = benchmark::RunSpecifiedBenchmarks(".");
  assert(ret > 0);

  // Setup/Teardown is called once for each arg group (1,3,5,7).
  assert(singlethreaded::setup_call == 4);
  assert(singlethreaded::teardown_call == 4);

  // 3 group of threads calling this function (3,5,10).
  assert(concurrent::setup_call.load(std::memory_order_relaxed) == 3);
  assert(concurrent::teardown_call.load(std::memory_order_relaxed) == 3);
  assert((5 + 10 + 15) ==
         concurrent::func_call.load(std::memory_order_relaxed));

  // Setup is called 4 times, once for each arg group (1,3,5,7)
  assert(fixture_interaction::setup == 4);
  // Fixture::Setup is called every time the bm routine is run.
  // The exact number is indeterministic, so we just assert that
  // it's more than setup.
  assert(fixture_interaction::fixture_setup > fixture_interaction::setup);

  // Setup is call once for each repetition * num_arg =  4 * 4 = 16.
  assert(repetitions::setup == 16);

  return 0;
}
