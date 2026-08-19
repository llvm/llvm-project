#include "benchmark/benchmark.h"

namespace {
// Simulate an overloaded function name.
// This version does nothing and is just here to create ambiguity for
// MyOverloadedBenchmark.
BENCHMARK_UNUSED void MyOverloadedBenchmark() {}

// This is the actual benchmark function we want to register.
// It has the signature void(benchmark::State&) required by the library.
void MyOverloadedBenchmark(benchmark::State& state) {
  for (auto _ : state) {
  }
}

// This macro invocation should compile correctly if benchmark.h
// contains the fix (using static_cast), but would fail to compile
// if the benchmark name were ambiguous (e.g., when using + or no cast
// with an overloaded function).
BENCHMARK(MyOverloadedBenchmark);

// Also test BENCHMARK_TEMPLATE with an overloaded name.
template <int N>
void MyTemplatedOverloadedBenchmark() {}

template <int N>
void MyTemplatedOverloadedBenchmark(benchmark::State& state) {
  for (auto _ : state) {
  }
}

BENCHMARK_TEMPLATE(MyTemplatedOverloadedBenchmark, 1);
}  // end namespace

BENCHMARK_MAIN();
