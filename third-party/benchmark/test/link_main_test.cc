#include "benchmark/benchmark.h"

void BM_empty(benchmark::State& state) {
  for (auto _ : state) {
    auto iterations = double(state.iterations()) * double(state.iterations());
    benchmark::DoNotOptimize(iterations);
  }
}
BENCHMARK(BM_empty);
