#undef NDEBUG

#include "../src/perf_counters.h"

#include "benchmark/benchmark.h"
#include "output_test.h"

static void BM_Simple(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(state.iterations());
  }
}
BENCHMARK(BM_Simple);
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_Simple\",$"}});

static void CheckSimple(Results const& e) {
  CHECK_COUNTER_VALUE(e, double, "CYCLES", GT, 0);
  CHECK_COUNTER_VALUE(e, double, "BRANCHES", GT, 0.0);
}
CHECK_BENCHMARK_RESULTS("BM_Simple", &CheckSimple);

int main(int argc, char* argv[]) {
  if (!benchmark::internal::PerfCounters::kSupported) {
    return 0;
  }
  RunOutputTests(argc, argv);
}
