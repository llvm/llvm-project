#include <string.h>

#include <iostream>

#include "benchmark/benchmark.h"

// Tests that the user specified verbosity level can be get.
static void BM_Verbosity(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_Verbosity);

int main(int argc, char** argv) {
  const int32_t flagv = 42;

  // Verify that argv specify --v=42.
  bool found = false;
  for (int i = 0; i < argc; ++i) {
    if (strcmp("--v=42", argv[i]) == 0) {
      found = true;
      break;
    }
  }
  if (!found) {
    std::cerr << "This test requires '--v=42' to be passed as a command-line "
              << "argument.\n";
    return 1;
  }

  benchmark::Initialize(&argc, argv);

  // Check that the current flag value is reported accurately via the
  // GetBenchmarkVerbosity() function.
  if (flagv != benchmark::GetBenchmarkVerbosity()) {
    std::cerr
        << "Seeing different value for flags. GetBenchmarkVerbosity() returns ["
        << benchmark::GetBenchmarkVerbosity() << "] expected flag=[" << flagv
        << "]\n";
    return 1;
  }
  return 0;
}
