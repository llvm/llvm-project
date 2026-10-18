//===- CodeLayout.cpp - Code layout benchmarks ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CodeLayout.h"
#include "benchmark/benchmark.h"

#include <cstdint>
#include <vector>

using namespace llvm;
using namespace llvm::codelayout;

namespace {

static void BM_ExtTSP(benchmark::State &State) {
  const size_t NumNodes = State.range(0);
  const std::vector<uint64_t> NodeSizes(NumNodes, 16);
  const std::vector<uint64_t> NodeCounts(NumNodes, 100);
  std::vector<EdgeCount> EdgeCounts;
  EdgeCounts.reserve(2 * NumNodes);

  // Use two successors and two predecessors for each node to prevent the
  // forced-pair pass from collapsing the graph before mergeChainPairs runs.
  for (size_t I = 0; I < NumNodes; ++I) {
    EdgeCounts.push_back({I, (I + 1) % NumNodes, 60});
    EdgeCounts.push_back({I, (I + 2) % NumNodes, 40});
  }

  for (auto _ : State) {
    auto Order = computeExtTspLayout(NodeSizes, NodeCounts, EdgeCounts);
    benchmark::DoNotOptimize(Order);
  }

  State.SetComplexityN(NumNodes);
}

} // namespace

BENCHMARK(BM_ExtTSP)->RangeMultiplier(2)->Range(1 << 10, 1 << 14)->Complexity();

BENCHMARK_MAIN();
