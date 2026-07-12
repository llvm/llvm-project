//===- ImmutableSetBuildBM.cpp - Benchmark ImmutableSet construction ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Benchmarks the mutating operations of ImmutableSet -- add and remove -- which
// exercise the ImutAVLFactory node bookkeeping (createNode / recoverNodes) that
// the iterator benchmark deliberately keeps out of its timed region. Run the
// binary before and after a change and compare the two reports (e.g. with
// llvm/utils/compare.py).
//
//   * Build       - build an N-element set from empty (N add calls).
//   * BuildRemove - build an N-element set, then remove every element.
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "llvm/ADT/ImmutableSet.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

using namespace llvm;

namespace {

// A shuffled [0, N) key sequence, so the tree is built in random order.
static std::vector<int> shuffledKeys(size_t N) {
  std::vector<int> Vals(N);
  std::iota(Vals.begin(), Vals.end(), 0);
  std::mt19937 Rng(0xC0FFEE);
  std::shuffle(Vals.begin(), Vals.end(), Rng);
  return Vals;
}

static void BM_Build(benchmark::State &State) {
  const size_t N = State.range(0);
  std::vector<int> Vals = shuffledKeys(N);
  for (auto _ : State) {
    ImmutableSet<int>::Factory F(/*canonicalize=*/false);
    ImmutableSet<int> S = F.getEmptySet();
    for (int V : Vals)
      S = F.add(S, V);
    benchmark::DoNotOptimize(S.getRootWithoutRetain());
  }
  State.SetItemsProcessed(State.iterations() * N);
}

static void BM_BuildRemove(benchmark::State &State) {
  const size_t N = State.range(0);
  std::vector<int> Vals = shuffledKeys(N);
  for (auto _ : State) {
    ImmutableSet<int>::Factory F(/*canonicalize=*/false);
    ImmutableSet<int> S = F.getEmptySet();
    for (int V : Vals)
      S = F.add(S, V);
    for (int V : Vals)
      S = F.remove(S, V);
    benchmark::DoNotOptimize(S.getRootWithoutRetain());
  }
  State.SetItemsProcessed(State.iterations() * N * 2);
}

} // namespace

#define BUILD_SIZES Arg(256)->Arg(4096)->Arg(65536)

BENCHMARK(BM_Build)->Name("Build")->BUILD_SIZES;
BENCHMARK(BM_BuildRemove)->Name("BuildRemove")->BUILD_SIZES;

BENCHMARK_MAIN();
