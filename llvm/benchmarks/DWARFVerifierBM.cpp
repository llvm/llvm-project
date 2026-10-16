//===- DWARFVerifierBM.cpp - DieRangeInfo::insert benchmark ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "llvm/DebugInfo/DWARF/DWARFVerifier.h"
#include <algorithm>
#include <random>
#include <vector>

using namespace llvm;
using DieRangeInfo = DWARFVerifier::DieRangeInfo;

static DieRangeInfo makeRI(uint64_t Lo, uint64_t Hi) {
  return DieRangeInfo({{Lo, Hi}});
}

// Insert N non-overlapping ranges in forward address order.
static void BM_DieRangeInfoInsertForward(benchmark::State &State) {
  const unsigned N = State.range(0);
  for (auto _ : State) {
    DieRangeInfo Parent;
    for (unsigned I = 0; I < N; ++I) {
      uint64_t Lo = I * 100;
      uint64_t Hi = Lo + 50;
      Parent.insert(makeRI(Lo, Hi));
    }
    benchmark::DoNotOptimize(Parent);
  }
}
BENCHMARK(BM_DieRangeInfoInsertForward)->Arg(1000)->Arg(10000)->Arg(100000);

// Insert N non-overlapping ranges in reverse address order.
static void BM_DieRangeInfoInsertReverse(benchmark::State &State) {
  const unsigned N = State.range(0);
  for (auto _ : State) {
    DieRangeInfo Parent;
    for (unsigned I = N; I > 0; --I) {
      uint64_t Lo = I * 100;
      uint64_t Hi = Lo + 50;
      Parent.insert(makeRI(Lo, Hi));
    }
    benchmark::DoNotOptimize(Parent);
  }
}
BENCHMARK(BM_DieRangeInfoInsertReverse)->Arg(1000)->Arg(10000)->Arg(100000);

// Insert N non-overlapping ranges in random order.
static void BM_DieRangeInfoInsertRandom(benchmark::State &State) {
  const unsigned N = State.range(0);

  std::vector<std::pair<uint64_t, uint64_t>> Ranges;
  Ranges.reserve(N);
  for (unsigned I = 0; I < N; ++I)
    Ranges.push_back({I * 100, I * 100 + 50});
  std::mt19937 RNG(42);
  std::shuffle(Ranges.begin(), Ranges.end(), RNG);

  for (auto _ : State) {
    DieRangeInfo Parent;
    for (const auto &[Lo, Hi] : Ranges)
      Parent.insert(makeRI(Lo, Hi));
    benchmark::DoNotOptimize(Parent);
  }
}
BENCHMARK(BM_DieRangeInfoInsertRandom)->Arg(1000)->Arg(10000)->Arg(100000);

// Measure single overlap detection after N-1 insertions.
static void BM_DieRangeInfoOverlapDetection(benchmark::State &State) {
  const unsigned N = State.range(0);

  // Pre-build the parent with N-1 non-overlapping ranges.
  DieRangeInfo Base;
  for (unsigned I = 0; I < N - 1; ++I) {
    uint64_t Lo = I * 100;
    uint64_t Hi = Lo + 50;
    Base.insert(makeRI(Lo, Hi));
  }

  uint64_t Mid = (N / 2) * 100;
  for (auto _ : State) {
    DieRangeInfo Parent = Base;
    auto It = Parent.insert(makeRI(Mid + 10, Mid + 60));
    benchmark::DoNotOptimize(It);
  }
}
BENCHMARK(BM_DieRangeInfoOverlapDetection)->Arg(1000)->Arg(10000)->Arg(100000);

BENCHMARK_MAIN();
