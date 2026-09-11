//===- ImmutableMergeBM.cpp - Benchmark ImmutableSet/Map merges -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compares the single-pass, structure-sharing merge (ImmutableSet::unionSets /
// ImmutableMap::mergeWith) against the per-element add loop it replaced, on the
// operand shapes that dominate dataflow joins:
//
//   * Small     - both operands tiny (constant-factor sensitive).
//   * NearId    - both large but differing by only a few keys (the fixpoint
//                 case, where most subtrees are shared).
//
// It covers a plain set, a scalar-valued map, and the map-of-sets shape used by
// the Clang lifetime analysis (ImmutableMap<Origin, ImmutableSet<Loan>>, whose
// join key-wise unions the inner loan sets). The point of these benchmarks is
// to confirm the bulk merge is not slower than the add loop for small or nearly
// identical inputs. Run before/after and compare (e.g. llvm/utils/compare.py).
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/ImmutableSet.h"
#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

using namespace llvm;

namespace {

using IntSet =
    ImmutableSet<int, ImutContainerInfo<int>, /*Canonicalize=*/false>;
using IntMap =
    ImmutableMap<int, int, ImutKeyValueInfo<int, int>, /*Canonicalize=*/false>;
// The lifetime analysis' OriginLoanMap shape: a map whose values are sets.
using SetMap = ImmutableMap<int, IntSet, ImutKeyValueInfo<int, IntSet>,
                            /*Canonicalize=*/false>;

//===----------------------------------------------------------------------===//
// The per-element merges the bulk path replaced.
//===----------------------------------------------------------------------===//

static IntSet setUnionAdd(IntSet::Factory &F, IntSet A, IntSet B) {
  if (A.getRootWithoutRetain() == B.getRootWithoutRetain() || B.isEmpty())
    return A;
  if (A.isEmpty())
    return B;
  if (A.getHeight() < B.getHeight())
    std::swap(A, B);
  for (int E : B)
    A = F.add(A, E);
  return A;
}

static IntMap mapUnionAdd(IntMap::Factory &F, IntMap A, IntMap B) {
  if (A.getRootWithoutRetain() == B.getRootWithoutRetain())
    return A;
  if (A.getHeight() < B.getHeight())
    std::swap(A, B);
  IntMap Res = A;
  for (IntMap::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    const int *AV = A.lookup(I.getKey());
    Res = F.add(Res, I.getKey(), AV ? std::max(*AV, I.getData()) : I.getData());
  }
  return Res;
}

static SetMap setMapUnionAdd(SetMap::Factory &MF, IntSet::Factory &SF, SetMap A,
                             SetMap B) {
  if (A.getRootWithoutRetain() == B.getRootWithoutRetain())
    return A;
  if (A.getHeight() < B.getHeight())
    std::swap(A, B);
  SetMap Res = A;
  for (SetMap::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    const IntSet *AV = A.lookup(I.getKey());
    IntSet Merged = AV ? setUnionAdd(SF, *AV, I.getData()) : I.getData();
    Res = MF.add(Res, I.getKey(), Merged);
  }
  return Res;
}

//===----------------------------------------------------------------------===//
// The bulk merges (mirroring the caller-side short-circuit/swap the analyses
// perform around them, so both sides are compared on equal footing).
//===----------------------------------------------------------------------===//

static IntSet setUnionNew(IntSet::Factory &F, IntSet A, IntSet B) {
  return F.unionSets(A, B);
}

static IntMap mapUnionNew(IntMap::Factory &F, IntMap A, IntMap B) {
  if (A.getRootWithoutRetain() == B.getRootWithoutRetain())
    return A;
  if (A.getHeight() < B.getHeight())
    std::swap(A, B);
  auto Combine = [](const IntMap::value_type *AE,
                    const IntMap::value_type *BE) -> std::pair<int, int> {
    int Key = AE ? AE->first : BE->first;
    return {Key, std::max(AE ? AE->second : BE->second,
                          BE ? BE->second : AE->second)};
  };
  return F.mergeWith(A, B, Combine, /*KeepUnmatched=*/true);
}

static SetMap setMapUnionNew(SetMap::Factory &MF, IntSet::Factory &SF, SetMap A,
                             SetMap B) {
  if (A.getRootWithoutRetain() == B.getRootWithoutRetain())
    return A;
  if (A.getHeight() < B.getHeight())
    std::swap(A, B);
  auto Combine = [&SF](const SetMap::value_type *AE,
                       const SetMap::value_type *BE) -> std::pair<int, IntSet> {
    int Key = AE ? AE->first : BE->first;
    IntSet AS = AE ? AE->second : SF.getEmptySet();
    IntSet BS = BE ? BE->second : SF.getEmptySet();
    return {Key, SF.unionSets(AS, BS)};
  };
  return MF.mergeWith(A, B, Combine, /*KeepUnmatched=*/true);
}

//===----------------------------------------------------------------------===//
// Key generators. Small: two tiny half-overlapping sets. NearId: two size-N
// sets sharing all but D keys (D unique to each side).
//===----------------------------------------------------------------------===//

static void makeKeys(bool NearId, int N, int D, std::vector<int> &KA,
                     std::vector<int> &KB) {
  KA.resize(N);
  std::iota(KA.begin(), KA.end(), 0);
  KB = KA;
  if (NearId) {
    // Replace the first D keys of B with fresh keys, so A and B share N-D keys
    // and each has D unique ones.
    for (int I = 0; I < D && I < N; ++I)
      KB[I] = N + I;
  } else {
    // Half-overlap: shift B up by N/2.
    for (int &V : KB)
      V += N / 2;
  }
}

// Bound allocator growth: results accumulate in the factory, so periodically
// rebuild it (excluded from timing) rather than let it grow unboundedly.
static constexpr int ResetEvery = 4096;

static IntSet buildSet(IntSet::Factory &F, const std::vector<int> &Keys) {
  IntSet S = F.getEmptySet();
  for (int V : Keys)
    S = F.add(S, V);
  return S;
}

//===----------------------------------------------------------------------===//
// Benchmarks.
//===----------------------------------------------------------------------===//

static void BM_Set(benchmark::State &State, bool UseNew, bool NearId, int N,
                   int D) {
  std::vector<int> KA, KB;
  makeKeys(NearId, N, D, KA, KB);

  std::unique_ptr<IntSet::Factory> F;
  IntSet A{nullptr}, B{nullptr};
  auto reset = [&] {
    A = IntSet(nullptr);
    B = IntSet(nullptr);
    F = std::make_unique<IntSet::Factory>();
    A = buildSet(*F, KA);
    B = buildSet(*F, KB);
  };
  reset();

  int Since = 0;
  for (auto _ : State) {
    // Rebuild before creating this iteration's result, so no live result
    // references the factory being torn down.
    if (++Since == ResetEvery) {
      State.PauseTiming();
      reset();
      Since = 0;
      State.ResumeTiming();
    }
    IntSet U = UseNew ? setUnionNew(*F, A, B) : setUnionAdd(*F, A, B);
    benchmark::DoNotOptimize(U.getRootWithoutRetain());
  }
}

// Like BM_Set, but B is derived from A by adding D fresh keys, so B shares all
// of A's untouched subtrees by pointer -- the realistic "state after a small
// edit" shape that the pointer-equality skip in unionSets is meant to exploit.
static void BM_SetShared(benchmark::State &State, bool UseNew, int N, int D) {
  std::vector<int> KA(N);
  std::iota(KA.begin(), KA.end(), 0);

  std::unique_ptr<IntSet::Factory> F;
  IntSet A{nullptr}, B{nullptr};
  auto reset = [&] {
    A = IntSet(nullptr);
    B = IntSet(nullptr);
    F = std::make_unique<IntSet::Factory>();
    A = buildSet(*F, KA);
    B = A;
    for (int I = 0; I < D; ++I)
      B = F->add(B, N + I);
  };
  reset();

  int Since = 0;
  for (auto _ : State) {
    if (++Since == ResetEvery) {
      State.PauseTiming();
      reset();
      Since = 0;
      State.ResumeTiming();
    }
    IntSet U = UseNew ? setUnionNew(*F, A, B) : setUnionAdd(*F, A, B);
    benchmark::DoNotOptimize(U.getRootWithoutRetain());
  }
}

static void BM_Map(benchmark::State &State, bool UseNew, bool NearId, int N,
                   int D) {
  std::vector<int> KA, KB;
  makeKeys(NearId, N, D, KA, KB);

  std::unique_ptr<IntMap::Factory> F;
  IntMap A{nullptr}, B{nullptr};
  auto reset = [&] {
    A = IntMap(nullptr);
    B = IntMap(nullptr);
    F = std::make_unique<IntMap::Factory>();
    IntMap MA = F->getEmptyMap(), MB = F->getEmptyMap();
    for (int V : KA)
      MA = F->add(MA, V, V);
    for (int V : KB)
      MB = F->add(MB, V, V);
    A = MA;
    B = MB;
  };
  reset();

  int Since = 0;
  for (auto _ : State) {
    if (++Since == ResetEvery) {
      State.PauseTiming();
      reset();
      Since = 0;
      State.ResumeTiming();
    }
    IntMap U = UseNew ? mapUnionNew(*F, A, B) : mapUnionAdd(*F, A, B);
    benchmark::DoNotOptimize(U.getRootWithoutRetain());
  }
}

// Map<int, Set<int>>: each key maps to a small set; the merge key-wise unions
// the value sets (the lifetime OriginLoanMap join).
static void BM_MapOfSets(benchmark::State &State, bool UseNew, bool NearId,
                         int N, int D, int SetSize) {
  std::vector<int> KA, KB;
  makeKeys(NearId, N, D, KA, KB);

  std::unique_ptr<SetMap::Factory> MF;
  std::unique_ptr<IntSet::Factory> SF;
  SetMap A{nullptr}, B{nullptr};
  auto valueSet = [&](int Key) {
    // A deterministic small set derived from the key.
    IntSet S = SF->getEmptySet();
    for (int I = 0; I < SetSize; ++I)
      S = SF->add(S, Key * 31 + I);
    return S;
  };
  auto reset = [&] {
    A = SetMap(nullptr);
    B = SetMap(nullptr);
    MF = std::make_unique<SetMap::Factory>();
    SF = std::make_unique<IntSet::Factory>();
    SetMap MA = MF->getEmptyMap(), MB = MF->getEmptyMap();
    for (int V : KA)
      MA = MF->add(MA, V, valueSet(V));
    for (int V : KB)
      MB = MF->add(MB, V, valueSet(V));
    A = MA;
    B = MB;
  };
  reset();

  int Since = 0;
  for (auto _ : State) {
    if (++Since == ResetEvery) {
      State.PauseTiming();
      reset();
      Since = 0;
      State.ResumeTiming();
    }
    SetMap U = UseNew ? setMapUnionNew(*MF, *SF, A, B)
                      : setMapUnionAdd(*MF, *SF, A, B);
    benchmark::DoNotOptimize(U.getRootWithoutRetain());
  }
}

} // namespace

// Small: both operands tiny. NearId: size N, differ by D keys each side.
BENCHMARK_CAPTURE(BM_Set, "Set/Small/Add", false, false, 8, 0);
BENCHMARK_CAPTURE(BM_Set, "Set/Small/Union", true, false, 8, 0);
BENCHMARK_CAPTURE(BM_Set, "Set/Small16/Add", false, false, 16, 0);
BENCHMARK_CAPTURE(BM_Set, "Set/Small16/Union", true, false, 16, 0);
// Small and nearly identical (share all but one key each side) -- the common
// dataflow live-set shape, and the case most exposed to split overhead.
BENCHMARK_CAPTURE(BM_Set, "Set/SmallNearId8/Add", false, true, 8, 1);
BENCHMARK_CAPTURE(BM_Set, "Set/SmallNearId8/Union", true, true, 8, 1);
// Very small sets, delta 2 (and a couple of neighbours) to see if the Δ1
// regression persists at tiny sizes once the diff is at least 2.
BENCHMARK_CAPTURE(BM_Set, "Set/N8D2/Add", false, true, 8, 2);
BENCHMARK_CAPTURE(BM_Set, "Set/N8D2/Union", true, true, 8, 2);
BENCHMARK_CAPTURE(BM_Set, "Set/N16D2/Add", false, true, 16, 2);
BENCHMARK_CAPTURE(BM_Set, "Set/N16D2/Union", true, true, 16, 2);
BENCHMARK_CAPTURE(BM_Set, "Set/N32D2/Add", false, true, 32, 2);
BENCHMARK_CAPTURE(BM_Set, "Set/N32D2/Union", true, true, 32, 2);
BENCHMARK_CAPTURE(BM_Set, "Set/SmallNearId64/Add", false, true, 64, 1);
BENCHMARK_CAPTURE(BM_Set, "Set/SmallNearId64/Union", true, true, 64, 1);
// Delta sweep at N=64 (where the near-identical regression peaked): how does
// the gap behave as the two sets differ by more keys?
BENCHMARK_CAPTURE(BM_Set, "Set/N64D2/Add", false, true, 64, 2);
BENCHMARK_CAPTURE(BM_Set, "Set/N64D2/Union", true, true, 64, 2);
BENCHMARK_CAPTURE(BM_Set, "Set/N64D4/Add", false, true, 64, 4);
BENCHMARK_CAPTURE(BM_Set, "Set/N64D4/Union", true, true, 64, 4);
BENCHMARK_CAPTURE(BM_Set, "Set/N64D8/Add", false, true, 64, 8);
BENCHMARK_CAPTURE(BM_Set, "Set/N64D8/Union", true, true, 64, 8);
BENCHMARK_CAPTURE(BM_Set, "Set/N64D16/Add", false, true, 64, 16);
BENCHMARK_CAPTURE(BM_Set, "Set/N64D16/Union", true, true, 64, 16);
BENCHMARK_CAPTURE(BM_Set, "Set/N64D32/Add", false, true, 64, 32);
BENCHMARK_CAPTURE(BM_Set, "Set/N64D32/Union", true, true, 64, 32);
// Δ1 but B derived from A (shares subtrees by pointer): the case the
// pointer-equality skip targets. Compare against the independent-build
// Set/*NearId* numbers above.
BENCHMARK_CAPTURE(BM_SetShared, "SetShared/N8D1/Add", false, 8, 1);
BENCHMARK_CAPTURE(BM_SetShared, "SetShared/N8D1/Union", true, 8, 1);
BENCHMARK_CAPTURE(BM_SetShared, "SetShared/N64D1/Add", false, 64, 1);
BENCHMARK_CAPTURE(BM_SetShared, "SetShared/N64D1/Union", true, 64, 1);
BENCHMARK_CAPTURE(BM_SetShared, "SetShared/N1024D1/Add", false, 1024, 1);
BENCHMARK_CAPTURE(BM_SetShared, "SetShared/N1024D1/Union", true, 1024, 1);
BENCHMARK_CAPTURE(BM_Set, "Set/NearId128/Add", false, true, 128, 1);
BENCHMARK_CAPTURE(BM_Set, "Set/NearId128/Union", true, true, 128, 1);
BENCHMARK_CAPTURE(BM_Set, "Set/NearId256/Add", false, true, 256, 1);
BENCHMARK_CAPTURE(BM_Set, "Set/NearId256/Union", true, true, 256, 1);
BENCHMARK_CAPTURE(BM_Set, "Set/NearId512/Add", false, true, 512, 1);
BENCHMARK_CAPTURE(BM_Set, "Set/NearId512/Union", true, true, 512, 1);
BENCHMARK_CAPTURE(BM_Set, "Set/NearId/Add", false, true, 1024, 1);
BENCHMARK_CAPTURE(BM_Set, "Set/NearId/Union", true, true, 1024, 1);

BENCHMARK_CAPTURE(BM_Map, "Map/Small/Add", false, false, 8, 0);
BENCHMARK_CAPTURE(BM_Map, "Map/Small/Union", true, false, 8, 0);
BENCHMARK_CAPTURE(BM_Map, "Map/SmallNearId8/Add", false, true, 8, 1);
BENCHMARK_CAPTURE(BM_Map, "Map/SmallNearId8/Union", true, true, 8, 1);
BENCHMARK_CAPTURE(BM_Map, "Map/SmallNearId64/Add", false, true, 64, 1);
BENCHMARK_CAPTURE(BM_Map, "Map/SmallNearId64/Union", true, true, 64, 1);
BENCHMARK_CAPTURE(BM_Map, "Map/NearId/Add", false, true, 1024, 1);
BENCHMARK_CAPTURE(BM_Map, "Map/NearId/Union", true, true, 1024, 1);

BENCHMARK_CAPTURE(BM_MapOfSets, "MapOfSets/Small/Add", false, false, 8, 0, 4);
BENCHMARK_CAPTURE(BM_MapOfSets, "MapOfSets/Small/Union", true, false, 8, 0, 4);
BENCHMARK_CAPTURE(BM_MapOfSets, "MapOfSets/SmallNearId8/Add", false, true, 8, 1,
                  4);
BENCHMARK_CAPTURE(BM_MapOfSets, "MapOfSets/SmallNearId8/Union", true, true, 8,
                  1, 4);
BENCHMARK_CAPTURE(BM_MapOfSets, "MapOfSets/NearId/Add", false, true, 256, 1, 8);
BENCHMARK_CAPTURE(BM_MapOfSets, "MapOfSets/NearId/Union", true, true, 256, 1,
                  8);

BENCHMARK_MAIN();
