//===- ImmutableSetIteratorBM.cpp - Benchmark ImmutableSet iterators ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Benchmarks in-order traversal of the ImutAVLTree backing ImmutableSet using
// only its public iterator API. It does not compare iterator implementations
// directly; instead, run the binary before and after a change to the iterator
// and compare the two reports (e.g. with llvm/utils/compare.py) to see the
// effect of that change.
//
// Two access patterns are measured:
//   * Iterate - a plain forward walk over ImmutableSet::iterator, the common
//               client usage.
//   * Skip    - a walk that calls skipSubTree on the tree iterator at every
//               other node, the pattern used by ImutAVLTree::isEqual and the
//               tree canonicalization that the clang static analyzer relies on.
//
// It also measures tree canonicalization directly:
//   * CanonicalizeSharedEqual - re-canonicalize a freshly built tree that is
//               structurally equal to, and shares subtrees with, a tree already
//               in the factory's cache. This is the clang static analyzer's hot
//               path when equivalent ProgramStates are re-derived on many
//               paths, and it exercises ImutAVLFactory::getCanonicalTree ->
//               ImutAVLTree::isEqual.
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "llvm/ADT/ImmutableSet.h"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

using namespace llvm;

namespace {

using Tree = ImmutableSet<int>::TreeTy;

// Holds a factory plus a built set so the (non-trivial) tree construction is
// kept out of the timed region. The factory must outlive the tree.
struct Fixture {
  ImmutableSet<int>::Factory F{/*canonicalize=*/false};
  ImmutableSet<int> Set = F.getEmptySet();

  explicit Fixture(size_t N) {
    std::vector<int> Vals(N);
    std::iota(Vals.begin(), Vals.end(), 0);
    std::mt19937 Rng(0xC0FFEE);
    std::shuffle(Vals.begin(), Vals.end(), Rng);
    for (int V : Vals)
      Set = F.add(Set, V);
  }
};

// Plain forward iteration over the public ImmutableSet::iterator.
static void BM_Iterate(benchmark::State &State) {
  const size_t N = State.range(0);
  Fixture Fix(N);
  const ImmutableSet<int> &S = Fix.Set;
  benchmark::DoNotOptimize(S.getRootWithoutRetain());

  for (auto _ : State) {
    int64_t Sum = 0;
    for (int V : S)
      Sum += V;
    benchmark::DoNotOptimize(Sum);
  }
  State.SetItemsProcessed(State.iterations() * N);
}

// Iterate while skipping every other subtree, exercising skipSubTree (the
// pattern behind ImutAVLTree::isEqual / canonicalization) rather than plain ++.
static void BM_IterateWithSkips(benchmark::State &State) {
  const size_t N = State.range(0);
  Fixture Fix(N);
  const Tree *Root = Fix.Set.getRootWithoutRetain();
  benchmark::DoNotOptimize(Root);

  for (auto _ : State) {
    int64_t Sum = 0;
    unsigned K = 0;
    for (Tree::iterator I(Root), E; I != E; ++K) {
      Sum += I->getValue();
      if (K & 1)
        I.skipSubTree();
      else
        ++I;
    }
    benchmark::DoNotOptimize(Sum);
  }
}

// Compare two iterators positioned at the same (non-end) node on every step.
// This isolates operator==: it is the case that differs between comparing the
// current node only (O(1)) and comparing the whole ancestor path (O(depth)).
// It is a microbenchmark of the comparison itself, not a typical client loop
// (those compare against end(), which is O(1) either way).
static void BM_CompareSamePosition(benchmark::State &State) {
  const size_t N = State.range(0);
  Fixture Fix(N);
  const Tree *Root = Fix.Set.getRootWithoutRetain();
  benchmark::DoNotOptimize(Root);

  for (auto _ : State) {
    bool R = false;
    Tree::iterator E;
    for (Tree::iterator A(Root), B(Root); A != E; ++A, ++B)
      R ^= (A == B);
    benchmark::DoNotOptimize(R);
  }
  State.SetItemsProcessed(State.iterations() * N);
}

// Re-canonicalize a freshly built tree that is structurally equal to a tree
// already in the factory's cache and shares most of its subtrees. This mirrors
// the clang static analyzer, where equivalent ProgramStates re-derived on many
// paths are canonicalized against ones already cached. Adding then removing an
// absent (largest) key rebuilds only the right spine while sharing the rest of
// Base's subtrees, so getCanonicalTree confirms equality against a structurally
// shared equal tree -- exactly where isEqual's subtree pointer skipping
// matters.
static void BM_CanonicalizeSharedEqual(benchmark::State &State) {
  const size_t N = State.range(0);
  ImmutableSet<int>::Factory F(/*canonicalize=*/true);
  ImmutableSet<int> Base = F.getEmptySet();
  std::vector<int> Vals(N);
  std::iota(Vals.begin(), Vals.end(), 0);
  std::mt19937 Rng(0xC0FFEE);
  std::shuffle(Vals.begin(), Vals.end(), Rng);
  for (int V : Vals)
    Base = F.add(Base, V); // Base is canonical and lives in the factory cache.
  const int Absent = static_cast<int>(N) + 1;

  for (auto _ : State) {
    ImmutableSet<int> Tmp = F.add(Base, Absent);
    ImmutableSet<int> Eq = F.remove(Tmp, Absent); // structurally equal to Base
    benchmark::DoNotOptimize(Eq.getRootWithoutRetain());
  }
  State.SetItemsProcessed(State.iterations());
}

} // namespace

#define ITER_SIZES Arg(16)->Arg(256)->Arg(4096)->Arg(65536)

BENCHMARK(BM_Iterate)->Name("Iterate")->ITER_SIZES;
BENCHMARK(BM_IterateWithSkips)->Name("Skip")->ITER_SIZES;
BENCHMARK(BM_CompareSamePosition)->Name("CompareSamePos")->ITER_SIZES;
BENCHMARK(BM_CanonicalizeSharedEqual)
    ->Name("CanonicalizeSharedEqual")
    ->ITER_SIZES;

BENCHMARK_MAIN();
