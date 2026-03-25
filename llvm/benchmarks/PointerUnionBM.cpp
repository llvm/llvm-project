//===- PointerUnionBM.cpp - Benchmark for PointerUnion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Benchmarks for PointerUnion operations with randomized data.
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Casting.h"
#include <cstddef>
#include <random>
#include <vector>

using namespace llvm;

namespace llvm {
namespace {

// Aligned slot types with controlled NumLowBitsAvailable (3 low bits).
template <int N> struct alignas(8) A8 {
  int data;
};

template <typename T> T &getSlot() {
  static T S{};
  return S;
}

// Encodes a list of slot types for random fill.
template <typename... Ts> struct TypeList {};

template <typename UnionT, typename... SlotTs>
void fillRandom(UnionT *Arr, size_t Count, TypeList<SlotTs...>) {
  constexpr size_t N = sizeof...(SlotTs);
  UnionT Variants[N] = {UnionT(&getSlot<SlotTs>())...};
  std::mt19937 Rng(42);
  std::uniform_int_distribution<size_t> Dist(0, N - 1);
  for (size_t I = 0; I < Count; ++I)
    Arr[I] = Variants[Dist(Rng)];
}

template <typename UnionT, typename... SlotTs>
void fillRandomWithNulls(UnionT *Arr, size_t Count, TypeList<SlotTs...> TL) {
  fillRandom(Arr, Count, TL);
  std::mt19937 Rng(123);
  // 50% null mix to exercise both null and non-null paths equally.
  std::bernoulli_distribution Coin(0.5);
  for (size_t I = 0; I < Count; ++I)
    if (Coin(Rng))
      Arr[I] = nullptr;
}

// A8Types<N> = TypeList<A8<0>, A8<1>, ..., A8<N-1>>.
// Used to generate randomized arrays with a uniform mix of N distinct types.
template <size_t... Is>
TypeList<A8<Is>...> makeA8TypesImpl(std::index_sequence<Is...>);
template <size_t N>
using A8Types = decltype(makeA8TypesImpl(std::make_index_sequence<N>{}));

template <size_t... Is>
auto makePtrUnion(std::index_sequence<Is...>) -> PointerUnion<A8<Is> *...>;

template <size_t N>
using PtrUnion = decltype(makePtrUnion(std::make_index_sequence<N>{}));

// Isa: random type mix, uniform distribution (~1/N hit rate for each type).
template <typename UnionT, typename QueryT, typename TL>
static void BM_Isa(benchmark::State &State) {
  constexpr size_t Batch = 1024;
  std::vector<UnionT> Arr(Batch);
  fillRandom(Arr.data(), Batch, TL{});
  benchmark::DoNotOptimize(Arr.data());
  for (auto _ : State) {
    bool R = false;
    for (size_t I = 0; I < Batch; ++I)
      R ^= isa<QueryT *>(Arr[I]);
    benchmark::DoNotOptimize(R);
  }
  State.SetItemsProcessed(State.iterations() * Batch);
}

// IsNull: random mix with ~50% nulls.
template <typename UnionT, typename TL>
static void BM_IsNull(benchmark::State &State) {
  constexpr size_t Batch = 1024;
  std::vector<UnionT> Arr(Batch);
  fillRandomWithNulls(Arr.data(), Batch, TL{});
  benchmark::DoNotOptimize(Arr.data());
  for (auto _ : State) {
    bool R = false;
    for (size_t I = 0; I < Batch; ++I)
      R ^= Arr[I].isNull();
    benchmark::DoNotOptimize(R);
  }
  State.SetItemsProcessed(State.iterations() * Batch);
}

} // namespace
} // namespace llvm

// Registration -- N = 2, 4, 8.
BENCHMARK((BM_Isa<PtrUnion<2>, A8<0>, A8Types<2>>))->Name("Isa/PU/2/First");
BENCHMARK((BM_Isa<PtrUnion<2>, A8<1>, A8Types<2>>))->Name("Isa/PU/2/Last");

BENCHMARK((BM_Isa<PtrUnion<4>, A8<0>, A8Types<4>>))->Name("Isa/PU/4/First");
BENCHMARK((BM_Isa<PtrUnion<4>, A8<3>, A8Types<4>>))->Name("Isa/PU/4/Last");

BENCHMARK((BM_Isa<PtrUnion<8>, A8<0>, A8Types<8>>))->Name("Isa/PU/8/First");
BENCHMARK((BM_Isa<PtrUnion<8>, A8<7>, A8Types<8>>))->Name("Isa/PU/8/Last");

BENCHMARK((BM_IsNull<PtrUnion<2>, A8Types<2>>))->Name("IsNull/PU/2");
BENCHMARK((BM_IsNull<PtrUnion<4>, A8Types<4>>))->Name("IsNull/PU/4");
BENCHMARK((BM_IsNull<PtrUnion<8>, A8Types<8>>))->Name("IsNull/PU/8");

BENCHMARK_MAIN();
