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

// Aligned slot types with controlled NumLowBitsAvailable.
template <int N> struct alignas(4) A4 {
  int data;
}; // 2 bits.
template <int N> struct alignas(8) A8 {
  int data;
}; // 3 bits.
template <int N> struct alignas(32) A32 {
  int data;
}; // 5 bits.

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

// Splits N types into two tiers: N0 types in tier 0 (A4, 2-bit) and
// N1 types in tier 1 (A32, 5-bit). Tier 0 gets up to 3 types (the max
// for 2 low bits minus one escape code).
template <size_t N> struct TwoTierSplit {
  static constexpr size_t N0 = N < 4 ? N - 1 : 3;
  static constexpr size_t N1 = N - N0;
};

// Splits N types into three tiers: N0 types in tier 0 (A4, 2-bit),
// N1 = 1 type in tier 1 (A8, 3-bit), and N2 types in tier 2 (A32, 5-bit).
// Tier 0 gets up to 3 types; tier 1 always gets exactly 1.
template <size_t N> struct ThreeTierSplit {
  static constexpr size_t N0 = N < 5 ? N - 2 : 3;
  static constexpr size_t N1 = 1;
  static constexpr size_t N2 = N - N0 - N1;
};

// A4A32Types<N> = TypeList<A4<0>, ..., A4<N0-1>, A32<0>, ..., A32<N1-1>>.
// Used to generate randomized arrays with a mix of A4 and A32 types.
template <size_t... I0, size_t... I1>
TypeList<A4<I0>..., A32<I1>...> makeA4A32TypesImpl(std::index_sequence<I0...>,
                                                   std::index_sequence<I1...>);
template <size_t N>
using A4A32Types = decltype(makeA4A32TypesImpl(
    std::make_index_sequence<TwoTierSplit<N>::N0>{},
    std::make_index_sequence<TwoTierSplit<N>::N1>{}));

// A4A8A32Types<N> = TypeList<A4<0>, ..., A8<0>, ..., A32<0>, ...>.
// Used to generate randomized arrays with a mix of A4, A8, and A32 types.
template <size_t... I0, size_t... I1, size_t... I2>
TypeList<A4<I0>..., A8<I1>..., A32<I2>...>
    makeA4A8A32TypesImpl(std::index_sequence<I0...>, std::index_sequence<I1...>,
                         std::index_sequence<I2...>);
template <size_t N>
using A4A8A32Types = decltype(makeA4A8A32TypesImpl(
    std::make_index_sequence<ThreeTierSplit<N>::N0>{},
    std::make_index_sequence<ThreeTierSplit<N>::N1>{},
    std::make_index_sequence<ThreeTierSplit<N>::N2>{}));

// Union type aliases.
template <size_t... Is>
auto makePtrUnion(std::index_sequence<Is...>) -> PointerUnion<A8<Is> *...>;
template <size_t N>
using PtrUnion = decltype(makePtrUnion(std::make_index_sequence<N>{}));

template <size_t... I0, size_t... I1>
auto makePtrUnion2T(std::index_sequence<I0...>, std::index_sequence<I1...>)
    -> PointerUnion<A4<I0> *..., A32<I1> *...>;
template <size_t N>
using PtrUnion2T =
    decltype(makePtrUnion2T(std::make_index_sequence<TwoTierSplit<N>::N0>{},
                            std::make_index_sequence<TwoTierSplit<N>::N1>{}));

template <size_t... I0, size_t... I1, size_t... I2>
auto makePtrUnion3T(std::index_sequence<I0...>, std::index_sequence<I1...>,
                    std::index_sequence<I2...>)
    -> PointerUnion<A4<I0> *..., A8<I1> *..., A32<I2> *...>;
template <size_t N>
using PtrUnion3T =
    decltype(makePtrUnion3T(std::make_index_sequence<ThreeTierSplit<N>::N0>{},
                            std::make_index_sequence<ThreeTierSplit<N>::N1>{},
                            std::make_index_sequence<ThreeTierSplit<N>::N2>{}));

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

// Registration -- N = 2, 4, 8. PtrUnion3T uses N = 3, 4, 8.

// Isa: PtrUnion (fixed-width tag).
BENCHMARK((BM_Isa<PtrUnion<2>, A8<0>, A8Types<2>>))->Name("Isa/PU/2/First");
BENCHMARK((BM_Isa<PtrUnion<2>, A8<1>, A8Types<2>>))->Name("Isa/PU/2/Last");

BENCHMARK((BM_Isa<PtrUnion<4>, A8<0>, A8Types<4>>))->Name("Isa/PU/4/First");
BENCHMARK((BM_Isa<PtrUnion<4>, A8<3>, A8Types<4>>))->Name("Isa/PU/4/Last");

BENCHMARK((BM_Isa<PtrUnion<8>, A8<0>, A8Types<8>>))->Name("Isa/PU/8/First");
BENCHMARK((BM_Isa<PtrUnion<8>, A8<7>, A8Types<8>>))->Name("Isa/PU/8/Last");

// Isa: PtrUnion2T (variable-width, 2 alignment tiers).
// Note: PtrUnion2T<N> uses variable-width encoding only when N > 3 (when
// fixed-width tags don't fit in 2 bits). PtrUnion2T<2> is still fixed-width.
BENCHMARK((BM_Isa<PtrUnion2T<2>, A4<0>, A4A32Types<2>>))
    ->Name("Isa/PU2T/2/Tier0");
BENCHMARK((BM_Isa<PtrUnion2T<2>, A32<0>, A4A32Types<2>>))
    ->Name("Isa/PU2T/2/Tier1");

BENCHMARK((BM_Isa<PtrUnion2T<4>, A4<0>, A4A32Types<4>>))
    ->Name("Isa/PU2T/4/Tier0");
BENCHMARK((BM_Isa<PtrUnion2T<4>, A32<0>, A4A32Types<4>>))
    ->Name("Isa/PU2T/4/Tier1");

BENCHMARK((BM_Isa<PtrUnion2T<8>, A4<0>, A4A32Types<8>>))
    ->Name("Isa/PU2T/8/Tier0");
BENCHMARK((BM_Isa<PtrUnion2T<8>, A32<4>, A4A32Types<8>>))
    ->Name("Isa/PU2T/8/Tier1");

// Isa: PtrUnion3T (variable-width, 3 alignment tiers).
// Note: PtrUnion3T<N> uses variable-width encoding only when N > 4 (when
// fixed-width tags don't fit in 2 bits). PtrUnion3T<3> and <4> are
// still fixed-width.
BENCHMARK((BM_Isa<PtrUnion3T<3>, A4<0>, A4A8A32Types<3>>))
    ->Name("Isa/PU3T/3/Tier0");
BENCHMARK((BM_Isa<PtrUnion3T<3>, A8<0>, A4A8A32Types<3>>))
    ->Name("Isa/PU3T/3/Tier1");
BENCHMARK((BM_Isa<PtrUnion3T<3>, A32<0>, A4A8A32Types<3>>))
    ->Name("Isa/PU3T/3/Tier2");

BENCHMARK((BM_Isa<PtrUnion3T<4>, A4<0>, A4A8A32Types<4>>))
    ->Name("Isa/PU3T/4/Tier0");
BENCHMARK((BM_Isa<PtrUnion3T<4>, A8<0>, A4A8A32Types<4>>))
    ->Name("Isa/PU3T/4/Tier1");
BENCHMARK((BM_Isa<PtrUnion3T<4>, A32<0>, A4A8A32Types<4>>))
    ->Name("Isa/PU3T/4/Tier2");

BENCHMARK((BM_Isa<PtrUnion3T<8>, A4<0>, A4A8A32Types<8>>))
    ->Name("Isa/PU3T/8/Tier0");
BENCHMARK((BM_Isa<PtrUnion3T<8>, A8<0>, A4A8A32Types<8>>))
    ->Name("Isa/PU3T/8/Tier1");
BENCHMARK((BM_Isa<PtrUnion3T<8>, A32<3>, A4A8A32Types<8>>))
    ->Name("Isa/PU3T/8/Tier2");

// IsNull: all suites.
BENCHMARK((BM_IsNull<PtrUnion<2>, A8Types<2>>))->Name("IsNull/PU/2");
BENCHMARK((BM_IsNull<PtrUnion<4>, A8Types<4>>))->Name("IsNull/PU/4");
BENCHMARK((BM_IsNull<PtrUnion<8>, A8Types<8>>))->Name("IsNull/PU/8");

BENCHMARK((BM_IsNull<PtrUnion2T<2>, A4A32Types<2>>))->Name("IsNull/PU2T/2");
BENCHMARK((BM_IsNull<PtrUnion2T<4>, A4A32Types<4>>))->Name("IsNull/PU2T/4");
BENCHMARK((BM_IsNull<PtrUnion2T<8>, A4A32Types<8>>))->Name("IsNull/PU2T/8");

BENCHMARK((BM_IsNull<PtrUnion3T<3>, A4A8A32Types<3>>))->Name("IsNull/PU3T/3");
BENCHMARK((BM_IsNull<PtrUnion3T<4>, A4A8A32Types<4>>))->Name("IsNull/PU3T/4");
BENCHMARK((BM_IsNull<PtrUnion3T<8>, A4A8A32Types<8>>))->Name("IsNull/PU3T/8");

BENCHMARK_MAIN();
