//===----------- ImmutableMapTest.cpp - ImmutableMap unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ImmutableMap.h"
#include "gtest/gtest.h"
#include <map>
#include <random>
#include <utility>
#include <vector>

using namespace llvm;

// Non-canonicalizing map, matching how the clang lifetime analyses use it and
// the only configuration for which the bulk mergeWith path is valid.
template <typename K, typename V>
using NCMap =
    ImmutableMap<K, V, ImutKeyValueInfo<K, V>, /*Canonicalize=*/false>;

namespace {

TEST(ImmutableMapTest, EmptyIntMapTest) {
  ImmutableMap<int, int>::Factory f;

  EXPECT_TRUE(f.getEmptyMap() == f.getEmptyMap());
  EXPECT_FALSE(f.getEmptyMap() != f.getEmptyMap());
  EXPECT_TRUE(f.getEmptyMap().isEmpty());

  ImmutableMap<int, int> S = f.getEmptyMap();
  EXPECT_EQ(0u, S.getHeight());
  EXPECT_TRUE(S.begin() == S.end());
  EXPECT_FALSE(S.begin() != S.end());
}

TEST(ImmutableMapTest, MultiElemIntMapTest) {
  ImmutableMap<int, int>::Factory f;
  ImmutableMap<int, int> S = f.getEmptyMap();

  ImmutableMap<int, int> S2 = f.add(f.add(f.add(S, 3, 10), 4, 11), 5, 12);

  EXPECT_TRUE(S.isEmpty());
  EXPECT_FALSE(S2.isEmpty());

  EXPECT_EQ(nullptr, S.lookup(3));
  EXPECT_EQ(nullptr, S.lookup(9));

  EXPECT_EQ(10, *S2.lookup(3));
  EXPECT_EQ(11, *S2.lookup(4));
  EXPECT_EQ(12, *S2.lookup(5));

  EXPECT_EQ(5, S2.getMaxElement()->first);
  EXPECT_EQ(3U, S2.getHeight());
}

TEST(ImmutableMapTest, EmptyIntMapRefTest) {
  using int_int_map = ImmutableMapRef<int, int>;
  ImmutableMapRef<int, int>::FactoryTy f;

  EXPECT_TRUE(int_int_map::getEmptyMap(&f) == int_int_map::getEmptyMap(&f));
  EXPECT_FALSE(int_int_map::getEmptyMap(&f) != int_int_map::getEmptyMap(&f));
  EXPECT_TRUE(int_int_map::getEmptyMap(&f).isEmpty());

  int_int_map S = int_int_map::getEmptyMap(&f);
  EXPECT_EQ(0u, S.getHeight());
  EXPECT_TRUE(S.begin() == S.end());
  EXPECT_FALSE(S.begin() != S.end());
}

TEST(ImmutableMapTest, MultiElemIntMapRefTest) {
  ImmutableMapRef<int, int>::FactoryTy f;

  ImmutableMapRef<int, int> S = ImmutableMapRef<int, int>::getEmptyMap(&f);

  ImmutableMapRef<int, int> S2 = S.add(3, 10).add(4, 11).add(5, 12);

  EXPECT_TRUE(S.isEmpty());
  EXPECT_FALSE(S2.isEmpty());

  EXPECT_EQ(nullptr, S.lookup(3));
  EXPECT_EQ(nullptr, S.lookup(9));

  EXPECT_EQ(10, *S2.lookup(3));
  EXPECT_EQ(11, *S2.lookup(4));
  EXPECT_EQ(12, *S2.lookup(5));

  EXPECT_EQ(5, S2.getMaxElement()->first);
  EXPECT_EQ(3U, S2.getHeight());
}

  TEST(ImmutableMapTest, MapOfMapRefsTest) {
  ImmutableMap<int, ImmutableMapRef<int, int>>::Factory f;

  EXPECT_TRUE(f.getEmptyMap() == f.getEmptyMap());
  }

  // Exercises Factory::mergeWith against a std::map oracle across the
  // KeepA/KeepB configurations used by the set union (true/true), an asymmetric
  // map join (true/false), and a symmetric map join (false/false), with a
  // combiner that is order- and side-sensitive so the flag semantics are
  // actually tested.
  TEST(ImmutableMapTest, MergeWithStressTest) {
    using Map = NCMap<int, int>;
    using VT = Map::value_type; // const std::pair<int, int>

    auto Combine = [](const VT *A, const VT *B) -> std::pair<int, int> {
      int Key = A ? A->first : B->first;
      int AV = A ? A->second : 0;
      int BV = B ? B->second : 0;
      return {Key, 1 + 3 * AV + 5 * BV};
    };

    std::mt19937 Rng(7);
    std::uniform_int_distribution<int> SizeDist(0, 120);
    std::uniform_int_distribution<int> KeyDist(0, 200);
    std::uniform_int_distribution<int> ValDist(0, 1000);

    for (bool KeepUnmatched : {true, false}) {
      for (int Trial = 0; Trial < 300; ++Trial) {
        Map::Factory F;
        Map A = F.getEmptyMap(), B = F.getEmptyMap();
        std::map<int, int> RefA, RefB;

        int NA = SizeDist(Rng), NB = SizeDist(Rng);
        for (int I = 0; I < NA; ++I) {
          int K = KeyDist(Rng), V = ValDist(Rng);
          A = F.add(A, K, V);
          RefA[K] = V;
        }
        for (int I = 0; I < NB; ++I) {
          int K = KeyDist(Rng), V = ValDist(Rng);
          B = F.add(B, K, V);
          RefB[K] = V;
        }

        Map U = F.mergeWith(A, B, Combine, KeepUnmatched);
        if (const auto *Root = U.getRootWithoutRetain())
          Root->validateTree();

        // Reference result: matched keys always combined; unmatched keys are
        // kept as-is when KeepUnmatched, else passed through Combine.
        std::map<int, int> Exp;
        for (auto &[K, AV] : RefA) {
          auto It = RefB.find(K);
          if (It != RefB.end())
            Exp[K] = 1 + 3 * AV + 5 * It->second;
          else
            Exp[K] = KeepUnmatched ? AV : 1 + 3 * AV;
        }
        for (auto &[K, BV] : RefB)
          if (!RefA.count(K))
            Exp[K] = KeepUnmatched ? BV : 1 + 5 * BV;

        std::map<int, int> Got;
        for (Map::iterator It = U.begin(), E = U.end(); It != E; ++It)
          Got[It.getKey()] = It.getData();
        EXPECT_EQ(Exp, Got) << "KeepUnmatched=" << KeepUnmatched;

        // Inputs unchanged.
        std::map<int, int> GotA, GotB;
        for (Map::iterator It = A.begin(), E = A.end(); It != E; ++It)
          GotA[It.getKey()] = It.getData();
        for (Map::iterator It = B.begin(), E = B.end(); It != E; ++It)
          GotB[It.getKey()] = It.getData();
        EXPECT_EQ(RefA, GotA);
        EXPECT_EQ(RefB, GotB);
      }
    }
  }

  template <typename Map> void testFilter() {
    using VT = typename Map::value_type;
    std::mt19937 Rng(99);
    std::uniform_int_distribution<int> SizeDist(0, 150);
    std::uniform_int_distribution<int> KeyDist(0, 300);
    for (int Trial = 0; Trial < 200; ++Trial) {
      typename Map::Factory F;
      Map M = F.getEmptyMap();
      std::map<int, int> Ref;
      for (int I = 0, N = SizeDist(Rng); I < N; ++I) {
        int K = KeyDist(Rng);
        M = F.add(M, K, K * 2);
        Ref[K] = K * 2;
      }
      // Mode 0 keeps all keys, 1 drops all, otherwise a residue class is
      // dropped.
      int Mode = Trial % 8;
      auto Keeps = [&](int K) {
        return Mode == 0 || (Mode != 1 && K % Mode != Trial % Mode);
      };
      std::vector<int> Seen;
      Map Filtered = F.filter(M, [&](const VT &E) {
        Seen.push_back(E.first);
        return Keeps(E.first);
      });
      if (const auto *Root = Filtered.getRootWithoutRetain())
        Root->validateTree();

      std::vector<int> AllKeys;
      std::map<int, int> Exp;
      for (auto &[K, V] : Ref) {
        AllKeys.push_back(K);
        if (Keeps(K))
          Exp[K] = V;
      }
      EXPECT_EQ(AllKeys, Seen); // Pred is called once per entry, in order.
      std::map<int, int> Got;
      for (auto It = Filtered.begin(), E = Filtered.end(); It != E; ++It)
        Got[It.getKey()] = It.getData();
      EXPECT_EQ(Exp, Got);
      if (Mode == 0)
        EXPECT_EQ(M.getRootWithoutRetain(), Filtered.getRootWithoutRetain());
      if (Mode == 1)
        EXPECT_TRUE(Filtered.isEmpty());
      if constexpr (std::is_same_v<Map, ImmutableMap<int, int>>) {
        // A canonicalizing factory returns the same tree as for an equal map
        // built independently.
        Map Indep = F.getEmptyMap();
        for (auto &[K, V] : Exp)
          Indep = F.add(Indep, K, V);
        EXPECT_EQ(Indep.getRootWithoutRetain(),
                  Filtered.getRootWithoutRetain());
      }
    }
  }

  TEST(ImmutableMapTest, FilterTest) {
    testFilter<NCMap<int, int>>();
    testFilter<ImmutableMap<int, int>>();
  }

  // Removing the smallest key keeps the root's right subtree shared.
  TEST(ImmutableMapTest, FilterSharesUnchangedTest) {
    NCMap<int, int>::Factory F;
    NCMap<int, int> M = F.getEmptyMap();
    for (int I = 0; I < 100; ++I)
      M = F.add(M, I, I);
    NCMap<int, int> Filtered =
        F.filter(M, [](const auto &E) { return E.first != 0; });
    EXPECT_EQ(M.getRootWithoutRetain()->getRight(),
              Filtered.getRootWithoutRetain()->getRight());
  }
}
