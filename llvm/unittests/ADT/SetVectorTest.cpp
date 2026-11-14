//===- llvm/unittest/ADT/SetVector.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SetVector unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(SetVector, EraseTest) {
  SetVector<int> S;
  S.insert(0);
  S.insert(1);
  S.insert(2);

  auto I = S.erase(std::next(S.begin()));

  // Test that the returned iterator is the expected one-after-erase
  // and the size/contents is the expected sequence {0, 2}.
  EXPECT_EQ(std::next(S.begin()), I);
  EXPECT_EQ(2u, S.size());
  EXPECT_EQ(0, *S.begin());
  EXPECT_EQ(2, *std::next(S.begin()));
}

TEST(SetVector, ContainsTest) {
  SetVector<int> S;
  S.insert(0);
  S.insert(1);
  S.insert(2);

  EXPECT_TRUE(S.contains(0));
  EXPECT_TRUE(S.contains(1));
  EXPECT_TRUE(S.contains(2));
  EXPECT_FALSE(S.contains(-1));

  S.insert(2);
  EXPECT_TRUE(S.contains(2));

  S.remove(2);
  EXPECT_FALSE(S.contains(2));
}

TEST(SetVector, ConstPtrKeyTest) {
  SetVector<int *> S, T;
  int i, j, k, m, n;

  S.insert(&i);
  S.insert(&j);
  S.insert(&k);

  EXPECT_TRUE(S.contains(&i));
  EXPECT_TRUE(S.contains(&j));
  EXPECT_TRUE(S.contains(&k));

  EXPECT_TRUE(S.contains((const int *)&i));
  EXPECT_TRUE(S.contains((const int *)&j));
  EXPECT_TRUE(S.contains((const int *)&k));

  EXPECT_TRUE(S.contains(S[0]));
  EXPECT_TRUE(S.contains(S[1]));
  EXPECT_TRUE(S.contains(S[2]));

  S.remove(&k);
  EXPECT_FALSE(S.contains(&k));
  EXPECT_FALSE(S.contains((const int *)&k));

  T.insert(&j);
  T.insert(&m);
  T.insert(&n);

  EXPECT_TRUE(S.set_union(T));
  EXPECT_TRUE(S.contains(&m));
  EXPECT_TRUE(S.contains((const int *)&m));

  S.set_subtract(T);
  EXPECT_FALSE(S.contains(&j));
  EXPECT_FALSE(S.contains((const int *)&j));
}

TEST(SetVector, CtorRange) {
  constexpr unsigned Args[] = {3, 1, 2};
  SetVector<unsigned> Set(llvm::from_range, Args);
  EXPECT_THAT(Set, ::testing::ElementsAre(3, 1, 2));
}

TEST(SetVector, InsertRange) {
  SetVector<unsigned> Set;
  constexpr unsigned Args[] = {3, 1, 2};
  Set.insert_range(Args);
  EXPECT_THAT(Set, ::testing::ElementsAre(3, 1, 2));
}

TEST(SmallSetVector, CtorRange) {
  constexpr unsigned Args[] = {3, 1, 2};
  SmallSetVector<unsigned, 4> Set(llvm::from_range, Args);
  EXPECT_THAT(Set, ::testing::ElementsAre(3, 1, 2));
}

namespace {
struct PolymorphicBase {
  int Value = 0;
  PolymorphicBase() = default;
  explicit PolymorphicBase(int V) : Value(V) {}
  PolymorphicBase(const PolymorphicBase &) = default;
  PolymorphicBase(PolymorphicBase &&) noexcept = default;
  PolymorphicBase &operator=(const PolymorphicBase &) = default;
  PolymorphicBase &operator=(PolymorphicBase &&) noexcept = default;
  virtual ~PolymorphicBase() = default;
  virtual int get() const { return Value; }
  bool operator==(const PolymorphicBase &RHS) const {
    return Value == RHS.Value;
  }
};

struct PolymorphicDerived : PolymorphicBase {
  using PolymorphicBase::PolymorphicBase;
  int get() const override { return Value * 2; }
};
} // namespace

namespace llvm {
template <> struct DenseMapInfo<PolymorphicBase> {
  static unsigned getHashValue(const PolymorphicBase &Val) {
    return static_cast<unsigned>(Val.Value) * 37U;
  }
  static bool isEqual(const PolymorphicBase &LHS, const PolymorphicBase &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

TEST(SetVector, PolymorphicTypes) {
  SetVector<PolymorphicBase> SV;
  SmallSetVector<PolymorphicBase, 2> SSV;
  for (int I = 0; I < 6; ++I) {
    EXPECT_TRUE(SV.insert(PolymorphicBase(I)));
    EXPECT_TRUE(SSV.insert(PolymorphicBase(I)));
  }
  EXPECT_FALSE(SV.insert(PolymorphicBase(2)));
  EXPECT_FALSE(SSV.insert(PolymorphicBase(2)));
  EXPECT_EQ(6u, SV.size());
  EXPECT_EQ(6u, SSV.size());
  EXPECT_EQ(4, SV[4].get());
  EXPECT_EQ(4, SSV[4].get());

  EXPECT_TRUE(SV.remove(PolymorphicBase(2)));
  EXPECT_TRUE(SSV.remove(PolymorphicBase(2)));
  EXPECT_FALSE(SV.contains(PolymorphicBase(2)));
  EXPECT_FALSE(SSV.contains(PolymorphicBase(2)));

  PolymorphicBase B(3);
  PolymorphicDerived D(4);
  SetVector<PolymorphicBase *> PtrSV;
  SmallSetVector<PolymorphicBase *, 2> PtrSSV;
  EXPECT_TRUE(PtrSV.insert(&B));
  EXPECT_TRUE(PtrSV.insert(&D));
  EXPECT_TRUE(PtrSSV.insert(&B));
  EXPECT_TRUE(PtrSSV.insert(&D));
  const PolymorphicBase *ConstD = &D;
  EXPECT_TRUE(PtrSV.contains(ConstD));
  EXPECT_TRUE(PtrSSV.contains(ConstD));
  EXPECT_EQ(3, PtrSV[0]->get());
  EXPECT_EQ(8, PtrSV[1]->get());
  EXPECT_EQ(3, PtrSSV[0]->get());
  EXPECT_EQ(8, PtrSSV[1]->get());
}
