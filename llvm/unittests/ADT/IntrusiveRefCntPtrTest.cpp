//===- unittest/ADT/IntrusiveRefCntPtrTest.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "gtest/gtest.h"

namespace llvm {

namespace {
int NumInstances = 0;
template <template <typename> class Base>
struct SimpleRefCounted : Base<SimpleRefCounted<Base>> {
  SimpleRefCounted() { ++NumInstances; }
  SimpleRefCounted(const SimpleRefCounted &RHS) : Base<SimpleRefCounted>(RHS) {
    ++NumInstances;
  }
  ~SimpleRefCounted() { --NumInstances; }
};
} // anonymous namespace

template <typename T> struct IntrusiveRefCntPtrTest : testing::Test {};

typedef ::testing::Types<SimpleRefCounted<RefCountedBase>,
                         SimpleRefCounted<ThreadSafeRefCountedBase>>
    IntrusiveRefCntTypes;
TYPED_TEST_SUITE(IntrusiveRefCntPtrTest, IntrusiveRefCntTypes, );

TYPED_TEST(IntrusiveRefCntPtrTest, RefCountedBaseCopyDoesNotLeak) {
  EXPECT_EQ(0, NumInstances);
  {
    TypeParam *S1 = new TypeParam;
    IntrusiveRefCntPtr<TypeParam> R1 = S1;
    TypeParam *S2 = new TypeParam(*S1);
    IntrusiveRefCntPtr<TypeParam> R2 = S2;
    EXPECT_EQ(2, NumInstances);
  }
  EXPECT_EQ(0, NumInstances);
}

TYPED_TEST(IntrusiveRefCntPtrTest, InteropsWithUniquePtr) {
  EXPECT_EQ(0, NumInstances);
  {
    auto S1 = std::make_unique<TypeParam>();
    IntrusiveRefCntPtr<TypeParam> R1 = std::move(S1);
    EXPECT_EQ(1, NumInstances);
    EXPECT_EQ(S1, nullptr);
  }
  EXPECT_EQ(0, NumInstances);
}

TYPED_TEST(IntrusiveRefCntPtrTest, MakeIntrusiveRefCnt) {
  EXPECT_EQ(0, NumInstances);
  {
    auto S1 = makeIntrusiveRefCnt<TypeParam>();
    auto S2 = makeIntrusiveRefCnt<const TypeParam>();
    EXPECT_EQ(2, NumInstances);
    static_assert(
        std::is_same<decltype(S1), IntrusiveRefCntPtr<TypeParam>>::value,
        "Non-const type mismatch");
    static_assert(
        std::is_same<decltype(S2), IntrusiveRefCntPtr<const TypeParam>>::value,
        "Const type mismatch");
  }
  EXPECT_EQ(0, NumInstances);
}

struct InterceptRefCounted : public RefCountedBase<InterceptRefCounted> {
  InterceptRefCounted(bool *Released, bool *Retained)
    : Released(Released), Retained(Retained) {}
  bool * const Released;
  bool * const Retained;
};
template <> struct IntrusiveRefCntPtrInfo<InterceptRefCounted> {
  static void retain(InterceptRefCounted *I) {
    *I->Retained = true;
    I->Retain();
  }
  static void release(InterceptRefCounted *I) {
    *I->Released = true;
    I->Release();
  }
};
TEST(IntrusiveRefCntPtr, UsesTraitsToRetainAndRelease) {
  bool Released = false;
  bool Retained = false;
  {
    InterceptRefCounted *I = new InterceptRefCounted(&Released, &Retained);
    IntrusiveRefCntPtr<InterceptRefCounted> R = I;
  }
  EXPECT_TRUE(Released);
  EXPECT_TRUE(Retained);
}

// Test that the generic constructors use SFINAE to disable invalid
// conversions.
struct X : RefCountedBase<X> {};
struct Y : X {};
struct Z : RefCountedBase<Z> {};
static_assert(
    !std::is_convertible_v<IntrusiveRefCntPtr<X> &&, IntrusiveRefCntPtr<Y>>,
    "X&& -> Y should be rejected with SFINAE");
static_assert(!std::is_convertible_v<const IntrusiveRefCntPtr<X> &,
                                     IntrusiveRefCntPtr<Y>>,
              "const X& -> Y should be rejected with SFINAE");
static_assert(!std::is_convertible_v<std::unique_ptr<X>, IntrusiveRefCntPtr<Y>>,
              "X -> Y should be rejected with SFINAE");
static_assert(
    !std::is_convertible_v<IntrusiveRefCntPtr<X> &&, IntrusiveRefCntPtr<Z>>,
    "X&& -> Z should be rejected with SFINAE");
static_assert(!std::is_convertible_v<const IntrusiveRefCntPtr<X> &,
                                     IntrusiveRefCntPtr<Z>>,
              "const X& -> Z should be rejected with SFINAE");
static_assert(!std::is_convertible_v<std::unique_ptr<X>, IntrusiveRefCntPtr<Z>>,
              "X -> Z should be rejected with SFINAE");

TEST(IntrusiveRefCntPtr, InteropsWithConvertible) {
  // Check converting constructors and operator=.
  auto Y1 = makeIntrusiveRefCnt<Y>();
  auto Y2 = makeIntrusiveRefCnt<Y>();
  auto Y3 = makeIntrusiveRefCnt<Y>();
  auto Y4 = makeIntrusiveRefCnt<Y>();
  const void *P1 = Y1.get();
  const void *P2 = Y2.get();
  const void *P3 = Y3.get();
  const void *P4 = Y4.get();
  IntrusiveRefCntPtr<X> X1 = std::move(Y1);
  IntrusiveRefCntPtr<X> X2 = Y2;
  IntrusiveRefCntPtr<X> X3;
  IntrusiveRefCntPtr<X> X4;
  X3 = std::move(Y3);
  X4 = Y4;
  EXPECT_EQ(P1, X1.get());
  EXPECT_EQ(P2, X2.get());
  EXPECT_EQ(P3, X3.get());
  EXPECT_EQ(P4, X4.get());
}

TEST(IntrusiveRefCntPtrTest, Unique) {
  IntrusiveRefCntPtr<X> X1;
  EXPECT_EQ(X1.useCount(), 0);
  X1 = new X();
  EXPECT_EQ(X1.useCount(), 1);
  {
    IntrusiveRefCntPtr<X> X2 = X1;
    EXPECT_EQ(X1.useCount(), 2);
    EXPECT_EQ(X2.useCount(), 2);
  }
  EXPECT_EQ(X1.useCount(), 1);
}

} // end namespace llvm
