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
TYPED_TEST_CASE(IntrusiveRefCntPtrTest, IntrusiveRefCntTypes);

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

} // end namespace llvm
