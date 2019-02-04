//===- llvm/unittest/ADT/PointerSumTypeTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/PointerSumType.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

struct PointerSumTypeTest : public testing::Test {
  enum Kinds { Float, Int1, Int2 };
  float f;
  int i1, i2;

  typedef PointerSumType<Kinds, PointerSumTypeMember<Float, float *>,
                         PointerSumTypeMember<Int1, int *>,
                         PointerSumTypeMember<Int2, int *>>
      SumType;
  SumType a, b, c, n;

  PointerSumTypeTest()
      : f(3.14f), i1(42), i2(-1), a(SumType::create<Float>(&f)),
        b(SumType::create<Int1>(&i1)), c(SumType::create<Int2>(&i2)), n() {}
};

TEST_F(PointerSumTypeTest, NullTest) {
  EXPECT_TRUE(a);
  EXPECT_TRUE(b);
  EXPECT_TRUE(c);
  EXPECT_FALSE(n);
}

TEST_F(PointerSumTypeTest, GetTag) {
  EXPECT_EQ(Float, a.getTag());
  EXPECT_EQ(Int1, b.getTag());
  EXPECT_EQ(Int2, c.getTag());
  EXPECT_EQ((Kinds)0, n.getTag());
}

TEST_F(PointerSumTypeTest, Is) {
  EXPECT_TRUE(a.is<Float>());
  EXPECT_FALSE(a.is<Int1>());
  EXPECT_FALSE(a.is<Int2>());
  EXPECT_FALSE(b.is<Float>());
  EXPECT_TRUE(b.is<Int1>());
  EXPECT_FALSE(b.is<Int2>());
  EXPECT_FALSE(c.is<Float>());
  EXPECT_FALSE(c.is<Int1>());
  EXPECT_TRUE(c.is<Int2>());
}

TEST_F(PointerSumTypeTest, Get) {
  EXPECT_EQ(&f, a.get<Float>());
  EXPECT_EQ(nullptr, a.get<Int1>());
  EXPECT_EQ(nullptr, a.get<Int2>());
  EXPECT_EQ(nullptr, b.get<Float>());
  EXPECT_EQ(&i1, b.get<Int1>());
  EXPECT_EQ(nullptr, b.get<Int2>());
  EXPECT_EQ(nullptr, c.get<Float>());
  EXPECT_EQ(nullptr, c.get<Int1>());
  EXPECT_EQ(&i2, c.get<Int2>());

  // Note that we can use .get even on a null sum type. It just always produces
  // a null pointer, even if one of the discriminants is null.
  EXPECT_EQ(nullptr, n.get<Float>());
  EXPECT_EQ(nullptr, n.get<Int1>());
  EXPECT_EQ(nullptr, n.get<Int2>());
}

TEST_F(PointerSumTypeTest, Cast) {
  EXPECT_EQ(&f, a.cast<Float>());
  EXPECT_EQ(&i1, b.cast<Int1>());
  EXPECT_EQ(&i2, c.cast<Int2>());
}

TEST_F(PointerSumTypeTest, Assignment) {
  b = SumType::create<Int2>(&i2);
  EXPECT_EQ(nullptr, b.get<Float>());
  EXPECT_EQ(nullptr, b.get<Int1>());
  EXPECT_EQ(&i2, b.get<Int2>());

  b = SumType::create<Int2>(&i1);
  EXPECT_EQ(nullptr, b.get<Float>());
  EXPECT_EQ(nullptr, b.get<Int1>());
  EXPECT_EQ(&i1, b.get<Int2>());

  float Local = 1.616f;
  b = SumType::create<Float>(&Local);
  EXPECT_EQ(&Local, b.get<Float>());
  EXPECT_EQ(nullptr, b.get<Int1>());
  EXPECT_EQ(nullptr, b.get<Int2>());

  n = SumType::create<Int1>(&i2);
  EXPECT_TRUE(n);
  EXPECT_EQ(nullptr, n.get<Float>());
  EXPECT_EQ(&i2, n.get<Int1>());
  EXPECT_EQ(nullptr, n.get<Int2>());

  n = SumType::create<Float>(nullptr);
  EXPECT_FALSE(n);
  EXPECT_EQ(nullptr, n.get<Float>());
  EXPECT_EQ(nullptr, n.get<Int1>());
  EXPECT_EQ(nullptr, n.get<Int2>());
}


} // end anonymous namespace
