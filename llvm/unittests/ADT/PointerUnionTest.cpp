//===- llvm/unittest/ADT/PointerUnionTest.cpp - Optional unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/PointerUnion.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

typedef PointerUnion<int *, float *> PU;

struct PointerUnionTest : public testing::Test {
  float f;
  int i;

  PU a, b, c, n;

  PointerUnionTest() : f(3.14f), i(42), a(&f), b(&i), c(&i), n() {}
};

TEST_F(PointerUnionTest, Comparison) {
  EXPECT_TRUE(a == a);
  EXPECT_FALSE(a != a);
  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(b == c);
  EXPECT_FALSE(b != c);
  EXPECT_TRUE(b != n);
  EXPECT_FALSE(b == n);
}

TEST_F(PointerUnionTest, Null) {
  EXPECT_FALSE(a.isNull());
  EXPECT_FALSE(b.isNull());
  EXPECT_TRUE(n.isNull());
  EXPECT_FALSE(!a);
  EXPECT_FALSE(!b);
  EXPECT_TRUE(!n);
  // workaround an issue with EXPECT macros and explicit bool
  EXPECT_TRUE((bool)a);
  EXPECT_TRUE((bool)b);
  EXPECT_FALSE(n);

  EXPECT_NE(n, b);
  EXPECT_EQ(b, c);
  b = nullptr;
  EXPECT_EQ(n, b);
  EXPECT_NE(b, c);
}

TEST_F(PointerUnionTest, Is) {
  EXPECT_FALSE(a.is<int *>());
  EXPECT_TRUE(a.is<float *>());
  EXPECT_TRUE(b.is<int *>());
  EXPECT_FALSE(b.is<float *>());
  EXPECT_TRUE(n.is<int *>());
  EXPECT_FALSE(n.is<float *>());
}

TEST_F(PointerUnionTest, Get) {
  EXPECT_EQ(a.get<float *>(), &f);
  EXPECT_EQ(b.get<int *>(), &i);
  EXPECT_EQ(n.get<int *>(), (int *)nullptr);
}

} // end anonymous namespace
