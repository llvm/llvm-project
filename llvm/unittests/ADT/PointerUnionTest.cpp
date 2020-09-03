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
typedef PointerUnion<int *, float *, long long *> PU3;
typedef PointerUnion<int *, float *, long long *, double *> PU4;

struct PointerUnionTest : public testing::Test {
  float f;
  int i;
  double d;
  long long l;

  PU a, b, c, n;
  PU3 i3, f3, l3;
  PU4 i4, f4, l4, d4;
  PU4 i4null, f4null, l4null, d4null;

  PointerUnionTest()
      : f(3.14f), i(42), d(3.14), l(42), a(&f), b(&i), c(&i), n(), i3(&i),
        f3(&f), l3(&l), i4(&i), f4(&f), l4(&l), d4(&d), i4null((int *)nullptr),
        f4null((float *)nullptr), l4null((long long *)nullptr),
        d4null((double *)nullptr) {}
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
  EXPECT_TRUE(i3 == i3);
  EXPECT_FALSE(i3 != i3);
  EXPECT_TRUE(i3 != f3);
  EXPECT_TRUE(f3 != l3);
  EXPECT_TRUE(i4 == i4);
  EXPECT_FALSE(i4 != i4);
  EXPECT_TRUE(i4 != f4);
  EXPECT_TRUE(i4 != l4);
  EXPECT_TRUE(f4 != l4);
  EXPECT_TRUE(l4 != d4);
  EXPECT_TRUE(i4null != f4null);
  EXPECT_TRUE(i4null != l4null);
  EXPECT_TRUE(i4null != d4null);
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
  EXPECT_FALSE(i3.isNull());
  EXPECT_FALSE(f3.isNull());
  EXPECT_FALSE(l3.isNull());
  EXPECT_FALSE(i4.isNull());
  EXPECT_FALSE(f4.isNull());
  EXPECT_FALSE(l4.isNull());
  EXPECT_FALSE(d4.isNull());
  EXPECT_TRUE(i4null.isNull());
  EXPECT_TRUE(f4null.isNull());
  EXPECT_TRUE(l4null.isNull());
  EXPECT_TRUE(d4null.isNull());
}

TEST_F(PointerUnionTest, Is) {
  EXPECT_FALSE(a.is<int *>());
  EXPECT_TRUE(a.is<float *>());
  EXPECT_TRUE(b.is<int *>());
  EXPECT_FALSE(b.is<float *>());
  EXPECT_TRUE(n.is<int *>());
  EXPECT_FALSE(n.is<float *>());
  EXPECT_TRUE(i3.is<int *>());
  EXPECT_TRUE(f3.is<float *>());
  EXPECT_TRUE(l3.is<long long *>());
  EXPECT_TRUE(i4.is<int *>());
  EXPECT_TRUE(f4.is<float *>());
  EXPECT_TRUE(l4.is<long long *>());
  EXPECT_TRUE(d4.is<double *>());
  EXPECT_TRUE(i4null.is<int *>());
  EXPECT_TRUE(f4null.is<float *>());
  EXPECT_TRUE(l4null.is<long long *>());
  EXPECT_TRUE(d4null.is<double *>());
}

TEST_F(PointerUnionTest, Get) {
  EXPECT_EQ(a.get<float *>(), &f);
  EXPECT_EQ(b.get<int *>(), &i);
  EXPECT_EQ(n.get<int *>(), (int *)nullptr);
}

template<int I> struct alignas(8) Aligned {};

typedef PointerUnion<Aligned<0> *, Aligned<1> *, Aligned<2> *, Aligned<3> *,
                     Aligned<4> *, Aligned<5> *, Aligned<6> *, Aligned<7> *>
    PU8;

TEST_F(PointerUnionTest, ManyElements) {
  Aligned<0> a0;
  Aligned<7> a7;

  PU8 a = &a0;
  EXPECT_TRUE(a.is<Aligned<0>*>());
  EXPECT_FALSE(a.is<Aligned<1>*>());
  EXPECT_FALSE(a.is<Aligned<2>*>());
  EXPECT_FALSE(a.is<Aligned<3>*>());
  EXPECT_FALSE(a.is<Aligned<4>*>());
  EXPECT_FALSE(a.is<Aligned<5>*>());
  EXPECT_FALSE(a.is<Aligned<6>*>());
  EXPECT_FALSE(a.is<Aligned<7>*>());
  EXPECT_EQ(a.dyn_cast<Aligned<0>*>(), &a0);
  EXPECT_EQ(*a.getAddrOfPtr1(), &a0);

  a = &a7;
  EXPECT_FALSE(a.is<Aligned<0>*>());
  EXPECT_FALSE(a.is<Aligned<1>*>());
  EXPECT_FALSE(a.is<Aligned<2>*>());
  EXPECT_FALSE(a.is<Aligned<3>*>());
  EXPECT_FALSE(a.is<Aligned<4>*>());
  EXPECT_FALSE(a.is<Aligned<5>*>());
  EXPECT_FALSE(a.is<Aligned<6>*>());
  EXPECT_TRUE(a.is<Aligned<7>*>());
  EXPECT_EQ(a.dyn_cast<Aligned<7>*>(), &a7);

  EXPECT_TRUE(a == PU8(&a7));
  EXPECT_TRUE(a != PU8(&a0));
}

TEST_F(PointerUnionTest, GetAddrOfPtr1) {
  EXPECT_TRUE((void *)b.getAddrOfPtr1() == (void *)&b);
  EXPECT_TRUE((void *)n.getAddrOfPtr1() == (void *)&n);
}

} // end anonymous namespace
