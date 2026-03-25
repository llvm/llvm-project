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

using PU = PointerUnion<int *, float *>;
using PU3 = PointerUnion<int *, float *, long long *>;
using PU4 = PointerUnion<int *, float *, long long *, double *>;

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
  EXPECT_FALSE(isa<int *>(a));
  EXPECT_TRUE(isa<float *>(a));
  EXPECT_TRUE(isa<int *>(b));
  EXPECT_FALSE(isa<float *>(b));
  EXPECT_TRUE(isa<int *>(n));
  EXPECT_FALSE(isa<float *>(n));
  EXPECT_TRUE(isa<int *>(i3));
  EXPECT_TRUE(isa<float *>(f3));
  EXPECT_TRUE(isa<long long *>(l3));
  EXPECT_TRUE(isa<int *>(i4));
  EXPECT_TRUE(isa<float *>(f4));
  EXPECT_TRUE(isa<long long *>(l4));
  EXPECT_TRUE(isa<double *>(d4));
  EXPECT_TRUE(isa<int *>(i4null));
  EXPECT_TRUE(isa<float *>(f4null));
  EXPECT_TRUE(isa<long long *>(l4null));
  EXPECT_TRUE(isa<double *>(d4null));
}

TEST_F(PointerUnionTest, Get) {
  EXPECT_EQ(cast<float *>(a), &f);
  EXPECT_EQ(cast<int *>(b), &i);
  EXPECT_EQ(cast<int *>(n), (int *)nullptr);
}

template<int I> struct alignas(8) Aligned {};

using PU8 =
    PointerUnion<Aligned<0> *, Aligned<1> *, Aligned<2> *, Aligned<3> *,
                 Aligned<4> *, Aligned<5> *, Aligned<6> *, Aligned<7> *>;

TEST_F(PointerUnionTest, ManyElements) {
  Aligned<0> a0;
  Aligned<7> a7;

  PU8 a = &a0;
  EXPECT_TRUE(isa<Aligned<0> *>(a));
  EXPECT_FALSE(isa<Aligned<1> *>(a));
  EXPECT_FALSE(isa<Aligned<2> *>(a));
  EXPECT_FALSE(isa<Aligned<3> *>(a));
  EXPECT_FALSE(isa<Aligned<4> *>(a));
  EXPECT_FALSE(isa<Aligned<5> *>(a));
  EXPECT_FALSE(isa<Aligned<6> *>(a));
  EXPECT_FALSE(isa<Aligned<7> *>(a));
  EXPECT_EQ(dyn_cast_if_present<Aligned<0> *>(a), &a0);
  EXPECT_EQ(*a.getAddrOfPtr1(), &a0);

  a = &a7;
  EXPECT_FALSE(isa<Aligned<0> *>(a));
  EXPECT_FALSE(isa<Aligned<1> *>(a));
  EXPECT_FALSE(isa<Aligned<2> *>(a));
  EXPECT_FALSE(isa<Aligned<3> *>(a));
  EXPECT_FALSE(isa<Aligned<4> *>(a));
  EXPECT_FALSE(isa<Aligned<5> *>(a));
  EXPECT_FALSE(isa<Aligned<6> *>(a));
  EXPECT_TRUE(isa<Aligned<7> *>(a));
  EXPECT_EQ(dyn_cast_if_present<Aligned<7> *>(a), &a7);

  EXPECT_TRUE(a == PU8(&a7));
  EXPECT_TRUE(a != PU8(&a0));
}

TEST_F(PointerUnionTest, GetAddrOfPtr1) {
  EXPECT_TRUE((void *)b.getAddrOfPtr1() == (void *)&b);
  EXPECT_TRUE((void *)n.getAddrOfPtr1() == (void *)&n);
}

TEST_F(PointerUnionTest, NewCastInfra) {
  // test isa<>
  EXPECT_TRUE(isa<float *>(a));
  EXPECT_TRUE(isa<int *>(b));
  EXPECT_TRUE(isa<int *>(c));
  EXPECT_TRUE(isa<int *>(n));
  EXPECT_TRUE(isa<int *>(i3));
  EXPECT_TRUE(isa<float *>(f3));
  EXPECT_TRUE(isa<long long *>(l3));
  EXPECT_TRUE(isa<int *>(i4));
  EXPECT_TRUE(isa<float *>(f4));
  EXPECT_TRUE(isa<long long *>(l4));
  EXPECT_TRUE(isa<double *>(d4));
  EXPECT_TRUE(isa<int *>(i4null));
  EXPECT_TRUE(isa<float *>(f4null));
  EXPECT_TRUE(isa<long long *>(l4null));
  EXPECT_TRUE(isa<double *>(d4null));
  EXPECT_FALSE(isa<int *>(a));
  EXPECT_FALSE(isa<float *>(b));
  EXPECT_FALSE(isa<float *>(c));
  EXPECT_FALSE(isa<float *>(n));
  EXPECT_FALSE(isa<float *>(i3));
  EXPECT_FALSE(isa<long long *>(i3));
  EXPECT_FALSE(isa<int *>(f3));
  EXPECT_FALSE(isa<long long *>(f3));
  EXPECT_FALSE(isa<int *>(l3));
  EXPECT_FALSE(isa<float *>(l3));
  EXPECT_FALSE(isa<float *>(i4));
  EXPECT_FALSE(isa<long long *>(i4));
  EXPECT_FALSE(isa<double *>(i4));
  EXPECT_FALSE(isa<int *>(f4));
  EXPECT_FALSE(isa<long long *>(f4));
  EXPECT_FALSE(isa<double *>(f4));
  EXPECT_FALSE(isa<int *>(l4));
  EXPECT_FALSE(isa<float *>(l4));
  EXPECT_FALSE(isa<double *>(l4));
  EXPECT_FALSE(isa<int *>(d4));
  EXPECT_FALSE(isa<float *>(d4));
  EXPECT_FALSE(isa<long long *>(d4));
  EXPECT_FALSE(isa<float *>(i4null));
  EXPECT_FALSE(isa<long long *>(i4null));
  EXPECT_FALSE(isa<double *>(i4null));
  EXPECT_FALSE(isa<int *>(f4null));
  EXPECT_FALSE(isa<long long *>(f4null));
  EXPECT_FALSE(isa<double *>(f4null));
  EXPECT_FALSE(isa<int *>(l4null));
  EXPECT_FALSE(isa<float *>(l4null));
  EXPECT_FALSE(isa<double *>(l4null));
  EXPECT_FALSE(isa<int *>(d4null));
  EXPECT_FALSE(isa<float *>(d4null));
  EXPECT_FALSE(isa<long long *>(d4null));

  // test cast<>
  EXPECT_EQ(cast<float *>(a), &f);
  EXPECT_EQ(cast<int *>(b), &i);
  EXPECT_EQ(cast<int *>(c), &i);
  EXPECT_EQ(cast<int *>(i3), &i);
  EXPECT_EQ(cast<float *>(f3), &f);
  EXPECT_EQ(cast<long long *>(l3), &l);
  EXPECT_EQ(cast<int *>(i4), &i);
  EXPECT_EQ(cast<float *>(f4), &f);
  EXPECT_EQ(cast<long long *>(l4), &l);
  EXPECT_EQ(cast<double *>(d4), &d);

  // test dyn_cast
  EXPECT_EQ(dyn_cast<int *>(a), nullptr);
  EXPECT_EQ(dyn_cast<float *>(a), &f);
  EXPECT_EQ(dyn_cast<int *>(b), &i);
  EXPECT_EQ(dyn_cast<float *>(b), nullptr);
  EXPECT_EQ(dyn_cast<int *>(c), &i);
  EXPECT_EQ(dyn_cast<float *>(c), nullptr);
  EXPECT_EQ(dyn_cast_if_present<int *>(n), nullptr);
  EXPECT_EQ(dyn_cast_if_present<float *>(n), nullptr);
  EXPECT_EQ(dyn_cast<int *>(i3), &i);
  EXPECT_EQ(dyn_cast<float *>(i3), nullptr);
  EXPECT_EQ(dyn_cast<long long *>(i3), nullptr);
  EXPECT_EQ(dyn_cast<int *>(f3), nullptr);
  EXPECT_EQ(dyn_cast<float *>(f3), &f);
  EXPECT_EQ(dyn_cast<long long *>(f3), nullptr);
  EXPECT_EQ(dyn_cast<int *>(l3), nullptr);
  EXPECT_EQ(dyn_cast<float *>(l3), nullptr);
  EXPECT_EQ(dyn_cast<long long *>(l3), &l);
  EXPECT_EQ(dyn_cast<int *>(i4), &i);
  EXPECT_EQ(dyn_cast<float *>(i4), nullptr);
  EXPECT_EQ(dyn_cast<long long *>(i4), nullptr);
  EXPECT_EQ(dyn_cast<double *>(i4), nullptr);
  EXPECT_EQ(dyn_cast<int *>(f4), nullptr);
  EXPECT_EQ(dyn_cast<float *>(f4), &f);
  EXPECT_EQ(dyn_cast<long long *>(f4), nullptr);
  EXPECT_EQ(dyn_cast<double *>(f4), nullptr);
  EXPECT_EQ(dyn_cast<int *>(l4), nullptr);
  EXPECT_EQ(dyn_cast<float *>(l4), nullptr);
  EXPECT_EQ(dyn_cast<long long *>(l4), &l);
  EXPECT_EQ(dyn_cast<double *>(l4), nullptr);
  EXPECT_EQ(dyn_cast<int *>(d4), nullptr);
  EXPECT_EQ(dyn_cast<float *>(d4), nullptr);
  EXPECT_EQ(dyn_cast<long long *>(d4), nullptr);
  EXPECT_EQ(dyn_cast<double *>(d4), &d);
  EXPECT_EQ(dyn_cast_if_present<int *>(i4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<float *>(i4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<long long *>(i4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<double *>(i4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<int *>(f4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<float *>(f4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<long long *>(f4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<double *>(f4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<int *>(l4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<float *>(l4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<long long *>(l4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<double *>(l4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<int *>(d4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<float *>(d4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<long long *>(d4null), nullptr);
  EXPECT_EQ(dyn_cast_if_present<double *>(d4null), nullptr);

  // test for const
  const PU4 constd4(&d);
  EXPECT_TRUE(isa<double *>(constd4));
  EXPECT_FALSE(isa<int *>(constd4));
  EXPECT_EQ(cast<double *>(constd4), &d);
  EXPECT_EQ(dyn_cast<long long *>(constd4), nullptr);

  auto *result1 = cast<double *>(constd4);
  static_assert(std::is_same_v<double *, decltype(result1)>,
                "type mismatch for cast with PointerUnion");

  PointerUnion<int *, const double *> constd2(&d);
  auto *result2 = cast<const double *>(constd2);
  EXPECT_EQ(result2, &d);
  static_assert(std::is_same_v<const double *, decltype(result2)>,
                "type mismatch for cast with PointerUnion");
}

// Regression test: doCast must mask with minLowBitsAvailable(), not
// To::NumLowBitsAvailable, to avoid clearing inner PointerUnion tag bits.
// This reproduces the 32-bit crash from PR #187950 on any platform by
// using types whose alignment mimics the 32-bit DeclLink layout:
//   OuterPU<InnerPU, OverClaimWrapper>
// where OverClaimWrapper's PLTT claims more low bits than its inner
// PointerUnion actually has spare.

struct alignas(8) HighAlign {
  int x;
};
struct alignas(4) LowAlign {
  int x;
};

// Wrapper around a PointerUnion that over-claims NumLowBitsAvailable,
// mimicking LazyGenerationalUpdatePtr's PLTT on 32-bit.
struct OverClaimWrapper {
  PointerUnion<HighAlign *, LowAlign *> Value;

  OverClaimWrapper() = default;
  explicit OverClaimWrapper(decltype(Value) V) : Value(V) {}

  void *getOpaqueValue() { return Value.getOpaqueValue(); }
  static OverClaimWrapper getFromOpaqueValue(void *P) {
    return OverClaimWrapper(decltype(Value)::getFromOpaqueValue(P));
  }
};

} // end anonymous namespace

namespace llvm {
template <> struct PointerLikeTypeTraits<OverClaimWrapper> {
  static void *getAsVoidPointer(OverClaimWrapper W) {
    return W.getOpaqueValue();
  }
  static OverClaimWrapper getFromVoidPointer(void *P) {
    return OverClaimWrapper::getFromOpaqueValue(P);
  }
  // Inner PU<HighAlign*(3 bits), LowAlign*(2 bits)> has tagShift=1, so only
  // 1 spare bit. Claiming 2 bits mimics the LGUP over-claim on 32-bit.
  static constexpr int NumLowBitsAvailable = 2;
};
} // namespace llvm

namespace {

TEST(PointerUnionNestedTest, NestedTagPreservation) {
  // Inner PU: PointerUnion<HighAlign*, LowAlign*>
  //   minLowBits = min(3, 2) = 2, tagBits = 1, tagShift = 1
  //   Tag for LowAlign* (index 1) is in bit 1.
  //   NumLowBitsAvailable = 1

  // Outer PU: PointerUnion<InnerPU, OverClaimWrapper>
  //   InnerPU NumLowBitsAvailable = 1
  //   OverClaimWrapper NumLowBitsAvailable = 2 (over-claimed)
  //   minLowBits = 1, tagBits = 1, tagShift = 0

  using InnerPU = PointerUnion<HighAlign *, LowAlign *>;
  using OuterPU = PointerUnion<InnerPU, OverClaimWrapper>;

  LowAlign low;

  // Store LowAlign* in the inner PU (tag = 1, in bit 1).
  InnerPU inner(&low);
  ASSERT_TRUE(isa<LowAlign *>(inner));
  ASSERT_EQ(cast<LowAlign *>(inner), &low);

  // Wrap it and store in the outer PU.
  OverClaimWrapper wrapper(inner);
  OuterPU outer(wrapper);
  ASSERT_TRUE(isa<OverClaimWrapper>(outer));

  // Extract the wrapper back. Before the fix, doCast would clear bit 1
  // (the inner PU's tag), corrupting the type discriminator.
  OverClaimWrapper extracted = cast<OverClaimWrapper>(outer);
  InnerPU extractedInner = extracted.Value;

  EXPECT_TRUE(isa<LowAlign *>(extractedInner))
      << "Inner PointerUnion tag corrupted during doCast: expected LowAlign*, "
         "got HighAlign*. doCast must not clear bits beyond "
         "minLowBitsAvailable().";
  EXPECT_EQ(cast<LowAlign *>(extractedInner), &low);

  // Also verify the HighAlign* path (tag = 0) works.
  HighAlign high;
  InnerPU inner2(&high);
  OverClaimWrapper wrapper2(inner2);
  OuterPU outer2(wrapper2);
  OverClaimWrapper extracted2 = cast<OverClaimWrapper>(outer2);
  EXPECT_TRUE(isa<HighAlign *>(extracted2.Value));
  EXPECT_EQ(cast<HighAlign *>(extracted2.Value), &high);
}

} // end anonymous namespace
