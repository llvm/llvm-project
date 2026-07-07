//===- llvm/unittest/ADT/PointerUnionTest.cpp - PointerUnion unit tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/DenseMap.h"
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
        f3(&f), l3(&l), i4(&i), f4(&f), l4(&l), d4(&d),
        i4null(static_cast<int *>(nullptr)),
        f4null(static_cast<float *>(nullptr)),
        l4null(static_cast<long long *>(nullptr)),
        d4null(static_cast<double *>(nullptr)) {}
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
  EXPECT_TRUE(static_cast<bool>(a));
  EXPECT_TRUE(static_cast<bool>(b));
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
  EXPECT_EQ(cast<int *>(n), static_cast<int *>(nullptr));
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
  EXPECT_TRUE(static_cast<void *>(b.getAddrOfPtr1()) ==
              static_cast<void *>(&b));
  EXPECT_TRUE(static_cast<void *>(n.getAddrOfPtr1()) ==
              static_cast<void *>(&n));
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

//===----------------------------------------------------------------------===//
// Variable-width encoding PointerUnion tests
//===----------------------------------------------------------------------===//

template <int I> struct alignas(4) Align4 {};
template <int I> struct alignas(8) Align8 {};
template <int I> struct alignas(16) Align16 {};

TEST(PointerUnionEncodingTest, ExtendedTagsFit) {
  // Positive: 3 x 2-bit + 2 x 3-bit types.
  EXPECT_TRUE(
      (pointer_union_detail::computeExtendedTags<
           Align4<0> *, Align4<1> *, Align4<2> *, Align8<0> *, Align8<1> *>()
           .has_value()));
  // Negative: 4 x 2-bit types need 4 codes but only 3 are available
  // (2^2 - 1 escape = 3).
  EXPECT_FALSE(
      (pointer_union_detail::computeExtendedTags<
           Align4<0> *, Align4<1> *, Align4<2> *, Align4<3> *, Align8<0> *>()
           .has_value()));
}

TEST(PointerUnionEncodingTest, ComputeExtendedTags) {
  // 2-tier union: 3 x 2-bit + 2 x 3-bit.
  auto Tags = *pointer_union_detail::computeExtendedTags<
      Align4<0> *, Align4<1> *, Align4<2> *, Align8<0> *, Align8<1> *>();
  // Tier 0 (2-bit): codes 0b00, 0b01, 0b10; escape = 0b11.
  EXPECT_EQ(Tags[0].Value, 0b00u);
  EXPECT_EQ(Tags[0].Mask, 0b11u);
  EXPECT_EQ(Tags[1].Value, 0b01u);
  EXPECT_EQ(Tags[2].Value, 0b10u);
  // Tier 1 (3-bit): codes 0b011, 0b111; mask = 0b111.
  EXPECT_EQ(Tags[3].Value, 0b011u);
  EXPECT_EQ(Tags[3].Mask, 0b111u);
  EXPECT_EQ(Tags[4].Value, 0b111u);
}

TEST(PointerUnionEncodingTest, ComputeExtendedTags3Tier) {
  // 3-tier union: 3 x 2-bit + 1 x 3-bit + 2 x 4-bit.
  auto Tags =
      *pointer_union_detail::computeExtendedTags<Align4<0> *, Align4<1> *,
                                                 Align4<2> *, Align8<0> *,
                                                 Align16<0> *, Align16<1> *>();
  // Tier 0 (2-bit): codes 0b00, 0b01, 0b10; escape = 0b11.
  EXPECT_EQ(Tags[0].Value, 0b00u);
  EXPECT_EQ(Tags[0].Mask, 0b11u);
  EXPECT_EQ(Tags[1].Value, 0b01u);
  EXPECT_EQ(Tags[2].Value, 0b10u);
  // Tier 1 (3-bit): code 0b011; escape = 0b111. Mask = 0b111.
  EXPECT_EQ(Tags[3].Value, 0b011u);
  EXPECT_EQ(Tags[3].Mask, 0b111u);
  // Tier 2 (4-bit): codes 0b0111, 0b1111. Mask = 0b1111.
  EXPECT_EQ(Tags[4].Value, 0b0111u);
  EXPECT_EQ(Tags[4].Mask, 0b1111u);
  EXPECT_EQ(Tags[5].Value, 0b1111u);
  EXPECT_EQ(Tags[5].Mask, 0b1111u);
}

// 2-tier: 3 x 2-bit + 2 x 3-bit types.
using PU2Tier = PointerUnion<Align4<0> *, Align4<1> *, Align4<2> *, Align8<0> *,
                             Align8<1> *>;

// 3-tier: 3 x 2-bit + 1 x 3-bit + 2 x 4-bit types.
using PU3Tier = PointerUnion<Align4<0> *, Align4<1> *, Align4<2> *, Align8<0> *,
                             Align16<0> *, Align16<1> *>;

// Variable-width unions still fit in a single pointer.
static_assert(sizeof(PU2Tier) == sizeof(void *));
static_assert(sizeof(PU3Tier) == sizeof(void *));

// These unions actually use variable-width encoding (fixed-width tags don't
// fit because 5 types need 3 tag bits but Align4 only provides 2).
static_assert(
    !pointer_union_detail::useFixedWidthTags<
        Align4<0> *, Align4<1> *, Align4<2> *, Align8<0> *, Align8<1> *>());
static_assert(!pointer_union_detail::useFixedWidthTags<
              Align4<0> *, Align4<1> *, Align4<2> *, Align8<0> *, Align16<0> *,
              Align16<1> *>());

// NumLowBitsAvailable is 0 for variable-width PointerUnion.
static_assert(PointerLikeTypeTraits<PU2Tier>::NumLowBitsAvailable == 0);
static_assert(PointerLikeTypeTraits<PU3Tier>::NumLowBitsAvailable == 0);

struct PointerUnion2TierTest : public testing::Test {
  Align4<0> a0;
  Align4<1> a1;
  Align4<2> a2;
  Align8<0> b0;
  Align8<1> b1;

  PU2Tier pa0, pa1, pa2, pb0, pb1, null;
  PU2Tier na0, na1, na2, nb0, nb1;

  PointerUnion2TierTest()
      : pa0(&a0), pa1(&a1), pa2(&a2), pb0(&b0), pb1(&b1), null(),
        na0(static_cast<Align4<0> *>(nullptr)),
        na1(static_cast<Align4<1> *>(nullptr)),
        na2(static_cast<Align4<2> *>(nullptr)),
        nb0(static_cast<Align8<0> *>(nullptr)),
        nb1(static_cast<Align8<1> *>(nullptr)) {}
};

TEST_F(PointerUnion2TierTest, Isa) {
  // Tier 0 types
  EXPECT_TRUE(isa<Align4<0> *>(pa0));
  EXPECT_FALSE(isa<Align4<1> *>(pa0));
  EXPECT_FALSE(isa<Align4<2> *>(pa0));
  EXPECT_FALSE(isa<Align8<0> *>(pa0));
  EXPECT_FALSE(isa<Align8<1> *>(pa0));

  EXPECT_TRUE(isa<Align4<1> *>(pa1));
  EXPECT_TRUE(isa<Align4<2> *>(pa2));

  // Tier 1 types
  EXPECT_TRUE(isa<Align8<0> *>(pb0));
  EXPECT_FALSE(isa<Align4<0> *>(pb0));
  EXPECT_FALSE(isa<Align8<1> *>(pb0));

  EXPECT_TRUE(isa<Align8<1> *>(pb1));
  EXPECT_FALSE(isa<Align8<0> *>(pb1));

  // Null pointers preserve type identity
  EXPECT_TRUE(isa<Align4<0> *>(na0));
  EXPECT_TRUE(isa<Align8<1> *>(nb1));
  EXPECT_FALSE(isa<Align8<0> *>(na0));
}

TEST_F(PointerUnion2TierTest, Cast) {
  EXPECT_EQ(cast<Align4<0> *>(pa0), &a0);
  EXPECT_EQ(cast<Align4<1> *>(pa1), &a1);
  EXPECT_EQ(cast<Align4<2> *>(pa2), &a2);
  EXPECT_EQ(cast<Align8<0> *>(pb0), &b0);
  EXPECT_EQ(cast<Align8<1> *>(pb1), &b1);
}

TEST_F(PointerUnion2TierTest, DynCast) {
  EXPECT_EQ(dyn_cast<Align4<0> *>(pa0), &a0);
  EXPECT_EQ(dyn_cast<Align4<1> *>(pa0), nullptr);
  EXPECT_EQ(dyn_cast<Align8<0> *>(pa0), nullptr);

  EXPECT_EQ(dyn_cast<Align8<0> *>(pb0), &b0);
  EXPECT_EQ(dyn_cast<Align4<0> *>(pb0), nullptr);

  // pb1 has the all-ones tag -- most likely to expose masking bugs.
  EXPECT_EQ(dyn_cast<Align8<1> *>(pb1), &b1);
  EXPECT_EQ(dyn_cast<Align4<0> *>(pb1), nullptr);
  EXPECT_EQ(dyn_cast<Align4<1> *>(pb1), nullptr);
  EXPECT_EQ(dyn_cast<Align4<2> *>(pb1), nullptr);
  EXPECT_EQ(dyn_cast<Align8<0> *>(pb1), nullptr);

  EXPECT_EQ(dyn_cast_if_present<Align4<0> *>(na0), nullptr);
  EXPECT_EQ(dyn_cast_if_present<Align8<0> *>(na0), nullptr);
  EXPECT_EQ(dyn_cast_if_present<Align8<0> *>(nb0), nullptr);
}

TEST_F(PointerUnion2TierTest, Null) {
  EXPECT_FALSE(pa0.isNull());
  EXPECT_FALSE(pb0.isNull());
  EXPECT_TRUE(null.isNull());
  EXPECT_TRUE(!null);
  EXPECT_TRUE(static_cast<bool>(pa0));

  EXPECT_TRUE(na0.isNull());
  EXPECT_TRUE(na1.isNull());
  EXPECT_TRUE(na2.isNull());
  EXPECT_TRUE(nb0.isNull());
  EXPECT_TRUE(nb1.isNull());
}

TEST_F(PointerUnion2TierTest, NullDiscrimination) {
  // Null pointers of different types have different opaque values.
  EXPECT_NE(na0, na1);
  EXPECT_NE(na0, na2);
  EXPECT_NE(na0, nb0);
  EXPECT_NE(na1, nb0);
  EXPECT_NE(nb0, nb1);

  // Default-constructed is null of first type.
  EXPECT_EQ(null, na0);
}

TEST_F(PointerUnion2TierTest, Comparison) {
  EXPECT_EQ(pa0, pa0);
  EXPECT_NE(pa0, pa1);
  EXPECT_NE(pa0, pb0);

  PU2Tier other(&a0);
  EXPECT_EQ(pa0, other);
}

TEST_F(PointerUnion2TierTest, Assignment) {
  PU2Tier u;
  EXPECT_TRUE(u.isNull());

  u = &a0;
  EXPECT_TRUE(isa<Align4<0> *>(u));
  EXPECT_EQ(cast<Align4<0> *>(u), &a0);

  u = &b0;
  EXPECT_TRUE(isa<Align8<0> *>(u));
  EXPECT_EQ(cast<Align8<0> *>(u), &b0);

  u = &a2;
  EXPECT_TRUE(isa<Align4<2> *>(u));

  u = nullptr;
  EXPECT_TRUE(u.isNull());
}

TEST_F(PointerUnion2TierTest, GetAddrOfPtr1) {
  EXPECT_TRUE(static_cast<void *>(pa0.getAddrOfPtr1()) ==
              static_cast<void *>(&pa0));
  EXPECT_TRUE(static_cast<void *>(null.getAddrOfPtr1()) ==
              static_cast<void *>(&null));
}

TEST_F(PointerUnion2TierTest, OpaqueValueRoundTrip) {
  void *opaque = pa0.getOpaqueValue();
  PU2Tier restored = PU2Tier::getFromOpaqueValue(opaque);
  EXPECT_EQ(pa0, restored);
  EXPECT_EQ(cast<Align4<0> *>(restored), &a0);

  opaque = pb0.getOpaqueValue();
  restored = PU2Tier::getFromOpaqueValue(opaque);
  EXPECT_EQ(pb0, restored);
  EXPECT_EQ(cast<Align8<0> *>(restored), &b0);

  opaque = pb1.getOpaqueValue();
  restored = PU2Tier::getFromOpaqueValue(opaque);
  EXPECT_EQ(pb1, restored);
  EXPECT_EQ(cast<Align8<1> *>(restored), &b1);
}

// 3-tier tests

struct PointerUnion3TierTest : public testing::Test {
  Align4<0> a0;
  Align4<1> a1;
  Align4<2> a2;
  Align8<0> b0;
  Align16<0> c0;
  Align16<1> c1;

  PU3Tier pa0, pa1, pa2, pb0, pc0, pc1, null;

  PointerUnion3TierTest()
      : pa0(&a0), pa1(&a1), pa2(&a2), pb0(&b0), pc0(&c0), pc1(&c1), null() {}
};

TEST_F(PointerUnion3TierTest, Isa) {
  EXPECT_TRUE(isa<Align4<0> *>(pa0));
  EXPECT_FALSE(isa<Align8<0> *>(pa0));
  EXPECT_FALSE(isa<Align16<0> *>(pa0));

  EXPECT_TRUE(isa<Align8<0> *>(pb0));
  EXPECT_FALSE(isa<Align4<0> *>(pb0));
  EXPECT_FALSE(isa<Align16<0> *>(pb0));

  EXPECT_TRUE(isa<Align16<0> *>(pc0));
  EXPECT_FALSE(isa<Align4<0> *>(pc0));
  EXPECT_FALSE(isa<Align8<0> *>(pc0));
  EXPECT_FALSE(isa<Align16<1> *>(pc0));

  EXPECT_TRUE(isa<Align16<1> *>(pc1));
  EXPECT_FALSE(isa<Align16<0> *>(pc1));
}

TEST_F(PointerUnion3TierTest, Cast) {
  EXPECT_EQ(cast<Align4<0> *>(pa0), &a0);
  EXPECT_EQ(cast<Align4<1> *>(pa1), &a1);
  EXPECT_EQ(cast<Align4<2> *>(pa2), &a2);
  EXPECT_EQ(cast<Align8<0> *>(pb0), &b0);
  EXPECT_EQ(cast<Align16<0> *>(pc0), &c0);
  EXPECT_EQ(cast<Align16<1> *>(pc1), &c1);
}

TEST_F(PointerUnion3TierTest, DynCast) {
  EXPECT_EQ(dyn_cast<Align4<0> *>(pa0), &a0);
  EXPECT_EQ(dyn_cast<Align8<0> *>(pa0), nullptr);
  EXPECT_EQ(dyn_cast<Align16<0> *>(pa0), nullptr);

  EXPECT_EQ(dyn_cast<Align8<0> *>(pb0), &b0);
  EXPECT_EQ(dyn_cast<Align4<0> *>(pb0), nullptr);
  EXPECT_EQ(dyn_cast<Align16<0> *>(pb0), nullptr);

  EXPECT_EQ(dyn_cast<Align16<0> *>(pc0), &c0);
  EXPECT_EQ(dyn_cast<Align16<1> *>(pc0), nullptr);
  EXPECT_EQ(dyn_cast<Align4<0> *>(pc0), nullptr);

  EXPECT_EQ(dyn_cast<Align16<1> *>(pc1), &c1);
  EXPECT_EQ(dyn_cast<Align16<0> *>(pc1), nullptr);
}

TEST_F(PointerUnion3TierTest, Null) {
  EXPECT_TRUE(null.isNull());
  EXPECT_FALSE(pa0.isNull());
  EXPECT_FALSE(pb0.isNull());
  EXPECT_FALSE(pc0.isNull());
  EXPECT_FALSE(pc1.isNull());

  PU3Tier na0(static_cast<Align4<0> *>(nullptr));
  PU3Tier nb0(static_cast<Align8<0> *>(nullptr));
  PU3Tier nc0(static_cast<Align16<0> *>(nullptr));
  PU3Tier nc1(static_cast<Align16<1> *>(nullptr));
  EXPECT_TRUE(na0.isNull());
  EXPECT_TRUE(nb0.isNull());
  EXPECT_TRUE(nc0.isNull());
  EXPECT_TRUE(nc1.isNull());

  // Null discrimination across all three tiers.
  EXPECT_NE(na0, nb0);
  EXPECT_NE(nb0, nc0);
  EXPECT_NE(nc0, nc1);
  EXPECT_NE(na0, nc0);
}

TEST_F(PointerUnion3TierTest, Assignment) {
  PU3Tier u;
  EXPECT_TRUE(u.isNull());

  u = &a0;
  EXPECT_TRUE(isa<Align4<0> *>(u));
  EXPECT_EQ(cast<Align4<0> *>(u), &a0);

  u = &b0;
  EXPECT_TRUE(isa<Align8<0> *>(u));
  EXPECT_EQ(cast<Align8<0> *>(u), &b0);

  u = &c1;
  EXPECT_TRUE(isa<Align16<1> *>(u));
  EXPECT_EQ(cast<Align16<1> *>(u), &c1);

  u = nullptr;
  EXPECT_TRUE(u.isNull());
}

TEST_F(PointerUnion3TierTest, OpaqueValueRoundTrip) {
  // pb0's tag (0b011) contains the tier-0 escape prefix (0b11) in its low 2
  // bits.
  void *opaque = pb0.getOpaqueValue();
  PU3Tier restored = PU3Tier::getFromOpaqueValue(opaque);
  EXPECT_EQ(pb0, restored);
  EXPECT_EQ(cast<Align8<0> *>(restored), &b0);

  opaque = pc0.getOpaqueValue();
  restored = PU3Tier::getFromOpaqueValue(opaque);
  EXPECT_EQ(pc0, restored);
  EXPECT_EQ(cast<Align16<0> *>(restored), &c0);

  opaque = pc1.getOpaqueValue();
  restored = PU3Tier::getFromOpaqueValue(opaque);
  EXPECT_EQ(pc1, restored);
  EXPECT_EQ(cast<Align16<1> *>(restored), &c1);
}

TEST_F(PointerUnion3TierTest, ConstCast) {
  const PU3Tier cpc0(&c0);
  EXPECT_TRUE(isa<Align16<0> *>(cpc0));
  EXPECT_FALSE(isa<Align4<0> *>(cpc0));
  EXPECT_EQ(cast<Align16<0> *>(cpc0), &c0);
  EXPECT_EQ(dyn_cast<Align8<0> *>(cpc0), nullptr);
}

TEST(PointerUnionMultiTierDenseMapTest, BasicOperations) {
  Align4<0> a0;
  Align8<0> b0;
  Align8<1> b1;

  DenseMap<PU2Tier, int> map;
  PU2Tier ka(&a0), kb(&b0), kb1(&b1);

  map[ka] = 1;
  map[kb] = 2;
  map[kb1] = 3;

  EXPECT_EQ(map[ka], 1);
  EXPECT_EQ(map[kb], 2);
  EXPECT_EQ(map[kb1], 3);

  EXPECT_EQ(map.count(ka), 1u);
  map.erase(ka);
  EXPECT_EQ(map.count(ka), 0u);
  EXPECT_EQ(map.count(kb), 1u);
}

TEST(PointerUnionMixedAlignFixedWidth, BasicOperations) {
  // Align4 provides 2 low bits, Align8 provides 3. Two types need 1 tag bit,
  // so all types have enough bits for fixed-width encoding with spare bits.
  using MixedPU = PointerUnion<Align4<0> *, Align8<0> *>;
  static_assert(PointerLikeTypeTraits<MixedPU>::NumLowBitsAvailable > 0,
                "Mixed-alignment 2-type union should have spare low bits");

  Align4<0> a;
  Align8<0> b;

  MixedPU u;
  EXPECT_TRUE(u.isNull());

  u = &a;
  EXPECT_TRUE(isa<Align4<0> *>(u));
  EXPECT_FALSE(isa<Align8<0> *>(u));
  EXPECT_EQ(cast<Align4<0> *>(u), &a);

  u = &b;
  EXPECT_TRUE(isa<Align8<0> *>(u));
  EXPECT_FALSE(isa<Align4<0> *>(u));
  EXPECT_EQ(cast<Align8<0> *>(u), &b);

  u = nullptr;
  EXPECT_TRUE(u.isNull());
}

TEST(PointerUnionLargeTierJump, BasicOperations) {
  // 3 x 2-bit + 2 x 4-bit: skips the 3-bit tier entirely (tier jump 2->4).
  using JumpPU = PointerUnion<Align4<0> *, Align4<1> *, Align4<2> *,
                              Align16<0> *, Align16<1> *>;
  static_assert(
      !pointer_union_detail::useFixedWidthTags<
          Align4<0> *, Align4<1> *, Align4<2> *, Align16<0> *, Align16<1> *>(),
      "Should use variable-width encoding");

  Align4<0> a0;
  Align4<1> a1;
  Align4<2> a2;
  Align16<0> c0;
  Align16<1> c1;

  JumpPU u;
  EXPECT_TRUE(u.isNull());

  u = &a0;
  EXPECT_TRUE(isa<Align4<0> *>(u));
  EXPECT_EQ(cast<Align4<0> *>(u), &a0);

  u = &a1;
  EXPECT_TRUE(isa<Align4<1> *>(u));
  EXPECT_EQ(cast<Align4<1> *>(u), &a1);

  u = &a2;
  EXPECT_TRUE(isa<Align4<2> *>(u));
  EXPECT_EQ(cast<Align4<2> *>(u), &a2);

  u = &c0;
  EXPECT_TRUE(isa<Align16<0> *>(u));
  EXPECT_FALSE(isa<Align4<0> *>(u));
  EXPECT_EQ(cast<Align16<0> *>(u), &c0);

  u = &c1;
  EXPECT_TRUE(isa<Align16<1> *>(u));
  EXPECT_FALSE(isa<Align16<0> *>(u));
  EXPECT_EQ(cast<Align16<1> *>(u), &c1);

  // Typed nulls preserve type identity and are null.
  JumpPU na0(static_cast<Align4<0> *>(nullptr));
  JumpPU nc0(static_cast<Align16<0> *>(nullptr));
  JumpPU nc1(static_cast<Align16<1> *>(nullptr));
  EXPECT_TRUE(na0.isNull());
  EXPECT_TRUE(nc0.isNull());
  EXPECT_TRUE(nc1.isNull());
  EXPECT_TRUE(isa<Align4<0> *>(na0));
  EXPECT_TRUE(isa<Align16<0> *>(nc0));
  EXPECT_TRUE(isa<Align16<1> *>(nc1));
  EXPECT_NE(na0, nc0);
  EXPECT_NE(nc0, nc1);
}

} // end anonymous namespace
