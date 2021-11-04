//===- SequenceTest.cpp - Unit tests for a sequence abstraciton -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Sequence.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <numeric>

using namespace llvm;

using testing::ElementsAre;
using testing::IsEmpty;

namespace {

using detail::canTypeFitValue;
using detail::CheckedInt;

using IntegralTypes = testing::Types<uint8_t,   // 0
                                     uint16_t,  // 1
                                     uint32_t,  // 2
                                     uint64_t,  // 3
                                     uintmax_t, // 4
                                     int8_t,    // 5
                                     int16_t,   // 6
                                     int32_t,   // 7
                                     int64_t,   // 8
                                     intmax_t   // 9
                                     >;

template <class T> class StrongIntTest : public testing::Test {};
TYPED_TEST_SUITE(StrongIntTest, IntegralTypes, );
TYPED_TEST(StrongIntTest, Operations) {
  using T = TypeParam;
  auto Max = std::numeric_limits<T>::max();
  auto Min = std::numeric_limits<T>::min();

  // We bail out for types that are not entirely representable within intmax_t.
  if (!canTypeFitValue<intmax_t>(Max) || !canTypeFitValue<intmax_t>(Min))
    return;

  // All representable values convert back and forth.
  EXPECT_EQ(CheckedInt::from(Min).template to<T>(), Min);
  EXPECT_EQ(CheckedInt::from(Max).template to<T>(), Max);

  // Addition -2, -1, 0, 1, 2.
  const T Expected = Max / 2;
  const CheckedInt Actual = CheckedInt::from(Expected);
  EXPECT_EQ((Actual + -2).template to<T>(), Expected - 2);
  EXPECT_EQ((Actual + -1).template to<T>(), Expected - 1);
  EXPECT_EQ((Actual + 0).template to<T>(), Expected);
  EXPECT_EQ((Actual + 1).template to<T>(), Expected + 1);
  EXPECT_EQ((Actual + 2).template to<T>(), Expected + 2);

  // EQ/NEQ
  EXPECT_EQ(Actual, Actual);
  EXPECT_NE(Actual, Actual + 1);

  // Difference
  EXPECT_EQ(Actual - Actual, 0);
  EXPECT_EQ((Actual + 1) - Actual, 1);
  EXPECT_EQ(Actual - (Actual + 2), -2);
}

#if defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG)
TEST(StrongIntDeathTest, OutOfBounds) {
  // Values above 'INTMAX_MAX' are not representable.
  EXPECT_DEATH(CheckedInt::from<uintmax_t>(INTMAX_MAX + 1ULL), "Out of bounds");
  EXPECT_DEATH(CheckedInt::from<uintmax_t>(UINTMAX_MAX), "Out of bounds");
  // Casting to narrower type asserts when out of bounds.
  EXPECT_DEATH(CheckedInt::from(-1).to<uint8_t>(), "Out of bounds");
  EXPECT_DEATH(CheckedInt::from(256).to<uint8_t>(), "Out of bounds");
  // Operations leading to intmax_t overflow assert.
  EXPECT_DEATH(CheckedInt::from(INTMAX_MAX) + 1, "Out of bounds");
  EXPECT_DEATH(CheckedInt::from(INTMAX_MIN) + -1, "Out of bounds");
  EXPECT_DEATH(CheckedInt::from(INTMAX_MIN) - CheckedInt::from(INTMAX_MAX),
               "Out of bounds");
}
#endif

TEST(SafeIntIteratorTest, Operations) {
  detail::SafeIntIterator<int, false> Forward(0);
  detail::SafeIntIterator<int, true> Reverse(0);

  const auto SetToZero = [&]() {
    Forward = detail::SafeIntIterator<int, false>(0);
    Reverse = detail::SafeIntIterator<int, true>(0);
  };

  // Equality / Comparisons
  SetToZero();
  EXPECT_EQ(Forward, Forward);
  EXPECT_LT(Forward - 1, Forward);
  EXPECT_LE(Forward, Forward);
  EXPECT_LE(Forward - 1, Forward);
  EXPECT_GT(Forward + 1, Forward);
  EXPECT_GE(Forward, Forward);
  EXPECT_GE(Forward + 1, Forward);

  EXPECT_EQ(Reverse, Reverse);
  EXPECT_LT(Reverse - 1, Reverse);
  EXPECT_LE(Reverse, Reverse);
  EXPECT_LE(Reverse - 1, Reverse);
  EXPECT_GT(Reverse + 1, Reverse);
  EXPECT_GE(Reverse, Reverse);
  EXPECT_GE(Reverse + 1, Reverse);

  // Dereference
  SetToZero();
  EXPECT_EQ(*Forward, 0);
  EXPECT_EQ(*Reverse, 0);

  // Indexing
  SetToZero();
  EXPECT_EQ(Forward[2], 2);
  EXPECT_EQ(Reverse[2], -2);

  // Pre-increment
  SetToZero();
  ++Forward;
  EXPECT_EQ(*Forward, 1);
  ++Reverse;
  EXPECT_EQ(*Reverse, -1);

  // Pre-decrement
  SetToZero();
  --Forward;
  EXPECT_EQ(*Forward, -1);
  --Reverse;
  EXPECT_EQ(*Reverse, 1);

  // Post-increment
  SetToZero();
  EXPECT_EQ(*(Forward++), 0);
  EXPECT_EQ(*Forward, 1);
  EXPECT_EQ(*(Reverse++), 0);
  EXPECT_EQ(*Reverse, -1);

  // Post-decrement
  SetToZero();
  EXPECT_EQ(*(Forward--), 0);
  EXPECT_EQ(*Forward, -1);
  EXPECT_EQ(*(Reverse--), 0);
  EXPECT_EQ(*Reverse, 1);

  // Compound assignment operators
  SetToZero();
  Forward += 1;
  EXPECT_EQ(*Forward, 1);
  Reverse += 1;
  EXPECT_EQ(*Reverse, -1);
  SetToZero();
  Forward -= 2;
  EXPECT_EQ(*Forward, -2);
  Reverse -= 2;
  EXPECT_EQ(*Reverse, 2);

  // Arithmetic
  SetToZero();
  EXPECT_EQ(*(Forward + 3), 3);
  EXPECT_EQ(*(Reverse + 3), -3);
  SetToZero();
  EXPECT_EQ(*(Forward - 4), -4);
  EXPECT_EQ(*(Reverse - 4), 4);

  // Difference
  SetToZero();
  EXPECT_EQ(Forward - Forward, 0);
  EXPECT_EQ(Reverse - Reverse, 0);
  EXPECT_EQ((Forward + 1) - Forward, 1);
  EXPECT_EQ(Forward - (Forward + 1), -1);
  EXPECT_EQ((Reverse + 1) - Reverse, 1);
  EXPECT_EQ(Reverse - (Reverse + 1), -1);
}

TEST(SequenceTest, Iteration) {
  EXPECT_THAT(seq(-4, 5), ElementsAre(-4, -3, -2, -1, 0, 1, 2, 3, 4));
  EXPECT_THAT(reverse(seq(-4, 5)), ElementsAre(4, 3, 2, 1, 0, -1, -2, -3, -4));

  EXPECT_THAT(seq_inclusive(-4, 5),
              ElementsAre(-4, -3, -2, -1, 0, 1, 2, 3, 4, 5));
  EXPECT_THAT(reverse(seq_inclusive(-4, 5)),
              ElementsAre(5, 4, 3, 2, 1, 0, -1, -2, -3, -4));
}

TEST(SequenceTest, Distance) {
  const auto Forward = seq(0, 10);
  EXPECT_EQ(std::distance(Forward.begin(), Forward.end()), 10);
  EXPECT_EQ(std::distance(Forward.rbegin(), Forward.rend()), 10);
}

TEST(SequenceTest, Dereference) {
  const auto Forward = seq(0, 10).begin();
  EXPECT_EQ(Forward[0], 0);
  EXPECT_EQ(Forward[2], 2);
  const auto Backward = seq(0, 10).rbegin();
  EXPECT_EQ(Backward[0], 9);
  EXPECT_EQ(Backward[2], 7);
}

enum UntypedEnum { A = 3 };
enum TypedEnum : uint32_t { B = 3 };

namespace X {
enum class ScopedEnum : uint16_t { C = 3 };
} // namespace X

struct S {
  enum NestedEnum { D = 4 };
  enum NestedEnum2 { E = 5 };

private:
  enum NestedEnum3 { F = 6 };
  friend struct llvm::enum_iteration_traits<NestedEnum3>;

public:
  static auto getNestedEnum3() { return NestedEnum3::F; }
};

} // namespace

namespace llvm {

template <> struct enum_iteration_traits<UntypedEnum> {
  static constexpr bool is_iterable = true;
};

template <> struct enum_iteration_traits<TypedEnum> {
  static constexpr bool is_iterable = true;
};

template <> struct enum_iteration_traits<X::ScopedEnum> {
  static constexpr bool is_iterable = true;
};

template <> struct enum_iteration_traits<S::NestedEnum> {
  static constexpr bool is_iterable = true;
};

template <> struct enum_iteration_traits<S::NestedEnum3> {
  static constexpr bool is_iterable = true;
};

} // namespace llvm

namespace {

TEST(StrongIntTest, Enums) {
  EXPECT_EQ(CheckedInt::from(A).to<UntypedEnum>(), A);
  EXPECT_EQ(CheckedInt::from(B).to<TypedEnum>(), B);
  EXPECT_EQ(CheckedInt::from(X::ScopedEnum::C).to<X::ScopedEnum>(),
            X::ScopedEnum::C);
}

TEST(SequenceTest, IterableEnums) {
  EXPECT_THAT(enum_seq(UntypedEnum::A, UntypedEnum::A), IsEmpty());
  EXPECT_THAT(enum_seq_inclusive(UntypedEnum::A, UntypedEnum::A),
              ElementsAre(UntypedEnum::A));

  EXPECT_THAT(enum_seq(TypedEnum::B, TypedEnum::B), IsEmpty());
  EXPECT_THAT(enum_seq_inclusive(TypedEnum::B, TypedEnum::B),
              ElementsAre(TypedEnum::B));

  EXPECT_THAT(enum_seq(X::ScopedEnum::C, X::ScopedEnum::C), IsEmpty());
  EXPECT_THAT(enum_seq_inclusive(X::ScopedEnum::C, X::ScopedEnum::C),
              ElementsAre(X::ScopedEnum::C));

  EXPECT_THAT(enum_seq_inclusive(S::NestedEnum::D, S::NestedEnum::D),
              ElementsAre(S::NestedEnum::D));
  EXPECT_THAT(enum_seq_inclusive(S::getNestedEnum3(), S::getNestedEnum3()),
              ElementsAre(S::getNestedEnum3()));
}

TEST(SequenceTest, NonIterableEnums) {
  EXPECT_THAT(enum_seq(S::NestedEnum2::E, S::NestedEnum2::E,
                       force_iteration_on_noniterable_enum),
              IsEmpty());
  EXPECT_THAT(enum_seq_inclusive(S::NestedEnum2::E, S::NestedEnum2::E,
                                 force_iteration_on_noniterable_enum),
              ElementsAre(S::NestedEnum2::E));

  // Check that this also works with enums marked as iterable.
  EXPECT_THAT(enum_seq(UntypedEnum::A, UntypedEnum::A,
                       force_iteration_on_noniterable_enum),
              IsEmpty());
  EXPECT_THAT(enum_seq_inclusive(UntypedEnum::A, UntypedEnum::A,
                                 force_iteration_on_noniterable_enum),
              ElementsAre(UntypedEnum::A));
}

} // namespace
