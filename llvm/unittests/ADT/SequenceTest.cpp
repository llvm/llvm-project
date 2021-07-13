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

#include <list>

using namespace llvm;

using testing::ElementsAre;

namespace {

TEST(SequenceTest, Forward) {
  int X = 0;
  for (int I : seq(0, 10)) {
    EXPECT_EQ(X, I);
    ++X;
  }
  EXPECT_EQ(10, X);
}

TEST(SequenceTest, Backward) {
  int X = 9;
  for (int I : reverse(seq(0, 10))) {
    EXPECT_EQ(X, I);
    --X;
  }
  EXPECT_EQ(-1, X);
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

enum class CharEnum : char { A = 1, B, C, D, E };

TEST(SequenceTest, ForwardIteration) {
  EXPECT_THAT(seq_inclusive(CharEnum::C, CharEnum::E),
              ElementsAre(CharEnum::C, CharEnum::D, CharEnum::E));
}

TEST(SequenceTest, BackwardIteration) {
  EXPECT_THAT(reverse(seq_inclusive(CharEnum::B, CharEnum::D)),
              ElementsAre(CharEnum::D, CharEnum::C, CharEnum::B));
}

using IntegralTypes =
    testing::Types<uint8_t, uint16_t, uint32_t, uint64_t, uintmax_t, //
                   int8_t, int16_t, int32_t, int64_t, intmax_t>;

template <class T> class SequenceTest : public testing::Test {
public:
  const T min = std::numeric_limits<T>::min();
  const T minp1 = min + 1;
  const T max = std::numeric_limits<T>::max();
  const T maxm1 = max - 1;

  void checkIteration() const {
    // Forward
    EXPECT_THAT(seq(min, min), ElementsAre());
    EXPECT_THAT(seq(min, minp1), ElementsAre(min));
    EXPECT_THAT(seq(maxm1, max), ElementsAre(maxm1));
    EXPECT_THAT(seq(max, max), ElementsAre());
    // Reverse
    if (!std::is_same<T, intmax_t>::value) {
      EXPECT_THAT(reverse(seq(min, min)), ElementsAre());
      EXPECT_THAT(reverse(seq(min, minp1)), ElementsAre(min));
    }
    EXPECT_THAT(reverse(seq(maxm1, max)), ElementsAre(maxm1));
    EXPECT_THAT(reverse(seq(max, max)), ElementsAre());
    // Inclusive
    EXPECT_THAT(seq_inclusive(min, min), ElementsAre(min));
    EXPECT_THAT(seq_inclusive(min, minp1), ElementsAre(min, minp1));
    EXPECT_THAT(seq_inclusive(maxm1, maxm1), ElementsAre(maxm1));
    // Inclusive Reverse
    if (!std::is_same<T, intmax_t>::value) {
      EXPECT_THAT(reverse(seq_inclusive(min, min)), ElementsAre(min));
      EXPECT_THAT(reverse(seq_inclusive(min, minp1)), ElementsAre(minp1, min));
    }
    EXPECT_THAT(reverse(seq_inclusive(maxm1, maxm1)), ElementsAre(maxm1));
  }

  void checkIterators() const {
    auto checkValidIterators = [](auto sequence) {
      EXPECT_LE(sequence.begin(), sequence.end());
    };
    checkValidIterators(seq(min, min));
    checkValidIterators(seq(max, max));
    checkValidIterators(seq_inclusive(min, min));
    checkValidIterators(seq_inclusive(maxm1, maxm1));
  }
};
TYPED_TEST_SUITE(SequenceTest, IntegralTypes);
TYPED_TEST(SequenceTest, Boundaries) {
  this->checkIteration();
  this->checkIterators();
}

#if defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG)
template <class T> class SequenceDeathTest : public SequenceTest<T> {
public:
  using SequenceTest<T>::min;
  using SequenceTest<T>::minp1;
  using SequenceTest<T>::max;
  using SequenceTest<T>::maxm1;

  void checkInvalidOrder() const {
    EXPECT_DEATH(seq(max, min), "Begin must be less or equal to End.");
    EXPECT_DEATH(seq(minp1, min), "Begin must be less or equal to End.");
    EXPECT_DEATH(seq_inclusive(maxm1, min),
                 "Begin must be less or equal to End.");
    EXPECT_DEATH(seq_inclusive(minp1, min),
                 "Begin must be less or equal to End.");
  }
  void checkInvalidValues() const {
    if (std::is_same<T, intmax_t>::value || std::is_same<T, uintmax_t>::value) {
      EXPECT_DEATH(seq_inclusive(min, max),
                   "Forbidden End value for seq_inclusive.");
      EXPECT_DEATH(seq_inclusive(minp1, max),
                   "Forbidden End value for seq_inclusive.");
    }
    if (std::is_same<T, intmax_t>::value) {
      EXPECT_DEATH(reverse(seq(min, min)),
                   "Forbidden Begin value for reverse iteration");
      EXPECT_DEATH(reverse(seq_inclusive(min, min)),
                   "Forbidden Begin value for reverse iteration");
      // Note it is fine to use `Begin == 0` when `iota_range::numeric_type ==
      // uintmax_t` as unsigned integer underflow is well-defined.
    }
  }
};
TYPED_TEST_SUITE(SequenceDeathTest, IntegralTypes);
TYPED_TEST(SequenceDeathTest, DeathTests) {
  this->checkInvalidOrder();
  this->checkInvalidValues();
}
#endif // defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG)

} // anonymous namespace
