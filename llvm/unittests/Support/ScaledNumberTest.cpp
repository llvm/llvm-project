//===- llvm/unittest/Support/ScaledNumberTest.cpp - ScaledPair tests -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ScaledNumber.h"
#include "llvm/Support/DataTypes.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::ScaledNumbers;

namespace {

template <class UIntT> struct ScaledPair {
  UIntT D;
  int S;
  ScaledPair(const std::pair<UIntT, int16_t> &F) : D(F.first), S(F.second) {}
  ScaledPair(UIntT D, int S) : D(D), S(S) {}

  bool operator==(const ScaledPair<UIntT> &X) const {
    return D == X.D && S == X.S;
  }
};
template <class UIntT>
bool operator==(const std::pair<UIntT, int16_t> &L,
                const ScaledPair<UIntT> &R) {
  return ScaledPair<UIntT>(L) == R;
}
template <class UIntT>
void PrintTo(const ScaledPair<UIntT> &F, ::std::ostream *os) {
  *os << F.D << "*2^" << F.S;
}

typedef ScaledPair<uint32_t> SP32;
typedef ScaledPair<uint64_t> SP64;

TEST(ScaledNumberHelpersTest, getRounded) {
  EXPECT_EQ(getRounded32(0, 0, false), SP32(0, 0));
  EXPECT_EQ(getRounded32(0, 0, true), SP32(1, 0));
  EXPECT_EQ(getRounded32(20, 21, true), SP32(21, 21));
  EXPECT_EQ(getRounded32(UINT32_MAX, 0, false), SP32(UINT32_MAX, 0));
  EXPECT_EQ(getRounded32(UINT32_MAX, 0, true), SP32(1 << 31, 1));

  EXPECT_EQ(getRounded64(0, 0, false), SP64(0, 0));
  EXPECT_EQ(getRounded64(0, 0, true), SP64(1, 0));
  EXPECT_EQ(getRounded64(20, 21, true), SP64(21, 21));
  EXPECT_EQ(getRounded64(UINT32_MAX, 0, false), SP64(UINT32_MAX, 0));
  EXPECT_EQ(getRounded64(UINT32_MAX, 0, true), SP64(UINT64_C(1) << 32, 0));
  EXPECT_EQ(getRounded64(UINT64_MAX, 0, false), SP64(UINT64_MAX, 0));
  EXPECT_EQ(getRounded64(UINT64_MAX, 0, true), SP64(UINT64_C(1) << 63, 1));
}

TEST(ScaledNumberHelpersTest, getAdjusted) {
  const uint64_t Max32In64 = UINT32_MAX;
  EXPECT_EQ(getAdjusted32(0), SP32(0, 0));
  EXPECT_EQ(getAdjusted32(0, 5), SP32(0, 5));
  EXPECT_EQ(getAdjusted32(UINT32_MAX), SP32(UINT32_MAX, 0));
  EXPECT_EQ(getAdjusted32(Max32In64 << 1), SP32(UINT32_MAX, 1));
  EXPECT_EQ(getAdjusted32(Max32In64 << 1, 1), SP32(UINT32_MAX, 2));
  EXPECT_EQ(getAdjusted32(Max32In64 << 31), SP32(UINT32_MAX, 31));
  EXPECT_EQ(getAdjusted32(Max32In64 << 32), SP32(UINT32_MAX, 32));
  EXPECT_EQ(getAdjusted32(Max32In64 + 1), SP32(1u << 31, 1));
  EXPECT_EQ(getAdjusted32(UINT64_MAX), SP32(1u << 31, 33));

  EXPECT_EQ(getAdjusted64(0), SP64(0, 0));
  EXPECT_EQ(getAdjusted64(0, 5), SP64(0, 5));
  EXPECT_EQ(getAdjusted64(UINT32_MAX), SP64(UINT32_MAX, 0));
  EXPECT_EQ(getAdjusted64(Max32In64 << 1), SP64(Max32In64 << 1, 0));
  EXPECT_EQ(getAdjusted64(Max32In64 << 1, 1), SP64(Max32In64 << 1, 1));
  EXPECT_EQ(getAdjusted64(Max32In64 << 31), SP64(Max32In64 << 31, 0));
  EXPECT_EQ(getAdjusted64(Max32In64 << 32), SP64(Max32In64 << 32, 0));
  EXPECT_EQ(getAdjusted64(Max32In64 + 1), SP64(Max32In64 + 1, 0));
  EXPECT_EQ(getAdjusted64(UINT64_MAX), SP64(UINT64_MAX, 0));
}

TEST(ScaledNumberHelpersTest, getProduct) {
  // Zero.
  EXPECT_EQ(SP32(0, 0), getProduct32(0, 0));
  EXPECT_EQ(SP32(0, 0), getProduct32(0, 1));
  EXPECT_EQ(SP32(0, 0), getProduct32(0, 33));

  // Basic.
  EXPECT_EQ(SP32(6, 0), getProduct32(2, 3));
  EXPECT_EQ(SP32(UINT16_MAX / 3 * UINT16_MAX / 5 * 2, 0),
            getProduct32(UINT16_MAX / 3, UINT16_MAX / 5 * 2));

  // Overflow, no loss of precision.
  // ==> 0xf00010 * 0x1001
  // ==> 0xf00f00000 + 0x10010
  // ==> 0xf00f10010
  // ==> 0xf00f1001 * 2^4
  EXPECT_EQ(SP32(0xf00f1001, 4), getProduct32(0xf00010, 0x1001));

  // Overflow, loss of precision, rounds down.
  // ==> 0xf000070 * 0x1001
  // ==> 0xf00f000000 + 0x70070
  // ==> 0xf00f070070
  // ==> 0xf00f0700 * 2^8
  EXPECT_EQ(SP32(0xf00f0700, 8), getProduct32(0xf000070, 0x1001));

  // Overflow, loss of precision, rounds up.
  // ==> 0xf000080 * 0x1001
  // ==> 0xf00f000000 + 0x80080
  // ==> 0xf00f080080
  // ==> 0xf00f0801 * 2^8
  EXPECT_EQ(SP32(0xf00f0801, 8), getProduct32(0xf000080, 0x1001));

  // Reverse operand order.
  EXPECT_EQ(SP32(0, 0), getProduct32(1, 0));
  EXPECT_EQ(SP32(0, 0), getProduct32(33, 0));
  EXPECT_EQ(SP32(6, 0), getProduct32(3, 2));
  EXPECT_EQ(SP32(UINT16_MAX / 3 * UINT16_MAX / 5 * 2, 0),
            getProduct32(UINT16_MAX / 5 * 2, UINT16_MAX / 3));
  EXPECT_EQ(SP32(0xf00f1001, 4), getProduct32(0x1001, 0xf00010));
  EXPECT_EQ(SP32(0xf00f0700, 8), getProduct32(0x1001, 0xf000070));
  EXPECT_EQ(SP32(0xf00f0801, 8), getProduct32(0x1001, 0xf000080));

  // Round to overflow.
  EXPECT_EQ(SP64(UINT64_C(1) << 63, 64),
            getProduct64(UINT64_C(10376293541461622786),
                         UINT64_C(16397105843297379211)));

  // Big number with rounding.
  EXPECT_EQ(SP64(UINT64_C(9223372036854775810), 64),
            getProduct64(UINT64_C(18446744073709551556),
                         UINT64_C(9223372036854775840)));
}

TEST(ScaledNumberHelpersTest, getQuotient) {
  // Zero.
  EXPECT_EQ(SP32(0, 0), getQuotient32(0, 0));
  EXPECT_EQ(SP32(0, 0), getQuotient32(0, 1));
  EXPECT_EQ(SP32(0, 0), getQuotient32(0, 73));
  EXPECT_EQ(SP32(UINT32_MAX, MaxScale), getQuotient32(1, 0));
  EXPECT_EQ(SP32(UINT32_MAX, MaxScale), getQuotient32(6, 0));

  // Powers of two.
  EXPECT_EQ(SP32(1u << 31, -31), getQuotient32(1, 1));
  EXPECT_EQ(SP32(1u << 31, -30), getQuotient32(2, 1));
  EXPECT_EQ(SP32(1u << 31, -33), getQuotient32(4, 16));
  EXPECT_EQ(SP32(7u << 29, -29), getQuotient32(7, 1));
  EXPECT_EQ(SP32(7u << 29, -30), getQuotient32(7, 2));
  EXPECT_EQ(SP32(7u << 29, -33), getQuotient32(7, 16));

  // Divide evenly.
  EXPECT_EQ(SP32(3u << 30, -30), getQuotient32(9, 3));
  EXPECT_EQ(SP32(9u << 28, -28), getQuotient32(63, 7));

  // Divide unevenly.
  EXPECT_EQ(SP32(0xaaaaaaab, -33), getQuotient32(1, 3));
  EXPECT_EQ(SP32(0xd5555555, -31), getQuotient32(5, 3));

  // 64-bit division is hard to test, since divide64 doesn't canonicalize its
  // output.  However, this is the algorithm the implementation uses:
  //
  // - Shift divisor right.
  // - If we have 1 (power of 2), return early -- not canonicalized.
  // - Shift dividend left.
  // - 64-bit integer divide.
  // - If there's a remainder, continue with long division.
  //
  // TODO: require less knowledge about the implementation in the test.

  // Zero.
  EXPECT_EQ(SP64(0, 0), getQuotient64(0, 0));
  EXPECT_EQ(SP64(0, 0), getQuotient64(0, 1));
  EXPECT_EQ(SP64(0, 0), getQuotient64(0, 73));
  EXPECT_EQ(SP64(UINT64_MAX, MaxScale), getQuotient64(1, 0));
  EXPECT_EQ(SP64(UINT64_MAX, MaxScale), getQuotient64(6, 0));

  // Powers of two.
  EXPECT_EQ(SP64(1, 0), getQuotient64(1, 1));
  EXPECT_EQ(SP64(2, 0), getQuotient64(2, 1));
  EXPECT_EQ(SP64(4, -4), getQuotient64(4, 16));
  EXPECT_EQ(SP64(7, 0), getQuotient64(7, 1));
  EXPECT_EQ(SP64(7, -1), getQuotient64(7, 2));
  EXPECT_EQ(SP64(7, -4), getQuotient64(7, 16));

  // Divide evenly.
  EXPECT_EQ(SP64(UINT64_C(3) << 60, -60), getQuotient64(9, 3));
  EXPECT_EQ(SP64(UINT64_C(9) << 58, -58), getQuotient64(63, 7));

  // Divide unevenly.
  EXPECT_EQ(SP64(0xaaaaaaaaaaaaaaab, -65), getQuotient64(1, 3));
  EXPECT_EQ(SP64(0xd555555555555555, -63), getQuotient64(5, 3));
}

TEST(ScaledNumberHelpersTest, getLg) {
  EXPECT_EQ(0, getLg(UINT32_C(1), 0));
  EXPECT_EQ(1, getLg(UINT32_C(1), 1));
  EXPECT_EQ(1, getLg(UINT32_C(2), 0));
  EXPECT_EQ(3, getLg(UINT32_C(1), 3));
  EXPECT_EQ(3, getLg(UINT32_C(7), 0));
  EXPECT_EQ(3, getLg(UINT32_C(8), 0));
  EXPECT_EQ(3, getLg(UINT32_C(9), 0));
  EXPECT_EQ(3, getLg(UINT32_C(64), -3));
  EXPECT_EQ(31, getLg((UINT32_MAX >> 1) + 2, 0));
  EXPECT_EQ(32, getLg(UINT32_MAX, 0));
  EXPECT_EQ(-1, getLg(UINT32_C(1), -1));
  EXPECT_EQ(-1, getLg(UINT32_C(2), -2));
  EXPECT_EQ(INT32_MIN, getLg(UINT32_C(0), -1));
  EXPECT_EQ(INT32_MIN, getLg(UINT32_C(0), 0));
  EXPECT_EQ(INT32_MIN, getLg(UINT32_C(0), 1));

  EXPECT_EQ(0, getLg(UINT64_C(1), 0));
  EXPECT_EQ(1, getLg(UINT64_C(1), 1));
  EXPECT_EQ(1, getLg(UINT64_C(2), 0));
  EXPECT_EQ(3, getLg(UINT64_C(1), 3));
  EXPECT_EQ(3, getLg(UINT64_C(7), 0));
  EXPECT_EQ(3, getLg(UINT64_C(8), 0));
  EXPECT_EQ(3, getLg(UINT64_C(9), 0));
  EXPECT_EQ(3, getLg(UINT64_C(64), -3));
  EXPECT_EQ(63, getLg((UINT64_MAX >> 1) + 2, 0));
  EXPECT_EQ(64, getLg(UINT64_MAX, 0));
  EXPECT_EQ(-1, getLg(UINT64_C(1), -1));
  EXPECT_EQ(-1, getLg(UINT64_C(2), -2));
  EXPECT_EQ(INT32_MIN, getLg(UINT64_C(0), -1));
  EXPECT_EQ(INT32_MIN, getLg(UINT64_C(0), 0));
  EXPECT_EQ(INT32_MIN, getLg(UINT64_C(0), 1));
}

TEST(ScaledNumberHelpersTest, getLgFloor) {
  EXPECT_EQ(0, getLgFloor(UINT32_C(1), 0));
  EXPECT_EQ(1, getLgFloor(UINT32_C(1), 1));
  EXPECT_EQ(1, getLgFloor(UINT32_C(2), 0));
  EXPECT_EQ(2, getLgFloor(UINT32_C(7), 0));
  EXPECT_EQ(3, getLgFloor(UINT32_C(1), 3));
  EXPECT_EQ(3, getLgFloor(UINT32_C(8), 0));
  EXPECT_EQ(3, getLgFloor(UINT32_C(9), 0));
  EXPECT_EQ(3, getLgFloor(UINT32_C(64), -3));
  EXPECT_EQ(31, getLgFloor((UINT32_MAX >> 1) + 2, 0));
  EXPECT_EQ(31, getLgFloor(UINT32_MAX, 0));
  EXPECT_EQ(INT32_MIN, getLgFloor(UINT32_C(0), -1));
  EXPECT_EQ(INT32_MIN, getLgFloor(UINT32_C(0), 0));
  EXPECT_EQ(INT32_MIN, getLgFloor(UINT32_C(0), 1));

  EXPECT_EQ(0, getLgFloor(UINT64_C(1), 0));
  EXPECT_EQ(1, getLgFloor(UINT64_C(1), 1));
  EXPECT_EQ(1, getLgFloor(UINT64_C(2), 0));
  EXPECT_EQ(2, getLgFloor(UINT64_C(7), 0));
  EXPECT_EQ(3, getLgFloor(UINT64_C(1), 3));
  EXPECT_EQ(3, getLgFloor(UINT64_C(8), 0));
  EXPECT_EQ(3, getLgFloor(UINT64_C(9), 0));
  EXPECT_EQ(3, getLgFloor(UINT64_C(64), -3));
  EXPECT_EQ(63, getLgFloor((UINT64_MAX >> 1) + 2, 0));
  EXPECT_EQ(63, getLgFloor(UINT64_MAX, 0));
  EXPECT_EQ(INT32_MIN, getLgFloor(UINT64_C(0), -1));
  EXPECT_EQ(INT32_MIN, getLgFloor(UINT64_C(0), 0));
  EXPECT_EQ(INT32_MIN, getLgFloor(UINT64_C(0), 1));
}

TEST(ScaledNumberHelpersTest, getLgCeiling) {
  EXPECT_EQ(0, getLgCeiling(UINT32_C(1), 0));
  EXPECT_EQ(1, getLgCeiling(UINT32_C(1), 1));
  EXPECT_EQ(1, getLgCeiling(UINT32_C(2), 0));
  EXPECT_EQ(3, getLgCeiling(UINT32_C(1), 3));
  EXPECT_EQ(3, getLgCeiling(UINT32_C(7), 0));
  EXPECT_EQ(3, getLgCeiling(UINT32_C(8), 0));
  EXPECT_EQ(3, getLgCeiling(UINT32_C(64), -3));
  EXPECT_EQ(4, getLgCeiling(UINT32_C(9), 0));
  EXPECT_EQ(32, getLgCeiling(UINT32_MAX, 0));
  EXPECT_EQ(32, getLgCeiling((UINT32_MAX >> 1) + 2, 0));
  EXPECT_EQ(INT32_MIN, getLgCeiling(UINT32_C(0), -1));
  EXPECT_EQ(INT32_MIN, getLgCeiling(UINT32_C(0), 0));
  EXPECT_EQ(INT32_MIN, getLgCeiling(UINT32_C(0), 1));

  EXPECT_EQ(0, getLgCeiling(UINT64_C(1), 0));
  EXPECT_EQ(1, getLgCeiling(UINT64_C(1), 1));
  EXPECT_EQ(1, getLgCeiling(UINT64_C(2), 0));
  EXPECT_EQ(3, getLgCeiling(UINT64_C(1), 3));
  EXPECT_EQ(3, getLgCeiling(UINT64_C(7), 0));
  EXPECT_EQ(3, getLgCeiling(UINT64_C(8), 0));
  EXPECT_EQ(3, getLgCeiling(UINT64_C(64), -3));
  EXPECT_EQ(4, getLgCeiling(UINT64_C(9), 0));
  EXPECT_EQ(64, getLgCeiling((UINT64_MAX >> 1) + 2, 0));
  EXPECT_EQ(64, getLgCeiling(UINT64_MAX, 0));
  EXPECT_EQ(INT32_MIN, getLgCeiling(UINT64_C(0), -1));
  EXPECT_EQ(INT32_MIN, getLgCeiling(UINT64_C(0), 0));
  EXPECT_EQ(INT32_MIN, getLgCeiling(UINT64_C(0), 1));
}

TEST(ScaledNumberHelpersTest, compare) {
  EXPECT_EQ(0, compare(UINT32_C(0), 0, UINT32_C(0), 1));
  EXPECT_EQ(0, compare(UINT32_C(0), 0, UINT32_C(0), -10));
  EXPECT_EQ(0, compare(UINT32_C(0), 0, UINT32_C(0), 20));
  EXPECT_EQ(0, compare(UINT32_C(8), 0, UINT32_C(64), -3));
  EXPECT_EQ(0, compare(UINT32_C(8), 0, UINT32_C(32), -2));
  EXPECT_EQ(0, compare(UINT32_C(8), 0, UINT32_C(16), -1));
  EXPECT_EQ(0, compare(UINT32_C(8), 0, UINT32_C(8), 0));
  EXPECT_EQ(0, compare(UINT32_C(8), 0, UINT32_C(4), 1));
  EXPECT_EQ(0, compare(UINT32_C(8), 0, UINT32_C(2), 2));
  EXPECT_EQ(0, compare(UINT32_C(8), 0, UINT32_C(1), 3));
  EXPECT_EQ(-1, compare(UINT32_C(0), 0, UINT32_C(1), 3));
  EXPECT_EQ(-1, compare(UINT32_C(7), 0, UINT32_C(1), 3));
  EXPECT_EQ(-1, compare(UINT32_C(7), 0, UINT32_C(64), -3));
  EXPECT_EQ(1, compare(UINT32_C(9), 0, UINT32_C(1), 3));
  EXPECT_EQ(1, compare(UINT32_C(9), 0, UINT32_C(64), -3));
  EXPECT_EQ(1, compare(UINT32_C(9), 0, UINT32_C(0), 0));

  EXPECT_EQ(0, compare(UINT64_C(0), 0, UINT64_C(0), 1));
  EXPECT_EQ(0, compare(UINT64_C(0), 0, UINT64_C(0), -10));
  EXPECT_EQ(0, compare(UINT64_C(0), 0, UINT64_C(0), 20));
  EXPECT_EQ(0, compare(UINT64_C(8), 0, UINT64_C(64), -3));
  EXPECT_EQ(0, compare(UINT64_C(8), 0, UINT64_C(32), -2));
  EXPECT_EQ(0, compare(UINT64_C(8), 0, UINT64_C(16), -1));
  EXPECT_EQ(0, compare(UINT64_C(8), 0, UINT64_C(8), 0));
  EXPECT_EQ(0, compare(UINT64_C(8), 0, UINT64_C(4), 1));
  EXPECT_EQ(0, compare(UINT64_C(8), 0, UINT64_C(2), 2));
  EXPECT_EQ(0, compare(UINT64_C(8), 0, UINT64_C(1), 3));
  EXPECT_EQ(-1, compare(UINT64_C(0), 0, UINT64_C(1), 3));
  EXPECT_EQ(-1, compare(UINT64_C(7), 0, UINT64_C(1), 3));
  EXPECT_EQ(-1, compare(UINT64_C(7), 0, UINT64_C(64), -3));
  EXPECT_EQ(1, compare(UINT64_C(9), 0, UINT64_C(1), 3));
  EXPECT_EQ(1, compare(UINT64_C(9), 0, UINT64_C(64), -3));
  EXPECT_EQ(1, compare(UINT64_C(9), 0, UINT64_C(0), 0));
  EXPECT_EQ(-1, compare(UINT64_MAX, 0, UINT64_C(1), 64));
}

TEST(ScaledNumberHelpersTest, matchScales) {
#define MATCH_SCALES(T, LDIn, LSIn, RDIn, RSIn, LDOut, RDOut, SOut)            \
  do {                                                                         \
    T LDx = LDIn;                                                              \
    T RDx = RDIn;                                                              \
    T LDy = LDOut;                                                             \
    T RDy = RDOut;                                                             \
    int16_t LSx = LSIn;                                                        \
    int16_t RSx = RSIn;                                                        \
    int16_t Sy = SOut;                                                         \
                                                                               \
    EXPECT_EQ(SOut, matchScales(LDx, LSx, RDx, RSx));                          \
    EXPECT_EQ(LDy, LDx);                                                       \
    EXPECT_EQ(RDy, RDx);                                                       \
    if (LDy) {                                                                 \
      EXPECT_EQ(Sy, LSx);                                                      \
    }                                                                          \
    if (RDy) {                                                                 \
      EXPECT_EQ(Sy, RSx);                                                      \
    }                                                                          \
  } while (false)

  MATCH_SCALES(uint32_t, 0, 0, 0, 0, 0, 0, 0);
  MATCH_SCALES(uint32_t, 0, 50, 7, 1, 0, 7, 1);
  MATCH_SCALES(uint32_t, UINT32_C(1) << 31, 1, 9, 0, UINT32_C(1) << 31, 4, 1);
  MATCH_SCALES(uint32_t, UINT32_C(1) << 31, 2, 9, 0, UINT32_C(1) << 31, 2, 2);
  MATCH_SCALES(uint32_t, UINT32_C(1) << 31, 3, 9, 0, UINT32_C(1) << 31, 1, 3);
  MATCH_SCALES(uint32_t, UINT32_C(1) << 31, 4, 9, 0, UINT32_C(1) << 31, 0, 4);
  MATCH_SCALES(uint32_t, UINT32_C(1) << 30, 4, 9, 0, UINT32_C(1) << 31, 1, 3);
  MATCH_SCALES(uint32_t, UINT32_C(1) << 29, 4, 9, 0, UINT32_C(1) << 31, 2, 2);
  MATCH_SCALES(uint32_t, UINT32_C(1) << 28, 4, 9, 0, UINT32_C(1) << 31, 4, 1);
  MATCH_SCALES(uint32_t, UINT32_C(1) << 27, 4, 9, 0, UINT32_C(1) << 31, 9, 0);
  MATCH_SCALES(uint32_t, 7, 1, 0, 50, 7, 0, 1);
  MATCH_SCALES(uint32_t, 9, 0, UINT32_C(1) << 31, 1, 4, UINT32_C(1) << 31, 1);
  MATCH_SCALES(uint32_t, 9, 0, UINT32_C(1) << 31, 2, 2, UINT32_C(1) << 31, 2);
  MATCH_SCALES(uint32_t, 9, 0, UINT32_C(1) << 31, 3, 1, UINT32_C(1) << 31, 3);
  MATCH_SCALES(uint32_t, 9, 0, UINT32_C(1) << 31, 4, 0, UINT32_C(1) << 31, 4);
  MATCH_SCALES(uint32_t, 9, 0, UINT32_C(1) << 30, 4, 1, UINT32_C(1) << 31, 3);
  MATCH_SCALES(uint32_t, 9, 0, UINT32_C(1) << 29, 4, 2, UINT32_C(1) << 31, 2);
  MATCH_SCALES(uint32_t, 9, 0, UINT32_C(1) << 28, 4, 4, UINT32_C(1) << 31, 1);
  MATCH_SCALES(uint32_t, 9, 0, UINT32_C(1) << 27, 4, 9, UINT32_C(1) << 31, 0);

  MATCH_SCALES(uint64_t, 0, 0, 0, 0, 0, 0, 0);
  MATCH_SCALES(uint64_t, 0, 100, 7, 1, 0, 7, 1);
  MATCH_SCALES(uint64_t, UINT64_C(1) << 63, 1, 9, 0, UINT64_C(1) << 63, 4, 1);
  MATCH_SCALES(uint64_t, UINT64_C(1) << 63, 2, 9, 0, UINT64_C(1) << 63, 2, 2);
  MATCH_SCALES(uint64_t, UINT64_C(1) << 63, 3, 9, 0, UINT64_C(1) << 63, 1, 3);
  MATCH_SCALES(uint64_t, UINT64_C(1) << 63, 4, 9, 0, UINT64_C(1) << 63, 0, 4);
  MATCH_SCALES(uint64_t, UINT64_C(1) << 62, 4, 9, 0, UINT64_C(1) << 63, 1, 3);
  MATCH_SCALES(uint64_t, UINT64_C(1) << 61, 4, 9, 0, UINT64_C(1) << 63, 2, 2);
  MATCH_SCALES(uint64_t, UINT64_C(1) << 60, 4, 9, 0, UINT64_C(1) << 63, 4, 1);
  MATCH_SCALES(uint64_t, UINT64_C(1) << 59, 4, 9, 0, UINT64_C(1) << 63, 9, 0);
  MATCH_SCALES(uint64_t, 7, 1, 0, 100, 7, 0, 1);
  MATCH_SCALES(uint64_t, 9, 0, UINT64_C(1) << 63, 1, 4, UINT64_C(1) << 63, 1);
  MATCH_SCALES(uint64_t, 9, 0, UINT64_C(1) << 63, 2, 2, UINT64_C(1) << 63, 2);
  MATCH_SCALES(uint64_t, 9, 0, UINT64_C(1) << 63, 3, 1, UINT64_C(1) << 63, 3);
  MATCH_SCALES(uint64_t, 9, 0, UINT64_C(1) << 63, 4, 0, UINT64_C(1) << 63, 4);
  MATCH_SCALES(uint64_t, 9, 0, UINT64_C(1) << 62, 4, 1, UINT64_C(1) << 63, 3);
  MATCH_SCALES(uint64_t, 9, 0, UINT64_C(1) << 61, 4, 2, UINT64_C(1) << 63, 2);
  MATCH_SCALES(uint64_t, 9, 0, UINT64_C(1) << 60, 4, 4, UINT64_C(1) << 63, 1);
  MATCH_SCALES(uint64_t, 9, 0, UINT64_C(1) << 59, 4, 9, UINT64_C(1) << 63, 0);
}

TEST(ScaledNumberHelpersTest, getSum) {
  // Zero.
  EXPECT_EQ(SP32(1, 0), getSum32(0, 0, 1, 0));
  EXPECT_EQ(SP32(8, -3), getSum32(0, 0, 8, -3));
  EXPECT_EQ(SP32(UINT32_MAX, 0), getSum32(0, 0, UINT32_MAX, 0));

  // Basic.
  EXPECT_EQ(SP32(2, 0), getSum32(1, 0, 1, 0));
  EXPECT_EQ(SP32(3, 0), getSum32(1, 0, 2, 0));
  EXPECT_EQ(SP32(67, 0), getSum32(7, 0, 60, 0));

  // Different scales.
  EXPECT_EQ(SP32(3, 0), getSum32(1, 0, 1, 1));
  EXPECT_EQ(SP32(4, 0), getSum32(2, 0, 1, 1));

  // Loss of precision.
  EXPECT_EQ(SP32(UINT32_C(1) << 31, 1), getSum32(1, 32, 1, 0));
  EXPECT_EQ(SP32(UINT32_C(1) << 31, -31), getSum32(1, -32, 1, 0));

  // Not quite loss of precision.
  EXPECT_EQ(SP32((UINT32_C(1) << 31) + 1, 1), getSum32(1, 32, 1, 1));
  EXPECT_EQ(SP32((UINT32_C(1) << 31) + 1, -32), getSum32(1, -32, 1, -1));

  // Overflow.
  EXPECT_EQ(SP32(UINT32_C(1) << 31, 1), getSum32(1, 0, UINT32_MAX, 0));

  // Reverse operand order.
  EXPECT_EQ(SP32(1, 0), getSum32(1, 0, 0, 0));
  EXPECT_EQ(SP32(8, -3), getSum32(8, -3, 0, 0));
  EXPECT_EQ(SP32(UINT32_MAX, 0), getSum32(UINT32_MAX, 0, 0, 0));
  EXPECT_EQ(SP32(3, 0), getSum32(2, 0, 1, 0));
  EXPECT_EQ(SP32(67, 0), getSum32(60, 0, 7, 0));
  EXPECT_EQ(SP32(3, 0), getSum32(1, 1, 1, 0));
  EXPECT_EQ(SP32(4, 0), getSum32(1, 1, 2, 0));
  EXPECT_EQ(SP32(UINT32_C(1) << 31, 1), getSum32(1, 0, 1, 32));
  EXPECT_EQ(SP32(UINT32_C(1) << 31, -31), getSum32(1, 0, 1, -32));
  EXPECT_EQ(SP32((UINT32_C(1) << 31) + 1, 1), getSum32(1, 1, 1, 32));
  EXPECT_EQ(SP32((UINT32_C(1) << 31) + 1, -32), getSum32(1, -1, 1, -32));
  EXPECT_EQ(SP32(UINT32_C(1) << 31, 1), getSum32(UINT32_MAX, 0, 1, 0));

  // Zero.
  EXPECT_EQ(SP64(1, 0), getSum64(0, 0, 1, 0));
  EXPECT_EQ(SP64(8, -3), getSum64(0, 0, 8, -3));
  EXPECT_EQ(SP64(UINT64_MAX, 0), getSum64(0, 0, UINT64_MAX, 0));

  // Basic.
  EXPECT_EQ(SP64(2, 0), getSum64(1, 0, 1, 0));
  EXPECT_EQ(SP64(3, 0), getSum64(1, 0, 2, 0));
  EXPECT_EQ(SP64(67, 0), getSum64(7, 0, 60, 0));

  // Different scales.
  EXPECT_EQ(SP64(3, 0), getSum64(1, 0, 1, 1));
  EXPECT_EQ(SP64(4, 0), getSum64(2, 0, 1, 1));

  // Loss of precision.
  EXPECT_EQ(SP64(UINT64_C(1) << 63, 1), getSum64(1, 64, 1, 0));
  EXPECT_EQ(SP64(UINT64_C(1) << 63, -63), getSum64(1, -64, 1, 0));

  // Not quite loss of precision.
  EXPECT_EQ(SP64((UINT64_C(1) << 63) + 1, 1), getSum64(1, 64, 1, 1));
  EXPECT_EQ(SP64((UINT64_C(1) << 63) + 1, -64), getSum64(1, -64, 1, -1));

  // Overflow.
  EXPECT_EQ(SP64(UINT64_C(1) << 63, 1), getSum64(1, 0, UINT64_MAX, 0));

  // Reverse operand order.
  EXPECT_EQ(SP64(1, 0), getSum64(1, 0, 0, 0));
  EXPECT_EQ(SP64(8, -3), getSum64(8, -3, 0, 0));
  EXPECT_EQ(SP64(UINT64_MAX, 0), getSum64(UINT64_MAX, 0, 0, 0));
  EXPECT_EQ(SP64(3, 0), getSum64(2, 0, 1, 0));
  EXPECT_EQ(SP64(67, 0), getSum64(60, 0, 7, 0));
  EXPECT_EQ(SP64(3, 0), getSum64(1, 1, 1, 0));
  EXPECT_EQ(SP64(4, 0), getSum64(1, 1, 2, 0));
  EXPECT_EQ(SP64(UINT64_C(1) << 63, 1), getSum64(1, 0, 1, 64));
  EXPECT_EQ(SP64(UINT64_C(1) << 63, -63), getSum64(1, 0, 1, -64));
  EXPECT_EQ(SP64((UINT64_C(1) << 63) + 1, 1), getSum64(1, 1, 1, 64));
  EXPECT_EQ(SP64((UINT64_C(1) << 63) + 1, -64), getSum64(1, -1, 1, -64));
  EXPECT_EQ(SP64(UINT64_C(1) << 63, 1), getSum64(UINT64_MAX, 0, 1, 0));
}

TEST(ScaledNumberHelpersTest, getDifference) {
  // Basic.
  EXPECT_EQ(SP32(0, 0), getDifference32(1, 0, 1, 0));
  EXPECT_EQ(SP32(1, 0), getDifference32(2, 0, 1, 0));
  EXPECT_EQ(SP32(53, 0), getDifference32(60, 0, 7, 0));

  // Equals "0", different scales.
  EXPECT_EQ(SP32(0, 0), getDifference32(2, 0, 1, 1));

  // Subtract "0".
  EXPECT_EQ(SP32(1, 0), getDifference32(1, 0, 0, 0));
  EXPECT_EQ(SP32(8, -3), getDifference32(8, -3, 0, 0));
  EXPECT_EQ(SP32(UINT32_MAX, 0), getDifference32(UINT32_MAX, 0, 0, 0));

  // Loss of precision.
  EXPECT_EQ(SP32((UINT32_C(1) << 31) + 1, 1),
            getDifference32((UINT32_C(1) << 31) + 1, 1, 1, 0));
  EXPECT_EQ(SP32((UINT32_C(1) << 31) + 1, -31),
            getDifference32((UINT32_C(1) << 31) + 1, -31, 1, -32));

  // Not quite loss of precision.
  EXPECT_EQ(SP32(UINT32_MAX, 0), getDifference32(1, 32, 1, 0));
  EXPECT_EQ(SP32(UINT32_MAX, -32), getDifference32(1, 0, 1, -32));

  // Saturate to "0".
  EXPECT_EQ(SP32(0, 0), getDifference32(0, 0, 1, 0));
  EXPECT_EQ(SP32(0, 0), getDifference32(0, 0, 8, -3));
  EXPECT_EQ(SP32(0, 0), getDifference32(0, 0, UINT32_MAX, 0));
  EXPECT_EQ(SP32(0, 0), getDifference32(7, 0, 60, 0));
  EXPECT_EQ(SP32(0, 0), getDifference32(1, 0, 1, 1));
  EXPECT_EQ(SP32(0, 0), getDifference32(1, -32, 1, 0));
  EXPECT_EQ(SP32(0, 0), getDifference32(1, -32, 1, -1));

  // Regression tests for cases that failed during bringup.
  EXPECT_EQ(SP32(UINT32_C(1) << 26, -31),
            getDifference32(1, 0, UINT32_C(31) << 27, -32));

  // Basic.
  EXPECT_EQ(SP64(0, 0), getDifference64(1, 0, 1, 0));
  EXPECT_EQ(SP64(1, 0), getDifference64(2, 0, 1, 0));
  EXPECT_EQ(SP64(53, 0), getDifference64(60, 0, 7, 0));

  // Equals "0", different scales.
  EXPECT_EQ(SP64(0, 0), getDifference64(2, 0, 1, 1));

  // Subtract "0".
  EXPECT_EQ(SP64(1, 0), getDifference64(1, 0, 0, 0));
  EXPECT_EQ(SP64(8, -3), getDifference64(8, -3, 0, 0));
  EXPECT_EQ(SP64(UINT64_MAX, 0), getDifference64(UINT64_MAX, 0, 0, 0));

  // Loss of precision.
  EXPECT_EQ(SP64((UINT64_C(1) << 63) + 1, 1),
            getDifference64((UINT64_C(1) << 63) + 1, 1, 1, 0));
  EXPECT_EQ(SP64((UINT64_C(1) << 63) + 1, -63),
            getDifference64((UINT64_C(1) << 63) + 1, -63, 1, -64));

  // Not quite loss of precision.
  EXPECT_EQ(SP64(UINT64_MAX, 0), getDifference64(1, 64, 1, 0));
  EXPECT_EQ(SP64(UINT64_MAX, -64), getDifference64(1, 0, 1, -64));

  // Saturate to "0".
  EXPECT_EQ(SP64(0, 0), getDifference64(0, 0, 1, 0));
  EXPECT_EQ(SP64(0, 0), getDifference64(0, 0, 8, -3));
  EXPECT_EQ(SP64(0, 0), getDifference64(0, 0, UINT64_MAX, 0));
  EXPECT_EQ(SP64(0, 0), getDifference64(7, 0, 60, 0));
  EXPECT_EQ(SP64(0, 0), getDifference64(1, 0, 1, 1));
  EXPECT_EQ(SP64(0, 0), getDifference64(1, -64, 1, 0));
  EXPECT_EQ(SP64(0, 0), getDifference64(1, -64, 1, -1));
}

TEST(ScaledNumberHelpersTest, arithmeticOperators) {
  EXPECT_EQ(ScaledNumber<uint32_t>(10, 0),
            ScaledNumber<uint32_t>(1, 3) + ScaledNumber<uint32_t>(1, 1));
  EXPECT_EQ(ScaledNumber<uint32_t>(6, 0),
            ScaledNumber<uint32_t>(1, 3) - ScaledNumber<uint32_t>(1, 1));
  EXPECT_EQ(ScaledNumber<uint32_t>(2, 3),
            ScaledNumber<uint32_t>(1, 3) * ScaledNumber<uint32_t>(1, 1));
  EXPECT_EQ(ScaledNumber<uint32_t>(1, 2),
            ScaledNumber<uint32_t>(1, 3) / ScaledNumber<uint32_t>(1, 1));
  EXPECT_EQ(ScaledNumber<uint32_t>(1, 2), ScaledNumber<uint32_t>(1, 3) >> 1);
  EXPECT_EQ(ScaledNumber<uint32_t>(1, 4), ScaledNumber<uint32_t>(1, 3) << 1);

  EXPECT_EQ(ScaledNumber<uint64_t>(10, 0),
            ScaledNumber<uint64_t>(1, 3) + ScaledNumber<uint64_t>(1, 1));
  EXPECT_EQ(ScaledNumber<uint64_t>(6, 0),
            ScaledNumber<uint64_t>(1, 3) - ScaledNumber<uint64_t>(1, 1));
  EXPECT_EQ(ScaledNumber<uint64_t>(2, 3),
            ScaledNumber<uint64_t>(1, 3) * ScaledNumber<uint64_t>(1, 1));
  EXPECT_EQ(ScaledNumber<uint64_t>(1, 2),
            ScaledNumber<uint64_t>(1, 3) / ScaledNumber<uint64_t>(1, 1));
  EXPECT_EQ(ScaledNumber<uint64_t>(1, 2), ScaledNumber<uint64_t>(1, 3) >> 1);
  EXPECT_EQ(ScaledNumber<uint64_t>(1, 4), ScaledNumber<uint64_t>(1, 3) << 1);
}

TEST(ScaledNumberHelpersTest, toIntBug) {
  ScaledNumber<uint32_t> n(1, 0);
  EXPECT_EQ(1u, (n * n).toInt<uint32_t>());
}

static_assert(is_trivially_copyable<ScaledNumber<uint32_t>>::value,
              "trivially copyable");

} // end namespace
