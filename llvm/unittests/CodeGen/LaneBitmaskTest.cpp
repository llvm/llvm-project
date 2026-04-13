//===- llvm/unittest/CodeGen/LaneBitmaskTest.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/LaneBitmask.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Type aliases for clarity.
using LaneMask64 = detail::LaneBitmaskImpl<64>;
using LaneMask128 = detail::LaneBitmaskImpl<128>;

TEST(LaneBitmaskTest, ConstructorAndAssignment) {
  // Test 64-bit version.
  {
    LaneMask64 Default;
    EXPECT_TRUE(Default.none());
    EXPECT_FALSE(Default.any());

    LaneMask64 FromInt(0x123456789abcdef0);
    EXPECT_TRUE(FromInt.any());
    EXPECT_FALSE(FromInt.none());

    std::array<uint64_t, 1> Arr64 = {0x123456789abcdef0};
    LaneMask64 FromArray(Arr64);
    EXPECT_TRUE(FromArray.test(4));
    EXPECT_EQ(FromArray.count(), 32u);

    LaneMask64 None = LaneMask64::getNone();
    EXPECT_TRUE(None.none());

    LaneMask64 All = LaneMask64::getAll();
    EXPECT_TRUE(All.all());
    EXPECT_EQ(All.count(), LaneMask64::BitWidth);

    LaneMask64 Lane5 = LaneMask64::getLane(5);
    EXPECT_TRUE(Lane5.test(5));
    EXPECT_EQ(Lane5.getNumLanes(), 1u);

    LaneMask64 Bit0 = LaneMask64::getLane(0);
    EXPECT_TRUE(Bit0.test(0));
    EXPECT_FALSE(Bit0.test(1));
    EXPECT_EQ(Bit0.getHighestLane(), 0u);
    EXPECT_TRUE((Bit0 >> 1).none());
    EXPECT_TRUE(Bit0.rotateLeft(1).test(1));

    LaneMask64 BitMax = LaneMask64::getLane(LaneMask64::BitWidth - 1);
    EXPECT_TRUE(BitMax.test(LaneMask64::BitWidth - 1));
    EXPECT_FALSE(BitMax.test(LaneMask64::BitWidth - 2));
    EXPECT_EQ(BitMax.getHighestLane(), LaneMask64::BitWidth - 1);
    EXPECT_TRUE(BitMax.rotateLeft(1).test(0));
    EXPECT_TRUE((BitMax << 1).none());

    LaneMask64 Original(0xabcd);
    LaneMask64 Copied(Original);
    EXPECT_EQ(Copied, Original);

    LaneMask64 Assigned;
    Assigned = Original;
    EXPECT_EQ(Assigned, Original);
  }

  // Test 128-bit version.
  {
    LaneMask128 Default;
    EXPECT_TRUE(Default.none());
    EXPECT_FALSE(Default.any());

    LaneMask128 FromInt(0x123456789abcdef0);
    EXPECT_TRUE(FromInt.any());
    EXPECT_FALSE(FromInt.none());

    std::array<uint64_t, 2> Arr128 = {0xff, 0xff00};
    LaneMask128 FromArray(Arr128);
    EXPECT_TRUE(FromArray.test(0));
    EXPECT_TRUE(FromArray.test(7));
    EXPECT_TRUE(FromArray.test(64 + 8));
    EXPECT_EQ(FromArray.count(), 16u);

    LaneMask128 None = LaneMask128::getNone();
    EXPECT_TRUE(None.none());

    LaneMask128 All = LaneMask128::getAll();
    EXPECT_TRUE(All.all());
    EXPECT_EQ(All.count(), LaneMask128::BitWidth);

    LaneMask128 Lane5 = LaneMask128::getLane(5);
    EXPECT_TRUE(Lane5.test(5));
    EXPECT_EQ(Lane5.getNumLanes(), 1u);

    LaneMask128 Lane100 = LaneMask128::getLane(100);
    EXPECT_TRUE(Lane100.test(100));
    EXPECT_FALSE(Lane100.test(99));
    EXPECT_EQ(Lane100.getNumLanes(), 1u);

    LaneMask128 Bit127 = LaneMask128::getLane(127);
    EXPECT_TRUE(Bit127.test(127));
    EXPECT_FALSE(Bit127.test(126));
    EXPECT_EQ(Bit127.getHighestLane(), 127u);
    EXPECT_TRUE(Bit127.rotateLeft(1).test(0));
    EXPECT_TRUE((Bit127 << 1).none());

    LaneMask128 Bit63 = LaneMask128::getLane(63);
    EXPECT_TRUE(Bit63.test(63));
    EXPECT_FALSE(Bit63.test(62));
    EXPECT_FALSE(Bit63.test(64));
    EXPECT_EQ(Bit63.getHighestLane(), 63u);

    LaneMask128 Original = LaneMask128::getLane(100);
    LaneMask128 Copied(Original);
    EXPECT_EQ(Copied, Original);

    LaneMask128 Assigned;
    Assigned = Original;
    EXPECT_EQ(Assigned, Original);
  }

  // Constexpr tests for 64-bit.
  static_assert(LaneMask64::getNone().none(), "getNone() should be empty");
  static_assert(LaneMask64::getAll().all(), "getAll() should be full");
  static_assert(LaneMask64::getAll().count() == LaneMask64::BitWidth,
                "getAll() should have all bits set");
  static_assert(LaneMask64::getLane(5).test(5), "getLane() should set bit");
  static_assert(LaneMask64::getLane(0).count() == 1, "getLane() sets one bit");
  static_assert(LaneMask64::getLane(0).test(0), "Bit 0 is set");
  static_assert((LaneMask64::getLane(0) >> 1).none(), "Shift clears bit 0");
  static_assert(
      LaneMask64::getLane(LaneMask64::BitWidth - 1).rotateLeft(1).test(0),
      "Rotate highest bit wraps to 0");
  static_assert((LaneMask64::getLane(LaneMask64::BitWidth - 1) << 1).none(),
                "Shift clears highest bit");

  // Constexpr tests for 128-bit.
  static_assert(LaneMask128::getNone().none(), "getNone() should be empty");
  static_assert(LaneMask128::getAll().all(), "getAll() should be full");
  static_assert(LaneMask128::getAll().count() == LaneMask128::BitWidth,
                "getAll() should have all bits set");
  static_assert(LaneMask128::getLane(100).test(100),
                "getLane(100) should set bit 100");
  static_assert(LaneMask128::getLane(127).count() == 1,
                "getLane() sets one bit");
  static_assert(LaneMask128::getLane(127).test(127), "Bit 127 is set");
  static_assert(!LaneMask128::getLane(63).test(64), "Bit 64 not set");
  static_assert((LaneMask128::getLane(0) >> 1).none(), "Shift clears bit 0");
  static_assert(
      LaneMask128::getLane(LaneMask128::BitWidth - 1).rotateLeft(1).test(0),
      "Rotate highest bit wraps to 0");
  static_assert((LaneMask128::getLane(LaneMask128::BitWidth - 1) << 1).none(),
                "Shift clears highest bit");
  static_assert(
      []() constexpr {
        std::array<uint64_t, 1> Arr = {0xff};
        LaneMask64 M(Arr);
        return M.count() == 8;
      }(),
      "Constexpr array constructor");
  static_assert(
      []() constexpr {
        std::array<uint64_t, 2> Arr = {0xff, 0xff00};
        LaneMask128 M(Arr);
        return M.count() == 16;
      }(),
      "Constexpr array constructor 128-bit");
  static_assert(
      []() constexpr {
        LaneMask64 Original(0xff);
        LaneMask64 Copied(Original);
        return Copied == Original;
      }(),
      "Constexpr copy constructor");
  static_assert(
      []() constexpr {
        LaneMask64 Original(0xff);
        LaneMask64 Assigned;
        Assigned = Original;
        return Assigned == Original;
      }(),
      "Constexpr copy assignment");
  static_assert(
      []() constexpr {
        LaneMask128 Original = LaneMask128::getLane(100);
        LaneMask128 Copied(Original);
        return Copied == Original;
      }(),
      "Constexpr copy constructor 128-bit");
  static_assert(
      []() constexpr {
        LaneMask128 Original = LaneMask128::getLane(100);
        LaneMask128 Assigned;
        Assigned = Original;
        return Assigned == Original;
      }(),
      "Constexpr copy assignment 128-bit");
}

TEST(LaneBitmaskTest, ComparisonOperators) {
  // Test 64-bit version.
  {
    LaneMask64 A(0x1234);
    LaneMask64 B(0x1234);
    LaneMask64 C(0x5678);

    EXPECT_TRUE(A == B);
    EXPECT_FALSE(A == C);
    EXPECT_FALSE(A != B);
    EXPECT_TRUE(A != C);
    EXPECT_TRUE(A < C);
    EXPECT_FALSE(C < A);
    EXPECT_FALSE(A < B);
  }

  // Test 128-bit version.
  {
    LaneMask128 A(0x1234);
    LaneMask128 B(0x1234);
    LaneMask128 C(0x5678);

    EXPECT_TRUE(A == B);
    EXPECT_FALSE(A == C);
    EXPECT_FALSE(A != B);
    EXPECT_TRUE(A != C);
    EXPECT_TRUE(A < C);
    EXPECT_FALSE(C < A);
    EXPECT_FALSE(A < B);

    // Test comparison across word boundaries.
    LaneMask128 Low(0x1234);
    LaneMask128 High = LaneMask128::getLane(100);
    EXPECT_TRUE(Low < High);
    EXPECT_FALSE(High < Low);
  }

  // Constexpr comparison operators for 64-bit.
  static_assert(LaneMask64(0x1234) == LaneMask64(0x1234), "Equality");
  static_assert(LaneMask64(0x1234) != LaneMask64(0x5678), "Inequality");
  static_assert(LaneMask64(0x1000) < LaneMask64(0x2000), "Less than");

  // Constexpr comparison operators for 128-bit.
  static_assert(LaneMask128(0x1234) == LaneMask128(0x1234), "Equality");
  static_assert(LaneMask128(0x1234) != LaneMask128(0x5678), "Inequality");
  static_assert(LaneMask128::getLane(64) < LaneMask128::getLane(100),
                "Cross-word less than");
}

TEST(LaneBitmaskTest, BitwiseOperators) {
  // Test 64-bit version.
  {
    LaneMask64 A(0xff00);
    LaneMask64 B(0x0ff0);

    // Test OR.
    EXPECT_EQ(A | B, LaneMask64(0xfff0));

    // Test AND.
    EXPECT_EQ(A & B, LaneMask64(0x0f00));

    // Test XOR.
    EXPECT_EQ(A ^ B, LaneMask64(0xf0f0));
    EXPECT_EQ((LaneMask64(0xff) ^ LaneMask64(0xff)), LaneMask64::getNone());

    // Test NOT.
    LaneMask64 NotA = ~A;
    EXPECT_EQ(NotA.count(), LaneMask64::BitWidth - A.count());

    // Test OR assign.
    LaneMask64 A2(0xff00);
    A2 |= B;
    EXPECT_EQ(A2, LaneMask64(0xfff0));

    // Test AND assign.
    LaneMask64 A3(0xff00);
    A3 &= B;
    EXPECT_EQ(A3, LaneMask64(0x0f00));

    // Test XOR assign.
    LaneMask64 A4(0xff00);
    A4 ^= B;
    EXPECT_EQ(A4, LaneMask64(0xf0f0));
  }

  // Test 128-bit version.
  {
    LaneMask128 A(0xff00);
    LaneMask128 B(0x0ff0);

    // Test OR.
    EXPECT_EQ(A | B, LaneMask128(0xfff0));

    // Test AND.
    EXPECT_EQ(A & B, LaneMask128(0x0f00));

    // Test XOR.
    EXPECT_EQ(A ^ B, LaneMask128(0xf0f0));
    EXPECT_EQ((LaneMask128(0xff) ^ LaneMask128(0xff)), LaneMask128::getNone());

    // Test NOT.
    LaneMask128 NotA = ~A;
    EXPECT_EQ(NotA.count(), LaneMask128::BitWidth - A.count());

    // Test operations across word boundaries.
    LaneMask128 Low(0xff);
    LaneMask128 High = LaneMask128::getLane(100);
    LaneMask128 Combined = Low | High;
    EXPECT_TRUE(Combined.test(0));
    EXPECT_TRUE(Combined.test(100));
    EXPECT_EQ(Combined.count(), 9u);

    LaneMask128 AndResult = Combined & High;
    EXPECT_FALSE(AndResult.test(0));
    EXPECT_TRUE(AndResult.test(100));
    EXPECT_EQ(AndResult.count(), 1u);

    // Test XOR across word boundaries.
    EXPECT_TRUE((Combined ^ High).test(0));
    EXPECT_FALSE((Combined ^ High).test(100));
    EXPECT_EQ((Combined ^ High).count(), 8u);

    // Test XOR assign.
    LaneMask128 A4(0xff00);
    A4 ^= B;
    EXPECT_EQ(A4, LaneMask128(0xf0f0));
  }

  // Constexpr bitwise operations for 64-bit.
  static_assert((LaneMask64(0xff00) | LaneMask64(0x0ff0)) == LaneMask64(0xfff0),
                "Constexpr OR");
  static_assert((LaneMask64(0xff00) & LaneMask64(0x0ff0)) == LaneMask64(0x0f00),
                "Constexpr AND");
  static_assert((LaneMask64(0xff00) ^ LaneMask64(0x0ff0)) == LaneMask64(0xf0f0),
                "Constexpr XOR");
  static_assert((~LaneMask64(0xff)).any(), "Constexpr NOT");
  static_assert((~LaneMask64::getAll()).none(), "Constexpr NOT of all");
  static_assert(
      []() constexpr {
        LaneMask64 L(0xff00);
        L |= LaneMask64(0x0ff0);
        return L == LaneMask64(0xfff0);
      }(),
      "Constexpr OR assign");
  static_assert(
      []() constexpr {
        LaneMask64 L(0xff00);
        L &= LaneMask64(0x0ff0);
        return L == LaneMask64(0x0f00);
      }(),
      "Constexpr AND assign");
  static_assert(
      []() constexpr {
        LaneMask64 L(0xff00);
        L ^= LaneMask64(0x0ff0);
        return L == LaneMask64(0xf0f0);
      }(),
      "Constexpr XOR assign");

  // Constexpr bitwise operations for 128-bit.
  static_assert((LaneMask128(0xff00) | LaneMask128(0x0ff0)) ==
                    LaneMask128(0xfff0),
                "Constexpr OR");
  static_assert((LaneMask128(0xff00) ^ LaneMask128(0x0ff0)) ==
                    LaneMask128(0xf0f0),
                "Constexpr XOR");
  static_assert(
      (LaneMask128::getLane(64) | LaneMask128::getLane(100)).count() == 2,
      "Constexpr OR across words");
  static_assert((LaneMask128::getLane(64) ^ LaneMask128::getLane(64)).none(),
                "Constexpr XOR with self is empty");
  static_assert(
      []() constexpr {
        LaneMask128 L(0xff00);
        L |= LaneMask128::getLane(100);
        return L.count() == 9;
      }(),
      "Constexpr OR assign 128-bit");
  static_assert(
      []() constexpr {
        LaneMask128 L(0xff00);
        L &= LaneMask128(0x0ff0);
        return L == LaneMask128(0x0f00);
      }(),
      "Constexpr AND assign 128-bit");
  static_assert(
      []() constexpr {
        LaneMask128 L(0xff00);
        L ^= LaneMask128(0x0ff0);
        return L == LaneMask128(0xf0f0);
      }(),
      "Constexpr XOR assign 128-bit");
}

TEST(LaneBitmaskTest, QueryMethods) {
  // Test 64-bit version.
  {
    LaneMask64 Empty;
    EXPECT_TRUE(Empty.none());
    EXPECT_FALSE(Empty.any());
    EXPECT_FALSE(Empty.all());
    EXPECT_EQ(Empty.count(), 0u);
    EXPECT_EQ(Empty.getNumLanes(), 0u);

    LaneMask64 Partial(0x00ff);
    EXPECT_FALSE(Partial.none());
    EXPECT_TRUE(Partial.any());
    EXPECT_FALSE(Partial.all());
    EXPECT_EQ(Partial.count(), 8u);
    EXPECT_EQ(Partial.getNumLanes(), 8u);

    LaneMask64 Full = LaneMask64::getAll();
    EXPECT_FALSE(Full.none());
    EXPECT_TRUE(Full.any());
    EXPECT_TRUE(Full.all());
    EXPECT_EQ(Full.count(), LaneMask64::BitWidth);
    EXPECT_EQ(Full.getNumLanes(), LaneMask64::BitWidth);
    EXPECT_EQ(Full.size(), LaneMask64::BitWidth);
  }

  // Test 128-bit version.
  {
    LaneMask128 Empty;
    EXPECT_TRUE(Empty.none());
    EXPECT_FALSE(Empty.any());
    EXPECT_FALSE(Empty.all());
    EXPECT_EQ(Empty.count(), 0u);
    EXPECT_EQ(Empty.getNumLanes(), 0u);

    LaneMask128 Partial(0x00ff);
    EXPECT_FALSE(Partial.none());
    EXPECT_TRUE(Partial.any());
    EXPECT_FALSE(Partial.all());
    EXPECT_EQ(Partial.count(), 8u);
    EXPECT_EQ(Partial.getNumLanes(), 8u);

    LaneMask128 Full = LaneMask128::getAll();
    EXPECT_FALSE(Full.none());
    EXPECT_TRUE(Full.any());
    EXPECT_TRUE(Full.all());
    EXPECT_EQ(Full.count(), LaneMask128::BitWidth);
    EXPECT_EQ(Full.getNumLanes(), LaneMask128::BitWidth);

    LaneMask128 UpperBit = LaneMask128::getLane(127);
    EXPECT_FALSE(UpperBit.none());
    EXPECT_TRUE(UpperBit.any());
    EXPECT_FALSE(UpperBit.all());
  }

  // Constexpr query methods for 64-bit.
  static_assert(LaneMask64().none(), "Empty mask");
  static_assert(!LaneMask64(0x1).none(), "Non-empty mask");
  static_assert(LaneMask64(0x1).any(), "Any bit set");
  static_assert(!LaneMask64().any(), "No bits set");
  static_assert(LaneMask64::getAll().all(), "All bits set");
  static_assert(!LaneMask64(0xff).all(), "Not all bits set");
  static_assert(LaneMask64().count() == 0, "Count zero");
  static_assert(LaneMask64(0x7).count() == 3, "Count three");
  static_assert(LaneMask64(0x7).getNumLanes() == 3, "Num lanes three");
  static_assert(LaneMask64().size() == LaneMask64::BitWidth, "Size");
  static_assert(LaneMask64::getAll().getNumLanes() == LaneMask64::BitWidth,
                "Full mask num lanes");

  // Constexpr query methods for 128-bit.
  static_assert(LaneMask128().none(), "Empty mask");
  static_assert(LaneMask128::getLane(127).any(), "High bit set");
  static_assert(LaneMask128::getAll().all(), "All bits set");
  static_assert(LaneMask128().size() == LaneMask128::BitWidth, "Size");
  static_assert(LaneMask128::getAll().getNumLanes() == 128, "128 lanes");
}

TEST(LaneBitmaskTest, GetHighestLane) {
  // Test 64-bit version.
  EXPECT_EQ(LaneMask64(1).getHighestLane(), 0u);
  EXPECT_EQ(LaneMask64(1ull << 5).getHighestLane(), 5u);
  EXPECT_EQ(LaneMask64(1ull << 63).getHighestLane(), 63u);
  EXPECT_EQ(LaneMask64((1ull << 10) | (1ull << 30)).getHighestLane(), 30u);

  // Test 128-bit version.
  EXPECT_EQ(LaneMask128(1).getHighestLane(), 0u);
  EXPECT_EQ(LaneMask128(1ull << 5).getHighestLane(), 5u);
  EXPECT_EQ(LaneMask128::getLane(100).getHighestLane(), 100u);
  EXPECT_EQ(LaneMask128::getLane(127).getHighestLane(), 127u);
  EXPECT_EQ(
      (LaneMask128::getLane(10) | LaneMask128::getLane(100)).getHighestLane(),
      100u);
}

TEST(LaneBitmaskTest, ShiftAssignOperators) {
  // Test 64-bit version.
  {
    LaneMask64 A1(0xff);
    A1 <<= 8;
    EXPECT_EQ(A1, LaneMask64(0xff00));

    LaneMask64 A2(0xff00);
    A2 >>= 8;
    EXPECT_EQ(A2, LaneMask64(0xff));

    LaneMask64 A3(0x1);
    A3 <<= 4;
    A3 <<= 4;
    EXPECT_EQ(A3, LaneMask64(0x100));
  }

  // Test 128-bit version with cross-word shifts.
  {
    LaneMask128 A1(0xff);
    A1 <<= 8;
    EXPECT_EQ(A1, LaneMask128(0xff00));

    LaneMask128 A2(0xff);
    A2 <<= 64;
    EXPECT_FALSE(A2.test(0));
    EXPECT_TRUE(A2.test(64));

    LaneMask128 A3 = LaneMask128::getLane(100);
    A3 >>= 50;
    EXPECT_TRUE(A3.test(50));
    EXPECT_FALSE(A3.test(100));
  }

  static_assert(
      []() constexpr {
        LaneMask64 L(0xff);
        L <<= 8;
        return L == LaneMask64(0xff00);
      }(),
      "Constexpr shift left assign");

  static_assert(
      []() constexpr {
        LaneMask64 L(0xff00);
        L >>= 8;
        return L == LaneMask64(0xff);
      }(),
      "Constexpr shift right assign");

  static_assert(
      []() constexpr {
        LaneMask128 L(0xff);
        L <<= 64;
        return !L.test(0) && L.test(64);
      }(),
      "Constexpr shift left assign 128-bit");

  static_assert(
      []() constexpr {
        LaneMask128 L = LaneMask128::getLane(100);
        L >>= 50;
        return L.test(50) && !L.test(100);
      }(),
      "Constexpr shift right assign 128-bit");
}

TEST(LaneBitmaskTest, ShiftOperators) {
  // Test 64-bit version.
  {
    LaneMask64 A(0xff);

    EXPECT_EQ(A << 0, A);
    EXPECT_EQ(A << 8, LaneMask64(0xff00));
    EXPECT_TRUE((A << LaneMask64::BitWidth).none());
    EXPECT_TRUE((A << (LaneMask64::BitWidth + 10)).none());

    LaneMask64 B(0xff00);
    EXPECT_EQ(B >> 0, B);
    EXPECT_EQ(B >> 8, LaneMask64(0xff));
    EXPECT_TRUE((B >> LaneMask64::BitWidth).none());

    LaneMask64 Bit0 = LaneMask64::getLane(0);
    LaneMask64 ShiftedLeft = Bit0 << (LaneMask64::BitWidth - 1);
    EXPECT_FALSE(ShiftedLeft.none());
    EXPECT_TRUE(ShiftedLeft.test(LaneMask64::BitWidth - 1));
    EXPECT_TRUE((ShiftedLeft << 1).none());
  }

  // Test 128-bit version with cross-word shifts.
  {
    LaneMask128 A(0xff);

    // Shift left within word.
    EXPECT_EQ(A << 8, LaneMask128(0xff00));

    // Shift left across word boundary.
    LaneMask128 B = A << 60;
    EXPECT_TRUE(B.test(60));
    EXPECT_TRUE(B.test(67));

    // Shift left completely into upper word.
    LaneMask128 C = A << 64;
    EXPECT_FALSE(C.test(0));
    EXPECT_TRUE(C.test(64));
    EXPECT_TRUE(C.test(71));

    // Shift right across word boundary.
    LaneMask128 D = LaneMask128::getLane(100);
    LaneMask128 E = D >> 50;
    EXPECT_TRUE(E.test(50));
    EXPECT_FALSE(E.test(100));

    // Shift by full width.
    EXPECT_TRUE((A << LaneMask128::BitWidth).none());
    EXPECT_TRUE((A >> LaneMask128::BitWidth).none());
  }

  // Constexpr shift operations for 64-bit.
  static_assert((LaneMask64(0xff) << 8) == LaneMask64(0xff00),
                "Constexpr shift left");
  static_assert((LaneMask64(0xff00) >> 8) == LaneMask64(0xff),
                "Constexpr shift right");
  static_assert((LaneMask64(0xff) << LaneMask64::BitWidth).none(),
                "Shift left by BitWidth");
  static_assert((LaneMask64(0xff) << (LaneMask64::BitWidth + 10)).none(),
                "Shift left beyond BitWidth");
  static_assert(
      []() constexpr {
        LaneMask64 Bit0 = LaneMask64::getLane(0);
        LaneMask64 Shifted = Bit0 << (LaneMask64::BitWidth - 1);
        return !Shifted.none() && Shifted.test(LaneMask64::BitWidth - 1) &&
               (Shifted << 1).none();
      }(),
      "Shift by BitWidth-1 then by 1");

  // Constexpr shift operations for 128-bit.
  static_assert((LaneMask128(0xff) << 8) == LaneMask128(0xff00),
                "Constexpr shift left");
  static_assert((LaneMask128::getLane(64) >> 64).test(0),
                "Shift across word boundary");
}

TEST(LaneBitmaskTest, RotateOperators) {
  // Test 64-bit version.
  {
    LaneMask64 A(0xff);

    EXPECT_EQ(A.rotateLeft(0), A);
    EXPECT_EQ(A.rotateLeft(8), LaneMask64(0xff00));
    EXPECT_EQ(A.rotateLeft(LaneMask64::BitWidth), A);
    EXPECT_EQ(A.rotateLeft(LaneMask64::BitWidth * 2), A);
    EXPECT_EQ(A.rotateLeft(LaneMask64::BitWidth + 8), A.rotateLeft(8));

    LaneMask64 B(0xff00);
    EXPECT_EQ(B.rotateRight(0), B);
    EXPECT_EQ(B.rotateRight(8), LaneMask64(0xff));
    EXPECT_EQ(B.rotateRight(LaneMask64::BitWidth), B);
    EXPECT_EQ(B.rotateRight(LaneMask64::BitWidth * 3), B);
    EXPECT_EQ(B.rotateRight(LaneMask64::BitWidth + 8), B.rotateRight(8));

    LaneMask64 C(0x123456789abcdef0);
    EXPECT_EQ(C.rotateLeft(37).rotateRight(37), C);

    LaneMask64 HighBit = LaneMask64::getLane(LaneMask64::BitWidth - 1);
    EXPECT_TRUE((HighBit << 1).none());
    EXPECT_TRUE(HighBit.rotateLeft(1).test(0));
  }

  // Test 128-bit version with cross-word rotations.
  {
    LaneMask128 A(0xff);

    // Rotate left within and across words.
    EXPECT_EQ(A.rotateLeft(0), A);
    EXPECT_EQ(A.rotateLeft(8), LaneMask128(0xff00));
    EXPECT_EQ(A.rotateLeft(LaneMask128::BitWidth), A);

    // Rotate across word boundary.
    LaneMask128 B = A.rotateLeft(60);
    EXPECT_TRUE(B.test(60));
    EXPECT_TRUE(B.test(67));

    // Rotate highest bit wraps to bit 0.
    LaneMask128 HighBit = LaneMask128::getLane(127);
    EXPECT_TRUE(HighBit.rotateLeft(1).test(0));
    EXPECT_FALSE(HighBit.rotateLeft(1).test(127));

    // Rotate from upper word to lower word.
    LaneMask128 UpperBit = LaneMask128::getLane(100);
    LaneMask128 Rotated = UpperBit.rotateRight(50);
    EXPECT_TRUE(Rotated.test(50));
    EXPECT_FALSE(Rotated.test(100));

    // Verify rotate vs shift difference for 128-bit.
    EXPECT_TRUE((HighBit << 1).none());
    EXPECT_TRUE(HighBit.rotateLeft(1).test(0));
  }

  // Constexpr rotate operations for 64-bit.
  static_assert(LaneMask64(0xff).rotateLeft(8) == LaneMask64(0xff00),
                "Constexpr rotate left");
  static_assert(LaneMask64(0xff00).rotateRight(8) == LaneMask64(0xff),
                "Constexpr rotate right");
  static_assert(LaneMask64(0xff).rotateLeft(LaneMask64::BitWidth) ==
                    LaneMask64(0xff),
                "Rotate by BitWidth is identity");
  static_assert(LaneMask64(0xff).rotateLeft(LaneMask64::BitWidth * 2) ==
                    LaneMask64(0xff),
                "Rotate by multiple of BitWidth");
  static_assert(LaneMask64(0xff).rotateLeft(LaneMask64::BitWidth + 8) ==
                    LaneMask64(0xff).rotateLeft(8),
                "Rotate by BitWidth + N");
  static_assert(LaneMask64(0x1234).rotateLeft(37).rotateRight(37) ==
                    LaneMask64(0x1234),
                "Rotate roundtrip");
  static_assert(
      LaneMask64::getLane(LaneMask64::BitWidth - 1).rotateLeft(1).test(0),
      "Rotate wraps around");

  // Constexpr rotate operations for 128-bit.
  static_assert(LaneMask128(0xff).rotateLeft(8) == LaneMask128(0xff00),
                "Constexpr rotate left");
  static_assert(LaneMask128::getLane(127).rotateLeft(1).test(0),
                "Rotate highest bit wraps to 0");
  static_assert(LaneMask128(0xff).rotateLeft(LaneMask128::BitWidth) ==
                    LaneMask128(0xff),
                "Rotate by 128 is identity");
}

TEST(LaneBitmaskTest, InheritedBitsetOperations) {
  // Test 64-bit version.
  {
    LaneMask64 M;

    M.set(5);
    EXPECT_TRUE(M.test(5));
    EXPECT_TRUE(M[5]);
    EXPECT_EQ(M.count(), 1u);

    M.set(10);
    EXPECT_EQ(M.count(), 2u);

    M.reset(5);
    EXPECT_FALSE(M.test(5));
    EXPECT_FALSE(M[5]);
    EXPECT_EQ(M.count(), 1u);

    M.flip(5);
    EXPECT_TRUE(M.test(5));
    EXPECT_TRUE(M[5]);

    M.set();
    EXPECT_TRUE(M.all());
    EXPECT_EQ(M.getNumLanes(), LaneMask64::BitWidth);
  }

  // Test 128-bit version.
  {
    LaneMask128 M;

    M.set(100);
    EXPECT_TRUE(M.test(100));
    EXPECT_TRUE(M[100]);
    EXPECT_EQ(M.count(), 1u);

    M.set(10);
    EXPECT_EQ(M.count(), 2u);

    M.reset(100);
    EXPECT_FALSE(M.test(100));
    EXPECT_FALSE(M[100]);

    M.flip(127);
    EXPECT_TRUE(M.test(127));
    EXPECT_TRUE(M[127]);

    M.set();
    EXPECT_TRUE(M.all());
    EXPECT_EQ(M.getNumLanes(), LaneMask128::BitWidth);
  }

  // Constexpr set/reset/flip/test/operator[] operations for 64-bit.
  constexpr auto TestSet64 = []() constexpr {
    LaneMask64 L;
    L.set(5);
    return L.test(5) && L[5] && L.count() == 1;
  };
  static_assert(TestSet64(), "Constexpr set and test");

  constexpr auto TestReset64 = []() constexpr {
    LaneMask64 L;
    L.set(5);
    L.reset(5);
    return !L.test(5) && !L[5] && L.none();
  };
  static_assert(TestReset64(), "Constexpr reset");

  constexpr auto TestFlip64 = []() constexpr {
    LaneMask64 L;
    L.flip(5);
    return L.test(5) && L[5] && L.count() == 1;
  };
  static_assert(TestFlip64(), "Constexpr flip");

  constexpr auto TestSetAll64 = []() constexpr {
    LaneMask64 L;
    L.set();
    return L.all() && L.count() == LaneMask64::BitWidth;
  };
  static_assert(TestSetAll64(), "Constexpr set all");

  // Constexpr operations for 128-bit.
  constexpr auto TestSet128 = []() constexpr {
    LaneMask128 L;
    L.set(100);
    return L.test(100) && L[100] && L.count() == 1;
  };
  static_assert(TestSet128(), "Constexpr set bit 100");

  constexpr auto TestReset128 = []() constexpr {
    LaneMask128 L;
    L.set(100);
    L.reset(100);
    return !L.test(100) && L.none();
  };
  static_assert(TestReset128(), "Constexpr reset bit 100");

  constexpr auto TestFlip128 = []() constexpr {
    LaneMask128 L;
    L.flip(100);
    return L.test(100) && L.count() == 1;
  };
  static_assert(TestFlip128(), "Constexpr flip bit 100");

  constexpr auto TestSetAll128 = []() constexpr {
    LaneMask128 L;
    L.set();
    return L.all() && L.count() == LaneMask128::BitWidth;
  };
  static_assert(TestSetAll128(), "Constexpr set all 128-bit");
}

TEST(LaneBitmaskTest, APIntConstructor) {
  APInt Empty(64, 0);
  LaneMask64 MEmpty(Empty);
  EXPECT_TRUE(MEmpty.none());

  APInt Full(64, ~0ull);
  LaneMask64 MFull(Full);
  EXPECT_EQ(MFull.getNumLanes(), 64u);
  EXPECT_TRUE(MFull.all());

  APInt A64(64, 0x123456789abcdef0, false);
  LaneMask64 M64(A64);
  EXPECT_TRUE(M64.test(4));
  EXPECT_EQ(M64.getHighestLane(), 60u);

  APInt Small(16, 0xff);
  LaneMask64 MSmall(Small);
  EXPECT_EQ(MSmall.getNumLanes(), 8u);
  EXPECT_TRUE(MSmall.test(0));
  EXPECT_TRUE(MSmall.test(7));
  EXPECT_FALSE(MSmall.test(8));

  APInt A128(128, {0xff, 0xff00});
  LaneMask128 M128(A128);
  EXPECT_TRUE(M128.test(0));
  EXPECT_TRUE(M128.test(7));
  EXPECT_TRUE(M128.test(64 + 8));
  EXPECT_FALSE(M128.test(64 + 16));
}

TEST(LaneBitmaskTest, Printing) {
  std::string Str;
  raw_string_ostream OS(Str);

  LaneBitmask M(0xABCD);
  OS << PrintLaneMask(M);
  OS.flush();

  EXPECT_TRUE(Str.find("ABCD") != std::string::npos);
}

TEST(LaneBitmaskTest, Hashing) {
  // Test 64-bit version.
  {
    LaneMask64 A(0x1234);
    LaneMask64 B(0x1234);
    LaneMask64 C(0x5678);

    EXPECT_EQ(std::hash<LaneMask64>{}(A), std::hash<LaneMask64>{}(B));
    EXPECT_NE(std::hash<LaneMask64>{}(A), std::hash<LaneMask64>{}(C));
  }

  // Test 128-bit version.
  {
    LaneMask128 A = LaneMask128::getLane(100);
    LaneMask128 B = LaneMask128::getLane(100);
    LaneMask128 C = LaneMask128::getLane(50);

    EXPECT_EQ(std::hash<LaneMask128>{}(A), std::hash<LaneMask128>{}(B));
    EXPECT_NE(std::hash<LaneMask128>{}(A), std::hash<LaneMask128>{}(C));
  }
}

TEST(LaneBitmaskTest, MultiWordOperations) {
  // Test comprehensive operations on 128-bit LaneMask64 across word boundaries.
  LaneMask128 M;
  M.set(0);
  M.set(64);
  M.set(127);

  EXPECT_EQ(M.getNumLanes(), 3u);
  EXPECT_EQ(M.getHighestLane(), 127u);
  EXPECT_TRUE(M.test(0));
  EXPECT_TRUE(M.test(64));
  EXPECT_TRUE(M.test(127));
  EXPECT_FALSE(M.test(63));

  // Test bitwise operations across word boundaries.
  LaneMask128 N;
  N.set(64);
  N.set(100);

  LaneMask128 Or = M | N;
  EXPECT_EQ(Or.getNumLanes(), 4u);

  LaneMask128 And = M & N;
  EXPECT_EQ(And.getNumLanes(), 1u);
  EXPECT_TRUE(And.test(64));

  // Test NOT across multiple words.
  LaneMask128 NotM = ~M;
  EXPECT_FALSE(NotM.test(0));
  EXPECT_FALSE(NotM.test(64));
  EXPECT_FALSE(NotM.test(127));
  EXPECT_TRUE(NotM.test(63));
  EXPECT_TRUE(NotM.test(65));
  EXPECT_EQ(NotM.getNumLanes(), LaneMask128::BitWidth - 3);
}

} // namespace
