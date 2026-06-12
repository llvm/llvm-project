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
    EXPECT_EQ(FromArray.getNumLanes(), 32u);

    LaneMask64 None = LaneMask64::getNone();
    EXPECT_TRUE(None.none());

    LaneMask64 All = LaneMask64::getAll();
    EXPECT_TRUE(All.all());
    EXPECT_EQ(All.getNumLanes(), LaneMask64::BitWidth);

    LaneMask64 Lane5 = LaneMask64::getLane(5);
    EXPECT_EQ(Lane5.getNumLanes(), 1u);

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
    EXPECT_EQ(FromArray.getNumLanes(), 16u);

    LaneMask128 None = LaneMask128::getNone();
    EXPECT_TRUE(None.none());

    LaneMask128 All = LaneMask128::getAll();
    EXPECT_TRUE(All.all());
    EXPECT_EQ(All.getNumLanes(), LaneMask128::BitWidth);

    LaneMask128 Lane100 = LaneMask128::getLane(100);
    EXPECT_EQ(Lane100.getNumLanes(), 1u);
    EXPECT_EQ(Lane100.getHighestLane(), 100u);

    LaneMask128 Original = LaneMask128::getLane(100);
    LaneMask128 Copied(Original);
    EXPECT_EQ(Copied, Original);

    LaneMask128 Assigned;
    Assigned = Original;
    EXPECT_EQ(Assigned, Original);
  }

  // Constexpr tests.
  static_assert(LaneMask64::getNone().none());
  static_assert(LaneMask64::getAll().all());
  static_assert(LaneMask64::getAll().getNumLanes() == LaneMask64::BitWidth);
  static_assert(LaneMask128::getNone().none());
  static_assert(LaneMask128::getAll().all());
  static_assert(LaneMask128::getAll().getNumLanes() == LaneMask128::BitWidth);
  static_assert(
      []() constexpr {
        std::array<uint64_t, 1> Arr = {0xff};
        LaneMask64 M(Arr);
        return M.getNumLanes() == 8;
      }(),
      "Constexpr array constructor");
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
    LaneMask128 Low(0x1234);
    LaneMask128 High = LaneMask128::getLane(100);
    EXPECT_TRUE(Low < High);
    EXPECT_FALSE(High < Low);
  }

  static_assert(LaneMask64(0x1234) == LaneMask64(0x1234));
  static_assert(LaneMask64(0x1234) != LaneMask64(0x5678));
  static_assert(LaneMask64(0x1000) < LaneMask64(0x2000));
  static_assert(LaneMask128::getLane(64) < LaneMask128::getLane(100));
}

TEST(LaneBitmaskTest, BitwiseOperators) {
  // Test 64-bit version.
  {
    LaneMask64 A(0xff00);
    LaneMask64 B(0x0ff0);

    EXPECT_EQ(A | B, LaneMask64(0xfff0));
    EXPECT_EQ(A & B, LaneMask64(0x0f00));

    LaneMask64 NotA = ~A;
    EXPECT_EQ(NotA.getNumLanes(), LaneMask64::BitWidth - A.getNumLanes());

    LaneMask64 A2(0xff00);
    A2 |= B;
    EXPECT_EQ(A2, LaneMask64(0xfff0));

    LaneMask64 A3(0xff00);
    A3 &= B;
    EXPECT_EQ(A3, LaneMask64(0x0f00));
  }

  // Test 128-bit version across word boundaries.
  {
    LaneMask128 Low(0xff);
    LaneMask128 High = LaneMask128::getLane(100);
    LaneMask128 Combined = Low | High;
    EXPECT_TRUE(Combined.any());
    EXPECT_EQ(Combined.getNumLanes(), 9u);

    LaneMask128 AndResult = Combined & High;
    EXPECT_EQ(AndResult.getNumLanes(), 1u);
    EXPECT_EQ(AndResult.getHighestLane(), 100u);
  }

  static_assert((LaneMask64(0xff00) | LaneMask64(0x0ff0)) ==
                LaneMask64(0xfff0));
  static_assert((LaneMask64(0xff00) & LaneMask64(0x0ff0)) ==
                LaneMask64(0x0f00));
  static_assert((~LaneMask64::getAll()).none());
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
}

TEST(LaneBitmaskTest, QueryMethods) {
  // Test 64-bit version.
  {
    LaneMask64 Empty;
    EXPECT_TRUE(Empty.none());
    EXPECT_FALSE(Empty.any());
    EXPECT_FALSE(Empty.all());
    EXPECT_EQ(Empty.getNumLanes(), 0u);

    LaneMask64 Partial(0x00ff);
    EXPECT_FALSE(Partial.none());
    EXPECT_TRUE(Partial.any());
    EXPECT_FALSE(Partial.all());
    EXPECT_EQ(Partial.getNumLanes(), 8u);

    LaneMask64 Full = LaneMask64::getAll();
    EXPECT_FALSE(Full.none());
    EXPECT_TRUE(Full.any());
    EXPECT_TRUE(Full.all());
    EXPECT_EQ(Full.getNumLanes(), LaneMask64::BitWidth);
  }

  // Test 128-bit version.
  {
    LaneMask128 Empty;
    EXPECT_TRUE(Empty.none());
    EXPECT_FALSE(Empty.any());
    EXPECT_FALSE(Empty.all());
    EXPECT_EQ(Empty.getNumLanes(), 0u);

    LaneMask128 Full = LaneMask128::getAll();
    EXPECT_TRUE(Full.all());
    EXPECT_EQ(Full.getNumLanes(), LaneMask128::BitWidth);

    LaneMask128 UpperBit = LaneMask128::getLane(127);
    EXPECT_TRUE(UpperBit.any());
    EXPECT_FALSE(UpperBit.all());
  }

  static_assert(LaneMask64().none());
  static_assert(!LaneMask64(0x1).none());
  static_assert(LaneMask64(0x1).any());
  static_assert(LaneMask64::getAll().all());
  static_assert(!LaneMask64(0xff).all());
  static_assert(LaneMask64(0x7).getNumLanes() == 3);
  static_assert(LaneMask128::getLane(127).any());
  static_assert(LaneMask128::getAll().all());
}

TEST(LaneBitmaskTest, GetHighestLane) {
  EXPECT_EQ(LaneMask64(1).getHighestLane(), 0u);
  EXPECT_EQ(LaneMask64(1ull << 5).getHighestLane(), 5u);
  EXPECT_EQ(LaneMask64(1ull << 63).getHighestLane(), 63u);
  EXPECT_EQ(LaneMask64((1ull << 10) | (1ull << 30)).getHighestLane(), 30u);

  EXPECT_EQ(LaneMask128(1).getHighestLane(), 0u);
  EXPECT_EQ(LaneMask128::getLane(100).getHighestLane(), 100u);
  EXPECT_EQ(LaneMask128::getLane(127).getHighestLane(), 127u);
  EXPECT_EQ(
      (LaneMask128::getLane(10) | LaneMask128::getLane(100)).getHighestLane(),
      100u);
}

TEST(LaneBitmaskTest, RotateOperators) {
  // Test 64-bit version.
  {
    LaneMask64 A(0xff);
    EXPECT_EQ(A.rotateLeft(0), A);
    EXPECT_EQ(A.rotateLeft(8), LaneMask64(0xff00));
    EXPECT_EQ(A.rotateLeft(LaneMask64::BitWidth), A);

    LaneMask64 B(0xff00);
    EXPECT_EQ(B.rotateRight(0), B);
    EXPECT_EQ(B.rotateRight(8), LaneMask64(0xff));
    EXPECT_EQ(B.rotateRight(LaneMask64::BitWidth), B);

    LaneMask64 C(0x123456789abcdef0);
    EXPECT_EQ(C.rotateLeft(37).rotateRight(37), C);

    LaneMask64 HighBit = LaneMask64::getLane(LaneMask64::BitWidth - 1);
    EXPECT_TRUE(HighBit.rotateLeft(1).any());
  }

  // Test 128-bit version.
  {
    LaneMask128 A(0xff);
    EXPECT_EQ(A.rotateLeft(0), A);
    EXPECT_EQ(A.rotateLeft(8), LaneMask128(0xff00));
    EXPECT_EQ(A.rotateLeft(LaneMask128::BitWidth), A);

    LaneMask128 HighBit = LaneMask128::getLane(127);
    LaneMask128 Rotated = HighBit.rotateLeft(1);
    EXPECT_EQ(Rotated.getNumLanes(), 1u);
    EXPECT_EQ(Rotated.getHighestLane(), 0u);

    LaneMask128 UpperBit = LaneMask128::getLane(100);
    LaneMask128 RotatedDown = UpperBit.rotateRight(50);
    EXPECT_EQ(RotatedDown.getHighestLane(), 50u);
  }

  static_assert(LaneMask64(0xff).rotateLeft(8) == LaneMask64(0xff00));
  static_assert(LaneMask64(0xff00).rotateRight(8) == LaneMask64(0xff));
  static_assert(LaneMask64(0xff).rotateLeft(LaneMask64::BitWidth) ==
                LaneMask64(0xff));
  static_assert(LaneMask64(0x1234).rotateLeft(37).rotateRight(37) ==
                LaneMask64(0x1234));
  static_assert(LaneMask128(0xff).rotateLeft(8) == LaneMask128(0xff00));
  static_assert(LaneMask128(0xff).rotateLeft(LaneMask128::BitWidth) ==
                LaneMask128(0xff));
}

TEST(LaneBitmaskTest, GetWord) {
  // Test 64-bit version.
  {
    LaneMask64 M(0xdeadbeefcafe1234);
    EXPECT_EQ(M.getWord(0), 0xdeadbeefcafe1234ULL);
  }

  // Test 128-bit version.
  {
    std::array<uint64_t, 2> Arr = {0x1111222233334444, 0xaaaabbbbccccdddd};
    LaneMask128 M(Arr);
    EXPECT_EQ(M.getWord(0), 0x1111222233334444ULL);
    EXPECT_EQ(M.getWord(1), 0xaaaabbbbccccddddULL);
  }

  // Empty mask.
  EXPECT_EQ(LaneMask64().getWord(0), 0ULL);
  EXPECT_EQ(LaneMask128().getWord(0), 0ULL);
  EXPECT_EQ(LaneMask128().getWord(1), 0ULL);

  static_assert(LaneMask64(0xff).getWord(0) == 0xff);
  static_assert(LaneMask128::getAll().getWord(0) == ~0ULL);
  static_assert(LaneMask128::getAll().getWord(1) == ~0ULL);
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
  EXPECT_EQ(M64.getHighestLane(), 60u);

  APInt Small(16, 0xff);
  LaneMask64 MSmall(Small);
  EXPECT_EQ(MSmall.getNumLanes(), 8u);

  APInt A128(128, {0xff, 0xff00});
  LaneMask128 M128(A128);
  EXPECT_EQ(M128.getNumLanes(), 16u);
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
  LaneMask128 A = LaneMask128::getLane(0) | LaneMask128::getLane(64) |
                  LaneMask128::getLane(127);

  EXPECT_EQ(A.getNumLanes(), 3u);
  EXPECT_EQ(A.getHighestLane(), 127u);

  LaneMask128 B = LaneMask128::getLane(64) | LaneMask128::getLane(100);

  LaneMask128 Or = A | B;
  EXPECT_EQ(Or.getNumLanes(), 4u);

  LaneMask128 And = A & B;
  EXPECT_EQ(And.getNumLanes(), 1u);
  EXPECT_EQ(And.getHighestLane(), 64u);

  LaneMask128 NotA = ~A;
  EXPECT_FALSE(NotA.any() && NotA.none()); // sanity
  EXPECT_EQ(NotA.getNumLanes(), LaneMask128::BitWidth - 3);
}

} // namespace
