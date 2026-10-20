//===- llvm/unittest/Support/BitsetTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Bitset.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <unsigned NumBits>
bool verifyBitsetValue(const Bitset<NumBits> &Bits,
                       const std::array<uint64_t, (NumBits + 63) / 64> &Ref) {
  for (unsigned I = 0; I != NumBits; ++I) {
    bool ReferenceVal =
        (Ref[I / 64] & (static_cast<uint64_t>(1) << (I % 64))) != 0;
    if (ReferenceVal != Bits.test(I))
      return false;
  }
  return true;
}

template <unsigned NumBits>
void verifyBitsetStorageSize(size_t Elements64, size_t Elements32) {
  if constexpr (sizeof(uintptr_t) == sizeof(uint64_t))
    EXPECT_EQ(sizeof(Bitset<NumBits>), Elements64 * sizeof(uintptr_t));
  else
    EXPECT_EQ(sizeof(Bitset<NumBits>), Elements32 * sizeof(uintptr_t));
}

TEST(BitsetTest, Construction) {
  std::array<uint64_t, 2> TestVals = {0x123456789abcdef3, 0x1337d3a0b22c24};
  Bitset<96> Test(TestVals);
  EXPECT_TRUE(verifyBitsetValue(Test, TestVals));
  verifyBitsetStorageSize<96>(2, 3);

  Bitset<65> Test1(TestVals);
  EXPECT_TRUE(verifyBitsetValue(Test1, TestVals));
  verifyBitsetStorageSize<65>(2, 3);

  std::array<uint64_t, 1> TestSingleVal = {0x12345678abcdef99};

  Bitset<64> Test64(TestSingleVal);
  EXPECT_TRUE(verifyBitsetValue(Test64, TestSingleVal));
  verifyBitsetStorageSize<64>(1, 2);

  Bitset<30> Test30(TestSingleVal);
  EXPECT_TRUE(verifyBitsetValue(Test30, TestSingleVal));
  verifyBitsetStorageSize<30>(1, 1);

  Bitset<32> Test32(TestSingleVal);
  EXPECT_TRUE(verifyBitsetValue(Test32, TestSingleVal));
  verifyBitsetStorageSize<32>(1, 1);

  Bitset<33> Test33(TestSingleVal);
  EXPECT_TRUE(verifyBitsetValue(Test33, TestSingleVal));
  verifyBitsetStorageSize<33>(1, 2);
}

TEST(BitsetTest, SetAndQuery) {
  // Test set() with all bits.
  Bitset<64> A;
  A.set();
  EXPECT_TRUE(A.all());
  EXPECT_TRUE(A.any());
  EXPECT_FALSE(A.none());

  static_assert(Bitset<64>().set().all());
  EXPECT_TRUE(Bitset<33>().set().all());

  // Test set() with single bit.
  Bitset<64> B;
  B.set(10);
  B.set(20);
  EXPECT_TRUE(B.test(10));
  EXPECT_TRUE(B.test(20));
  EXPECT_FALSE(B.test(15));

  static_assert(Bitset<64>().set(10).test(10));
  EXPECT_TRUE(Bitset<64>().set(0).set(63).test(0));
  EXPECT_TRUE(Bitset<64>().set(0).set(63).test(63));
  EXPECT_TRUE(Bitset<33>().set(32).test(32));
  EXPECT_TRUE(Bitset<128>().set(64).set(127).test(64));
  EXPECT_TRUE(Bitset<128>().set(64).set(127).test(127));

  // Test reset() with single bit.
  Bitset<64> C({10, 20, 30});
  C.reset(20);
  EXPECT_TRUE(C.test(10));
  EXPECT_FALSE(C.test(20));
  EXPECT_TRUE(C.test(30));

  static_assert(!Bitset<64>({10, 20}).reset(10).test(10));
  EXPECT_TRUE(Bitset<64>({10, 20}).reset(10).test(20));
  EXPECT_FALSE(Bitset<96>({31, 32, 63}).reset(32).test(32));
  EXPECT_TRUE(Bitset<33>({0, 32}).reset(0).test(32));

  // Test flip() with single bit.
  Bitset<64> D({10, 20});
  D.flip(10);
  D.flip(30);
  EXPECT_FALSE(D.test(10));
  EXPECT_TRUE(D.test(20));
  EXPECT_TRUE(D.test(30));

  static_assert(!Bitset<64>({10, 20}).flip(10).test(10));
  EXPECT_TRUE(Bitset<64>({10, 20}).flip(30).test(30));
  EXPECT_TRUE(Bitset<100>({50, 99}).flip(50).test(99));
  EXPECT_FALSE(Bitset<100>({50, 99}).flip(50).test(50));
  EXPECT_TRUE(Bitset<33>().flip(32).test(32));

  // Test operator[].
  Bitset<64> E({5, 15, 25});
  EXPECT_TRUE(E[5]);
  EXPECT_FALSE(E[10]);
  EXPECT_TRUE(E[15]);

  static_assert(Bitset<64>({10, 20})[10]);
  EXPECT_FALSE(Bitset<64>({10, 20})[15]);
  EXPECT_TRUE(Bitset<128>({127})[127]);
  EXPECT_TRUE(Bitset<96>({63, 64})[63]);
  EXPECT_TRUE(Bitset<96>({63, 64})[64]);

  // Test size().
  EXPECT_EQ(A.size(), 64u);
  Bitset<33> F;
  EXPECT_EQ(F.size(), 33u);

  static_assert(Bitset<64>().size() == 64);
  EXPECT_EQ(Bitset<128>().size(), 128u);
  EXPECT_EQ(Bitset<33>().size(), 33u);

  // Test any() and none().
  static_assert(!Bitset<64>().any());
  EXPECT_TRUE(Bitset<64>().none());
  EXPECT_TRUE(Bitset<64>({10}).any());
  EXPECT_FALSE(Bitset<64>({10}).none());
}

TEST(BitsetTest, ComparisonOperators) {
  // Test operator==.
  Bitset<64> A({10, 20, 30});
  Bitset<64> B({10, 20, 30});
  Bitset<64> C({10, 20, 31});
  EXPECT_TRUE(A == B);
  EXPECT_FALSE(A == C);

  static_assert(Bitset<64>({10, 20}) == Bitset<64>({10, 20}));
  EXPECT_TRUE(Bitset<64>({10, 20}) != Bitset<64>({10, 21}));

  // Test operator< (lexicographic comparison, bit 0 is least significant).
  static_assert(Bitset<64>({5, 11}) <
                Bitset<64>({5, 10})); // At bit 10: A=0, B=1.
  EXPECT_FALSE(Bitset<64>({5, 10}) < Bitset<64>({5, 10}));
}

TEST(BitsetTest, BitwiseNot) {
  // Test operator~.
  Bitset<64> A;
  A.set();
  Bitset<64> B = ~A;
  EXPECT_TRUE(B.none());

  static_assert((~Bitset<64>()).all());
  EXPECT_TRUE((~Bitset<64>().set()).none());
  EXPECT_TRUE((~Bitset<33>().set()).none());
}

TEST(BitsetTest, BitwiseOperators) {
  // Test operator&.
  Bitset<64> A({10, 20, 30});
  Bitset<64> B({20, 30, 40});
  Bitset<64> Result1 = A & B;
  EXPECT_FALSE(Result1.test(10));
  EXPECT_TRUE(Result1.test(20));
  EXPECT_TRUE(Result1.test(30));
  EXPECT_FALSE(Result1.test(40));
  EXPECT_EQ(Result1.count(), 2u);

  static_assert((Bitset<64>({10, 20}) & Bitset<64>({20, 30})).test(20));
  EXPECT_FALSE((Bitset<64>({10, 20}) & Bitset<64>({20, 30})).test(10));
  EXPECT_EQ((Bitset<64>({10, 20}) & Bitset<64>({20, 30})).count(), 1u);
  EXPECT_EQ((Bitset<96>({31, 32, 63, 64}) & Bitset<96>({32, 64, 95})).count(),
            2u);
  EXPECT_TRUE((Bitset<33>({0, 32}) & Bitset<33>({32})).test(32));

  // Test operator&=.
  Bitset<64> C({10, 20, 30});
  C &= Bitset<64>({20, 30, 40});
  EXPECT_FALSE(C.test(10));
  EXPECT_TRUE(C.test(20));
  EXPECT_TRUE(C.test(30));
  EXPECT_FALSE(C.test(40));

  static_assert([] {
    Bitset<64> X({10, 20, 30});
    X &= Bitset<64>({20, 30, 40});
    return X.test(20) && X.test(30) && !X.test(10);
  }());

  Bitset<100> TestAnd100({10, 50, 99});
  TestAnd100 &= Bitset<100>({50, 99});
  EXPECT_EQ(TestAnd100.count(), 2u);
  EXPECT_TRUE(TestAnd100.test(50));
  EXPECT_TRUE(TestAnd100.test(99));

  // Test operator|.
  Bitset<64> D({10, 20});
  Bitset<64> E({20, 30});
  Bitset<64> Result2 = D | E;
  EXPECT_TRUE(Result2.test(10));
  EXPECT_TRUE(Result2.test(20));
  EXPECT_TRUE(Result2.test(30));
  EXPECT_EQ(Result2.count(), 3u);

  static_assert((Bitset<64>({10}) | Bitset<64>({20})).count() == 2);
  EXPECT_EQ((Bitset<128>({0, 64, 127}) | Bitset<128>({64, 100})).count(), 4u);
  EXPECT_EQ((Bitset<33>({0, 16}) | Bitset<33>({16, 32})).count(), 3u);

  // Test operator|=.
  Bitset<64> F({10, 20});
  F |= Bitset<64>({20, 30});
  EXPECT_TRUE(F.test(10));
  EXPECT_TRUE(F.test(20));
  EXPECT_TRUE(F.test(30));

  static_assert([] {
    Bitset<64> X({10});
    X |= Bitset<64>({20});
    return X.test(10) && X.test(20);
  }());

  Bitset<96> TestOr96({31, 63});
  TestOr96 |= Bitset<96>({32, 64});
  EXPECT_EQ(TestOr96.count(), 4u);

  // Test operator^.
  Bitset<64> G({10, 20, 30});
  Bitset<64> H({20, 30, 40});
  Bitset<64> Result3 = G ^ H;
  EXPECT_TRUE(Result3.test(10));
  EXPECT_FALSE(Result3.test(20));
  EXPECT_FALSE(Result3.test(30));
  EXPECT_TRUE(Result3.test(40));
  EXPECT_EQ(Result3.count(), 2u);

  static_assert((Bitset<64>({10, 20}) ^ Bitset<64>({20, 30})).test(10));
  EXPECT_FALSE((Bitset<64>({10, 20}) ^ Bitset<64>({20, 30})).test(20));
  EXPECT_TRUE((Bitset<64>({10, 20}) ^ Bitset<64>({20, 30})).test(30));
  EXPECT_EQ((Bitset<64>({10, 20}) ^ Bitset<64>({20, 30})).count(), 2u);
  EXPECT_EQ((Bitset<100>({0, 50, 99}) ^ Bitset<100>({50})).count(), 2u);
  EXPECT_EQ((Bitset<33>({0, 32}) ^ Bitset<33>({0, 16})).count(), 2u);

  // Test operator^=.
  Bitset<64> I({10, 20, 30});
  I ^= Bitset<64>({20, 30, 40});
  EXPECT_TRUE(I.test(10));
  EXPECT_FALSE(I.test(20));
  EXPECT_FALSE(I.test(30));
  EXPECT_TRUE(I.test(40));

  static_assert([] {
    Bitset<64> X({10, 20});
    X ^= Bitset<64>({20, 30});
    return X.test(10) && !X.test(20) && X.test(30);
  }());

  Bitset<128> TestXor128({0, 64, 127});
  TestXor128 ^= Bitset<128>({64});
  EXPECT_EQ(TestXor128.count(), 2u);
  EXPECT_TRUE(TestXor128.test(0));
  EXPECT_TRUE(TestXor128.test(127));
}

TEST(BitsetTest, ShiftOperators) {
  // Test left shift.
  static_assert((Bitset<64>({0}) << 10).test(10));
  EXPECT_FALSE((Bitset<64>({0}) << 10).test(0));
  EXPECT_TRUE((Bitset<64>({63}) << 1).none());
  EXPECT_TRUE((Bitset<128>({0}) << 64).test(64));
  EXPECT_TRUE((Bitset<128>({63}) << 1).test(64));
  EXPECT_TRUE((Bitset<128>({127}) << 1).none());

  // Test right shift.
  static_assert((Bitset<64>({10}) >> 10).test(0));
  EXPECT_FALSE((Bitset<64>({10}) >> 10).test(10));
  EXPECT_TRUE((Bitset<64>({0}) >> 1).none());
  EXPECT_TRUE((Bitset<128>({64}) >> 64).test(0));
  EXPECT_TRUE((Bitset<128>({64}) >> 1).test(63));
  EXPECT_TRUE((Bitset<128>({0}) >> 1).none());

  // Test shift by 0.
  EXPECT_TRUE((Bitset<64>({10, 20}) << 0) == Bitset<64>({10, 20}));
  EXPECT_TRUE((Bitset<64>({10, 20}) >> 0) == Bitset<64>({10, 20}));

  // Test shift by NumBits (clears all).
  EXPECT_TRUE((Bitset<64>({0, 63}) << 64).none());
  EXPECT_TRUE((Bitset<64>({0, 63}) >> 64).none());
  EXPECT_TRUE((Bitset<128>({0, 127}) << 128).none());
  EXPECT_TRUE((Bitset<128>({0, 127}) >> 128).none());
}

TEST(BitsetTest, GetNumWords64) {
  static_assert(Bitset<1>::getNumWords64() == 1);
  EXPECT_EQ(Bitset<32>::getNumWords64(), 1u);
  EXPECT_EQ(Bitset<64>::getNumWords64(), 1u);
  EXPECT_EQ(Bitset<65>::getNumWords64(), 2u);
  EXPECT_EQ(Bitset<96>::getNumWords64(), 2u);
  EXPECT_EQ(Bitset<128>::getNumWords64(), 2u);
  EXPECT_EQ(Bitset<129>::getNumWords64(), 3u);
}

TEST(BitsetTest, GetWord64) {
  // Single-word bitset.
  constexpr auto B64 = Bitset<64>(std::array<uint64_t, 1>{0xdeadbeefcafe1234});
  static_assert(B64.getWord64(0) == 0xdeadbeefcafe1234);

  // Multi-word bitset.
  Bitset<128> B128(
      std::array<uint64_t, 2>{0x1111222233334444, 0xaaaabbbbccccdddd});
  EXPECT_EQ(B128.getWord64(0), 0x1111222233334444u);
  EXPECT_EQ(B128.getWord64(1), uint64_t(0xaaaabbbbccccdddd));

  // Partial last word - high bits should be masked off.
  Bitset<96> B96(
      std::array<uint64_t, 2>{0xffffffffffffffff, 0xffffffffffffffff});
  EXPECT_EQ(B96.getWord64(0), uint64_t(0xffffffffffffffff));
  // Only lower 32 bits.
  EXPECT_EQ(B96.getWord64(1), uint64_t(0x00000000ffffffff));

  // Empty bitset.
  EXPECT_EQ(Bitset<64>().getWord64(0), 0u);
  EXPECT_EQ(Bitset<128>().getWord64(0), 0u);
  EXPECT_EQ(Bitset<128>().getWord64(1), 0u);
}

TEST(BitsetTest, FindLastSet) {
  // Empty bitset returns -1.
  static_assert(Bitset<64>().findLastSet() == -1);
  EXPECT_EQ(Bitset<128>().findLastSet(), -1);

  // Single bit set.
  EXPECT_EQ(Bitset<64>({0}).findLastSet(), 0);
  EXPECT_EQ(Bitset<64>({63}).findLastSet(), 63);
  EXPECT_EQ(Bitset<64>({31}).findLastSet(), 31);
  EXPECT_EQ(Bitset<128>({0}).findLastSet(), 0);
  EXPECT_EQ(Bitset<128>({64}).findLastSet(), 64);
  EXPECT_EQ(Bitset<128>({127}).findLastSet(), 127);

  // Multiple bits - returns highest.
  EXPECT_EQ(Bitset<64>({0, 10, 50}).findLastSet(), 50);
  EXPECT_EQ(Bitset<128>({0, 63, 64, 100}).findLastSet(), 100);

  // All bits set.
  EXPECT_EQ(Bitset<64>().set().findLastSet(), 63);
  EXPECT_EQ(Bitset<128>().set().findLastSet(), 127);
  EXPECT_EQ(Bitset<96>().set().findLastSet(), 95);

  // Non-power-of-2 sizes.
  EXPECT_EQ(Bitset<33>({32}).findLastSet(), 32);
  EXPECT_EQ(Bitset<33>({0, 32}).findLastSet(), 32);
  EXPECT_EQ(Bitset<65>({64}).findLastSet(), 64);
}

TEST(BitsetTest, ShiftMultiWords) {
  constexpr auto B192 = Bitset<192>({0, 64, 128});
  static_assert((B192 << 1) == Bitset<192>({1, 65, 129}));
  EXPECT_TRUE((B192 >> 1) == Bitset<192>({63, 127}));
  EXPECT_TRUE((B192 << 64) == Bitset<192>({64, 128}));
  EXPECT_TRUE((B192 >> 64) == Bitset<192>({0, 64}));
  EXPECT_TRUE((Bitset<192>({63, 127}) << 1) == Bitset<192>({64, 128}));
  EXPECT_TRUE((Bitset<192>({64, 128}) >> 1) == Bitset<192>({63, 127}));
}

TEST(BitsetTest, ShiftBoundaryBitShifts) {
  static_assert((Bitset<128>({1}) << 63) == Bitset<128>({64}));
  EXPECT_TRUE((Bitset<128>({64}) >> 63) == Bitset<128>({1}));
  EXPECT_TRUE((Bitset<192>({1, 65}) << 63) == Bitset<192>({64, 128}));
  // Shift by NumBits - 1.
  EXPECT_TRUE((Bitset<64>({0}) << 63) == Bitset<64>({63}));
  EXPECT_TRUE((Bitset<64>({63}) >> 63) == Bitset<64>({0}));
  EXPECT_TRUE((Bitset<33>({0}) << 32) == Bitset<33>({32}));
  // Full-width shift of a fully-set bitset loses exactly one bit, and the
  // bit that is lost must be the boundary bit.
  EXPECT_EQ((Bitset<128>().set() << 1).count(), 127u);
  EXPECT_EQ((Bitset<128>().set() >> 1).count(), 127u);
  EXPECT_EQ((Bitset<100>().set() >> 1).count(), 99u);
  EXPECT_FALSE((Bitset<100>().set() >> 1).test(99));
  EXPECT_FALSE((Bitset<100>().set() << 1).test(0));
  EXPECT_FALSE((Bitset<128>().set() >> 1).test(127));
  EXPECT_FALSE((Bitset<128>().set() << 1).test(0));
}

TEST(BitsetTest, ShiftExcessAmount) {
  static_assert((Bitset<64>().set() << 65).none());
  EXPECT_TRUE((Bitset<64>().set() >> 200).none());
  EXPECT_TRUE((Bitset<33>({0, 10, 32}) << 1000).none());
  EXPECT_TRUE((Bitset<128>({0, 127}) >> 1000).none());
  EXPECT_TRUE((Bitset<192>().set() << 193).none());
}

TEST(BitsetTest, ShiftDoesNotMutateOperand) {
  // Non-mutating operator<< / operator>> must leave the source unchanged.
  Bitset<128> X({5, 70});
  Bitset<128> YL = X << 1;
  EXPECT_TRUE(YL == Bitset<128>({6, 71}));
  EXPECT_TRUE(X == Bitset<128>({5, 70}));

  Bitset<128> YR = X >> 1;
  EXPECT_TRUE(YR == Bitset<128>({4, 69}));
  EXPECT_TRUE(X == Bitset<128>({5, 70}));

  static_assert([] {
    Bitset<128> X({5, 70});
    Bitset<128> Y = X << 1;
    return Y == Bitset<128>({6, 71}) && X == Bitset<128>({5, 70});
  }());
}

TEST(BitsetTest, ShiftAssignReturnsReference) {
  static_assert([] {
    Bitset<64> X({0});
    (X <<= 3) <<= 2;
    return X == Bitset<64>({5});
  }());

  Bitset<128> R({100});
  (R >>= 30) >>= 10;
  EXPECT_TRUE(R == Bitset<128>({60}));
}

TEST(BitsetTest, GetWord64ConsistencyWithTest) {
  // For every set bit, getWord64 must report it in the expected 64-bit word.
  constexpr auto B100 = Bitset<100>({0, 50, 64, 99});
  static_assert((B100.getWord64(0) & 1) != 0);
  EXPECT_NE(B100.getWord64(0) & (uint64_t(1) << 50), 0u);
  EXPECT_NE(B100.getWord64(1) & 1, 0u);
  EXPECT_NE(B100.getWord64(1) & (uint64_t(1) << 35), 0u);
}

TEST(BitsetTest, GetWord64AfterMutation) {
  // getWord64() reflects subsequent set / shift.
  static_assert([] {
    Bitset<128> X;
    X.set(5).set(70);
    return X.getWord64(0) == (uint64_t(1) << 5) &&
           X.getWord64(1) == (uint64_t(1) << 6);
  }());

  Bitset<128> Shifted = Bitset<128>({5}) << 64;
  EXPECT_EQ(Shifted.getWord64(0), 0u);
  EXPECT_EQ(Shifted.getWord64(1), uint64_t(1) << 5);
}

TEST(BitsetTest, GetNumWords64MoreWidths) {
  static_assert(Bitset<2>::getNumWords64() == 1);
  EXPECT_EQ(Bitset<192>::getNumWords64(), 3u);
  EXPECT_EQ(Bitset<193>::getNumWords64(), 4u);
  EXPECT_EQ(Bitset<256>::getNumWords64(), 4u);
}

TEST(BitsetTest, FindLastSetSmallWidths) {
  static_assert(Bitset<1>().findLastSet() == -1);
  EXPECT_EQ(Bitset<1>({0}).findLastSet(), 0);
  EXPECT_EQ(Bitset<2>({0, 1}).findLastSet(), 1);
  EXPECT_EQ(Bitset<32>({31}).findLastSet(), 31);
  EXPECT_EQ(Bitset<32>().set().findLastSet(), 31);
}

TEST(BitsetTest, FindLastSetMultiWordScan) {
  static_assert(Bitset<192>({70}).findLastSet() == 70);
  EXPECT_EQ(Bitset<192>({64, 70, 127}).findLastSet(), 127);
  EXPECT_EQ(Bitset<192>({3}).findLastSet(), 3);
  EXPECT_EQ(Bitset<100>({99}).findLastSet(), 99);
  // Highest set bit lives in the lowest word; the loop must scan past
  // multiple empty trailing words.
  EXPECT_EQ(Bitset<192>({0}).findLastSet(), 0);
  EXPECT_EQ(Bitset<256>({1}).findLastSet(), 1);
}

TEST(BitsetTest, FindLastSetAfterMutation) {
  static_assert(Bitset<128>({0, 50, 100}).reset(100).findLastSet() == 50);

  Bitset<64> B = Bitset<64>({10}) << 20;
  EXPECT_EQ(B.findLastSet(), 30);

  Bitset<64> C = Bitset<64>({63}) >> 10;
  EXPECT_EQ(C.findLastSet(), 53);

  Bitset<64> D = Bitset<64>({63}) << 1;
  EXPECT_EQ(D.findLastSet(), -1);
}

} // namespace
