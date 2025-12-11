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
class TestBitsetUInt64Array : public Bitset<NumBits> {
  static constexpr unsigned NumElts = (NumBits + 63) / 64;

public:
  TestBitsetUInt64Array(const std::array<uint64_t, NumElts> &B)
      : Bitset<NumBits>(B) {}

  bool verifyValue(const std::array<uint64_t, NumElts> &B) const {
    for (unsigned I = 0; I != NumBits; ++I) {
      bool ReferenceVal =
          (B[(I / 64)] & (static_cast<uint64_t>(1) << (I % 64))) != 0;
      if (ReferenceVal != this->test(I))
        return false;
    }

    return true;
  }

  void verifyStorageSize(size_t elements_64_bit, size_t elements_32_bit) {
    if constexpr (sizeof(uintptr_t) == sizeof(uint64_t))
      EXPECT_EQ(sizeof(*this), elements_64_bit * sizeof(uintptr_t));
    else
      EXPECT_EQ(sizeof(*this), elements_32_bit * sizeof(uintptr_t));
  }
};

TEST(BitsetTest, Construction) {
  std::array<uint64_t, 2> TestVals = {0x123456789abcdef3, 0x1337d3a0b22c24};
  TestBitsetUInt64Array<96> Test(TestVals);
  EXPECT_TRUE(Test.verifyValue(TestVals));
  Test.verifyStorageSize(2, 3);

  TestBitsetUInt64Array<65> Test1(TestVals);
  EXPECT_TRUE(Test1.verifyValue(TestVals));
  Test1.verifyStorageSize(2, 3);

  std::array<uint64_t, 1> TestSingleVal = {0x12345678abcdef99};

  TestBitsetUInt64Array<64> Test64(TestSingleVal);
  EXPECT_TRUE(Test64.verifyValue(TestSingleVal));
  Test64.verifyStorageSize(1, 2);

  TestBitsetUInt64Array<30> Test30(TestSingleVal);
  EXPECT_TRUE(Test30.verifyValue(TestSingleVal));
  Test30.verifyStorageSize(1, 1);

  TestBitsetUInt64Array<32> Test32(TestSingleVal);
  EXPECT_TRUE(Test32.verifyValue(TestSingleVal));
  Test32.verifyStorageSize(1, 1);

  TestBitsetUInt64Array<33> Test33(TestSingleVal);
  EXPECT_TRUE(Test33.verifyValue(TestSingleVal));
  Test33.verifyStorageSize(1, 2);
}

TEST(BitsetTest, SetAndQuery) {
  // Test set() with all bits.
  Bitset<64> A;
  A.set();
  EXPECT_TRUE(A.all());
  EXPECT_TRUE(A.any());
  EXPECT_FALSE(A.none());

  static_assert(Bitset<64>().set().all());
  static_assert(Bitset<33>().set().all());

  // Test set() with single bit.
  Bitset<64> B;
  B.set(10);
  B.set(20);
  EXPECT_TRUE(B.test(10));
  EXPECT_TRUE(B.test(20));
  EXPECT_FALSE(B.test(15));

  static_assert(Bitset<64>().set(10).test(10));
  static_assert(Bitset<64>().set(0).set(63).test(0) &&
                Bitset<64>().set(0).set(63).test(63));
  static_assert(Bitset<33>().set(32).test(32));
  static_assert(Bitset<128>().set(64).set(127).test(64) &&
                Bitset<128>().set(64).set(127).test(127));

  // Test reset() with single bit.
  Bitset<64> C({10, 20, 30});
  C.reset(20);
  EXPECT_TRUE(C.test(10));
  EXPECT_FALSE(C.test(20));
  EXPECT_TRUE(C.test(30));

  static_assert(!Bitset<64>({10, 20}).reset(10).test(10));
  static_assert(Bitset<64>({10, 20}).reset(10).test(20));
  static_assert(!Bitset<96>({31, 32, 63}).reset(32).test(32));
  static_assert(Bitset<33>({0, 32}).reset(0).test(32));

  // Test flip() with single bit.
  Bitset<64> D({10, 20});
  D.flip(10);
  D.flip(30);
  EXPECT_FALSE(D.test(10));
  EXPECT_TRUE(D.test(20));
  EXPECT_TRUE(D.test(30));

  static_assert(!Bitset<64>({10, 20}).flip(10).test(10));
  static_assert(Bitset<64>({10, 20}).flip(30).test(30));
  static_assert(Bitset<100>({50, 99}).flip(50).test(99) &&
                !Bitset<100>({50, 99}).flip(50).test(50));
  static_assert(Bitset<33>().flip(32).test(32));

  // Test operator[].
  Bitset<64> E({5, 15, 25});
  EXPECT_TRUE(E[5]);
  EXPECT_FALSE(E[10]);
  EXPECT_TRUE(E[15]);

  static_assert(Bitset<64>({10, 20})[10]);
  static_assert(!Bitset<64>({10, 20})[15]);
  static_assert(Bitset<128>({127})[127]);
  static_assert(Bitset<96>({63, 64})[63] && Bitset<96>({63, 64})[64]);

  // Test size().
  EXPECT_EQ(A.size(), 64u);
  Bitset<33> F;
  EXPECT_EQ(F.size(), 33u);

  static_assert(Bitset<64>().size() == 64);
  static_assert(Bitset<128>().size() == 128);
  static_assert(Bitset<33>().size() == 33);

  // Test any() and none().
  static_assert(!Bitset<64>().any());
  static_assert(Bitset<64>().none());
  static_assert(Bitset<64>({10}).any());
  static_assert(!Bitset<64>({10}).none());
}

TEST(BitsetTest, ComparisonOperators) {
  // Test operator==.
  Bitset<64> A({10, 20, 30});
  Bitset<64> B({10, 20, 30});
  Bitset<64> C({10, 20, 31});
  EXPECT_TRUE(A == B);
  EXPECT_FALSE(A == C);

  static_assert(Bitset<64>({10, 20}) == Bitset<64>({10, 20}));
  static_assert(Bitset<64>({10, 20}) != Bitset<64>({10, 21}));

  // Test operator< (lexicographic comparison, bit 0 is least significant).
  static_assert(Bitset<64>({5, 11}) <
                Bitset<64>({5, 10})); // At bit 10: A=0, B=1.
  static_assert(!(Bitset<64>({5, 10}) < Bitset<64>({5, 10})));
}

TEST(BitsetTest, BitwiseNot) {
  // Test operator~.
  Bitset<64> A;
  A.set();
  Bitset<64> B = ~A;
  EXPECT_TRUE(B.none());

  static_assert((~Bitset<64>()).all());
  static_assert((~Bitset<64>().set()).none());
  static_assert((~Bitset<33>().set()).none());
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
  static_assert(!(Bitset<64>({10, 20}) & Bitset<64>({20, 30})).test(10));
  static_assert((Bitset<64>({10, 20}) & Bitset<64>({20, 30})).count() == 1);
  static_assert(
      (Bitset<96>({31, 32, 63, 64}) & Bitset<96>({32, 64, 95})).count() == 2);
  static_assert((Bitset<33>({0, 32}) & Bitset<33>({32})).test(32));

  // Test operator&=.
  Bitset<64> C({10, 20, 30});
  C &= Bitset<64>({20, 30, 40});
  EXPECT_FALSE(C.test(10));
  EXPECT_TRUE(C.test(20));
  EXPECT_TRUE(C.test(30));
  EXPECT_FALSE(C.test(40));

  constexpr Bitset<64> TestAnd = [] {
    Bitset<64> X({10, 20, 30});
    X &= Bitset<64>({20, 30, 40});
    return X;
  }();
  static_assert(TestAnd.test(20) && TestAnd.test(30) && !TestAnd.test(10));

  constexpr Bitset<100> TestAnd100 = [] {
    Bitset<100> X({10, 50, 99});
    X &= Bitset<100>({50, 99});
    return X;
  }();
  static_assert(TestAnd100.count() == 2 && TestAnd100.test(50) &&
                TestAnd100.test(99));

  // Test operator|.
  Bitset<64> D({10, 20});
  Bitset<64> E({20, 30});
  Bitset<64> Result2 = D | E;
  EXPECT_TRUE(Result2.test(10));
  EXPECT_TRUE(Result2.test(20));
  EXPECT_TRUE(Result2.test(30));
  EXPECT_EQ(Result2.count(), 3u);

  static_assert((Bitset<64>({10}) | Bitset<64>({20})).count() == 2);
  static_assert((Bitset<128>({0, 64, 127}) | Bitset<128>({64, 100})).count() ==
                4);
  static_assert((Bitset<33>({0, 16}) | Bitset<33>({16, 32})).count() == 3);

  // Test operator|=.
  Bitset<64> F({10, 20});
  F |= Bitset<64>({20, 30});
  EXPECT_TRUE(F.test(10));
  EXPECT_TRUE(F.test(20));
  EXPECT_TRUE(F.test(30));

  constexpr Bitset<64> TestOr = [] {
    Bitset<64> X({10});
    X |= Bitset<64>({20});
    return X;
  }();
  static_assert(TestOr.test(10) && TestOr.test(20));

  constexpr Bitset<96> TestOr96 = [] {
    Bitset<96> X({31, 63});
    X |= Bitset<96>({32, 64});
    return X;
  }();
  static_assert(TestOr96.count() == 4);

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
  static_assert(!(Bitset<64>({10, 20}) ^ Bitset<64>({20, 30})).test(20));
  static_assert((Bitset<64>({10, 20}) ^ Bitset<64>({20, 30})).test(30));
  static_assert((Bitset<64>({10, 20}) ^ Bitset<64>({20, 30})).count() == 2);
  static_assert((Bitset<100>({0, 50, 99}) ^ Bitset<100>({50})).count() == 2);
  static_assert((Bitset<33>({0, 32}) ^ Bitset<33>({0, 16})).count() == 2);

  // Test operator^=.
  Bitset<64> I({10, 20, 30});
  I ^= Bitset<64>({20, 30, 40});
  EXPECT_TRUE(I.test(10));
  EXPECT_FALSE(I.test(20));
  EXPECT_FALSE(I.test(30));
  EXPECT_TRUE(I.test(40));

  constexpr Bitset<64> TestXor = [] {
    Bitset<64> X({10, 20});
    X ^= Bitset<64>({20, 30});
    return X;
  }();
  static_assert(TestXor.test(10) && !TestXor.test(20) && TestXor.test(30));

  constexpr Bitset<128> TestXor128 = [] {
    Bitset<128> X({0, 64, 127});
    X ^= Bitset<128>({64});
    return X;
  }();
  static_assert(TestXor128.count() == 2 && TestXor128.test(0) &&
                TestXor128.test(127));
}

TEST(BitsetTest, LeftShiftOperator) {
  // Test shift by 0 (should be identity).
  Bitset<64> A({0, 10, 20, 30});
  Bitset<64> Result0 = A << 0;
  EXPECT_TRUE(Result0 == A);

  static_assert((Bitset<64>({0, 10}) << 0) == Bitset<64>({0, 10}));

  // Test simple left shift.
  Bitset<64> B({0, 10, 20});
  Bitset<64> Result1 = B << 5;
  EXPECT_TRUE(Result1.test(5));
  EXPECT_TRUE(Result1.test(15));
  EXPECT_TRUE(Result1.test(25));
  EXPECT_FALSE(Result1.test(0));
  EXPECT_FALSE(Result1.test(10));
  EXPECT_FALSE(Result1.test(20));
  EXPECT_EQ(Result1.count(), 3u);

  constexpr Bitset<64> TestShift = Bitset<64>({0, 10, 20}) << 5;
  static_assert(TestShift.test(5) && TestShift.test(15) && TestShift.test(25) &&
                !TestShift.test(0));
  static_assert(TestShift.count() == 3);

  // Test shift across word boundary (32-bit and 64-bit).
  Bitset<64> C({0, 31});
  Bitset<64> Result2 = C << 1;
  EXPECT_TRUE(Result2.test(1));
  EXPECT_TRUE(Result2.test(32));
  EXPECT_FALSE(Result2.test(0));
  EXPECT_FALSE(Result2.test(31));

  // Test word-aligned shift.
  Bitset<128> D({0, 10, 20});
  Bitset<128> Result3 = D << 64;
  EXPECT_TRUE(Result3.test(64));
  EXPECT_TRUE(Result3.test(74));
  EXPECT_TRUE(Result3.test(84));
  EXPECT_FALSE(Result3.test(0));
  EXPECT_FALSE(Result3.test(10));
  EXPECT_FALSE(Result3.test(20));

  // Test shift that moves bits out of range.
  Bitset<64> E({50, 60, 63});
  Bitset<64> Result4 = E << 10;
  EXPECT_TRUE(Result4.test(60));
  EXPECT_EQ(Result4.count(), 1u);

  static_assert((Bitset<64>({50, 60, 63}) << 10).count() == 1);

  // Test shift by NumBits or more (should result in all zeros).
  Bitset<64> F({0, 10, 20, 30});
  Bitset<64> Result5 = F << 64;
  EXPECT_TRUE(Result5.none());

  static_assert((Bitset<64>({0, 10}) << 64).none());

  Bitset<64> G({0, 10, 20, 30});
  Bitset<64> Result6 = G << 100;
  EXPECT_TRUE(Result6.none());

  // Test with non-multiple of word size.
  Bitset<33> H({0, 16, 32});
  Bitset<33> Result7 = H << 1;
  EXPECT_TRUE(Result7.test(1));
  EXPECT_TRUE(Result7.test(17));
  EXPECT_EQ(Result7.count(), 2u);

  static_assert((Bitset<33>({0, 16, 32}) << 1).count() == 2);
  static_assert(Bitset<64>().count() == 0);
  static_assert(Bitset<64>().set().count() == 64);
  static_assert(Bitset<128>({0, 10, 64, 127}).count() == 4);
}

TEST(BitsetTest, LeftShiftAssignOperator) {
  // Test simple left shift assignment.
  Bitset<64> A({0, 10, 20});
  A <<= 5;
  EXPECT_TRUE(A.test(5));
  EXPECT_TRUE(A.test(15));
  EXPECT_TRUE(A.test(25));
  EXPECT_FALSE(A.test(0));
  EXPECT_EQ(A.count(), 3u);

  constexpr Bitset<64> TestShiftAssign = [] {
    Bitset<64> X({0, 10});
    X <<= 5;
    return X;
  }();
  static_assert(TestShiftAssign.test(5) && TestShiftAssign.test(15));

  // Test chained operations.
  Bitset<64> B({0});
  B <<= 1;
  B <<= 2;
  EXPECT_TRUE(B.test(3));
  EXPECT_FALSE(B.test(0));
  EXPECT_FALSE(B.test(1));

  // Test shift by 0.
  Bitset<64> C({5, 10, 15});
  Bitset<64> Original = C;
  C <<= 0;
  EXPECT_TRUE(C == Original);

  // Test shift to all zeros.
  Bitset<64> D({0, 10, 20});
  D <<= 64;
  EXPECT_TRUE(D.none());
}

TEST(BitsetTest, RightShiftOperator) {
  // Test shift by 0 (should be identity).
  Bitset<64> A({10, 20, 30, 40});
  Bitset<64> Result0 = A >> 0;
  EXPECT_TRUE(Result0 == A);

  static_assert((Bitset<64>({10, 20}) >> 0) == Bitset<64>({10, 20}));

  // Test simple right shift.
  Bitset<64> B({10, 20, 30});
  Bitset<64> Result1 = B >> 5;
  EXPECT_TRUE(Result1.test(5));
  EXPECT_TRUE(Result1.test(15));
  EXPECT_TRUE(Result1.test(25));
  EXPECT_FALSE(Result1.test(10));
  EXPECT_FALSE(Result1.test(20));
  EXPECT_FALSE(Result1.test(30));
  EXPECT_EQ(Result1.count(), 3u);

  constexpr Bitset<64> TestRShift = Bitset<64>({10, 20, 30}) >> 5;
  static_assert(TestRShift.test(5) && TestRShift.test(15) &&
                TestRShift.test(25) && !TestRShift.test(10));

  // Test shift across word boundary.
  Bitset<64> C({32, 33});
  Bitset<64> Result2 = C >> 1;
  EXPECT_TRUE(Result2.test(31));
  EXPECT_TRUE(Result2.test(32));
  EXPECT_FALSE(Result2.test(33));

  // Test word-aligned shift.
  Bitset<128> D({64, 74, 84});
  Bitset<128> Result3 = D >> 64;
  EXPECT_TRUE(Result3.test(0));
  EXPECT_TRUE(Result3.test(10));
  EXPECT_TRUE(Result3.test(20));
  EXPECT_FALSE(Result3.test(64));
  EXPECT_FALSE(Result3.test(74));

  // Test shift that moves bits out of range.
  Bitset<64> E({0, 5, 10});
  Bitset<64> Result4 = E >> 8;
  EXPECT_TRUE(Result4.test(2));
  EXPECT_FALSE(Result4.test(0));
  EXPECT_FALSE(Result4.test(5));
  EXPECT_EQ(Result4.count(), 1u);

  // Test shift by NumBits or more (should result in all zeros).
  Bitset<64> F({10, 20, 30, 40});
  Bitset<64> Result5 = F >> 64;
  EXPECT_TRUE(Result5.none());

  static_assert((Bitset<64>({10, 20}) >> 64).none());

  Bitset<64> G({10, 20, 30, 40});
  Bitset<64> Result6 = G >> 100;
  EXPECT_TRUE(Result6.none());

  // Test with non-multiple of word size.
  Bitset<33> H({1, 17, 32});
  Bitset<33> Result7 = H >> 1;
  EXPECT_TRUE(Result7.test(0));
  EXPECT_TRUE(Result7.test(16));
  EXPECT_TRUE(Result7.test(31));
  EXPECT_EQ(Result7.count(), 3u);

  // Test right shift of the last bit.
  Bitset<64> I({63});
  Bitset<64> Result8 = I >> 1;
  EXPECT_TRUE(Result8.test(62));
  EXPECT_FALSE(Result8.test(63));
}

TEST(BitsetTest, RightShiftAssignOperator) {
  // Test simple right shift assignment.
  Bitset<64> A({10, 20, 30});
  A >>= 5;
  EXPECT_TRUE(A.test(5));
  EXPECT_TRUE(A.test(15));
  EXPECT_TRUE(A.test(25));
  EXPECT_FALSE(A.test(10));
  EXPECT_EQ(A.count(), 3u);

  constexpr Bitset<64> TestRShiftAssign = [] {
    Bitset<64> X({10, 20});
    X >>= 5;
    return X;
  }();
  static_assert(TestRShiftAssign.test(5) && TestRShiftAssign.test(15));

  // Test chained operations.
  Bitset<64> B({8});
  B >>= 1;
  B >>= 2;
  EXPECT_TRUE(B.test(5));
  EXPECT_FALSE(B.test(7));
  EXPECT_FALSE(B.test(8));

  // Test shift by 0.
  Bitset<64> C({5, 10, 15});
  Bitset<64> Original = C;
  C >>= 0;
  EXPECT_TRUE(C == Original);

  // Test shift to all zeros.
  Bitset<64> D({10, 20, 30});
  D >>= 64;
  EXPECT_TRUE(D.none());
}

TEST(BitsetTest, ShiftEdgeCases) {
  // Test shift at exact word boundaries.
  Bitset<96> A({31, 32, 63, 64});
  Bitset<96> Result1 = A << 32;
  EXPECT_TRUE(Result1.test(63));
  EXPECT_TRUE(Result1.test(64));
  EXPECT_TRUE(Result1.test(95));
  // 64 << 32 = 96 which is out of range for Bitset<96>.
  EXPECT_EQ(Result1.count(), 3u);

  constexpr Bitset<128> TestWordShift = Bitset<128>({64, 74}) >> 64;
  static_assert(TestWordShift.test(0) && TestWordShift.test(10));

  // Test shift at exact word boundaries for 64-bit systems.
  Bitset<128> B({63, 64, 65});
  Bitset<128> Result2 = B >> 64;
  EXPECT_TRUE(Result2.test(0));
  EXPECT_TRUE(Result2.test(1));
  EXPECT_FALSE(Result2.test(63));
  EXPECT_FALSE(Result2.test(64));

  // Test multiple shifts.
  Bitset<64> C({10});
  Bitset<64> Result3 = (C << 5) << 3;
  EXPECT_TRUE(Result3.test(18));
  EXPECT_FALSE(Result3.test(10));

  // Test left then right shift (should restore bits).
  Bitset<64> D({10, 20, 30});
  Bitset<64> Result4 = (D << 5) >> 5;
  EXPECT_TRUE(Result4.test(10));
  EXPECT_TRUE(Result4.test(20));
  EXPECT_TRUE(Result4.test(30));

  // Test right then left shift with partial loss.
  Bitset<64> E({0, 5, 10});
  Bitset<64> Result5 = (E >> 3) << 3;
  EXPECT_FALSE(Result5.test(0));
  EXPECT_TRUE(Result5.test(5));
  EXPECT_TRUE(Result5.test(10));
  EXPECT_EQ(Result5.count(), 2u);

  // Test shift with single bit bitset.
  Bitset<1> F({0});
  Bitset<1> Result6 = F << 1;
  EXPECT_TRUE(Result6.none());

  // Test large shift values on small bitsets.
  Bitset<33> G({0, 10, 20});
  Bitset<33> Result7 = G << 33;
  EXPECT_TRUE(Result7.none());

  Bitset<33> H({10, 20, 30});
  Bitset<33> Result8 = H >> 50;
  EXPECT_TRUE(Result8.none());
}

TEST(BitsetTest, ShiftCombinedWithBitwiseOperators) {
  // Test alternating bit pattern shifted.
  Bitset<64> A;
  for (unsigned I = 0; I < 64; I += 2)
    A.set(I);
  Bitset<64> Result1 = A << 1;
  for (unsigned I = 1; I < 64; I += 2)
    EXPECT_TRUE(Result1.test(I));
  for (unsigned I = 0; I < 64; I += 2)
    EXPECT_FALSE(Result1.test(I));

  // Test all ones shifted.
  Bitset<64> B;
  B.set();
  Bitset<64> Result2 = B >> 10;
  EXPECT_EQ(Result2.count(), 54u);
  for (unsigned I = 0; I < 54; ++I)
    EXPECT_TRUE(Result2.test(I));
  for (unsigned I = 54; I < 64; ++I)
    EXPECT_FALSE(Result2.test(I));

  // Test shifting combined with OR operator.
  Bitset<64> C({10, 20, 30});
  Bitset<64> D({15, 25, 35});
  Bitset<64> Result3 = (C << 5) | (D >> 5);
  EXPECT_TRUE(Result3.test(10));
  EXPECT_TRUE(Result3.test(15));
  EXPECT_TRUE(Result3.test(20));
  EXPECT_TRUE(Result3.test(25));
  EXPECT_TRUE(Result3.test(30));
  EXPECT_TRUE(Result3.test(35));
  EXPECT_EQ(Result3.count(), 6u);

  constexpr Bitset<64> TestCombined =
      (Bitset<64>({10, 20}) << 5) | (Bitset<64>({15, 25}) >> 5);
  static_assert(TestCombined.test(10) && TestCombined.test(15) &&
                TestCombined.test(20) && TestCombined.test(25));
}

} // namespace
