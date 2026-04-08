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

} // namespace
