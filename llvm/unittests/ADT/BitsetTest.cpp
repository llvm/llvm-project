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
} // namespace
