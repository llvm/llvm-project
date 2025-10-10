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
};

TEST(BitsetTest, Construction) {
  std::array<uint64_t, 2> TestVals = {0x123456789abcdef3, 0x1337d3a0b22c24};
  TestBitsetUInt64Array<96> Test(TestVals);
  EXPECT_TRUE(Test.verifyValue(TestVals));

  TestBitsetUInt64Array<65> Test1(TestVals);
  EXPECT_TRUE(Test1.verifyValue(TestVals));
}
} // namespace
