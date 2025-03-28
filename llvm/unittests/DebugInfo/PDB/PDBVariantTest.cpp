//===- llvm/unittest/DebugInfo/PDB/PDBVariantTest.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "llvm/DebugInfo/PDB/PDBTypes.h"

using namespace llvm;
using namespace llvm::pdb;

namespace {

template <typename T> class PDBVariantIntegerTest : public testing::Test {
public:
  std::vector<T> getTestIntegers() {
    if constexpr (std::is_same_v<T, bool>) {
      return {true, false};
    } else {
      std::vector<T> Integers{0, 1, 32, std::numeric_limits<T>::min(),
                              std::numeric_limits<T>::max()};
      if constexpr (std::is_signed_v<T>) {
        Integers.emplace_back(-1);
        Integers.emplace_back(-65);
      }
      return Integers;
    }
  }
};

using TestTypes = testing::Types<bool, int8_t, uint8_t, int16_t, uint16_t,
                                 int32_t, uint32_t, int64_t, uint64_t>;

} // namespace

TYPED_TEST_SUITE(PDBVariantIntegerTest, TestTypes, );

TYPED_TEST(PDBVariantIntegerTest, ToAPSInt) {
  for (TypeParam IntegerValue : this->getTestIntegers()) {
    const Variant VariantValue(IntegerValue);

    const APSInt APSIntValue = VariantValue.toAPSInt();
    EXPECT_EQ(APSIntValue.isSigned(), std::is_signed_v<TypeParam>)
        << "Unexpected 'isSigned()' result for '" << IntegerValue << "'";
    bool IsNegative = false;
    if constexpr (!std::is_same_v<TypeParam, bool>) {
      IsNegative = IntegerValue < 0;
    }
    EXPECT_EQ(APSIntValue.isNegative(), IsNegative)
        << "Unexpected 'isNegative()' result for '" << IntegerValue << "'";

    SmallString<20> String;
    APSIntValue.toString(String);
    EXPECT_EQ(String.str().str(), std::to_string(IntegerValue));
  }
}
