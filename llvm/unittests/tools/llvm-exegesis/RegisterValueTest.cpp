//===-- RegisterValueTest.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterValue.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {

namespace {

#define CHECK(EXPECTED, ACTUAL)                                                \
  EXPECT_EQ(llvm::APInt(SizeInBits, EXPECTED, 16),                             \
            bitcastFloatValue(Semantic, PredefinedValues::ACTUAL))

TEST(RegisterValueTest, Half) {
  const size_t SizeInBits = 16;
  const auto &Semantic = llvm::APFloatBase::IEEEhalf();
  CHECK("0000", POS_ZERO);
  CHECK("8000", NEG_ZERO);
  CHECK("3C00", ONE);
  CHECK("4000", TWO);
  CHECK("7C00", INF);
  CHECK("7E00", QNAN);
  CHECK("7BFF", LARGEST);
  CHECK("0400", SMALLEST_NORM);
  CHECK("0001", SMALLEST);
  CHECK("0001", ULP);
  CHECK("3C01", ONE_PLUS_ULP);
}

TEST(RegisterValueTest, Single) {
  const size_t SizeInBits = 32;
  const auto &Semantic = llvm::APFloatBase::IEEEsingle();
  CHECK("00000000", POS_ZERO);
  CHECK("80000000", NEG_ZERO);
  CHECK("3F800000", ONE);
  CHECK("40000000", TWO);
  CHECK("7F800000", INF);
  CHECK("7FC00000", QNAN);
  CHECK("7F7FFFFF", LARGEST);
  CHECK("00800000", SMALLEST_NORM);
  CHECK("00000001", SMALLEST);
  CHECK("00000001", ULP);
  CHECK("3F800001", ONE_PLUS_ULP);
}

TEST(RegisterValueTest, Double) {
  const size_t SizeInBits = 64;
  const auto &Semantic = llvm::APFloatBase::IEEEdouble();
  CHECK("0000000000000000", POS_ZERO);
  CHECK("8000000000000000", NEG_ZERO);
  CHECK("3FF0000000000000", ONE);
  CHECK("4000000000000000", TWO);
  CHECK("7FF0000000000000", INF);
  CHECK("7FF8000000000000", QNAN);
  CHECK("7FEFFFFFFFFFFFFF", LARGEST);
  CHECK("0010000000000000", SMALLEST_NORM);
  CHECK("0000000000000001", SMALLEST);
  CHECK("0000000000000001", ULP);
  CHECK("3FF0000000000001", ONE_PLUS_ULP);
}

} // namespace
} // namespace exegesis
} // namespace llvm
