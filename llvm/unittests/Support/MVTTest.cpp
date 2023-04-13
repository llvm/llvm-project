//===- llvm/unittest/Support/MVTTest.cpp - Test compatibility -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Make sure the generated version of MachineValueType.h to be equivalent to
// the constant version of llvm/Support/MachineValueType.h.
//
//===----------------------------------------------------------------------===//

#include "MachineValueType.h"
#include "llvm/Support/MachineValueType.h"
#include "gtest/gtest.h"
#include <limits>
#include <string>

using namespace llvm;

namespace {

TEST(MVTTest, Properties) {
  for (int i = 0; i <= std::numeric_limits<uint8_t>::max(); ++i) {
    SCOPED_TRACE("i=" + std::to_string(i));
    auto Org = MVT(MVT::SimpleValueType(i));
    auto New = tmp::MVT(tmp::MVT::SimpleValueType(i));

#define MVTTEST_EXPECT_EQ_M(LHS, RHS, M) EXPECT_EQ((LHS).M(), (RHS).M())
#define MVTTEST_EXPECT_EQ_SVT(LHS, RHS)                                        \
  EXPECT_EQ(int((LHS).SimpleTy), int((RHS).SimpleTy))
#define MVTTEST_EXPECT_EQ_SVT_M(LHS, RHS, M)                                   \
  MVTTEST_EXPECT_EQ_SVT((LHS).M(), (RHS).M())
#define MVTTEST_EXPECT_EQ_SVT_F(F, ...)                                        \
  MVTTEST_EXPECT_EQ_SVT(MVT::F(__VA_ARGS__), tmp::MVT::F(__VA_ARGS__))

    MVTTEST_EXPECT_EQ_M(New, Org, isValid);
    MVTTEST_EXPECT_EQ_M(New, Org, isFloatingPoint);
    MVTTEST_EXPECT_EQ_M(New, Org, isInteger);
    MVTTEST_EXPECT_EQ_M(New, Org, isScalarInteger);
    MVTTEST_EXPECT_EQ_M(New, Org, isVector);
    MVTTEST_EXPECT_EQ_M(New, Org, isScalableVector);
    MVTTEST_EXPECT_EQ_M(New, Org, isScalableTargetExtVT);
    MVTTEST_EXPECT_EQ_M(New, Org, isScalableVT);
    MVTTEST_EXPECT_EQ_M(New, Org, isFixedLengthVector);
    MVTTEST_EXPECT_EQ_M(New, Org, is16BitVector);
    MVTTEST_EXPECT_EQ_M(New, Org, is32BitVector);
    MVTTEST_EXPECT_EQ_M(New, Org, is64BitVector);
    MVTTEST_EXPECT_EQ_M(New, Org, is128BitVector);
    MVTTEST_EXPECT_EQ_M(New, Org, is256BitVector);
    MVTTEST_EXPECT_EQ_M(New, Org, is512BitVector);
    MVTTEST_EXPECT_EQ_M(New, Org, is1024BitVector);
    MVTTEST_EXPECT_EQ_M(New, Org, is2048BitVector);
    MVTTEST_EXPECT_EQ_M(New, Org, isOverloaded);
    if (New.isVector()) {
      MVTTEST_EXPECT_EQ_SVT_M(New, Org, changeVectorElementTypeToInteger);
      MVTTEST_EXPECT_EQ_SVT_M(New, Org, changeTypeToInteger);
      if (New.getVectorElementCount().isKnownEven()) {
        MVTTEST_EXPECT_EQ_SVT_M(New, Org, getHalfNumVectorElementsVT);
      }
      MVTTEST_EXPECT_EQ_M(New, Org, isPow2VectorType);
      MVTTEST_EXPECT_EQ_SVT_M(New, Org, getPow2VectorType);
      MVTTEST_EXPECT_EQ_SVT_M(New, Org, getVectorElementType);
      MVTTEST_EXPECT_EQ_M(New, Org, getVectorMinNumElements);
      MVTTEST_EXPECT_EQ_M(New, Org, getVectorElementCount);

      auto n = New.getVectorMinNumElements();
      auto sc = New.isScalableVector();
      auto LHS = tmp::MVT::getVectorVT(New.getVectorElementType(), n, sc);
      auto RHS = MVT::getVectorVT(Org.getVectorElementType(), n, sc);
      MVTTEST_EXPECT_EQ_SVT(LHS, RHS);
    } else if (New.isInteger()) {
      auto bw = New.getSizeInBits();
      MVTTEST_EXPECT_EQ_SVT_F(getIntegerVT, bw);
    } else if (New.isFloatingPoint()) {
      auto bw = New.getSizeInBits();
      MVTTEST_EXPECT_EQ_SVT_F(getFloatingPointVT, bw);
    }
    MVTTEST_EXPECT_EQ_SVT_M(New, Org, getScalarType);
    if (New.isValid()) {
      switch (New.SimpleTy) {
      case tmp::MVT::Other:
      case tmp::MVT::Glue:
      case tmp::MVT::isVoid:
      case tmp::MVT::Untyped:
      case tmp::MVT::spirvbuiltin:
        break;
      case tmp::MVT::aarch64svcount:
        break;
      default:
        MVTTEST_EXPECT_EQ_M(New, Org, getSizeInBits);
        MVTTEST_EXPECT_EQ_M(New, Org, getScalarSizeInBits);
        MVTTEST_EXPECT_EQ_M(New, Org, getStoreSize);
        MVTTEST_EXPECT_EQ_M(New, Org, getScalarStoreSize);
        MVTTEST_EXPECT_EQ_M(New, Org, getStoreSizeInBits);
        MVTTEST_EXPECT_EQ_M(New, Org, isByteSized);
        if (!New.isScalableVector()) {
          MVTTEST_EXPECT_EQ_M(New, Org, getFixedSizeInBits);
        }
        break;
      }
    }
  }
}
} // namespace
