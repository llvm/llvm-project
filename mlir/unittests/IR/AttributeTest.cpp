//===- AttributeTest.cpp - Attribute unit tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/StandardTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::detail;

template <typename EltTy>
static void testSplat(Type eltType, const EltTy &splatElt) {
  RankedTensorType shape = RankedTensorType::get({2, 1}, eltType);

  // Check that the generated splat is the same for 1 element and N elements.
  DenseElementsAttr splat = DenseElementsAttr::get(shape, splatElt);
  EXPECT_TRUE(splat.isSplat());

  auto detectedSplat =
      DenseElementsAttr::get(shape, llvm::makeArrayRef({splatElt, splatElt}));
  EXPECT_EQ(detectedSplat, splat);

  for (auto newValue : detectedSplat.template getValues<EltTy>())
    EXPECT_TRUE(newValue == splatElt);
}

namespace {
TEST(DenseSplatTest, BoolSplat) {
  MLIRContext context;
  IntegerType boolTy = IntegerType::get(1, &context);
  RankedTensorType shape = RankedTensorType::get({2, 2}, boolTy);

  // Check that splat is automatically detected for boolean values.
  /// True.
  DenseElementsAttr trueSplat = DenseElementsAttr::get(shape, true);
  EXPECT_TRUE(trueSplat.isSplat());
  /// False.
  DenseElementsAttr falseSplat = DenseElementsAttr::get(shape, false);
  EXPECT_TRUE(falseSplat.isSplat());
  EXPECT_NE(falseSplat, trueSplat);

  /// Detect and handle splat within 8 elements (bool values are bit-packed).
  /// True.
  auto detectedSplat = DenseElementsAttr::get(shape, {true, true, true, true});
  EXPECT_EQ(detectedSplat, trueSplat);
  /// False.
  detectedSplat = DenseElementsAttr::get(shape, {false, false, false, false});
  EXPECT_EQ(detectedSplat, falseSplat);
}

TEST(DenseSplatTest, LargeBoolSplat) {
  constexpr int64_t boolCount = 56;

  MLIRContext context;
  IntegerType boolTy = IntegerType::get(1, &context);
  RankedTensorType shape = RankedTensorType::get({boolCount}, boolTy);

  // Check that splat is automatically detected for boolean values.
  /// True.
  DenseElementsAttr trueSplat = DenseElementsAttr::get(shape, true);
  DenseElementsAttr falseSplat = DenseElementsAttr::get(shape, false);
  EXPECT_TRUE(trueSplat.isSplat());
  EXPECT_TRUE(falseSplat.isSplat());

  /// Detect that the large boolean arrays are properly splatted.
  /// True.
  SmallVector<bool, 64> trueValues(boolCount, true);
  auto detectedSplat = DenseElementsAttr::get(shape, trueValues);
  EXPECT_EQ(detectedSplat, trueSplat);
  /// False.
  SmallVector<bool, 64> falseValues(boolCount, false);
  detectedSplat = DenseElementsAttr::get(shape, falseValues);
  EXPECT_EQ(detectedSplat, falseSplat);
}

TEST(DenseSplatTest, BoolNonSplat) {
  MLIRContext context;
  IntegerType boolTy = IntegerType::get(1, &context);
  RankedTensorType shape = RankedTensorType::get({6}, boolTy);

  // Check that we properly handle non-splat values.
  DenseElementsAttr nonSplat =
      DenseElementsAttr::get(shape, {false, false, true, false, false, true});
  EXPECT_FALSE(nonSplat.isSplat());
}

TEST(DenseSplatTest, OddIntSplat) {
  // Test detecting a splat with an odd(non 8-bit) integer bitwidth.
  MLIRContext context;
  constexpr size_t intWidth = 19;
  IntegerType intTy = IntegerType::get(intWidth, &context);
  APInt value(intWidth, 10);

  testSplat(intTy, value);
}

TEST(DenseSplatTest, Int32Splat) {
  MLIRContext context;
  IntegerType intTy = IntegerType::get(32, &context);
  int value = 64;

  testSplat(intTy, value);
}

TEST(DenseSplatTest, IntAttrSplat) {
  MLIRContext context;
  IntegerType intTy = IntegerType::get(85, &context);
  Attribute value = IntegerAttr::get(intTy, 109);

  testSplat(intTy, value);
}

TEST(DenseSplatTest, F32Splat) {
  MLIRContext context;
  FloatType floatTy = FloatType::getF32(&context);
  float value = 10.0;

  testSplat(floatTy, value);
}

TEST(DenseSplatTest, F64Splat) {
  MLIRContext context;
  FloatType floatTy = FloatType::getF64(&context);
  double value = 10.0;

  testSplat(floatTy, APFloat(value));
}

TEST(DenseSplatTest, FloatAttrSplat) {
  MLIRContext context;
  FloatType floatTy = FloatType::getF32(&context);
  Attribute value = FloatAttr::get(floatTy, 10.0);

  testSplat(floatTy, value);
}

TEST(DenseSplatTest, BF16Splat) {
  MLIRContext context;
  FloatType floatTy = FloatType::getBF16(&context);
  Attribute value = FloatAttr::get(floatTy, 10.0);

  testSplat(floatTy, value);
}

TEST(DenseSplatTest, StringSplat) {
  MLIRContext context;
  Type stringType =
      OpaqueType::get(Identifier::get("test", &context), "string", &context);
  StringRef value = "test-string";
  testSplat(stringType, value);
}

TEST(DenseSplatTest, StringAttrSplat) {
  MLIRContext context;
  Type stringType =
      OpaqueType::get(Identifier::get("test", &context), "string", &context);
  Attribute stringAttr = StringAttr::get("test-string", stringType);
  testSplat(stringType, stringAttr);
}

TEST(DenseComplexTest, ComplexFloatSplat) {
  MLIRContext context;
  ComplexType complexType = ComplexType::get(FloatType::getF32(&context));
  std::complex<float> value(10.0, 15.0);
  testSplat(complexType, value);
}

TEST(DenseComplexTest, ComplexIntSplat) {
  MLIRContext context;
  ComplexType complexType = ComplexType::get(IntegerType::get(64, &context));
  std::complex<int64_t> value(10, 15);
  testSplat(complexType, value);
}

TEST(DenseComplexTest, ComplexAPFloatSplat) {
  MLIRContext context;
  ComplexType complexType = ComplexType::get(FloatType::getF32(&context));
  std::complex<APFloat> value(APFloat(10.0f), APFloat(15.0f));
  testSplat(complexType, value);
}

TEST(DenseComplexTest, ComplexAPIntSplat) {
  MLIRContext context;
  ComplexType complexType = ComplexType::get(IntegerType::get(64, &context));
  std::complex<APInt> value(APInt(64, 10), APInt(64, 15));
  testSplat(complexType, value);
}

} // end namespace
