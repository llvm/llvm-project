//===- AttributeTest.cpp - Attribute unit tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "gtest/gtest.h"
#include <optional>

#include "../../test/lib/Dialect/Test/TestDialect.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// DenseElementsAttr
//===----------------------------------------------------------------------===//

template <typename EltTy>
static void testSplat(Type eltType, const EltTy &splatElt) {
  RankedTensorType shape = RankedTensorType::get({2, 1}, eltType);

  // Check that the generated splat is the same for 1 element and N elements.
  DenseElementsAttr splat = DenseElementsAttr::get(shape, splatElt);
  EXPECT_TRUE(splat.isSplat());

  auto detectedSplat =
      DenseElementsAttr::get(shape, llvm::ArrayRef({splatElt, splatElt}));
  EXPECT_EQ(detectedSplat, splat);

  for (auto newValue : detectedSplat.template getValues<EltTy>())
    EXPECT_TRUE(newValue == splatElt);
}

namespace {
TEST(DenseSplatTest, BoolSplat) {
  MLIRContext context;
  IntegerType boolTy = IntegerType::get(&context, 1);
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
TEST(DenseSplatTest, BoolSplatRawRoundtrip) {
  MLIRContext context;
  IntegerType boolTy = IntegerType::get(&context, 1);
  RankedTensorType shape = RankedTensorType::get({2, 2}, boolTy);

  // Check that splat booleans properly round trip via the raw API.
  DenseElementsAttr trueSplat = DenseElementsAttr::get(shape, true);
  EXPECT_TRUE(trueSplat.isSplat());
  DenseElementsAttr trueSplatFromRaw =
      DenseElementsAttr::getFromRawBuffer(shape, trueSplat.getRawData());
  EXPECT_TRUE(trueSplatFromRaw.isSplat());

  EXPECT_EQ(trueSplat, trueSplatFromRaw);
}

TEST(DenseSplatTest, BoolSplatSmall) {
  MLIRContext context;
  Builder builder(&context);

  // Check that splats that don't fill entire byte are handled properly.
  auto tensorType = RankedTensorType::get({4}, builder.getI1Type());
  std::vector<char> data{0b00001111};
  auto trueSplatFromRaw =
      DenseIntOrFPElementsAttr::getFromRawBuffer(tensorType, data);
  EXPECT_TRUE(trueSplatFromRaw.isSplat());
  DenseElementsAttr trueSplat = DenseElementsAttr::get(tensorType, true);
  EXPECT_EQ(trueSplat, trueSplatFromRaw);
}

TEST(DenseSplatTest, LargeBoolSplat) {
  constexpr int64_t boolCount = 56;

  MLIRContext context;
  IntegerType boolTy = IntegerType::get(&context, 1);
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
  IntegerType boolTy = IntegerType::get(&context, 1);
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
  IntegerType intTy = IntegerType::get(&context, intWidth);
  APInt value(intWidth, 10);

  testSplat(intTy, value);
}

TEST(DenseSplatTest, Int32Splat) {
  MLIRContext context;
  IntegerType intTy = IntegerType::get(&context, 32);
  int value = 64;

  testSplat(intTy, value);
}

TEST(DenseSplatTest, IntAttrSplat) {
  MLIRContext context;
  IntegerType intTy = IntegerType::get(&context, 85);
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
  context.allowUnregisteredDialects();
  Type stringType =
      OpaqueType::get(StringAttr::get(&context, "test"), "string");
  StringRef value = "test-string";
  testSplat(stringType, value);
}

TEST(DenseSplatTest, StringAttrSplat) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  Type stringType =
      OpaqueType::get(StringAttr::get(&context, "test"), "string");
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
  ComplexType complexType = ComplexType::get(IntegerType::get(&context, 64));
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
  ComplexType complexType = ComplexType::get(IntegerType::get(&context, 64));
  std::complex<APInt> value(APInt(64, 10), APInt(64, 15));
  testSplat(complexType, value);
}

TEST(DenseScalarTest, ExtractZeroRankElement) {
  MLIRContext context;
  const int elementValue = 12;
  IntegerType intTy = IntegerType::get(&context, 32);
  Attribute value = IntegerAttr::get(intTy, elementValue);
  RankedTensorType shape = RankedTensorType::get({}, intTy);

  auto attr = DenseElementsAttr::get(shape, llvm::ArrayRef({elementValue}));
  EXPECT_TRUE(attr.getValues<Attribute>()[0] == value);
}

TEST(DenseSplatMapValuesTest, I32ToTrue) {
  MLIRContext context;
  const int elementValue = 12;
  IntegerType boolTy = IntegerType::get(&context, 1);
  IntegerType intTy = IntegerType::get(&context, 32);
  RankedTensorType shape = RankedTensorType::get({4}, intTy);

  auto attr =
      DenseElementsAttr::get(shape, llvm::ArrayRef({elementValue}))
          .mapValues(boolTy, [](const APInt &x) {
            return x.isZero() ? APInt::getZero(1) : APInt::getAllOnes(1);
          });
  EXPECT_EQ(attr.getNumElements(), 4);
  EXPECT_TRUE(attr.isSplat());
  EXPECT_TRUE(attr.getSplatValue<BoolAttr>().getValue());
}

TEST(DenseSplatMapValuesTest, I32ToFalse) {
  MLIRContext context;
  const int elementValue = 0;
  IntegerType boolTy = IntegerType::get(&context, 1);
  IntegerType intTy = IntegerType::get(&context, 32);
  RankedTensorType shape = RankedTensorType::get({4}, intTy);

  auto attr =
      DenseElementsAttr::get(shape, llvm::ArrayRef({elementValue}))
          .mapValues(boolTy, [](const APInt &x) {
            return x.isZero() ? APInt::getZero(1) : APInt::getAllOnes(1);
          });
  EXPECT_EQ(attr.getNumElements(), 4);
  EXPECT_TRUE(attr.isSplat());
  EXPECT_FALSE(attr.getSplatValue<BoolAttr>().getValue());
}
} // namespace

//===----------------------------------------------------------------------===//
// DenseResourceElementsAttr
//===----------------------------------------------------------------------===//

template <typename AttrT, typename T>
static void checkNativeAccess(MLIRContext *ctx, ArrayRef<T> data,
                              Type elementType) {
  auto type = RankedTensorType::get(data.size(), elementType);
  auto attr = AttrT::get(type, "resource",
                         UnmanagedAsmResourceBlob::allocateInferAlign(data));

  // Check that we can access and iterate the data properly.
  std::optional<ArrayRef<T>> attrData = attr.tryGetAsArrayRef();
  EXPECT_TRUE(attrData.has_value());
  EXPECT_EQ(*attrData, data);

  // Check that we cast to this attribute when possible.
  Attribute genericAttr = attr;
  EXPECT_TRUE(isa<AttrT>(genericAttr));
}
template <typename AttrT, typename T>
static void checkNativeIntAccess(Builder &builder, size_t intWidth) {
  T data[] = {0, 1, 2};
  checkNativeAccess<AttrT, T>(builder.getContext(), llvm::ArrayRef(data),
                              builder.getIntegerType(intWidth));
}

namespace {
TEST(DenseResourceElementsAttrTest, CheckNativeAccess) {
  MLIRContext context;
  Builder builder(&context);

  // Bool
  bool boolData[] = {true, false, true};
  checkNativeAccess<DenseBoolResourceElementsAttr>(
      &context, llvm::ArrayRef(boolData), builder.getI1Type());

  // Unsigned integers
  checkNativeIntAccess<DenseUI8ResourceElementsAttr, uint8_t>(builder, 8);
  checkNativeIntAccess<DenseUI16ResourceElementsAttr, uint16_t>(builder, 16);
  checkNativeIntAccess<DenseUI32ResourceElementsAttr, uint32_t>(builder, 32);
  checkNativeIntAccess<DenseUI64ResourceElementsAttr, uint64_t>(builder, 64);

  // Signed integers
  checkNativeIntAccess<DenseI8ResourceElementsAttr, int8_t>(builder, 8);
  checkNativeIntAccess<DenseI16ResourceElementsAttr, int16_t>(builder, 16);
  checkNativeIntAccess<DenseI32ResourceElementsAttr, int32_t>(builder, 32);
  checkNativeIntAccess<DenseI64ResourceElementsAttr, int64_t>(builder, 64);

  // Float
  float floatData[] = {0, 1, 2};
  checkNativeAccess<DenseF32ResourceElementsAttr>(
      &context, llvm::ArrayRef(floatData), builder.getF32Type());

  // Double
  double doubleData[] = {0, 1, 2};
  checkNativeAccess<DenseF64ResourceElementsAttr>(
      &context, llvm::ArrayRef(doubleData), builder.getF64Type());
}

TEST(DenseResourceElementsAttrTest, CheckNoCast) {
  MLIRContext context;
  Builder builder(&context);

  // Create a i32 attribute.
  ArrayRef<uint32_t> data;
  auto type = RankedTensorType::get(data.size(), builder.getI32Type());
  Attribute i32ResourceAttr = DenseI32ResourceElementsAttr::get(
      type, "resource", UnmanagedAsmResourceBlob::allocateInferAlign(data));

  EXPECT_TRUE(isa<DenseI32ResourceElementsAttr>(i32ResourceAttr));
  EXPECT_FALSE(isa<DenseF32ResourceElementsAttr>(i32ResourceAttr));
  EXPECT_FALSE(isa<DenseBoolResourceElementsAttr>(i32ResourceAttr));
}

TEST(DenseResourceElementsAttrTest, CheckInvalidData) {
  MLIRContext context;
  Builder builder(&context);

  // Create a bool attribute with data of the incorrect type.
  ArrayRef<uint32_t> data;
  auto type = RankedTensorType::get(data.size(), builder.getI32Type());
  EXPECT_DEBUG_DEATH(
      {
        DenseBoolResourceElementsAttr::get(
            type, "resource",
            UnmanagedAsmResourceBlob::allocateInferAlign(data));
      },
      "alignment mismatch between expected alignment and blob alignment");
}

TEST(DenseResourceElementsAttrTest, CheckInvalidType) {
  MLIRContext context;
  Builder builder(&context);

  // Create a bool attribute with incorrect type.
  ArrayRef<bool> data;
  auto type = RankedTensorType::get(data.size(), builder.getI32Type());
  EXPECT_DEBUG_DEATH(
      {
        DenseBoolResourceElementsAttr::get(
            type, "resource",
            UnmanagedAsmResourceBlob::allocateInferAlign(data));
      },
      "invalid shape element type for provided type `T`");
}
} // namespace

//===----------------------------------------------------------------------===//
// SparseElementsAttr
//===----------------------------------------------------------------------===//

namespace {
TEST(SparseElementsAttrTest, GetZero) {
  MLIRContext context;
  context.allowUnregisteredDialects();

  IntegerType intTy = IntegerType::get(&context, 32);
  FloatType floatTy = FloatType::getF32(&context);
  Type stringTy = OpaqueType::get(StringAttr::get(&context, "test"), "string");

  ShapedType tensorI32 = RankedTensorType::get({2, 2}, intTy);
  ShapedType tensorF32 = RankedTensorType::get({2, 2}, floatTy);
  ShapedType tensorString = RankedTensorType::get({2, 2}, stringTy);

  auto indicesType =
      RankedTensorType::get({1, 2}, IntegerType::get(&context, 64));
  auto indices =
      DenseIntElementsAttr::get(indicesType, {APInt(64, 0), APInt(64, 0)});

  RankedTensorType intValueTy = RankedTensorType::get({1}, intTy);
  auto intValue = DenseIntElementsAttr::get(intValueTy, {1});

  RankedTensorType floatValueTy = RankedTensorType::get({1}, floatTy);
  auto floatValue = DenseFPElementsAttr::get(floatValueTy, {1.0f});

  RankedTensorType stringValueTy = RankedTensorType::get({1}, stringTy);
  auto stringValue = DenseElementsAttr::get(stringValueTy, {StringRef("foo")});

  auto sparseInt = SparseElementsAttr::get(tensorI32, indices, intValue);
  auto sparseFloat = SparseElementsAttr::get(tensorF32, indices, floatValue);
  auto sparseString =
      SparseElementsAttr::get(tensorString, indices, stringValue);

  // Only index (0, 0) contains an element, others are supposed to return
  // the zero/empty value.
  auto zeroIntValue =
      cast<IntegerAttr>(sparseInt.getValues<Attribute>()[{1, 1}]);
  EXPECT_EQ(zeroIntValue.getInt(), 0);
  EXPECT_TRUE(zeroIntValue.getType() == intTy);

  auto zeroFloatValue =
      cast<FloatAttr>(sparseFloat.getValues<Attribute>()[{1, 1}]);
  EXPECT_EQ(zeroFloatValue.getValueAsDouble(), 0.0f);
  EXPECT_TRUE(zeroFloatValue.getType() == floatTy);

  auto zeroStringValue =
      cast<StringAttr>(sparseString.getValues<Attribute>()[{1, 1}]);
  EXPECT_TRUE(zeroStringValue.empty());
  EXPECT_TRUE(zeroStringValue.getType() == stringTy);
}

//===----------------------------------------------------------------------===//
// SubElements
//===----------------------------------------------------------------------===//

TEST(SubElementTest, Nested) {
  MLIRContext context;
  Builder builder(&context);

  BoolAttr trueAttr = builder.getBoolAttr(true);
  BoolAttr falseAttr = builder.getBoolAttr(false);
  ArrayAttr boolArrayAttr =
      builder.getArrayAttr({trueAttr, falseAttr, trueAttr});
  StringAttr strAttr = builder.getStringAttr("array");
  DictionaryAttr dictAttr =
      builder.getDictionaryAttr(builder.getNamedAttr(strAttr, boolArrayAttr));

  SmallVector<Attribute> subAttrs;
  dictAttr.walk([&](Attribute attr) { subAttrs.push_back(attr); });
  // Note that trueAttr appears only once, identical subattributes are skipped.
  EXPECT_EQ(llvm::ArrayRef(subAttrs),
            ArrayRef<Attribute>(
                {strAttr, trueAttr, falseAttr, boolArrayAttr, dictAttr}));
}

// Test how many times we call copy-ctor when building an attribute.
TEST(CopyCountAttr, CopyCount) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  test::CopyCount::counter = 0;
  test::CopyCount copyCount("hello");
  test::TestCopyCountAttr::get(&context, std::move(copyCount));
  int counter1 = test::CopyCount::counter;
  test::CopyCount::counter = 0;
  test::TestCopyCountAttr::get(&context, std::move(copyCount));
#ifndef NDEBUG
  // One verification enabled only in assert-mode requires a copy.
  EXPECT_EQ(counter1, 1);
  EXPECT_EQ(test::CopyCount::counter, 1);
#else
  EXPECT_EQ(counter1, 0);
  EXPECT_EQ(test::CopyCount::counter, 0);
#endif
}

// Test stripped printing using test dialect attribute.
TEST(CopyCountAttr, PrintStripped) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();
  // Doesn't matter which dialect attribute is used, just chose TestCopyCount
  // given proximity.
  test::CopyCount::counter = 0;
  test::CopyCount copyCount("hello");
  Attribute res = test::TestCopyCountAttr::get(&context, std::move(copyCount));

  std::string str;
  llvm::raw_string_ostream os(str);
  os << "|" << res << "|";
  res.printStripped(os << "[");
  os << "]";
  EXPECT_EQ(os.str(), "|#test.copy_count<hello>|[copy_count<hello>]");
}

} // namespace
