//===- StructsGenTest.cpp - TableGen StructsGen Tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSwitch.h"
#include "gmock/gmock.h"
#include <type_traits>

namespace mlir {

/// Pull in generated enum utility declarations and definitions.
#include "StructAttrGenTest.h.inc" // NOLINT
#include "StructAttrGenTest.cpp.inc"

/// Helper that returns an example test::TestStruct for testing its
/// implementation.
static test::TestStruct getTestStruct(mlir::MLIRContext *context) {
  auto integerType = mlir::IntegerType::get(context, 32);
  auto integerAttr = mlir::IntegerAttr::get(integerType, 127);

  auto floatType = mlir::FloatType::getF32(context);
  auto floatAttr = mlir::FloatAttr::get(floatType, 0.25);

  auto elementsType = mlir::RankedTensorType::get({2, 3}, integerType);
  auto elementsAttr =
      mlir::DenseIntElementsAttr::get(elementsType, {1, 2, 3, 4, 5, 6});
  auto optionalAttr = nullptr;
  auto defaultValuedAttr = nullptr;

  return test::TestStruct::get(integerAttr, floatAttr, elementsAttr,
                               optionalAttr, defaultValuedAttr, context);
}

/// Validates that test::TestStruct::classof correctly identifies a valid
/// test::TestStruct.
TEST(StructsGenTest, ClassofTrue) {
  mlir::MLIRContext context;
  auto structAttr = getTestStruct(&context);
  ASSERT_TRUE(test::TestStruct::classof(structAttr));
}

/// Validates that test::TestStruct::classof fails when an extra attribute is in
/// the class.
TEST(StructsGenTest, ClassofExtraFalse) {
  mlir::MLIRContext context;
  mlir::DictionaryAttr structAttr = getTestStruct(&context);
  auto expectedValues = structAttr.getValue();
  ASSERT_EQ(expectedValues.size(), 3u);

  // Copy the set of named attributes.
  llvm::SmallVector<mlir::NamedAttribute, 5> newValues(expectedValues.begin(),
                                                       expectedValues.end());

  // Add an extra NamedAttribute.
  auto wrongId = mlir::StringAttr::get(&context, "wrong");
  auto wrongAttr = mlir::NamedAttribute(wrongId, expectedValues[0].getValue());
  newValues.push_back(wrongAttr);

  // Make a new DictionaryAttr and validate.
  auto badDictionary = mlir::DictionaryAttr::get(&context, newValues);
  ASSERT_FALSE(test::TestStruct::classof(badDictionary));
}

/// Validates that test::TestStruct::classof fails when a NamedAttribute has an
/// incorrect name.
TEST(StructsGenTest, ClassofBadNameFalse) {
  mlir::MLIRContext context;
  mlir::DictionaryAttr structAttr = getTestStruct(&context);
  auto expectedValues = structAttr.getValue();
  ASSERT_EQ(expectedValues.size(), 3u);

  // Create a copy of all but the first NamedAttributes.
  llvm::SmallVector<mlir::NamedAttribute, 4> newValues(
      expectedValues.begin() + 1, expectedValues.end());

  // Add a copy of the first attribute with the wrong name.
  auto wrongId = mlir::StringAttr::get(&context, "wrong");
  auto wrongAttr = mlir::NamedAttribute(wrongId, expectedValues[0].getValue());
  newValues.push_back(wrongAttr);

  auto badDictionary = mlir::DictionaryAttr::get(&context, newValues);
  ASSERT_FALSE(test::TestStruct::classof(badDictionary));
}

/// Validates that test::TestStruct::classof fails when a NamedAttribute has an
/// incorrect type.
TEST(StructsGenTest, ClassofBadTypeFalse) {
  mlir::MLIRContext context;
  mlir::DictionaryAttr structAttr = getTestStruct(&context);
  auto expectedValues = structAttr.getValue();
  ASSERT_EQ(expectedValues.size(), 3u);

  // Create a copy of all but the last NamedAttributes.
  llvm::SmallVector<mlir::NamedAttribute, 4> newValues(
      expectedValues.begin(), expectedValues.end() - 1);

  // Add a copy of the last attribute with the wrong type.
  auto i64Type = mlir::IntegerType::get(&context, 64);
  auto elementsType = mlir::RankedTensorType::get({3}, i64Type);
  auto elementsAttr =
      mlir::DenseIntElementsAttr::get(elementsType, ArrayRef<int64_t>{1, 2, 3});
  mlir::StringAttr id = expectedValues.back().getName();
  auto wrongAttr = mlir::NamedAttribute(id, elementsAttr);
  newValues.push_back(wrongAttr);

  auto badDictionary = mlir::DictionaryAttr::get(&context, newValues);
  ASSERT_FALSE(test::TestStruct::classof(badDictionary));
}

/// Validates that test::TestStruct::classof fails when a NamedAttribute is
/// missing.
TEST(StructsGenTest, ClassofMissingFalse) {
  mlir::MLIRContext context;
  mlir::DictionaryAttr structAttr = getTestStruct(&context);
  auto expectedValues = structAttr.getValue();
  ASSERT_EQ(expectedValues.size(), 3u);

  // Copy a subset of the structures Named Attributes.
  llvm::SmallVector<mlir::NamedAttribute, 3> newValues(
      expectedValues.begin() + 1, expectedValues.end());

  // Make a new DictionaryAttr and validate it is not a validate TestStruct.
  auto badDictionary = mlir::DictionaryAttr::get(&context, newValues);
  ASSERT_FALSE(test::TestStruct::classof(badDictionary));
}

/// Validate the accessor for the FloatAttr value.
TEST(StructsGenTest, GetFloat) {
  mlir::MLIRContext context;
  auto structAttr = getTestStruct(&context);
  auto returnedAttr = structAttr.sample_float();
  EXPECT_EQ(returnedAttr.getValueAsDouble(), 0.25);
}

/// Validate the accessor for the IntegerAttr value.
TEST(StructsGenTest, GetInteger) {
  mlir::MLIRContext context;
  auto structAttr = getTestStruct(&context);
  auto returnedAttr = structAttr.sample_integer();
  EXPECT_EQ(returnedAttr.getInt(), 127);
}

/// Validate the accessor for the ElementsAttr value.
TEST(StructsGenTest, GetElements) {
  mlir::MLIRContext context;
  auto structAttr = getTestStruct(&context);
  auto returnedAttr = structAttr.sample_elements();
  auto denseAttr = returnedAttr.dyn_cast<mlir::DenseElementsAttr>();
  ASSERT_TRUE(denseAttr);

  for (const auto &valIndexIt : llvm::enumerate(denseAttr.getValues<APInt>())) {
    EXPECT_EQ(valIndexIt.value(), valIndexIt.index() + 1);
  }
}

TEST(StructsGenTest, EmptyOptional) {
  mlir::MLIRContext context;
  auto structAttr = getTestStruct(&context);
  EXPECT_EQ(structAttr.sample_optional_integer(), nullptr);
}

TEST(StructsGenTest, GetDefaultValuedAttr) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  auto structAttr = getTestStruct(&context);
  EXPECT_EQ(structAttr.sample_default_valued_integer(),
            builder.getI32IntegerAttr(42));
}

} // namespace mlir
