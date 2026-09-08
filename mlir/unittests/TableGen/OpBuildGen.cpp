//===- OpBuildGen.cpp - TableGen OpBuildGen Tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test TableGen generated build() methods on Operations.
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "llvm/Support/Compiler.h"
#include "gmock/gmock.h"
#include <array>
#include <vector>

namespace mlir {

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

static MLIRContext &getContext() {
  static MLIRContext ctx;
  ctx.getOrLoadDialect<test::TestDialect>();
  return ctx;
}
/// Test fixture for providing basic utilities for testing.
class OpBuildGenTest : public ::testing::Test {
protected:
  static NamedAttrList collectAttrs(Operation *op) {
    NamedAttrList attrs(op->getDiscardableAttrDictionary());
    if (op->getPropertiesStorageSize())
      op->getName().walkInherentAttrs(op, [&](StringRef name, Attribute &attr) {
        attrs.append(name, attr);
      });
    return NamedAttrList(attrs.getDictionary(op->getContext()));
  }

  OpBuildGenTest()
      : ctx(getContext()), builder(&ctx), loc(builder.getUnknownLoc()),
        i32Ty(builder.getI32Type()), f32Ty(builder.getF32Type()),
        cstI32(test::TableGenConstant::create(builder, loc, i32Ty)),
        cstF32(test::TableGenConstant::create(builder, loc, f32Ty)), noAttrs(),
        attrStorage{
            builder.getNamedAttr("attr0", builder.getBoolAttr(true)),
            builder.getNamedAttr("attr1", builder.getI32IntegerAttr(33))},
        attrs(attrStorage) {}

  // Verify that `op` has the given set of result types, operands, and
  // attributes.
  template <typename OpTy>
  void verifyOp(OpTy &&concreteOp, std::vector<Type> resultTypes,
                std::vector<Value> operands,
                std::vector<NamedAttribute> attrs) {
    ASSERT_NE(concreteOp, nullptr);
    Operation *op = concreteOp.getOperation();

    EXPECT_EQ(op->getNumResults(), resultTypes.size());
    for (unsigned idx : llvm::seq(0U, op->getNumResults()))
      EXPECT_EQ(op->getResult(idx).getType(), resultTypes[idx]);

    EXPECT_EQ(op->getNumOperands(), operands.size());
    for (unsigned idx : llvm::seq(0U, op->getNumOperands()))
      EXPECT_EQ(op->getOperand(idx), operands[idx]);

    NamedAttrList actualAttrs = collectAttrs(op);
    EXPECT_EQ(actualAttrs.getAttrs().size(), attrs.size());
    for (unsigned idx : llvm::seq<unsigned>(0U, attrs.size()))
      EXPECT_EQ(actualAttrs.get(attrs[idx].getName()), attrs[idx].getValue());

    EXPECT_TRUE(mlir::succeeded(concreteOp.verify()));
    concreteOp.erase();
  }

  template <typename OpTy>
  void verifyOp(OpTy &&concreteOp, std::vector<Type> resultTypes,
                std::vector<Value> operands1, std::vector<Value> operands2,
                std::vector<NamedAttribute> attrs) {
    ASSERT_NE(concreteOp, nullptr);
    Operation *op = concreteOp.getOperation();

    EXPECT_EQ(op->getNumResults(), resultTypes.size());
    for (unsigned idx : llvm::seq(0U, op->getNumResults()))
      EXPECT_EQ(op->getResult(idx).getType(), resultTypes[idx]);

    auto operands = llvm::to_vector(llvm::concat<Value>(operands1, operands2));
    EXPECT_EQ(op->getNumOperands(), operands.size());
    for (unsigned idx : llvm::seq(0U, op->getNumOperands()))
      EXPECT_EQ(op->getOperand(idx), operands[idx]);

    NamedAttrList actualAttrs = collectAttrs(op);
    EXPECT_EQ(actualAttrs.getAttrs().size(), attrs.size());
    if (actualAttrs.getAttrs().size() != attrs.size()) {
      // Simple export where there is mismatch count.
      llvm::errs() << "Op attrs:\n";
      for (auto it : actualAttrs)
        llvm::errs() << "\t" << it.getName() << " = " << it.getValue() << "\n";

      llvm::errs() << "Expected attrs:\n";
      for (auto it : attrs)
        llvm::errs() << "\t" << it.getName() << " = " << it.getValue() << "\n";
    } else {
      for (unsigned idx : llvm::seq<unsigned>(0U, attrs.size()))
        EXPECT_EQ(actualAttrs.get(attrs[idx].getName()), attrs[idx].getValue());
    }

    EXPECT_TRUE(mlir::succeeded(concreteOp.verify()));
    concreteOp.erase();
  }

protected:
  MLIRContext &ctx;
  OpBuilder builder;
  Location loc;
  Type i32Ty;
  Type f32Ty;
  OwningOpRef<test::TableGenConstant> cstI32;
  OwningOpRef<test::TableGenConstant> cstF32;

  ArrayRef<NamedAttribute> noAttrs;
  std::vector<NamedAttribute> attrStorage;
  ArrayRef<NamedAttribute> attrs;
};

/// Test basic build methods.
TEST_F(OpBuildGenTest, BasicBuildMethods) {
  // Test separate args, separate results build method.
  auto op = test::TableGenBuildOp0::create(builder, loc, i32Ty, *cstI32);
  verifyOp(op, {i32Ty}, {*cstI32}, noAttrs);

  // Test separate args, collective results build method.
  op = test::TableGenBuildOp0::create(builder, loc, TypeRange{i32Ty}, *cstI32);
  verifyOp(op, {i32Ty}, {*cstI32}, noAttrs);

  // Test collective args, collective params build method.
  op = test::TableGenBuildOp0::create(builder, loc, TypeRange{i32Ty},
                                      ValueRange{*cstI32});
  verifyOp(op, {i32Ty}, {*cstI32}, noAttrs);

  // Test collective args, collective results, non-empty attributes
  op = test::TableGenBuildOp0::create(builder, loc, TypeRange{i32Ty},
                                      ValueRange{*cstI32}, attrs);
  verifyOp(op, {i32Ty}, {*cstI32}, attrs);
}

/// The following 3 tests exercise build methods generated for operations
/// with a combination of:
///
/// single variadic arg x
/// {single variadic result, non-variadic result, multiple variadic results}
///
/// Specifically to test that ODS framework does not generate ambiguous
/// build() methods that fail to compile.

/// Test build methods for an Op with a single varadic arg and a single
/// variadic result.
TEST_F(OpBuildGenTest, BuildMethodsSingleVariadicArgAndResult) {
  // Test collective args, collective results method, building a unary op.
  auto op = test::TableGenBuildOp1::create(builder, loc, TypeRange{i32Ty},
                                           ValueRange{*cstI32});
  verifyOp(op, {i32Ty}, {*cstI32}, noAttrs);

  // Test collective args, collective results method, building a unary op with
  // named attributes.
  op = test::TableGenBuildOp1::create(builder, loc, TypeRange{i32Ty},
                                      ValueRange{*cstI32}, attrs);
  verifyOp(op, {i32Ty}, {*cstI32}, attrs);

  // Test collective args, collective results method, building a binary op.
  op = test::TableGenBuildOp1::create(builder, loc, TypeRange{i32Ty, f32Ty},
                                      ValueRange{*cstI32, *cstF32});
  verifyOp(op, {i32Ty, f32Ty}, {*cstI32, *cstF32}, noAttrs);

  // Test collective args, collective results method, building a binary op with
  // named attributes.
  op = test::TableGenBuildOp1::create(builder, loc, TypeRange{i32Ty, f32Ty},
                                      ValueRange{*cstI32, *cstF32}, attrs);
  verifyOp(op, {i32Ty, f32Ty}, {*cstI32, *cstF32}, attrs);
}

/// Test build methods for an Op with a single varadic arg and a non-variadic
/// result.
TEST_F(OpBuildGenTest, BuildMethodsSingleVariadicArgNonVariadicResults) {
  // Test separate arg, separate param build method.
  auto op =
      test::TableGenBuildOp1::create(builder, loc, i32Ty, ValueRange{*cstI32});
  verifyOp(op, {i32Ty}, {*cstI32}, noAttrs);

  // Test collective params build method, no attributes.
  op = test::TableGenBuildOp1::create(builder, loc, TypeRange{i32Ty},
                                      ValueRange{*cstI32});
  verifyOp(op, {i32Ty}, {*cstI32}, noAttrs);

  // Test collective params build method no attributes, 2 inputs.
  op = test::TableGenBuildOp1::create(builder, loc, TypeRange{i32Ty},
                                      ValueRange{*cstI32, *cstF32});
  verifyOp(op, {i32Ty}, {*cstI32, *cstF32}, noAttrs);

  // Test collective params build method, non-empty attributes.
  op = test::TableGenBuildOp1::create(builder, loc, TypeRange{i32Ty},
                                      ValueRange{*cstI32, *cstF32}, attrs);
  verifyOp(op, {i32Ty}, {*cstI32, *cstF32}, attrs);
}

/// Test build methods for an Op with a single varadic arg and multiple variadic
/// result.
TEST_F(OpBuildGenTest,
       BuildMethodsSingleVariadicArgAndMultipleVariadicResults) {
  // Test separate arg, separate param build method.
  auto op = test::TableGenBuildOp3::create(
      builder, loc, TypeRange{i32Ty}, TypeRange{f32Ty}, ValueRange{*cstI32});
  verifyOp(op, {i32Ty, f32Ty}, {*cstI32}, noAttrs);

  // Test collective params build method, no attributes.
  op = test::TableGenBuildOp3::create(builder, loc, TypeRange{i32Ty, f32Ty},
                                      ValueRange{*cstI32});
  verifyOp(op, {i32Ty, f32Ty}, {*cstI32}, noAttrs);

  // Test collective params build method, with attributes.
  op = test::TableGenBuildOp3::create(builder, loc, TypeRange{i32Ty, f32Ty},
                                      ValueRange{*cstI32}, attrs);
  verifyOp(op, {i32Ty, f32Ty}, {*cstI32}, attrs);
}

// The next test checks suppression of ambiguous build methods for ops that
// have a single variadic input, and single non-variadic result, and which
// support the SameOperandsAndResultType trait and optionally the
// InferOpTypeInterface interface. For such ops, the ODS framework generates
// build methods with no result types as they are inferred from the input types.
TEST_F(OpBuildGenTest, BuildMethodsSameOperandsAndResultTypeSuppression) {
  // Test separate arg, separate param build method.
  auto op = test::TableGenBuildOp4::create(builder, loc, i32Ty,
                                           ValueRange{*cstI32, *cstI32});
  verifyOp(std::move(op), {i32Ty}, {*cstI32, *cstI32}, noAttrs);

  // Test collective params build method.
  op = test::TableGenBuildOp4::create(builder, loc, TypeRange{i32Ty},
                                      ValueRange{*cstI32, *cstI32});
  verifyOp(std::move(op), {i32Ty}, {*cstI32, *cstI32}, noAttrs);

  // Test build method with no result types, default value of attributes.
  op = test::TableGenBuildOp4::create(builder, loc,
                                      ValueRange{*cstI32, *cstI32});
  verifyOp(std::move(op), {i32Ty}, {*cstI32, *cstI32}, noAttrs);

  // Test build method with no result types and supplied attributes.
  op = test::TableGenBuildOp4::create(builder, loc,
                                      ValueRange{*cstI32, *cstI32}, attrs);
  verifyOp(std::move(op), {i32Ty}, {*cstI32, *cstI32}, attrs);
}

TEST_F(OpBuildGenTest, BuildMethodsRegionsAndInferredType) {
  auto op = test::TableGenBuildOp5::create(
      builder, loc, ValueRange{*cstI32, *cstF32}, /*attributes=*/noAttrs);
  ASSERT_EQ(op->getNumRegions(), 1u);
  verifyOp(op, {i32Ty}, {*cstI32, *cstF32}, noAttrs);
}

TEST_F(OpBuildGenTest, BuildMethodsVariadicProperties) {
  // Account for conversion as part of getAttrs().
  std::vector<NamedAttribute> noAttrsStorage;
  auto segmentSize = builder.getNamedAttr("operandSegmentSizes",
                                          builder.getDenseI32ArrayAttr({1, 1}));
  noAttrsStorage.push_back(segmentSize);
  ArrayRef<NamedAttribute> noAttrs(noAttrsStorage);
  std::vector<NamedAttribute> attrsStorage = this->attrStorage;
  attrsStorage.push_back(segmentSize);
  ArrayRef<NamedAttribute> attrs(attrsStorage);

  // Test separate arg, separate param build method.
  auto op = test::TableGenBuildOp6::create(
      builder, loc, f32Ty, ValueRange{*cstI32}, ValueRange{*cstI32});
  verifyOp(std::move(op), {f32Ty}, {*cstI32}, {*cstI32}, noAttrs);

  // Test build method with no result types, default value of attributes.
  op = test::TableGenBuildOp6::create(builder, loc, ValueRange{*cstI32},
                                      ValueRange{*cstI32});
  verifyOp(std::move(op), {f32Ty}, {*cstI32}, {*cstI32}, noAttrs);

  // Test collective params build method.
  op = test::TableGenBuildOp6::create(builder, loc, TypeRange{f32Ty},
                                      ValueRange{*cstI32}, ValueRange{*cstI32});
  verifyOp(std::move(op), {f32Ty}, {*cstI32}, {*cstI32}, noAttrs);

  // Test build method with result types, supplied attributes.
  op = test::TableGenBuildOp6::create(builder, loc, TypeRange{f32Ty},
                                      ValueRange{*cstI32, *cstI32}, attrs);
  verifyOp(std::move(op), {f32Ty}, {*cstI32}, {*cstI32}, attrs);

  // Test build method with no result types and supplied attributes.
  op = test::TableGenBuildOp6::create(builder, loc,
                                      ValueRange{*cstI32, *cstI32}, attrs);
  verifyOp(std::move(op), {f32Ty}, {*cstI32}, {*cstI32}, attrs);

  // Test replacing an inherent attribute backed by a native property.
  op = test::TableGenBuildOp6::create(builder, loc, f32Ty, ValueRange{*cstI32},
                                      ValueRange{*cstI32});
  DenseI32ArrayAttr replacement = builder.getDenseI32ArrayAttr({0, 2});
  op->getName().walkInherentAttrs(op, [&](StringRef name, Attribute &attr) {
    if (name == "operandSegmentSizes")
      attr = replacement;
  });
  EXPECT_EQ(op.getProperties().operandSegmentSizes[0], 0);
  EXPECT_EQ(op.getProperties().operandSegmentSizes[1], 2);
  op.erase();
}

TEST_F(OpBuildGenTest, BuildMethodsInherentDiscardableAttrs) {
  test::TableGenBuildOp7::Properties props;
  props.attr0 = cast<BoolAttr>(attrs[0].getValue());
  ArrayRef<NamedAttribute> discardableAttrs = attrs.drop_front();
  auto op7 = test::TableGenBuildOp7::create(
      builder, loc, TypeRange{}, ValueRange{}, props, discardableAttrs);
  unsigned numInherentAttrs = 0;
  BoolAttr replacement = builder.getBoolAttr(false);
  op7->getName().walkInherentAttrs(op7, [&](StringRef name, Attribute &attr) {
    EXPECT_EQ(name, attrs[0].getName());
    EXPECT_EQ(attr, attrs[0].getValue());
    attr = replacement;
    ++numInherentAttrs;
  });
  EXPECT_EQ(numInherentAttrs, 1u);
  EXPECT_EQ(op7.getProperties().getAttr0(), replacement);
  std::vector<NamedAttribute> replacedAttrs(attrs.begin(), attrs.end());
  replacedAttrs[0].setValue(replacement);
  verifyOp(op7, {}, {}, replacedAttrs);

  // Check that the old-style builder partitions the attributes and populates
  // properties before Operation::create.
  OperationState state(loc, test::TableGenBuildOp7::getOperationName());
  test::TableGenBuildOp7::build(builder, state, TypeRange{}, ValueRange{},
                                attrs);
  ASSERT_TRUE(state.getRawProperties());
  EXPECT_EQ(state.attributes.getAttrs().size(), 1u);
  EXPECT_EQ(state.attributes.getAttrs()[0], attrs[1]);
  EXPECT_EQ(
      state.getOrAddProperties<test::TableGenBuildOp7::Properties>().getAttr0(),
      attrs[0].getValue());

  auto op7FromState = cast<test::TableGenBuildOp7>(builder.create(state));
  verifyOp(op7FromState, {}, {}, attrs);

  // Check that the legacy create forwarder remains compatible.
  auto op7b = test::TableGenBuildOp7::create(builder, loc, TypeRange{},
                                             ValueRange{}, attrs);
  // Note: this goes before verifyOp() because verifyOp() calls erase(), causing
  // use-after-free.
  ASSERT_EQ(op7b.getProperties().getAttr0(), attrs[0].getValue());
  verifyOp(op7b, {}, {}, attrs);
}

TEST_F(OpBuildGenTest, BuildMethodsLegacyMixedProperties) {
  SmallVector<NamedAttribute> mixedAttrs{
      builder.getNamedAttr("attr0", builder.getBoolAttr(true)),
      builder.getNamedAttr("nativeProp", builder.getI64IntegerAttr(42)),
      builder.getNamedAttr("operand_segment_sizes",
                           builder.getDenseI32ArrayAttr({1, 1})),
      builder.getNamedAttr("result_segment_sizes",
                           builder.getDenseI32ArrayAttr({1, 0})),
      builder.getNamedAttr("unknown", builder.getStringAttr("discardable"))};
  OperationState state(loc, test::TableGenBuildOp8::getOperationName());

  test::TableGenBuildOp8::build(builder, state, ValueRange{*cstI32, *cstF32},
                                mixedAttrs);

  ASSERT_TRUE(state.getRawProperties());
  ASSERT_EQ(state.attributes.getAttrs().size(), 1u);
  EXPECT_EQ(state.attributes.getAttrs()[0], mixedAttrs.back());
  ASSERT_EQ(state.types.size(), 1u);
  EXPECT_EQ(state.types[0], i32Ty);
  const auto &properties =
      state.getOrAddProperties<test::TableGenBuildOp8::Properties>();
  EXPECT_TRUE(properties.attr0.getValue());
  EXPECT_EQ(properties.defaultAttr.getInt(), 7);
  EXPECT_EQ(properties.nativeProp, 42);
  EXPECT_EQ(properties.operandSegmentSizes, (std::array<int32_t, 2>{1, 1}));
  EXPECT_EQ(properties.resultSegmentSizes, (std::array<int32_t, 2>{1, 0}));

  auto op = cast<test::TableGenBuildOp8>(builder.create(state));
  EXPECT_EQ(op->getDiscardableAttrDictionary().size(), 1u);
  EXPECT_EQ(op.getNativeProp(), 42);
  EXPECT_EQ(op.getDefaultAttr(), 7u);
  EXPECT_EQ(op->getResult(0).getType(), i32Ty);
  EXPECT_TRUE(succeeded(op.verify()));
  op.erase();
}

TEST_F(OpBuildGenTest, BuildMethodsLegacyDefaultsWithoutAttributes) {
  OperationState state(loc, test::TableGenBuildOp8::getOperationName());
  test::TableGenBuildOp8::build(builder, state, TypeRange{i32Ty},
                                ValueRange{*cstI32, *cstF32},
                                ArrayRef<NamedAttribute>{});

  ASSERT_TRUE(state.getRawProperties());
  const auto &properties =
      state.getOrAddProperties<test::TableGenBuildOp8::Properties>();
  EXPECT_EQ(properties.defaultAttr.getInt(), 7);
}

TEST_F(OpBuildGenTest, BuildMethodsLegacySameOperandAndResultType) {
  SmallVector<NamedAttribute> mixedAttrs{
      builder.getNamedAttr("attr0", builder.getBoolAttr(true)),
      builder.getNamedAttr("unknown", builder.getUnitAttr())};
  auto op = test::TableGenBuildOp9::create(builder, loc, ValueRange{*cstI32},
                                           mixedAttrs);
  EXPECT_EQ(op.getResult().getType(), i32Ty);
  EXPECT_TRUE(op.getAttr0());
  EXPECT_EQ(op->getDiscardableAttrDictionary().size(), 1u);
  EXPECT_TRUE(succeeded(op.verify()));
  op.erase();
}

TEST_F(OpBuildGenTest, BuildMethodsLegacyFirstAttrDerivedResultType) {
  SmallVector<NamedAttribute> mixedAttrs{
      builder.getNamedAttr("type", TypeAttr::get(f32Ty)),
      builder.getNamedAttr("unknown", builder.getUnitAttr())};
  auto op = test::TableGenBuildOp10::create(builder, loc, ValueRange{*cstI32},
                                            mixedAttrs);
  EXPECT_EQ(op.getResult().getType(), f32Ty);
  EXPECT_EQ(op.getType(), f32Ty);
  EXPECT_EQ(op->getDiscardableAttrDictionary().size(), 1u);
  EXPECT_TRUE(succeeded(op.verify()));
  op.erase();
}

TEST_F(OpBuildGenTest, BuildMethodsEmptyPropertiesKeepMixedAttributes) {
  OperationState state(loc, test::TableGenBuildOp0::getOperationName());
  test::TableGenBuildOp0::build(builder, state, TypeRange{i32Ty},
                                ValueRange{*cstI32}, attrs);
  EXPECT_FALSE(state.getRawProperties());
  EXPECT_EQ(state.attributes.getAttrs(), attrs);
}

TEST_F(OpBuildGenTest, BuildMethodsInvalidLegacyPropertyConversion) {
  SmallVector<NamedAttribute> badAttrs{
      builder.getNamedAttr("attr0", builder.getStringAttr("not-a-bool"))};
  OperationState state(loc, test::TableGenBuildOp7::getOperationName());
  EXPECT_DEATH_IF_SUPPORTED(
      test::TableGenBuildOp7::build(builder, state, TypeRange{}, ValueRange{},
                                    badAttrs),
      "Invalid attribute.*attr0");
}

} // namespace mlir
