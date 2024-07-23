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
#include "gmock/gmock.h"
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
  OpBuildGenTest()
      : ctx(getContext()), builder(&ctx), loc(builder.getUnknownLoc()),
        i32Ty(builder.getI32Type()), f32Ty(builder.getF32Type()),
        cstI32(builder.create<test::TableGenConstant>(loc, i32Ty)),
        cstF32(builder.create<test::TableGenConstant>(loc, f32Ty)),
        noAttrs(), attrStorage{builder.getNamedAttr("attr0",
                                                    builder.getBoolAttr(true)),
                               builder.getNamedAttr(
                                   "attr1", builder.getI32IntegerAttr(33))},
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

    EXPECT_EQ(op->getAttrs().size(), attrs.size());
    for (unsigned idx : llvm::seq<unsigned>(0U, attrs.size()))
      EXPECT_EQ(op->getAttr(attrs[idx].getName().strref()),
                attrs[idx].getValue());

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

    EXPECT_EQ(op->getAttrs().size(), attrs.size());
    if (op->getAttrs().size() != attrs.size()) {
      // Simple export where there is mismatch count.
      llvm::errs() << "Op attrs:\n";
      for (auto it : op->getAttrs())
        llvm::errs() << "\t" << it.getName() << " = " << it.getValue() << "\n";

      llvm::errs() << "Expected attrs:\n";
      for (auto it : attrs)
        llvm::errs() << "\t" << it.getName() << " = " << it.getValue() << "\n";
    } else {
      for (unsigned idx : llvm::seq<unsigned>(0U, attrs.size()))
        EXPECT_EQ(op->getAttr(attrs[idx].getName().strref()),
                  attrs[idx].getValue());
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
  auto op = builder.create<test::TableGenBuildOp0>(loc, i32Ty, *cstI32);
  verifyOp(op, {i32Ty}, {*cstI32}, noAttrs);

  // Test separate args, collective results build method.
  op = builder.create<test::TableGenBuildOp0>(loc, TypeRange{i32Ty}, *cstI32);
  verifyOp(op, {i32Ty}, {*cstI32}, noAttrs);

  // Test collective args, collective params build method.
  op = builder.create<test::TableGenBuildOp0>(loc, TypeRange{i32Ty},
                                              ValueRange{*cstI32});
  verifyOp(op, {i32Ty}, {*cstI32}, noAttrs);

  // Test collective args, collective results, non-empty attributes
  op = builder.create<test::TableGenBuildOp0>(loc, TypeRange{i32Ty},
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
  auto op = builder.create<test::TableGenBuildOp1>(loc, TypeRange{i32Ty},
                                                   ValueRange{*cstI32});
  verifyOp(op, {i32Ty}, {*cstI32}, noAttrs);

  // Test collective args, collective results method, building a unary op with
  // named attributes.
  op = builder.create<test::TableGenBuildOp1>(loc, TypeRange{i32Ty},
                                              ValueRange{*cstI32}, attrs);
  verifyOp(op, {i32Ty}, {*cstI32}, attrs);

  // Test collective args, collective results method, building a binary op.
  op = builder.create<test::TableGenBuildOp1>(loc, TypeRange{i32Ty, f32Ty},
                                              ValueRange{*cstI32, *cstF32});
  verifyOp(op, {i32Ty, f32Ty}, {*cstI32, *cstF32}, noAttrs);

  // Test collective args, collective results method, building a binary op with
  // named attributes.
  op = builder.create<test::TableGenBuildOp1>(
      loc, TypeRange{i32Ty, f32Ty}, ValueRange{*cstI32, *cstF32}, attrs);
  verifyOp(op, {i32Ty, f32Ty}, {*cstI32, *cstF32}, attrs);
}

/// Test build methods for an Op with a single varadic arg and a non-variadic
/// result.
TEST_F(OpBuildGenTest, BuildMethodsSingleVariadicArgNonVariadicResults) {
  // Test separate arg, separate param build method.
  auto op =
      builder.create<test::TableGenBuildOp1>(loc, i32Ty, ValueRange{*cstI32});
  verifyOp(op, {i32Ty}, {*cstI32}, noAttrs);

  // Test collective params build method, no attributes.
  op = builder.create<test::TableGenBuildOp1>(loc, TypeRange{i32Ty},
                                              ValueRange{*cstI32});
  verifyOp(op, {i32Ty}, {*cstI32}, noAttrs);

  // Test collective params build method no attributes, 2 inputs.
  op = builder.create<test::TableGenBuildOp1>(loc, TypeRange{i32Ty},
                                              ValueRange{*cstI32, *cstF32});
  verifyOp(op, {i32Ty}, {*cstI32, *cstF32}, noAttrs);

  // Test collective params build method, non-empty attributes.
  op = builder.create<test::TableGenBuildOp1>(
      loc, TypeRange{i32Ty}, ValueRange{*cstI32, *cstF32}, attrs);
  verifyOp(op, {i32Ty}, {*cstI32, *cstF32}, attrs);
}

/// Test build methods for an Op with a single varadic arg and multiple variadic
/// result.
TEST_F(OpBuildGenTest,
       BuildMethodsSingleVariadicArgAndMultipleVariadicResults) {
  // Test separate arg, separate param build method.
  auto op = builder.create<test::TableGenBuildOp3>(
      loc, TypeRange{i32Ty}, TypeRange{f32Ty}, ValueRange{*cstI32});
  verifyOp(op, {i32Ty, f32Ty}, {*cstI32}, noAttrs);

  // Test collective params build method, no attributes.
  op = builder.create<test::TableGenBuildOp3>(loc, TypeRange{i32Ty, f32Ty},
                                              ValueRange{*cstI32});
  verifyOp(op, {i32Ty, f32Ty}, {*cstI32}, noAttrs);

  // Test collective params build method, with attributes.
  op = builder.create<test::TableGenBuildOp3>(loc, TypeRange{i32Ty, f32Ty},
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
  auto op = builder.create<test::TableGenBuildOp4>(
      loc, i32Ty, ValueRange{*cstI32, *cstI32});
  verifyOp(std::move(op), {i32Ty}, {*cstI32, *cstI32}, noAttrs);

  // Test collective params build method.
  op = builder.create<test::TableGenBuildOp4>(loc, TypeRange{i32Ty},
                                              ValueRange{*cstI32, *cstI32});
  verifyOp(std::move(op), {i32Ty}, {*cstI32, *cstI32}, noAttrs);

  // Test build method with no result types, default value of attributes.
  op =
      builder.create<test::TableGenBuildOp4>(loc, ValueRange{*cstI32, *cstI32});
  verifyOp(std::move(op), {i32Ty}, {*cstI32, *cstI32}, noAttrs);

  // Test build method with no result types and supplied attributes.
  op = builder.create<test::TableGenBuildOp4>(loc, ValueRange{*cstI32, *cstI32},
                                              attrs);
  verifyOp(std::move(op), {i32Ty}, {*cstI32, *cstI32}, attrs);
}

TEST_F(OpBuildGenTest, BuildMethodsRegionsAndInferredType) {
  auto op = builder.create<test::TableGenBuildOp5>(
      loc, ValueRange{*cstI32, *cstF32}, /*attributes=*/noAttrs);
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
  auto op = builder.create<test::TableGenBuildOp6>(
      loc, f32Ty, ValueRange{*cstI32}, ValueRange{*cstI32});
  verifyOp(std::move(op), {f32Ty}, {*cstI32}, {*cstI32}, noAttrs);

  // Test build method with no result types, default value of attributes.
  op = builder.create<test::TableGenBuildOp6>(loc, ValueRange{*cstI32},
                                              ValueRange{*cstI32});
  verifyOp(std::move(op), {f32Ty}, {*cstI32}, {*cstI32}, noAttrs);

  // Test collective params build method.
  op = builder.create<test::TableGenBuildOp6>(
      loc, TypeRange{f32Ty}, ValueRange{*cstI32}, ValueRange{*cstI32});
  verifyOp(std::move(op), {f32Ty}, {*cstI32}, {*cstI32}, noAttrs);

  // Test build method with result types, supplied attributes.
  op = builder.create<test::TableGenBuildOp6>(
      loc, TypeRange{f32Ty}, ValueRange{*cstI32, *cstI32}, attrs);
  verifyOp(std::move(op), {f32Ty}, {*cstI32}, {*cstI32}, attrs);

  // Test build method with no result types and supplied attributes.
  op = builder.create<test::TableGenBuildOp6>(loc, ValueRange{*cstI32, *cstI32},
                                              attrs);
  verifyOp(std::move(op), {f32Ty}, {*cstI32}, {*cstI32}, attrs);
}

} // namespace mlir
