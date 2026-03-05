//===- mlir/unittest/IR/VerifierTest.cpp - Verifier unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Verifier.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {

TEST(VerifierTest, CrossContextResultType) {
  MLIRContext ctxA, ctxB;
  ctxA.allowUnregisteredDialects();
  ctxB.allowUnregisteredDialects();

  // Result type comes from ctxB but the op lives in ctxA.
  Type i32B = IntegerType::get(&ctxB, 32);
  Operation *op =
      Operation::create(UnknownLoc::get(&ctxA), OperationName("foo.bar", &ctxA),
                        {i32B}, {}, NamedAttrList(), nullptr, {}, 0);

  ScopedDiagnosticHandler suppress(&ctxA,
                                   [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verify(op)));
  op->destroy();
}

TEST(VerifierTest, CrossContextDiscardableAttr) {
  MLIRContext ctxA, ctxB;
  ctxA.allowUnregisteredDialects();
  ctxB.allowUnregisteredDialects();

  // Attribute value comes from ctxB but the op lives in ctxA.
  IntegerAttr attrB = IntegerAttr::get(IntegerType::get(&ctxB, 32), 42);
  NamedAttrList attrs;
  attrs.append(StringAttr::get(&ctxA, "my_attr"), attrB);

  Operation *op =
      Operation::create(UnknownLoc::get(&ctxA), OperationName("foo.bar", &ctxA),
                        {}, {}, std::move(attrs), nullptr, {}, 0);

  ScopedDiagnosticHandler suppress(&ctxA,
                                   [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verify(op)));
  op->destroy();
}

TEST(VerifierTest, CrossContextOperand) {
  MLIRContext ctxA, ctxB;
  ctxA.allowUnregisteredDialects();
  ctxB.allowUnregisteredDialects();

  // Producer op lives in ctxB; its result type is also from ctxB.
  Type i32B = IntegerType::get(&ctxB, 32);
  Operation *producerOp = Operation::create(
      UnknownLoc::get(&ctxB), OperationName("foo.producer", &ctxB), {i32B}, {},
      NamedAttrList(), nullptr, {}, 0);
  Value valFromCtxB = producerOp->getResult(0);

  // Consumer op lives in ctxA but uses the value (whose type is in ctxB).
  Operation *consumerOp = Operation::create(
      UnknownLoc::get(&ctxA), OperationName("foo.consumer", &ctxA), {},
      {valFromCtxB}, NamedAttrList(), nullptr, {}, 0);

  ScopedDiagnosticHandler suppress(&ctxA,
                                   [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verify(consumerOp)));
  consumerOp->destroy();
  producerOp->destroy();
}

TEST(VerifierTest, CrossContextOperationName) {
  MLIRContext ctxA, ctxB;
  ctxA.allowUnregisteredDialects();
  ctxB.allowUnregisteredDialects();

  // Location (and thus op.getContext()) is from ctxA; OperationName is from
  // ctxB.  op.getContext() == op.getLoc().getContext() by definition, so the
  // OperationName is the independent source of a cross-context violation here.
  Operation *op =
      Operation::create(UnknownLoc::get(&ctxA), OperationName("foo.bar", &ctxB),
                        {}, {}, NamedAttrList(), nullptr, {}, 0);

  ScopedDiagnosticHandler suppress(&ctxA,
                                   [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verify(op)));
  op->destroy();
}

TEST(VerifierTest, CrossContextOperationLocation) {
  MLIRContext ctxA, ctxB;
  ctxA.allowUnregisteredDialects();
  ctxB.allowUnregisteredDialects();

  // Create an outer op in ctxA with one region.
  Operation *outerOp = Operation::create(
      UnknownLoc::get(&ctxA), OperationName("foo.outer", &ctxA), {}, {},
      NamedAttrList(), nullptr, {}, /*numRegions=*/1);

  Block *block = new Block();
  outerOp->getRegion(0).push_back(block);

  // Create an inner op whose location (and thus context) comes from ctxB, and
  // place it inside the ctxA block.  This is the cross-context violation:
  // op.getContext() == ctxB but the enclosing block belongs to ctxA.
  Operation *innerOp = Operation::create(UnknownLoc::get(&ctxB),
                                         OperationName("foo.inner", &ctxB), {},
                                         {}, NamedAttrList(), nullptr, {}, 0);
  block->push_back(innerOp);

  // The verifier emits via the parent op's location (ctxA), so only ctxA's
  // handler needs to suppress it.
  ScopedDiagnosticHandler suppress(&ctxA,
                                   [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verify(outerOp)));
  outerOp->destroy();
}

TEST(VerifierTest, CrossContextBlockArgType) {
  MLIRContext ctxA, ctxB;
  ctxA.allowUnregisteredDialects();
  ctxB.allowUnregisteredDialects();

  // Create an unregistered op in ctxA with one region.
  Operation *op =
      Operation::create(UnknownLoc::get(&ctxA), OperationName("foo.bar", &ctxA),
                        {}, {}, NamedAttrList(), nullptr, {}, /*numRegions=*/1);

  // Add a block with one argument whose type comes from ctxB.
  Block *block = new Block();
  op->getRegion(0).push_back(block);
  Type i32B = IntegerType::get(&ctxB, 32);
  block->addArgument(i32B, UnknownLoc::get(&ctxA));

  ScopedDiagnosticHandler suppress(&ctxA,
                                   [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verify(op)));
  op->destroy();
}

TEST(VerifierTest, CrossContextBlockArgLocation) {
  MLIRContext ctxA, ctxB;
  ctxA.allowUnregisteredDialects();
  ctxB.allowUnregisteredDialects();

  // Create an unregistered op in ctxA with one region.
  Operation *op =
      Operation::create(UnknownLoc::get(&ctxA), OperationName("foo.bar", &ctxA),
                        {}, {}, NamedAttrList(), nullptr, {}, /*numRegions=*/1);

  // Add a block with one argument whose location comes from ctxB.
  Block *block = new Block();
  op->getRegion(0).push_back(block);
  Type i32A = IntegerType::get(&ctxA, 32);
  block->addArgument(i32A, UnknownLoc::get(&ctxB));

  // The verifier emits the diagnostic via the parent op's location (ctxA).
  ScopedDiagnosticHandler suppress(&ctxA,
                                   [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verify(op)));
  op->destroy();
}

} // namespace
