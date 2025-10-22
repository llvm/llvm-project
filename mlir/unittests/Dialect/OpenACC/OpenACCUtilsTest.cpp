//===- OpenACCUtilsTest.cpp - Unit tests for OpenACC utilities -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class OpenACCUtilsTest : public ::testing::Test {
protected:
  OpenACCUtilsTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<acc::OpenACCDialect, arith::ArithDialect,
                        memref::MemRefDialect, func::FuncDialect>();
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
};

//===----------------------------------------------------------------------===//
// getEnclosingComputeOp Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsTest, getEnclosingComputeOpParallel) {
  // Create a parallel op with a region
  OwningOpRef<ParallelOp> parallelOp =
      ParallelOp::create(b, loc, TypeRange{}, ValueRange{});
  Region &parallelRegion = parallelOp->getRegion();
  parallelRegion.emplaceBlock();

  // Test that we can find the parallel op from its region
  Operation *enclosingOp = getEnclosingComputeOp(parallelRegion);
  EXPECT_EQ(enclosingOp, parallelOp.get());
}

TEST_F(OpenACCUtilsTest, getEnclosingComputeOpKernels) {
  // Create a kernels op with a region
  OwningOpRef<KernelsOp> kernelsOp =
      KernelsOp::create(b, loc, TypeRange{}, ValueRange{});
  Region &kernelsRegion = kernelsOp->getRegion();
  kernelsRegion.emplaceBlock();

  // Test that we can find the kernels op from its region
  Operation *enclosingOp = getEnclosingComputeOp(kernelsRegion);
  EXPECT_EQ(enclosingOp, kernelsOp.get());
}

TEST_F(OpenACCUtilsTest, getEnclosingComputeOpSerial) {
  // Create a serial op with a region
  OwningOpRef<SerialOp> serialOp =
      SerialOp::create(b, loc, TypeRange{}, ValueRange{});
  Region &serialRegion = serialOp->getRegion();
  serialRegion.emplaceBlock();

  // Test that we can find the serial op from its region
  Operation *enclosingOp = getEnclosingComputeOp(serialRegion);
  EXPECT_EQ(enclosingOp, serialOp.get());
}

TEST_F(OpenACCUtilsTest, getEnclosingComputeOpNested) {
  // Create nested ops: parallel containing a loop op
  OwningOpRef<ParallelOp> parallelOp =
      ParallelOp::create(b, loc, TypeRange{}, ValueRange{});
  Region &parallelRegion = parallelOp->getRegion();
  Block *parallelBlock = &parallelRegion.emplaceBlock();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(parallelBlock);

  // Create a loop op inside the parallel region
  OwningOpRef<LoopOp> loopOp =
      LoopOp::create(b, loc, TypeRange{}, ValueRange{});
  Region &loopRegion = loopOp->getRegion();
  loopRegion.emplaceBlock();

  // Test that from the loop region, we find the parallel op (loop is not a
  // compute op)
  Operation *enclosingOp = getEnclosingComputeOp(loopRegion);
  EXPECT_EQ(enclosingOp, parallelOp.get());
}

TEST_F(OpenACCUtilsTest, getEnclosingComputeOpNone) {
  // Create a module with a region that's not inside a compute construct
  OwningOpRef<ModuleOp> moduleOp = ModuleOp::create(loc);
  Region &moduleRegion = moduleOp->getBodyRegion();

  // Test that we get nullptr when there's no enclosing compute op
  Operation *enclosingOp = getEnclosingComputeOp(moduleRegion);
  EXPECT_EQ(enclosingOp, nullptr);
}

//===----------------------------------------------------------------------===//
// isOnlyUsedByPrivateClauses Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsTest, isOnlyUsedByPrivateClausesTrue) {
  // Create a value (memref) outside the compute region
  auto memrefTy = MemRefType::get({10}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, memrefTy);
  TypedValue<PointerLikeType> varPtr =
      cast<TypedValue<PointerLikeType>>(allocOp->getResult());

  // Create a parallel op with a region
  OwningOpRef<ParallelOp> parallelOp =
      ParallelOp::create(b, loc, TypeRange{}, ValueRange{});
  Region &parallelRegion = parallelOp->getRegion();
  Block *parallelBlock = &parallelRegion.emplaceBlock();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(parallelBlock);

  // Create a private op using the value
  OwningOpRef<PrivateOp> privateOp = PrivateOp::create(
      b, loc, varPtr, /*structured=*/true, /*implicit=*/false);

  // Test that the value is only used by private clauses
  EXPECT_TRUE(isOnlyUsedByPrivateClauses(varPtr, parallelRegion));
}

TEST_F(OpenACCUtilsTest, isOnlyUsedByPrivateClausesFalse) {
  // Create a value (memref) outside the compute region
  auto memrefTy = MemRefType::get({10}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, memrefTy);
  TypedValue<PointerLikeType> varPtr =
      cast<TypedValue<PointerLikeType>>(allocOp->getResult());

  // Create a parallel op with a region
  OwningOpRef<ParallelOp> parallelOp =
      ParallelOp::create(b, loc, TypeRange{}, ValueRange{});
  Region &parallelRegion = parallelOp->getRegion();
  Block *parallelBlock = &parallelRegion.emplaceBlock();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(parallelBlock);

  // Create a private op using the value
  OwningOpRef<PrivateOp> privateOp = PrivateOp::create(
      b, loc, varPtr, /*structured=*/true, /*implicit=*/false);

  // Also use the value in a function call (escape)
  OwningOpRef<func::CallOp> callOp = func::CallOp::create(
      b, loc, "some_func", TypeRange{}, ValueRange{varPtr});

  // Test that the value is NOT only used by private clauses (it escapes via
  // call)
  EXPECT_FALSE(isOnlyUsedByPrivateClauses(varPtr, parallelRegion));
}

TEST_F(OpenACCUtilsTest, isOnlyUsedByPrivateClausesMultiple) {
  // Create a value (memref) outside the compute region
  auto memrefTy = MemRefType::get({10}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, memrefTy);
  TypedValue<PointerLikeType> varPtr =
      cast<TypedValue<PointerLikeType>>(allocOp->getResult());

  // Create a parallel op with a region
  OwningOpRef<ParallelOp> parallelOp =
      ParallelOp::create(b, loc, TypeRange{}, ValueRange{});
  Region &parallelRegion = parallelOp->getRegion();
  Block *parallelBlock = &parallelRegion.emplaceBlock();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(parallelBlock);

  // Create multiple private ops using the value
  OwningOpRef<PrivateOp> privateOp1 = PrivateOp::create(
      b, loc, varPtr, /*structured=*/true, /*implicit=*/false);
  OwningOpRef<PrivateOp> privateOp2 = PrivateOp::create(
      b, loc, varPtr, /*structured=*/true, /*implicit=*/false);

  // Test that the value is only used by private clauses even with multiple uses
  EXPECT_TRUE(isOnlyUsedByPrivateClauses(varPtr, parallelRegion));
}

//===----------------------------------------------------------------------===//
// isOnlyUsedByReductionClauses Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsTest, isOnlyUsedByReductionClausesTrue) {
  // Create a value (memref) outside the compute region
  auto memrefTy = MemRefType::get({10}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, memrefTy);
  TypedValue<PointerLikeType> varPtr =
      cast<TypedValue<PointerLikeType>>(allocOp->getResult());

  // Create a parallel op with a region
  OwningOpRef<ParallelOp> parallelOp =
      ParallelOp::create(b, loc, TypeRange{}, ValueRange{});
  Region &parallelRegion = parallelOp->getRegion();
  Block *parallelBlock = &parallelRegion.emplaceBlock();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(parallelBlock);

  // Create a reduction op using the value
  OwningOpRef<ReductionOp> reductionOp = ReductionOp::create(
      b, loc, varPtr, /*structured=*/true, /*implicit=*/false);

  // Test that the value is only used by reduction clauses
  EXPECT_TRUE(isOnlyUsedByReductionClauses(varPtr, parallelRegion));
}

TEST_F(OpenACCUtilsTest, isOnlyUsedByReductionClausesFalse) {
  // Create a value (memref) outside the compute region
  auto memrefTy = MemRefType::get({10}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, memrefTy);
  TypedValue<PointerLikeType> varPtr =
      cast<TypedValue<PointerLikeType>>(allocOp->getResult());

  // Create a parallel op with a region
  OwningOpRef<ParallelOp> parallelOp =
      ParallelOp::create(b, loc, TypeRange{}, ValueRange{});
  Region &parallelRegion = parallelOp->getRegion();
  Block *parallelBlock = &parallelRegion.emplaceBlock();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(parallelBlock);

  // Create a reduction op using the value
  OwningOpRef<ReductionOp> reductionOp = ReductionOp::create(
      b, loc, varPtr, /*structured=*/true, /*implicit=*/false);

  // Also use the value in a function call (escape)
  OwningOpRef<func::CallOp> callOp = func::CallOp::create(
      b, loc, "some_func", TypeRange{}, ValueRange{varPtr});

  // Test that the value is NOT only used by reduction clauses (it escapes via
  // call)
  EXPECT_FALSE(isOnlyUsedByReductionClauses(varPtr, parallelRegion));
}

TEST_F(OpenACCUtilsTest, isOnlyUsedByReductionClausesMultiple) {
  // Create a value (memref) outside the compute region
  auto memrefTy = MemRefType::get({10}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, memrefTy);
  TypedValue<PointerLikeType> varPtr =
      cast<TypedValue<PointerLikeType>>(allocOp->getResult());

  // Create a parallel op with a region
  OwningOpRef<ParallelOp> parallelOp =
      ParallelOp::create(b, loc, TypeRange{}, ValueRange{});
  Region &parallelRegion = parallelOp->getRegion();
  Block *parallelBlock = &parallelRegion.emplaceBlock();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(parallelBlock);

  // Create multiple reduction ops using the value
  OwningOpRef<ReductionOp> reductionOp1 = ReductionOp::create(
      b, loc, varPtr, /*structured=*/true, /*implicit=*/false);
  OwningOpRef<ReductionOp> reductionOp2 = ReductionOp::create(
      b, loc, varPtr, /*structured=*/true, /*implicit=*/false);

  // Test that the value is only used by reduction clauses even with multiple
  // uses
  EXPECT_TRUE(isOnlyUsedByReductionClauses(varPtr, parallelRegion));
}

//===----------------------------------------------------------------------===//
// getDefaultAttr Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsTest, getDefaultAttrOnParallel) {
  // Create a parallel op with a default attribute
  OwningOpRef<ParallelOp> parallelOp =
      ParallelOp::create(b, loc, TypeRange{}, ValueRange{});
  parallelOp->setDefaultAttr(ClauseDefaultValue::None);

  // Test that we can retrieve the default attribute
  std::optional<ClauseDefaultValue> defaultAttr =
      getDefaultAttr(parallelOp.get());
  EXPECT_TRUE(defaultAttr.has_value());
  EXPECT_EQ(defaultAttr.value(), ClauseDefaultValue::None);
}

TEST_F(OpenACCUtilsTest, getDefaultAttrOnKernels) {
  // Create a kernels op with a default attribute
  OwningOpRef<KernelsOp> kernelsOp =
      KernelsOp::create(b, loc, TypeRange{}, ValueRange{});
  kernelsOp->setDefaultAttr(ClauseDefaultValue::Present);

  // Test that we can retrieve the default attribute
  std::optional<ClauseDefaultValue> defaultAttr =
      getDefaultAttr(kernelsOp.get());
  EXPECT_TRUE(defaultAttr.has_value());
  EXPECT_EQ(defaultAttr.value(), ClauseDefaultValue::Present);
}

TEST_F(OpenACCUtilsTest, getDefaultAttrOnSerial) {
  // Create a serial op with a default attribute
  OwningOpRef<SerialOp> serialOp =
      SerialOp::create(b, loc, TypeRange{}, ValueRange{});
  serialOp->setDefaultAttr(ClauseDefaultValue::None);

  // Test that we can retrieve the default attribute
  std::optional<ClauseDefaultValue> defaultAttr =
      getDefaultAttr(serialOp.get());
  EXPECT_TRUE(defaultAttr.has_value());
  EXPECT_EQ(defaultAttr.value(), ClauseDefaultValue::None);
}

TEST_F(OpenACCUtilsTest, getDefaultAttrOnData) {
  // Create a data op with a default attribute
  OwningOpRef<DataOp> dataOp =
      DataOp::create(b, loc, TypeRange{}, ValueRange{});
  dataOp->setDefaultAttr(ClauseDefaultValue::Present);

  // Test that we can retrieve the default attribute
  std::optional<ClauseDefaultValue> defaultAttr = getDefaultAttr(dataOp.get());
  EXPECT_TRUE(defaultAttr.has_value());
  EXPECT_EQ(defaultAttr.value(), ClauseDefaultValue::Present);
}

TEST_F(OpenACCUtilsTest, getDefaultAttrNone) {
  // Create a parallel op without setting a default attribute
  OwningOpRef<ParallelOp> parallelOp =
      ParallelOp::create(b, loc, TypeRange{}, ValueRange{});
  // Do not set default attribute

  // Test that we get std::nullopt when there's no default attribute
  std::optional<ClauseDefaultValue> defaultAttr =
      getDefaultAttr(parallelOp.get());
  EXPECT_FALSE(defaultAttr.has_value());
}

TEST_F(OpenACCUtilsTest, getDefaultAttrNearest) {
  // Create a data op with a default attribute
  OwningOpRef<DataOp> dataOp =
      DataOp::create(b, loc, TypeRange{}, ValueRange{});
  dataOp->setDefaultAttr(ClauseDefaultValue::Present);

  Region &dataRegion = dataOp->getRegion();
  Block *dataBlock = &dataRegion.emplaceBlock();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(dataBlock);

  // Create a parallel op inside the data region with NO default attribute
  OwningOpRef<ParallelOp> parallelOp =
      ParallelOp::create(b, loc, TypeRange{}, ValueRange{});
  // Do not set default attribute on parallel op

  Region &parallelRegion = parallelOp->getRegion();
  Block *parallelBlock = &parallelRegion.emplaceBlock();

  b.setInsertionPointToStart(parallelBlock);

  // Create a loop op inside the parallel region
  OwningOpRef<LoopOp> loopOp =
      LoopOp::create(b, loc, TypeRange{}, ValueRange{});

  // Test that from the loop op, we find the nearest default attribute (from
  // data op)
  std::optional<ClauseDefaultValue> defaultAttr = getDefaultAttr(loopOp.get());
  EXPECT_TRUE(defaultAttr.has_value());
  EXPECT_EQ(defaultAttr.value(), ClauseDefaultValue::Present);
}

//===----------------------------------------------------------------------===//
// getTypeCategory Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsTest, getTypeCategoryScalar) {
  // Create a scalar memref (no dimensions)
  auto scalarMemrefTy = MemRefType::get({}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, scalarMemrefTy);
  Value varPtr = allocOp->getResult();

  // Test that a scalar memref returns scalar category
  VariableTypeCategory category = getTypeCategory(varPtr);
  EXPECT_EQ(category, VariableTypeCategory::scalar);
}

TEST_F(OpenACCUtilsTest, getTypeCategoryArray) {
  // Create an array memref (with dimensions)
  auto arrayMemrefTy = MemRefType::get({10}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, arrayMemrefTy);
  Value varPtr = allocOp->getResult();

  // Test that an array memref returns array category
  VariableTypeCategory category = getTypeCategory(varPtr);
  EXPECT_EQ(category, VariableTypeCategory::array);
}
