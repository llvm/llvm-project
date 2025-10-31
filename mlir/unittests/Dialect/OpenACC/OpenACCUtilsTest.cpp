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

//===----------------------------------------------------------------------===//
// getVariableName Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsTest, getVariableNameDirect) {
  // Create a memref with acc.var_name attribute
  auto memrefTy = MemRefType::get({10}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, memrefTy);

  // Set the acc.var_name attribute
  auto varNameAttr = VarNameAttr::get(&context, "my_variable");
  allocOp.get()->setAttr(getVarNameAttrName(), varNameAttr);

  Value varPtr = allocOp->getResult();

  // Test that getVariableName returns the variable name
  std::string varName = getVariableName(varPtr);
  EXPECT_EQ(varName, "my_variable");
}

TEST_F(OpenACCUtilsTest, getVariableNameThroughCast) {
  // Create a 5x2 memref with acc.var_name attribute
  auto memrefTy = MemRefType::get({5, 2}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, memrefTy);

  // Set the acc.var_name attribute on the alloca
  auto varNameAttr = VarNameAttr::get(&context, "casted_variable");
  allocOp.get()->setAttr(getVarNameAttrName(), varNameAttr);

  Value allocResult = allocOp->getResult();

  // Create a memref.cast operation to a flattened 10-element array
  auto castedMemrefTy = MemRefType::get({10}, b.getI32Type());
  OwningOpRef<memref::CastOp> castOp =
      memref::CastOp::create(b, loc, castedMemrefTy, allocResult);

  Value castedPtr = castOp->getResult();

  // Test that getVariableName walks through the cast to find the variable name
  std::string varName = getVariableName(castedPtr);
  EXPECT_EQ(varName, "casted_variable");
}

TEST_F(OpenACCUtilsTest, getVariableNameNotFound) {
  // Create a memref without acc.var_name attribute
  auto memrefTy = MemRefType::get({10}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, memrefTy);

  Value varPtr = allocOp->getResult();

  // Test that getVariableName returns empty string when no name is found
  std::string varName = getVariableName(varPtr);
  EXPECT_EQ(varName, "");
}

TEST_F(OpenACCUtilsTest, getVariableNameFromCopyin) {
  // Create a memref
  auto memrefTy = MemRefType::get({10}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, memrefTy);

  Value varPtr = allocOp->getResult();
  StringRef name = "data_array";
  OwningOpRef<CopyinOp> copyinOp =
      CopyinOp::create(b, loc, varPtr, /*structured=*/true, /*implicit=*/true,
                       /*name=*/name);

  // Test that getVariableName extracts the name from the copyin operation
  std::string varName = getVariableName(copyinOp->getAccVar());
  EXPECT_EQ(varName, name);
}

//===----------------------------------------------------------------------===//
// getRecipeName Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsTest, getRecipeNamePrivateScalarMemref) {
  // Create a scalar memref type
  auto scalarMemrefTy = MemRefType::get({}, b.getI32Type());

  // Test private recipe with scalar memref
  std::string recipeName =
      getRecipeName(RecipeKind::private_recipe, scalarMemrefTy);
  EXPECT_EQ(recipeName, "privatization_memref_i32_");
}

TEST_F(OpenACCUtilsTest, getRecipeNameFirstprivateScalarMemref) {
  // Create a scalar memref type
  auto scalarMemrefTy = MemRefType::get({}, b.getF32Type());

  // Test firstprivate recipe with scalar memref
  std::string recipeName =
      getRecipeName(RecipeKind::firstprivate_recipe, scalarMemrefTy);
  EXPECT_EQ(recipeName, "firstprivatization_memref_f32_");
}

TEST_F(OpenACCUtilsTest, getRecipeNameReductionScalarMemref) {
  // Create a scalar memref type
  auto scalarMemrefTy = MemRefType::get({}, b.getI64Type());

  // Test reduction recipe with scalar memref
  std::string recipeName =
      getRecipeName(RecipeKind::reduction_recipe, scalarMemrefTy);
  EXPECT_EQ(recipeName, "reduction_memref_i64_");
}

TEST_F(OpenACCUtilsTest, getRecipeNamePrivate2DMemref) {
  // Create a 2D memref type
  auto memref2DTy = MemRefType::get({5, 10}, b.getF32Type());

  // Test private recipe with 2D memref
  std::string recipeName =
      getRecipeName(RecipeKind::private_recipe, memref2DTy);
  EXPECT_EQ(recipeName, "privatization_memref_5x10xf32_");
}

TEST_F(OpenACCUtilsTest, getRecipeNameFirstprivate2DMemref) {
  // Create a 2D memref type
  auto memref2DTy = MemRefType::get({8, 16}, b.getF64Type());

  // Test firstprivate recipe with 2D memref
  std::string recipeName =
      getRecipeName(RecipeKind::firstprivate_recipe, memref2DTy);
  EXPECT_EQ(recipeName, "firstprivatization_memref_8x16xf64_");
}

TEST_F(OpenACCUtilsTest, getRecipeNameReduction2DMemref) {
  // Create a 2D memref type
  auto memref2DTy = MemRefType::get({4, 8}, b.getI32Type());

  // Test reduction recipe with 2D memref
  std::string recipeName =
      getRecipeName(RecipeKind::reduction_recipe, memref2DTy);
  EXPECT_EQ(recipeName, "reduction_memref_4x8xi32_");
}

TEST_F(OpenACCUtilsTest, getRecipeNamePrivateDynamicMemref) {
  // Create a memref with dynamic dimensions
  auto dynamicMemrefTy =
      MemRefType::get({ShapedType::kDynamic, 10}, b.getI32Type());

  // Test private recipe with dynamic memref
  std::string recipeName =
      getRecipeName(RecipeKind::private_recipe, dynamicMemrefTy);
  EXPECT_EQ(recipeName, "privatization_memref_Ux10xi32_");
}

TEST_F(OpenACCUtilsTest, getRecipeNamePrivateUnrankedMemref) {
  // Create an unranked memref type
  auto unrankedMemrefTy = UnrankedMemRefType::get(b.getI32Type(), 0);

  // Test private recipe with unranked memref
  std::string recipeName =
      getRecipeName(RecipeKind::private_recipe, unrankedMemrefTy);
  EXPECT_EQ(recipeName, "privatization_memref_Zxi32_");
}

//===----------------------------------------------------------------------===//
// getBaseEntity Tests
//===----------------------------------------------------------------------===//

// Local implementation of PartialEntityAccessOpInterface for memref.subview.
// This is implemented locally in the test rather than officially because memref
// operations already have ViewLikeOpInterface, which serves a similar purpose
// for walking through views to the base entity. This test demonstrates how
// getBaseEntity() would work if the interface were attached to memref.subview.
namespace {
struct SubViewOpPartialEntityAccessOpInterface
    : public acc::PartialEntityAccessOpInterface::ExternalModel<
          SubViewOpPartialEntityAccessOpInterface, memref::SubViewOp> {
  Value getBaseEntity(Operation *op) const {
    auto subviewOp = cast<memref::SubViewOp>(op);
    return subviewOp.getSource();
  }

  bool isCompleteView(Operation *op) const {
    // For testing purposes, we'll consider it a partial view (return false).
    // The real implementation would need to look at the offsets.
    return false;
  }
};
} // namespace

TEST_F(OpenACCUtilsTest, getBaseEntityFromSubview) {
  // Register the local interface implementation for memref.subview
  memref::SubViewOp::attachInterface<SubViewOpPartialEntityAccessOpInterface>(
      context);

  // Create a base memref
  auto memrefTy = MemRefType::get({10, 20}, b.getF32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, memrefTy);
  Value baseMemref = allocOp->getResult();

  // Create a subview of the base memref with non-zero offsets
  // This creates a 5x10 view starting at [2, 3] in the original 10x20 memref
  SmallVector<OpFoldResult> offsets = {b.getIndexAttr(2), b.getIndexAttr(3)};
  SmallVector<OpFoldResult> sizes = {b.getIndexAttr(5), b.getIndexAttr(10)};
  SmallVector<OpFoldResult> strides = {b.getIndexAttr(1), b.getIndexAttr(1)};

  OwningOpRef<memref::SubViewOp> subviewOp =
      memref::SubViewOp::create(b, loc, baseMemref, offsets, sizes, strides);
  Value subview = subviewOp->getResult();

  // Test that getBaseEntity returns the base memref, not the subview
  Value baseEntity = getBaseEntity(subview);
  EXPECT_EQ(baseEntity, baseMemref);
}

TEST_F(OpenACCUtilsTest, getBaseEntityNoInterface) {
  // Create a memref without the interface
  auto memrefTy = MemRefType::get({10}, b.getI32Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, memrefTy);
  Value varPtr = allocOp->getResult();

  // Test that getBaseEntity returns the value itself when there's no interface
  Value baseEntity = getBaseEntity(varPtr);
  EXPECT_EQ(baseEntity, varPtr);
}

TEST_F(OpenACCUtilsTest, getBaseEntityChainedSubviews) {
  // Register the local interface implementation for memref.subview
  memref::SubViewOp::attachInterface<SubViewOpPartialEntityAccessOpInterface>(
      context);

  // Create a base memref
  auto memrefTy = MemRefType::get({100, 200}, b.getI64Type());
  OwningOpRef<memref::AllocaOp> allocOp =
      memref::AllocaOp::create(b, loc, memrefTy);
  Value baseMemref = allocOp->getResult();

  // Create first subview
  SmallVector<OpFoldResult> offsets1 = {b.getIndexAttr(10), b.getIndexAttr(20)};
  SmallVector<OpFoldResult> sizes1 = {b.getIndexAttr(50), b.getIndexAttr(80)};
  SmallVector<OpFoldResult> strides1 = {b.getIndexAttr(1), b.getIndexAttr(1)};

  OwningOpRef<memref::SubViewOp> subview1Op =
      memref::SubViewOp::create(b, loc, baseMemref, offsets1, sizes1, strides1);
  Value subview1 = subview1Op->getResult();

  // Create second subview (subview of subview)
  SmallVector<OpFoldResult> offsets2 = {b.getIndexAttr(5), b.getIndexAttr(10)};
  SmallVector<OpFoldResult> sizes2 = {b.getIndexAttr(20), b.getIndexAttr(30)};
  SmallVector<OpFoldResult> strides2 = {b.getIndexAttr(1), b.getIndexAttr(1)};

  OwningOpRef<memref::SubViewOp> subview2Op =
      memref::SubViewOp::create(b, loc, subview1, offsets2, sizes2, strides2);
  Value subview2 = subview2Op->getResult();

  // Test that getBaseEntity on the nested subview returns the first subview
  // (since our implementation returns the immediate source, not the ultimate
  // base)
  Value baseEntity = getBaseEntity(subview2);
  EXPECT_EQ(baseEntity, subview1);

  // Test that calling getBaseEntity again returns the original base
  Value ultimateBase = getBaseEntity(baseEntity);
  EXPECT_EQ(ultimateBase, baseMemref);
}
