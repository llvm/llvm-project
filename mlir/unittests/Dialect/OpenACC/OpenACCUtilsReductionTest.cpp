//===- OpenACCUtilsReductionTest.cpp - OpenACC reduction utility tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsReduction.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class OpenACCUtilsReductionTest : public ::testing::Test {
protected:
  OpenACCUtilsReductionTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<acc::OpenACCDialect, arith::ArithDialect,
                        complex::ComplexDialect, memref::MemRefDialect>();
  }

  void SetUp() override {
    module = ModuleOp::create(b, loc);
    b.setInsertionPointToStart(module->getBody());
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
  OwningOpRef<ModuleOp> module;
};

//===----------------------------------------------------------------------===//
// translateAtomicRMWKind / translateACCReductionOperator Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsReductionTest, translateAtomicRMWKind) {
  EXPECT_EQ(translateAtomicRMWKind(arith::AtomicRMWKind::addi),
            ReductionOperator::AccAdd);
  EXPECT_EQ(translateAtomicRMWKind(arith::AtomicRMWKind::addf),
            ReductionOperator::AccAdd);
  EXPECT_EQ(translateAtomicRMWKind(arith::AtomicRMWKind::muli),
            ReductionOperator::AccMul);
  EXPECT_EQ(translateAtomicRMWKind(arith::AtomicRMWKind::maxnumf),
            ReductionOperator::AccMax);
  EXPECT_EQ(translateAtomicRMWKind(arith::AtomicRMWKind::minnumf),
            ReductionOperator::AccMin);
  EXPECT_EQ(translateAtomicRMWKind(arith::AtomicRMWKind::andi),
            ReductionOperator::AccIand);
}

TEST_F(OpenACCUtilsReductionTest, translateACCReductionOperator) {
  EXPECT_EQ(
      *translateACCReductionOperator(ReductionOperator::AccAdd, b.getI32Type()),
      arith::AtomicRMWKind::addi);
  EXPECT_EQ(
      *translateACCReductionOperator(ReductionOperator::AccAdd, b.getF32Type()),
      arith::AtomicRMWKind::addf);
  EXPECT_EQ(
      *translateACCReductionOperator(ReductionOperator::AccMul, b.getI64Type()),
      arith::AtomicRMWKind::muli);
  EXPECT_EQ(
      *translateACCReductionOperator(ReductionOperator::AccMax, b.getF32Type()),
      arith::AtomicRMWKind::maxnumf);
  EXPECT_FALSE(translateACCReductionOperator(ReductionOperator::AccAdd,
                                             b.getIntegerType(32, false)));
}

//===----------------------------------------------------------------------===//
// getReductionCombineParDims Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsReductionTest, getReductionCombineParDimsFromCombineOp) {
  MemRefType memTy = MemRefType::get({}, b.getI32Type());
  auto dest = memref::AllocaOp::create(b, loc, memTy);
  auto src = memref::AllocaOp::create(b, loc, memTy);
  auto combine =
      ReductionCombineOp::create(b, loc, dest, src, ReductionOperator::AccAdd);
  GPUParallelDimsAttr parDims = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::blockXDim(&context),
                 GPUParallelDimAttr::threadXDim(&context)});
  setParDimsAttr(combine, parDims);

  SmallVector<GPUParallelDimAttr> result = getReductionCombineParDims(combine);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], GPUParallelDimAttr::blockXDim(&context));
  EXPECT_EQ(result[1], GPUParallelDimAttr::threadXDim(&context));
}

TEST_F(OpenACCUtilsReductionTest,
       getReductionCombineParDimsFromCombineRegionViaAccumulate) {
  MemRefType memTy = MemRefType::get({}, b.getI32Type());
  auto dest = memref::AllocaOp::create(b, loc, memTy);
  auto src = memref::AllocaOp::create(b, loc, memTy);
  auto combineRegion = ReductionCombineRegionOp::create(b, loc, dest, src);
  combineRegion.getRegion().emplaceBlock();
  b.setInsertionPointToStart(&combineRegion.getRegion().front());
  YieldOp::create(b, loc);

  GPUParallelDimsAttr accDims = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::threadYDim(&context)});
  Value partial = arith::ConstantIntOp::create(b, loc, b.getI32Type(), 1);
  ReductionAccumulateOp::create(b, loc, partial, src.getResult(),
                                ReductionOperator::AccAdd, accDims);

  SmallVector<GPUParallelDimAttr> result =
      getReductionCombineParDims(combineRegion);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], GPUParallelDimAttr::threadYDim(&context));
}

TEST_F(OpenACCUtilsReductionTest,
       getReductionCombineParDimsFromCombineRegionAttribute) {
  MemRefType memTy = MemRefType::get({}, b.getI32Type());
  auto dest = memref::AllocaOp::create(b, loc, memTy);
  auto src = memref::AllocaOp::create(b, loc, memTy);
  auto combineRegion = ReductionCombineRegionOp::create(b, loc, dest, src);
  GPUParallelDimsAttr parDims = GPUParallelDimsAttr::get(
      &context, {GPUParallelDimAttr::blockZDim(&context)});
  setParDimsAttr(combineRegion, parDims);
  combineRegion.getRegion().emplaceBlock();
  b.setInsertionPointToStart(&combineRegion.getRegion().front());
  YieldOp::create(b, loc);

  SmallVector<GPUParallelDimAttr> result =
      getReductionCombineParDims(combineRegion);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], GPUParallelDimAttr::blockZDim(&context));
}

//===----------------------------------------------------------------------===//
// createIdentityValue / generateReductionOp Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsReductionTest, createIdentityValueIntegerAdd) {
  Value ident =
      createIdentityValue(b, loc, b.getI32Type(), arith::AtomicRMWKind::addi);
  auto cst = ident.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(cst);
  EXPECT_EQ(cast<IntegerAttr>(cst.getValue()).getInt(), 0);
}

TEST_F(OpenACCUtilsReductionTest, createIdentityValueFloatMul) {
  Value ident =
      createIdentityValue(b, loc, b.getF32Type(), arith::AtomicRMWKind::mulf);
  auto cst = ident.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(cst);
  EXPECT_EQ(cast<FloatAttr>(cst.getValue()).getValueAsDouble(), 1.0);
}

TEST_F(OpenACCUtilsReductionTest, createIdentityValueComplexAdd) {
  auto complexTy = ComplexType::get(b.getF32Type());
  Value ident =
      createIdentityValue(b, loc, complexTy, arith::AtomicRMWKind::addf);
  auto createOp = ident.getDefiningOp<complex::CreateOp>();
  ASSERT_TRUE(createOp);
  auto realCst = createOp.getReal().getDefiningOp<arith::ConstantOp>();
  auto imagCst = createOp.getImaginary().getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(realCst);
  ASSERT_TRUE(imagCst);
  EXPECT_EQ(cast<FloatAttr>(realCst.getValue()).getValueAsDouble(), 0.0);
  EXPECT_EQ(cast<FloatAttr>(imagCst.getValue()).getValueAsDouble(), 0.0);
}

TEST_F(OpenACCUtilsReductionTest, generateReductionOpIntegerAdd) {
  Value lhs = arith::ConstantIntOp::create(b, loc, b.getI32Type(), 3);
  Value rhs = arith::ConstantIntOp::create(b, loc, b.getI32Type(), 5);
  Value sum = generateReductionOp(b, loc, lhs, rhs, arith::AtomicRMWKind::addi);
  EXPECT_TRUE(isa<arith::AddIOp>(sum.getDefiningOp()));
}

TEST_F(OpenACCUtilsReductionTest, generateReductionOpComplexMul) {
  auto complexTy = ComplexType::get(b.getF32Type());
  Value lhs = complex::CreateOp::create(
      b, loc, complexTy,
      arith::ConstantOp::create(b, loc, b.getF32FloatAttr(2.0)),
      arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1.0)));
  Value rhs = complex::CreateOp::create(
      b, loc, complexTy,
      arith::ConstantOp::create(b, loc, b.getF32FloatAttr(3.0)),
      arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.0)));
  Value product =
      generateReductionOp(b, loc, lhs, rhs, arith::AtomicRMWKind::mulf);
  EXPECT_TRUE(isa<complex::MulOp>(product.getDefiningOp()));
}
