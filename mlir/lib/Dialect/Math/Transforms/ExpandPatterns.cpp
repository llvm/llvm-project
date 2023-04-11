//===- ExpandTanh.cpp - Code to perform expanding tanh op -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements expansion of tanh op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/// Create a float constant.
static Value createFloatConst(Location loc, Type type, double value,
                              OpBuilder &b) {
  auto attr = b.getFloatAttr(getElementTypeOrSelf(type), value);
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    return b.create<arith::ConstantOp>(loc,
                                       DenseElementsAttr::get(shapedTy, attr));
  }

  return b.create<arith::ConstantOp>(loc, attr);
}

/// Create a float constant.
static Value createIntConst(Location loc, Type type, int64_t value,
                            OpBuilder &b) {
  auto attr = b.getIntegerAttr(getElementTypeOrSelf(type), value);
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    return b.create<arith::ConstantOp>(loc,
                                       DenseElementsAttr::get(shapedTy, attr));
  }

  return b.create<arith::ConstantOp>(loc, attr);
}

static Value createTruncatedFPValue(Value operand, ImplicitLocOpBuilder &b) {
  Type opType = operand.getType();
  Value fixedConvert = b.create<arith::FPToSIOp>(b.getI64Type(), operand);
  Value fpFixedConvert = b.create<arith::SIToFPOp>(opType, fixedConvert);
  return fpFixedConvert;
}

/// Expands tanh op into
///   1) 1-exp^{-2x} / 1+exp^{-2x}, if x => 0
///   2) exp^{2x}-1 / exp^{2x}+1  , if x < 0
static LogicalResult convertTanhOp(math::TanhOp op, PatternRewriter &rewriter) {
  auto floatType = op.getOperand().getType();
  Location loc = op.getLoc();
  Value one = createFloatConst(loc, floatType, 1.0, rewriter);
  Value two = createFloatConst(loc, floatType, 2.0, rewriter);
  Value doubledX = rewriter.create<arith::MulFOp>(loc, op.getOperand(), two);

  // Case 1: tanh(x) = 1-exp^{-2x} / 1+exp^{-2x}
  Value negDoubledX = rewriter.create<arith::NegFOp>(loc, doubledX);
  Value exp2x = rewriter.create<math::ExpOp>(loc, negDoubledX);
  Value dividend = rewriter.create<arith::SubFOp>(loc, one, exp2x);
  Value divisor = rewriter.create<arith::AddFOp>(loc, one, exp2x);
  Value positiveRes = rewriter.create<arith::DivFOp>(loc, dividend, divisor);

  // Case 2: tanh(x) = exp^{2x}-1 / exp^{2x}+1
  exp2x = rewriter.create<math::ExpOp>(loc, doubledX);
  dividend = rewriter.create<arith::SubFOp>(loc, exp2x, one);
  divisor = rewriter.create<arith::AddFOp>(loc, exp2x, one);
  Value negativeRes = rewriter.create<arith::DivFOp>(loc, dividend, divisor);

  // tanh(x) = x >= 0 ? positiveRes : negativeRes
  Value zero = createFloatConst(loc, floatType, 0.0, rewriter);
  Value cmpRes = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE,
                                                op.getOperand(), zero);
  rewriter.replaceOpWithNewOp<arith::SelectOp>(op, cmpRes, positiveRes,
                                               negativeRes);
  return success();
}

// Converts math.tan to math.sin, math.cos, and arith.divf.
static LogicalResult convertTanOp(math::TanOp op, PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type type = operand.getType();
  Value sin = b.create<math::SinOp>(type, operand);
  Value cos = b.create<math::CosOp>(type, operand);
  Value div = b.create<arith::DivFOp>(type, sin, cos);
  rewriter.replaceOp(op, div);
  return success();
}

static LogicalResult convertFmaFOp(math::FmaOp op, PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operandA = op.getOperand(0);
  Value operandB = op.getOperand(1);
  Value operandC = op.getOperand(2);
  Type type = op.getType();
  Value mult = b.create<arith::MulFOp>(type, operandA, operandB);
  Value add = b.create<arith::AddFOp>(type, mult, operandC);
  rewriter.replaceOp(op, add);
  return success();
}

// Converts a floorf() function to the following:
// floorf(float x) ->
//     y = (float)(int) x
//     if (x < 0) then incr = -1 else incr = 0
//     y = y + incr    <= replace this op with the floorf op.
static LogicalResult convertFloorOp(math::FloorOp op,
                                    PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type opType = operand.getType();
  Value fpFixedConvert = createTruncatedFPValue(operand, b);

  // Creating constants for later use.
  Value zero = createFloatConst(op->getLoc(), opType, 0.00, rewriter);
  Value negOne = createFloatConst(op->getLoc(), opType, -1.00, rewriter);

  Value negCheck =
      b.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, operand, zero);
  Value incrValue =
      b.create<arith::SelectOp>(op->getLoc(), negCheck, negOne, zero);
  Value ret = b.create<arith::AddFOp>(opType, fpFixedConvert, incrValue);
  rewriter.replaceOp(op, ret);
  return success();
}

// Converts a ceilf() function to the following:
// ceilf(float x) ->
//      y = (float)(int) x
//      if (x > y) then incr = 1 else incr = 0
//      y = y + incr   <= replace this op with the ceilf op.
static LogicalResult convertCeilOp(math::CeilOp op, PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type opType = operand.getType();
  Value fpFixedConvert = createTruncatedFPValue(operand, b);

  // Creating constants for later use.
  Value zero = createFloatConst(op->getLoc(), opType, 0.00, rewriter);
  Value one = createFloatConst(op->getLoc(), opType, 1.00, rewriter);

  Value gtCheck = b.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, operand,
                                          fpFixedConvert);
  Value incrValue = b.create<arith::SelectOp>(op->getLoc(), gtCheck, one, zero);

  Value ret = b.create<arith::AddFOp>(opType, fpFixedConvert, incrValue);
  rewriter.replaceOp(op, ret);
  return success();
}

// Converts math.ctlz to scf and arith operations. This is done
// by performing a binary search on the bits.
static LogicalResult convertCtlzOp(math::CountLeadingZerosOp op,
                                   PatternRewriter &rewriter) {
  auto operand = op.getOperand();
  auto operandTy = operand.getType();
  auto eTy = getElementTypeOrSelf(operandTy);
  Location loc = op.getLoc();

  int32_t bitwidth = eTy.getIntOrFloatBitWidth();
  if (bitwidth > 64)
    return failure();

  uint64_t allbits = -1;
  if (bitwidth < 64) {
    allbits = allbits >> (64 - bitwidth);
  }

  Value x = operand;
  Value count = createIntConst(loc, operandTy, 0, rewriter);
  for (int32_t bw = bitwidth; bw > 1; bw = bw / 2) {
    auto half = bw / 2;
    auto bits = createIntConst(loc, operandTy, half, rewriter);
    auto mask = createIntConst(loc, operandTy, allbits >> half, rewriter);

    Value pred =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, x, mask);
    Value add = rewriter.create<arith::AddIOp>(loc, count, bits);
    Value shift = rewriter.create<arith::ShLIOp>(loc, x, bits);

    x = rewriter.create<arith::SelectOp>(loc, pred, shift, x);
    count = rewriter.create<arith::SelectOp>(loc, pred, add, count);
  }

  Value zero = createIntConst(loc, operandTy, 0, rewriter);
  Value pred = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              operand, zero);

  Value bwval = createIntConst(loc, operandTy, bitwidth, rewriter);
  Value sel = rewriter.create<arith::SelectOp>(loc, pred, bwval, count);
  rewriter.replaceOp(op, sel);
  return success();
}

void mlir::populateExpandCtlzPattern(RewritePatternSet &patterns) {
  patterns.add(convertCtlzOp);
}

void mlir::populateExpandTanPattern(RewritePatternSet &patterns) {
  patterns.add(convertTanOp);
}

void mlir::populateExpandTanhPattern(RewritePatternSet &patterns) {
  patterns.add(convertTanhOp);
}

void mlir::populateExpandFmaFPattern(RewritePatternSet &patterns) {
  patterns.add(convertFmaFOp);
}

void mlir::populateExpandCeilFPattern(RewritePatternSet &patterns) {
  patterns.add(convertCeilOp);
}

void mlir::populateExpandFloorFPattern(RewritePatternSet &patterns) {
  patterns.add(convertFloorOp);
}
