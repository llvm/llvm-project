//===- LowerVectorScam.cpp - Lower 'vector.scan' operation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.scan' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "vector-broadcast-lowering"

using namespace mlir;
using namespace mlir::vector;

/// This function checks to see if the vector combining kind
/// is consistent with the integer or float element type.
static bool isValidKind(bool isInt, vector::CombiningKind kind) {
  using vector::CombiningKind;
  enum class KindType { FLOAT, INT, INVALID };
  KindType type{KindType::INVALID};
  switch (kind) {
  case CombiningKind::MINNUMF:
  case CombiningKind::MINIMUMF:
  case CombiningKind::MAXNUMF:
  case CombiningKind::MAXIMUMF:
    type = KindType::FLOAT;
    break;
  case CombiningKind::MINUI:
  case CombiningKind::MINSI:
  case CombiningKind::MAXUI:
  case CombiningKind::MAXSI:
  case CombiningKind::AND:
  case CombiningKind::OR:
  case CombiningKind::XOR:
    type = KindType::INT;
    break;
  case CombiningKind::ADD:
  case CombiningKind::MUL:
    type = isInt ? KindType::INT : KindType::FLOAT;
    break;
  }
  bool isValidIntKind = (type == KindType::INT) && isInt;
  bool isValidFloatKind = (type == KindType::FLOAT) && (!isInt);
  return (isValidIntKind || isValidFloatKind);
}

namespace {
/// Convert vector.scan op into arith ops and vector.insert_strided_slice /
/// vector.extract_strided_slice.
///
/// Example:
///
/// ```
///   %0:2 = vector.scan <add>, %arg0, %arg1
///     {inclusive = true, reduction_dim = 1} :
///     (vector<2x3xi32>, vector<2xi32>) to (vector<2x3xi32>, vector<2xi32>)
/// ```
///
/// is converted to:
///
/// ```
///   %cst = arith.constant dense<0> : vector<2x3xi32>
///   %0 = vector.extract_strided_slice %arg0
///     {offsets = [0, 0], sizes = [2, 1], strides = [1, 1]}
///       : vector<2x3xi32> to vector<2x1xi32>
///   %1 = vector.insert_strided_slice %0, %cst
///     {offsets = [0, 0], strides = [1, 1]}
///       : vector<2x1xi32> into vector<2x3xi32>
///   %2 = vector.extract_strided_slice %arg0
///     {offsets = [0, 1], sizes = [2, 1], strides = [1, 1]}
///       : vector<2x3xi32> to vector<2x1xi32>
///   %3 = arith.muli %0, %2 : vector<2x1xi32>
///   %4 = vector.insert_strided_slice %3, %1
///     {offsets = [0, 1], strides = [1, 1]}
///       : vector<2x1xi32> into vector<2x3xi32>
///   %5 = vector.extract_strided_slice %arg0
///     {offsets = [0, 2], sizes = [2, 1], strides = [1, 1]}
///       : vector<2x3xi32> to vector<2x1xi32>
///   %6 = arith.muli %3, %5 : vector<2x1xi32>
///   %7 = vector.insert_strided_slice %6, %4
///     {offsets = [0, 2], strides = [1, 1]}
///       : vector<2x1xi32> into vector<2x3xi32>
///   %8 = vector.shape_cast %6 : vector<2x1xi32> to vector<2xi32>
///   return %7, %8 : vector<2x3xi32>, vector<2xi32>
/// ```
struct ScanToArithOps : public OpRewritePattern<vector::ScanOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ScanOp scanOp,
                                PatternRewriter &rewriter) const override {
    auto loc = scanOp.getLoc();
    VectorType destType = scanOp.getDestType();
    ArrayRef<int64_t> destShape = destType.getShape();
    auto elType = destType.getElementType();
    bool isInt = elType.isIntOrIndex();
    if (!isValidKind(isInt, scanOp.getKind()))
      return failure();

    VectorType resType = VectorType::get(destShape, elType);
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resType, rewriter.getZeroAttr(resType));
    int64_t reductionDim = scanOp.getReductionDim();
    bool inclusive = scanOp.getInclusive();
    int64_t destRank = destType.getRank();
    VectorType initialValueType = scanOp.getInitialValueType();
    int64_t initialValueRank = initialValueType.getRank();

    SmallVector<int64_t> reductionShape(destShape.begin(), destShape.end());
    reductionShape[reductionDim] = 1;
    VectorType reductionType = VectorType::get(reductionShape, elType);
    SmallVector<int64_t> offsets(destRank, 0);
    SmallVector<int64_t> strides(destRank, 1);
    SmallVector<int64_t> sizes(destShape.begin(), destShape.end());
    sizes[reductionDim] = 1;
    ArrayAttr scanSizes = rewriter.getI64ArrayAttr(sizes);
    ArrayAttr scanStrides = rewriter.getI64ArrayAttr(strides);

    Value lastOutput, lastInput;
    for (int i = 0; i < destShape[reductionDim]; i++) {
      offsets[reductionDim] = i;
      ArrayAttr scanOffsets = rewriter.getI64ArrayAttr(offsets);
      Value input = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, reductionType, scanOp.getSource(), scanOffsets, scanSizes,
          scanStrides);
      Value output;
      if (i == 0) {
        if (inclusive) {
          output = input;
        } else {
          if (initialValueRank == 0) {
            // ShapeCastOp cannot handle 0-D vectors
            output = rewriter.create<vector::BroadcastOp>(
                loc, input.getType(), scanOp.getInitialValue());
          } else {
            output = rewriter.create<vector::ShapeCastOp>(
                loc, input.getType(), scanOp.getInitialValue());
          }
        }
      } else {
        Value y = inclusive ? input : lastInput;
        output = vector::makeArithReduction(rewriter, loc, scanOp.getKind(),
                                            lastOutput, y);
      }
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, output, result, offsets, strides);
      lastOutput = output;
      lastInput = input;
    }

    Value reduction;
    if (initialValueRank == 0) {
      Value v = rewriter.create<vector::ExtractOp>(loc, lastOutput, 0);
      reduction =
          rewriter.create<vector::BroadcastOp>(loc, initialValueType, v);
    } else {
      reduction = rewriter.create<vector::ShapeCastOp>(loc, initialValueType,
                                                       lastOutput);
    }

    rewriter.replaceOp(scanOp, {result, reduction});
    return success();
  }
};
} // namespace

void mlir::vector::populateVectorScanLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ScanToArithOps>(patterns.getContext(), benefit);
}
