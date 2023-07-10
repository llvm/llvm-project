//===- LowerVectorOps.cpp - Lower vector ops to SME -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewrite patterns to lower vector dialect ops to ArmSME.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Transforms.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::arm_sme;

static constexpr unsigned kMinNumElts = 16;
static constexpr unsigned kZeroZAMask = 255;

/// Returns true if 'val' is a splat of zero, false otherwise.
static bool isSplatZero(Type elemType, DenseElementsAttr val) {
  if (llvm::isa<FloatType>(elemType))
    return val && val.isSplat() && val.getSplatValue<APFloat>().isZero();
  if (llvm::isa<IntegerType>(elemType))
    return val && val.isSplat() && val.getSplatValue<APInt>().isZero();
  return false;
}

namespace {
/// Lower 'vector.transfer_write' op to 'arm_sme.intr.zero' op. Currently only
/// supports 2d scalable vector type 'vector<[16x16]xi8>' that maps to the ZA0.B
/// SME virtual tile. This will be extended to support more element types.
struct TransferWriteToArmSMEZeroLowering
    : public ConvertOpToLLVMPattern<vector::TransferWriteOp> {
  using ConvertOpToLLVMPattern<vector::TransferWriteOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::TransferWriteOp write, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vType = write.getVectorType();
    if (vType.getRank() != 2)
      return failure();
    if (vType.getShape() != ArrayRef<int64_t>({kMinNumElts, kMinNumElts}))
      return failure();
    if (vType.getElementType() != rewriter.getI8Type())
      return failure();
    if (vType.getScalableDims().size() != 2)
      return failure();

    auto memRefType = llvm::dyn_cast<MemRefType>(write.getSource().getType());
    if (!memRefType)
      return failure();

    auto constant = write.getVector().getDefiningOp<arith::ConstantOp>();
    if (!constant)
      return failure();

    auto denseAttr = dyn_cast<DenseElementsAttr>(constant.getValueAttr());
    if (!denseAttr || !isSplatZero(vType.getElementType(), denseAttr))
      return failure();

    auto loc = write.getLoc();

    // Create 'arm_sme.intr.zero' intrinsic to zero ZA.
    auto tile = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(kZeroZAMask));
    rewriter.create<arm_sme::aarch64_sme_zero>(loc, tile);

    // Create loop that iterates from 0 to SVLB-1 inclusive (the number of
    // vectors in ZA) and stores each ZA vector to memory.
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto minElems = rewriter.create<arith::ConstantIndexOp>(loc, kMinNumElts);
    auto vscale =
        rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto upperBound = rewriter.create<arith::MulIOp>(loc, minElems, vscale);
    auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
    rewriter.setInsertionPointToStart(forOp.getBody());

    // Create 'arm_sme.intr.str' intrinsic to store ZA vector.
    auto vnumI64 = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getI64Type(), forOp.getInductionVar());
    auto offset =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0);
    Value ptr = getStridedElementPtr(loc, memRefType, adaptor.getSource(),
                                     ValueRange{vnumI64, offset}, rewriter);
    auto vnumI32 = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getI32Type(), forOp.getInductionVar());
    rewriter.create<arm_sme::aarch64_sme_str>(loc, vnumI32, ptr);

    rewriter.eraseOp(write);

    return success();
  }
};
} // namespace

void mlir::arm_sme::populateVectorTransferLoweringPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<TransferWriteToArmSMEZeroLowering>(converter);
}
