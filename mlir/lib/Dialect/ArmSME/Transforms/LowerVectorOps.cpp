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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSMEDialect.h"
#include "mlir/Dialect/ArmSME/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::arm_sme;

static constexpr unsigned kZeroZAMask = 255;

namespace {
/// Lower `vector.transfer_write` op to `arm_sme.intr.zero` op. Currently only
/// supports 2d scalable vector type `vector<[16x16]xi8>` that maps to the ZA0.B
/// SME tile. This will be extended to support more element types.
struct TransferWriteToArmSMEZeroLowering
    : public OpRewritePattern<vector::TransferWriteOp> {
  TransferWriteToArmSMEZeroLowering(MLIRContext *context)
      : OpRewritePattern<vector::TransferWriteOp>(context) {}

  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    auto vType = write.getVectorType();
    if (vType.getRank() != 2)
      return failure();
    if (vType.getShape() != ArrayRef<int64_t>({16, 16}))
      return failure();
    if (vType.getElementType() != rewriter.getI8Type())
      return failure();
    if (vType.getNumScalableDims() != 2)
      return failure();
    auto tile = rewriter.create<arith::ConstantOp>(
        write.getLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(kZeroZAMask));
    rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_zero>(write, tile);
    return success();
  }
};
} // namespace

void mlir::arm_sme::populateVectorTransferLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TransferWriteToArmSMEZeroLowering>(patterns.getContext());
}
