//===- ArithToArmSME.cpp - Arith to ArmSME dialect conversion -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToArmSME/ArithToArmSME.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOARMSMECONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "arith-to-arm-sme"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Conversion helpers
//===----------------------------------------------------------------------===//

/// Returns true if 'val' is a splat of zero, false otherwise.
static bool isSplatZero(Type elemType, DenseElementsAttr val) {
  if (llvm::isa<FloatType>(elemType))
    return val && val.isSplat() && val.getSplatValue<APFloat>().isZero();
  if (llvm::isa<IntegerType>(elemType))
    return val && val.isSplat() && val.getSplatValue<APInt>().isZero();
  return false;
}

namespace {

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Conversion pattern for dense arith.constant.
struct ConstantOpToArmSMELowering : public OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                PatternRewriter &rewriter) const final {
    auto tileType = dyn_cast<VectorType>(constantOp.getType());
    if (!tileType || !arm_sme::isValidSMETileVectorType(tileType))
      return failure();

    auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValueAttr());
    if (!denseAttr || !denseAttr.isSplat())
      return failure();

    auto tileElementType = tileType.getElementType();

    // Lower 'arith.constant dense<0>' to 'arm_sme.zero' op.
    if (isSplatZero(tileElementType, denseAttr)) {
      rewriter.replaceOpWithNewOp<arm_sme::ZeroOp>(constantOp, tileType);
      return success();
    }

    // Lower non-zero constants to a loop of 'arm_sme.move_vector_to_tile_slice'
    // ops that broadcast the constant to each tile slice.
    auto loc = constantOp.getLoc();

    // To fill a tile with a constant, we create a 1-D splat of the constant,
    // then move that into each tile slice (the largest unit we can set at once,
    // outside of operations like the outerproduct).
    VectorType tileSliceType = VectorType::Builder(tileType).dropDim(0);
    auto denseAttr1D = DenseElementsAttr::get(
        tileSliceType, denseAttr.getSplatValue<Attribute>());
    auto constantOp1D = rewriter.create<arith::ConstantOp>(loc, denseAttr1D);

    auto initTile = rewriter.create<arm_sme::GetTileOp>(loc, tileType);
    auto makeLoopBody = [&](OpBuilder &b, Location loc, Value tileSliceIndex,
                            Value currentTile) {
      // Create 'arm_sme.move_vector_to_tile_slice' to write vector to tile
      // slice.
      auto nextTile = b.create<arm_sme::MoveVectorToTileSliceOp>(
          loc, tileType, constantOp1D, currentTile, tileSliceIndex);
      return nextTile.getResult();
    };
    auto forOp = mlir::arm_sme::createLoopOverTileSlices(
        rewriter, loc, initTile, makeLoopBody);
    rewriter.replaceOp(constantOp, forOp.getResult(0));

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::arith::populateArithToArmSMEConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ConstantOpToArmSMELowering>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {
struct ArithToArmSMEConversionPass final
    : impl::ArithToArmSMEConversionPassBase<ArithToArmSMEConversionPass> {
  using impl::ArithToArmSMEConversionPassBase<
      ArithToArmSMEConversionPass>::ArithToArmSMEConversionPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    arith::populateArithToArmSMEConversionPatterns(patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
