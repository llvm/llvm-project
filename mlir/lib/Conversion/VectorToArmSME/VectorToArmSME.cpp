//===- VectorToArmSME.cpp - Conversion from Vector to the ArmSME dialect --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToArmSME/VectorToArmSME.h"

#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

/// Returns true if 'val' is a splat of zero, false otherwise.
static bool isSplatZero(Type elemType, DenseElementsAttr val) {
  if (llvm::isa<FloatType>(elemType))
    return val && val.isSplat() && val.getSplatValue<APFloat>().isZero();
  if (llvm::isa<IntegerType>(elemType))
    return val && val.isSplat() && val.getSplatValue<APInt>().isZero();
  return false;
}

namespace {

/// Conversion pattern for vector.transfer_write.
///
///   vector.transfer_write %vector, %source[%c0, %c0] : vector<[16]x[16]xi8>,
///                                                      memref<?x?xi8>
///
/// is converted to:
///
///   arm_sme.tile_store %vector, %source[%c0, %c0] : memref<?x?xi8>,
///                                                   vector<[16]x[16]xi8>
struct TransferWriteToArmSMELowering
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const final {
    auto vType = writeOp.getVectorType();
    if (!arm_sme::isValidSMETileVectorType(vType))
      return failure();

    if (!llvm::isa<MemRefType>(writeOp.getSource().getType()))
      return failure();

    rewriter.replaceOpWithNewOp<arm_sme::TileStoreOp>(
        writeOp, writeOp.getVector(), writeOp.getSource(),
        writeOp.getIndices());
    return success();
  }
};

/// Conversion pattern for vector.load.
struct VectorLoadToArmSMELowering : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp load,
                                PatternRewriter &rewriter) const override {
    if (!arm_sme::isValidSMETileVectorType(load.getVectorType()))
      return failure();

    rewriter.replaceOpWithNewOp<arm_sme::TileLoadOp>(
        load, load.getVectorType(), load.getBase(), load.getIndices());

    return success();
  }
};

/// Conversion pattern for vector.store.
struct VectorStoreToArmSMELowering : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern<vector::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp store,
                                PatternRewriter &rewriter) const override {
    if (!arm_sme::isValidSMETileVectorType(store.getVectorType()))
      return failure();

    rewriter.replaceOpWithNewOp<arm_sme::TileStoreOp>(
        store, store.getValueToStore(), store.getBase(), store.getIndices());

    return success();
  }
};

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
    OpBuilder::InsertionGuard g(rewriter);
    auto loc = constantOp.getLoc();

    // Unpack 1-d vector type from 2-d vector type.
    auto tileSliceType =
        VectorType::get(tileType.getShape().drop_front(), tileElementType,
                        /*scalableDims=*/{true});
    auto denseAttr1D = DenseElementsAttr::get(
        tileSliceType, denseAttr.getSplatValue<Attribute>());
    auto constantOp1D = rewriter.create<arith::ConstantOp>(loc, denseAttr1D);

    unsigned tileElementWidth = tileElementType.getIntOrFloatBitWidth();

    // Create 'arm_sme.get_tile' op.
    auto tileId = rewriter.create<arm_sme::GetTileID>(
        loc, rewriter.getIntegerType(tileElementWidth));

    // Create `arm_sme.cast_tile_to_vector` to cast tile ID to a vector type to
    // use as input tile to 'arm_sme.move_vector_to_tile_slice' ops.
    auto tile =
        rewriter.create<arm_sme::CastTileToVector>(loc, tileType, tileId);

    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto minTileSlices = rewriter.create<arith::ConstantIndexOp>(
        loc, arm_sme::getSMETileSliceMinNumElts(tileElementType));
    auto vscale =
        rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto numTileSlices =
        rewriter.create<arith::MulIOp>(loc, minTileSlices, vscale);
    // Create a loop that broadcasts the constant to each ZA tile slice.
    auto forOp =
        rewriter.create<scf::ForOp>(loc, lowerBound, numTileSlices, step);
    rewriter.setInsertionPointToStart(forOp.getBody());
    auto tileSliceIndex = forOp.getInductionVar();

    // Create 'arm_sme.move_vector_to_tile_slice' to write vector to tile slice.
    rewriter.create<arm_sme::MoveVectorToTileSliceOp>(
        loc, tileType, constantOp1D, tile, tileSliceIndex);

    rewriter.setInsertionPointAfter(forOp);

    rewriter.replaceOp(constantOp, tile);

    return success();
  }
};

} // namespace

void mlir::populateVectorToArmSMEPatterns(RewritePatternSet &patterns,
                                          MLIRContext &ctx) {
  patterns.add<TransferWriteToArmSMELowering, VectorLoadToArmSMELowering,
               VectorStoreToArmSMELowering, ConstantOpToArmSMELowering>(&ctx);
}
