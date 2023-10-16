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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

/// Generates a for loop over ZA tile slices where the induction variable is
/// the tile slice index. Sets the IR Builder insertion point as the loop body.
/// Callers of this method are responsible for restoring it if needed.
static scf::ForOp getLoopOverTileSlices(PatternRewriter &rewriter, Location loc,
                                        Type eltType) {
  auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto minTileSlices = rewriter.create<arith::ConstantIndexOp>(
      loc, arm_sme::getSMETileSliceMinNumElts(eltType));
  auto vscale =
      rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
  auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto numTileSlices =
      rewriter.create<arith::MulIOp>(loc, minTileSlices, vscale);
  auto forOp =
      rewriter.create<scf::ForOp>(loc, lowerBound, numTileSlices, step);
  rewriter.setInsertionPointToStart(forOp.getBody());
  return forOp;
}

/// Returns a tile of the given vector type.
static arm_sme::CastTileToVector
getSMETileAndCastToVector(PatternRewriter &rewriter, Location loc,
                          VectorType type) {
  unsigned tileElementWidth = type.getElementType().getIntOrFloatBitWidth();

  // Create 'arm_sme.get_tile' op.
  auto tileId = rewriter.create<arm_sme::GetTileID>(
      loc, rewriter.getIntegerType(tileElementWidth));

  // Create `arm_sme.cast_tile_to_vector` to cast tile ID to a vector type.
  return rewriter.create<arm_sme::CastTileToVector>(loc, type, tileId);
}

namespace {

/// Conversion pattern for vector.transfer_read op with transpose permutation
/// map to vertical arm_sme.tile_load (in-flight transpose).
///
///   vector.transfer_read ...  permutation_map: (d0, d1) -> (d1, d0)
///
/// is converted to:
///
///   arm_sme.tile_load ... layout<vertical>
struct TransferReadPermutationToArmSMELowering
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp transferReadOp,
                                PatternRewriter &rewriter) const final {
    // The permutation map must have two results.
    if (transferReadOp.getTransferRank() != 2)
      return rewriter.notifyMatchFailure(transferReadOp,
                                         "not a 2 result permutation map");

    AffineMap map = transferReadOp.getPermutationMap();

    // Permutation map doesn't perform permutation, can be lowered to
    // vector.load by TransferReadToVectorLoadLowering and then
    // arm_sme.tile_load by VectorLoadToArmSMELowering.
    if (map.isIdentity())
      return rewriter.notifyMatchFailure(
          transferReadOp, "map is an identity, apply another pattern");

    auto vectorType = transferReadOp.getVectorType();
    if (!arm_sme::isValidSMETileVectorType(vectorType))
      return rewriter.notifyMatchFailure(transferReadOp,
                                         "not a valid vector type for SME");

    if (!llvm::isa<MemRefType>(transferReadOp.getSource().getType()))
      return rewriter.notifyMatchFailure(transferReadOp, "not a memref source");

    if (transferReadOp.getMask())
      // TODO: support masking.
      return rewriter.notifyMatchFailure(transferReadOp,
                                         "masking not yet supported");

    // Out-of-bounds dims are not supported.
    if (transferReadOp.hasOutOfBoundsDim())
      return rewriter.notifyMatchFailure(transferReadOp,
                                         "not inbounds transfer read");

    AffineExpr d0, d1;
    bindDims(transferReadOp.getContext(), d0, d1);
    if (map != AffineMap::get(map.getNumDims(), 0, {d1, d0},
                              transferReadOp.getContext()))
      return rewriter.notifyMatchFailure(transferReadOp,
                                         "not true 2-D matrix transpose");

    rewriter.replaceOpWithNewOp<arm_sme::TileLoadOp>(
        transferReadOp, vectorType, transferReadOp.getSource(),
        transferReadOp.getIndices(), arm_sme::TileSliceLayout::Vertical);

    return success();
  }
};

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

    arm_sme::CastTileToVector tile =
        getSMETileAndCastToVector(rewriter, loc, tileType);

    auto forOp = getLoopOverTileSlices(rewriter, loc, tileElementType);
    auto tileSliceIndex = forOp.getInductionVar();

    // Create 'arm_sme.move_vector_to_tile_slice' to write vector to tile slice.
    rewriter.create<arm_sme::MoveVectorToTileSliceOp>(
        loc, tileType, constantOp1D, tile, tileSliceIndex);

    rewriter.replaceOp(constantOp, tile);

    return success();
  }
};

/// Conversion pattern for vector.broadcast.
///
/// Example:
///
///   %broadcast_to_tile = vector.broadcast %src : i32 to vector<[4]x[4]xi32>
///
/// is converted to:
///
///   %broadcast_to_1d = vector.broadcast %src : i32 to vector<[4]xi32>
///   scf.for %tile_slice_index = %c0 to %num_tile_slices step %c1 {
///     arm_sme.move_vector_to_tile_slice %broadcast_to_1d, %tile,
///       %tile_slice_index : vector<[4]xi32> into vector<[4]x[4]xi32>
///   }
///
/// Supports scalar, 0-d vector, and 1-d vector broadcasts.
struct BroadcastOpToArmSMELowering
    : public OpRewritePattern<vector::BroadcastOp> {
  using OpRewritePattern<vector::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const final {
    auto tileType = broadcastOp.getResultVectorType();
    if (!tileType || !arm_sme::isValidSMETileVectorType(tileType))
      return failure();

    OpBuilder::InsertionGuard g(rewriter);
    auto loc = broadcastOp.getLoc();

    auto srcType = broadcastOp.getSourceType();
    auto srcVectorType = dyn_cast<VectorType>(srcType);
    auto tileElementType = tileType.getElementType();

    Value broadcastOp1D;
    if (srcType.isIntOrFloat() ||
        (srcVectorType && (srcVectorType.getRank() == 0))) {
      // Broadcast scalar or 0-d vector to 1-d vector.
      auto tileSliceType =
          VectorType::get(tileType.getShape().drop_front(), tileElementType,
                          /*scalableDims=*/{true});
      broadcastOp1D = rewriter.create<vector::BroadcastOp>(
          loc, tileSliceType, broadcastOp.getSource());
    } else if (srcVectorType && (srcVectorType.getRank() == 1))
      // Value to broadcast is already a 1-d vector, nothing to do.
      broadcastOp1D = broadcastOp.getSource();
    else
      return failure();

    arm_sme::CastTileToVector tile =
        getSMETileAndCastToVector(rewriter, loc, tileType);

    // Create a loop over ZA tile slices.
    auto forOp = getLoopOverTileSlices(rewriter, loc, tileElementType);
    auto tileSliceIndex = forOp.getInductionVar();

    // Create 'arm_sme.move_vector_to_tile_slice' to broadcast the value to each
    // tile slice.
    rewriter.create<arm_sme::MoveVectorToTileSliceOp>(
        loc, tileType, broadcastOp1D, tile, tileSliceIndex);

    rewriter.replaceOp(broadcastOp, tile);

    return success();
  }
};

/// Conversion pattern for vector.splat.
///
/// Example:
///
///   %splat_to_tile = vector.splat %src : i32 to vector<[4]x[4]xi32>
///
/// is converted to:
///
///   %broadcast_to_1d = vector.broadcast %src : i32 to vector<[4]xi32>
///   scf.for %tile_slice_index = %c0 to %num_tile_slices step %c1 {
///     arm_sme.move_vector_to_tile_slice %broadcast_to_1d, %tile,
///       %tile_slice_index : vector<[4]xi32> into vector<[4]x[4]xi32>
///   }
///
/// This is identical to vector.broadcast of a scalar.
struct SplatOpToArmSMELowering : public OpRewritePattern<vector::SplatOp> {
  using OpRewritePattern<vector::SplatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::SplatOp splatOp,
                                PatternRewriter &rewriter) const final {
    auto tileType = splatOp.getResult().getType();
    if (!tileType || !arm_sme::isValidSMETileVectorType(tileType))
      return failure();

    OpBuilder::InsertionGuard g(rewriter);
    auto loc = splatOp.getLoc();

    auto srcType = splatOp.getOperand().getType();
    auto tileElementType = tileType.getElementType();

    assert(srcType.isIntOrFloat() && "Invalid source type for vector.splat");
    // Avoid unused-variable warning when building without assertions.
    (void)srcType;

    // First, broadcast the scalar to a 1-d vector.
    VectorType tileSliceType = VectorType::Builder(tileType).dropDim(0);
    Value broadcastOp1D = rewriter.create<vector::BroadcastOp>(
        loc, tileSliceType, splatOp.getInput());

    arm_sme::CastTileToVector tile =
        getSMETileAndCastToVector(rewriter, loc, tileType);

    // Next, create a loop over ZA tile slices and "move" the generated 1-d
    // vector to each slice.
    auto forOp = getLoopOverTileSlices(rewriter, loc, tileElementType);
    auto tileSliceIndex = forOp.getInductionVar();

    rewriter.create<arm_sme::MoveVectorToTileSliceOp>(
        loc, tileType, broadcastOp1D, tile, tileSliceIndex);

    rewriter.replaceOp(splatOp, tile);

    return success();
  }
};

/// Conversion pattern for vector.transpose.
///
/// Stores the input tile to memory and reloads vertically.
///
/// Example:
///
///   %transposed_src = vector.transpose %src, [1, 0]
///     : vector<[4]x[4]xi32> to vector<[4]x[4]xi32>
///
/// is converted to:
///
///   %alloca = memref.alloca(%svl_s, %svl_s) : memref<?x?xi32>
///   %arm_sme.tile_store %src, <hor>, %alloca[%c0, %c0]
///     : memref<?x?xi32>, vector<[4]x[4]xi32>
///   %transposed_src = arm_sme.tile_load %alloca[%c0, %c0]
///     layout<vertical> : memref<?x?xi32>, vector<[4]x[4]xi32>
///
/// NOTE: Tranposing via memory is obviously expensive, the current intention
/// is to avoid the transpose if possible, this is therefore intended as a
/// fallback and to provide base support for Vector ops. If it turns out
/// transposes can't be avoided then this should be replaced with a more optimal
/// implementation, perhaps with tile <-> vector (MOVA) ops.
struct TransposeOpToArmSMELowering
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const final {
    auto tileType = transposeOp.getResultVectorType();
    if (!tileType || !arm_sme::isValidSMETileVectorType(tileType))
      return failure();

    SmallVector<int64_t> transp;
    for (auto attr : transposeOp.getTransp())
      transp.push_back(cast<IntegerAttr>(attr).getInt());

    // Bail unless this is a true 2-D matrix transpose.
    if (transp[0] != 1 || transp[1] != 0)
      return failure();

    OpBuilder::InsertionGuard g(rewriter);
    auto loc = transposeOp.getLoc();

    // Allocate buffer to store input tile to.
    Value vscale =
        rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
    Value minTileSlices = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(tileType.getDimSize(0)));
    Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    Value numTileSlices =
        rewriter.create<arith::MulIOp>(loc, vscale, minTileSlices);
    auto bufferType =
        MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                        tileType.getElementType());
    auto buffer = rewriter.create<memref::AllocaOp>(
        loc, bufferType, ValueRange{numTileSlices, numTileSlices});

    Value input = transposeOp.getVector();

    // Store input tile.
    auto tileStoreOp = rewriter.create<arm_sme::TileStoreOp>(
        loc, input, buffer, ValueRange{c0, c0});

    // Reload input tile vertically.
    rewriter.replaceOpWithNewOp<arm_sme::TileLoadOp>(
        transposeOp, tileType, tileStoreOp.getBase(), tileStoreOp.getIndices(),
        arm_sme::TileSliceLayout::Vertical);

    return success();
  }
};

} // namespace

void mlir::populateVectorToArmSMEPatterns(RewritePatternSet &patterns,
                                          MLIRContext &ctx) {
  patterns.add<BroadcastOpToArmSMELowering, ConstantOpToArmSMELowering,
               SplatOpToArmSMELowering, TransferReadPermutationToArmSMELowering,
               TransferWriteToArmSMELowering, TransposeOpToArmSMELowering,
               VectorLoadToArmSMELowering, VectorStoreToArmSMELowering>(&ctx);
}
