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
/// the tile slice index and each iteration yields a new tile. Loop body is
/// built via the callback, which returns the next tile value.
template <typename LoopBodyCallback>
static scf::ForOp createLoopOverTileSlices(PatternRewriter &rewriter,
                                           Location loc, Value initTile,
                                           LoopBodyCallback callback) {
  OpBuilder::InsertionGuard g(rewriter);
  auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto minTileSlices = rewriter.create<arith::ConstantIndexOp>(
      loc, llvm::cast<VectorType>(initTile.getType()).getDimSize(0));
  auto vscale =
      rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
  auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto numTileSlices =
      rewriter.create<arith::MulIOp>(loc, minTileSlices, vscale);
  auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, numTileSlices, step,
                                           ValueRange{initTile});
  rewriter.setInsertionPointToStart(forOp.getBody());
  auto nextTile = callback(forOp);
  rewriter.create<scf::YieldOp>(loc, nextTile.getResult());
  return forOp;
}

namespace {

/// Conversion pattern for vector.transfer_read.
///
/// ---
///
/// Example 1: op with identity permutation map to horizontal
///            arm_sme.tile_load:
///
///   vector.transfer_read ...  permutation_map: (d0, d1) -> (d0, d1)
///
/// is converted to:
///
///   arm_sme.tile_load ...
///
/// ---
///
/// Example 2: op with transpose permutation map to vertical arm_sme.tile_load
///            (in-flight transpose):
///
///   vector.transfer_read ...  permutation_map: (d0, d1) -> (d1, d0)
///
/// is converted to:
///
///   arm_sme.tile_load ... layout<vertical>
struct TransferReadToArmSMELowering
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp transferReadOp,
                                PatternRewriter &rewriter) const final {
    // The permutation map must have two results.
    if (transferReadOp.getTransferRank() != 2)
      return rewriter.notifyMatchFailure(transferReadOp,
                                         "not a 2 result permutation map");

    auto vectorType = transferReadOp.getVectorType();
    if (!arm_sme::isValidSMETileVectorType(vectorType))
      return rewriter.notifyMatchFailure(transferReadOp,
                                         "not a valid vector type for SME");

    if (!llvm::isa<MemRefType>(transferReadOp.getSource().getType()))
      return rewriter.notifyMatchFailure(transferReadOp, "not a memref source");

    // Out-of-bounds dims are not supported.
    if (transferReadOp.hasOutOfBoundsDim())
      return rewriter.notifyMatchFailure(transferReadOp,
                                         "not inbounds transfer read");

    arm_sme::TileSliceLayout layout;

    AffineExpr d0, d1;
    bindDims(transferReadOp.getContext(), d0, d1);
    AffineMap map = transferReadOp.getPermutationMap();
    if (map.isIdentity())
      layout = arm_sme::TileSliceLayout::Horizontal;
    else if (map == AffineMap::get(map.getNumDims(), 0, {d1, d0},
                                   transferReadOp.getContext()))
      layout = arm_sme::TileSliceLayout::Vertical;
    else
      return rewriter.notifyMatchFailure(transferReadOp,
                                         "unsupported permutation map");

    // Padding isn't optional for transfer_read, but is only used in the case
    // of out-of-bounds accesses (not supported here) and/or masking. Mask is
    // optional, if it's not present don't pass padding.
    auto mask = transferReadOp.getMask();
    auto padding = mask ? transferReadOp.getPadding() : nullptr;
    rewriter.replaceOpWithNewOp<arm_sme::TileLoadOp>(
        transferReadOp, vectorType, transferReadOp.getSource(),
        transferReadOp.getIndices(), padding, mask, layout);

    return success();
  }
};

/// Conversion pattern for vector.transfer_write.
///
/// ---
///
/// Example 1: op with identity permutation map to horizontal
///            arm_sme.tile_store:
///
///   vector.transfer_write %vector, %source[%c0, %c0]
///     {in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>
///
/// is converted to:
///
///   arm_sme.tile_store %vector, %source[%c0, %c0] : memref<?x?xi8>,
///                                                   vector<[16]x[16]xi8>
/// ---
///
/// Example 2: op with transpose permutation map to vertical arm_sme.tile_store
///            (in-flight transpose):
///
///   vector.transfer_write %vector, %source[%c0, %c0]
///     {permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
///      in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>
///
/// is converted to:
///
///   arm_sme.tile_store %vector, %source[%c0, %c0] layout<vertical>
///     : memref<?x?xi8>, vector<[16]x[16]xi8>
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

    // Out-of-bounds dims are not supported.
    if (writeOp.hasOutOfBoundsDim())
      return rewriter.notifyMatchFailure(writeOp,
                                         "not inbounds transfer write");

    AffineExpr d0, d1;
    bindDims(writeOp.getContext(), d0, d1);
    AffineMap map = writeOp.getPermutationMap();
    bool isTranspose = (map == AffineMap::get(map.getNumDims(), 0, {d1, d0},
                                              writeOp.getContext()));

    if (!map.isIdentity() && !isTranspose)
      return rewriter.notifyMatchFailure(writeOp,
                                         "unsupported permutation map");

    arm_sme::TileSliceLayout layout =
        isTranspose ? arm_sme::TileSliceLayout::Vertical
                    : arm_sme::TileSliceLayout::Horizontal;

    rewriter.replaceOpWithNewOp<arm_sme::TileStoreOp>(
        writeOp, writeOp.getVector(), writeOp.getSource(), writeOp.getIndices(),
        writeOp.getMask(), layout);
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
    auto loc = constantOp.getLoc();

    // To fill a tile with a constant, we create a 1-D splat of the constant,
    // then move that into each tile slice (the largest unit we can set at once,
    // outside of operations like the outerproduct).
    VectorType tileSliceType = VectorType::Builder(tileType).dropDim(0);
    auto denseAttr1D = DenseElementsAttr::get(
        tileSliceType, denseAttr.getSplatValue<Attribute>());
    auto constantOp1D = rewriter.create<arith::ConstantOp>(loc, denseAttr1D);

    auto initTile = rewriter.create<arm_sme::GetTileOp>(loc, tileType);
    auto forOp =
        createLoopOverTileSlices(rewriter, loc, initTile, [&](auto forOp) {
          auto tileSliceIndex = forOp.getInductionVar();
          auto currentTile = forOp.getRegionIterArg(0);
          // Create 'arm_sme.move_vector_to_tile_slice' to write vector to tile
          // slice.
          return rewriter.create<arm_sme::MoveVectorToTileSliceOp>(
              loc, tileType, constantOp1D, currentTile, tileSliceIndex);
        });
    rewriter.replaceOp(constantOp, forOp.getResult(0));

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
///   %broadcast_to_tile = scf.for %tile_slice_index = %c0 to %num_tile_slices
///       step %c1 iter_args(%iter_tile = %init_tile) -> (vector<[4]x[4]xi32>)
///   {
///     %tile_update = arm_sme.move_vector_to_tile_slice
///        %broadcast_to_1d, %iter_tile, %tile_slice_index :
///        vector<[4]xi32> into vector<[4]x[4]xi32>
///     scf.yield %tile_update : vector<[4]x[4]xi32>
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

    auto loc = broadcastOp.getLoc();

    auto srcType = broadcastOp.getSourceType();
    auto srcVectorType = dyn_cast<VectorType>(srcType);

    Value broadcastOp1D;
    if (srcType.isIntOrFloat() ||
        (srcVectorType && (srcVectorType.getRank() == 0))) {
      // Broadcast scalar or 0-d vector to 1-d vector.
      VectorType tileSliceType = VectorType::Builder(tileType).dropDim(0);
      broadcastOp1D = rewriter.create<vector::BroadcastOp>(
          loc, tileSliceType, broadcastOp.getSource());
    } else if (srcVectorType && (srcVectorType.getRank() == 1))
      // Value to broadcast is already a 1-d vector, nothing to do.
      broadcastOp1D = broadcastOp.getSource();
    else
      return failure();

    auto initTile = rewriter.create<arm_sme::GetTileOp>(loc, tileType);

    // Create a loop over ZA tile slices.
    auto forOp =
        createLoopOverTileSlices(rewriter, loc, initTile, [&](auto forOp) {
          auto tileSliceIndex = forOp.getInductionVar();
          auto currentTile = forOp.getRegionIterArg(0);
          // Create 'arm_sme.move_vector_to_tile_slice' to broadcast the value
          // to each tile slice.
          return rewriter.create<arm_sme::MoveVectorToTileSliceOp>(
              loc, tileType, broadcastOp1D, currentTile, tileSliceIndex);
        });

    rewriter.replaceOp(broadcastOp, forOp.getResult(0));

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
///   %broadcast_to_tile = scf.for %tile_slice_index = %c0 to %num_tile_slices
///       step %c1 iter_args(%iter_tile = %init_tile) -> (vector<[4]x[4]xi32>)
///   {
///     %tile_update = arm_sme.move_vector_to_tile_slice
///        %broadcast_to_1d, %iter_tile, %tile_slice_index :
///        vector<[4]xi32> into vector<[4]x[4]xi32>
///     scf.yield %tile_update : vector<[4]x[4]xi32>
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

    auto loc = splatOp.getLoc();
    auto srcType = splatOp.getOperand().getType();

    assert(srcType.isIntOrFloat() && "Invalid source type for vector.splat");
    // Avoid unused-variable warning when building without assertions.
    (void)srcType;

    // First, broadcast the scalar to a 1-d vector.
    VectorType tileSliceType = VectorType::Builder(tileType).dropDim(0);
    Value broadcastOp1D = rewriter.create<vector::BroadcastOp>(
        loc, tileSliceType, splatOp.getInput());

    auto initTile = rewriter.create<arm_sme::GetTileOp>(loc, tileType);

    // Next, create a loop over ZA tile slices and "move" the generated 1-d
    // vector to each slice.
    auto forOp =
        createLoopOverTileSlices(rewriter, loc, initTile, [&](auto forOp) {
          auto tileSliceIndex = forOp.getInductionVar();
          auto currentTile = forOp.getRegionIterArg(0);
          return rewriter.create<arm_sme::MoveVectorToTileSliceOp>(
              loc, tileType, broadcastOp1D, currentTile, tileSliceIndex);
        });

    rewriter.replaceOp(splatOp, forOp.getResult(0));

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

    // Bail unless this is a true 2-D matrix transpose.
    ArrayRef<int64_t> permutation = transposeOp.getPermutation();
    if (permutation[0] != 1 || permutation[1] != 0)
      return failure();

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

/// Conversion pattern for vector.outerproduct.
///
/// If the vector.outerproduct is masked (and the mask is from a
/// vector.create_mask), then the mask is decomposed into two 1-D masks for the
/// operands.
///
/// Example:
///
///   %mask = vector.create_mask %dimA, %dimB : vector<[4]x[4]xi1>
///   %result = vector.mask %mask {
///                vector.outerproduct %vecA, %vecB
///                 : vector<[4]xf32>, vector<[4]xf32>
///             } : vector<[4]x[4]xi1> -> vector<[4]x[4]xf32>
///
/// is converted to:
///
///    %maskA = vector.create_mask %dimA : vector<[4]xi1>
///    %maskB = vector.create_mask %dimB : vector<[4]xi1>
///    %result = arm_sme.outerproduct %vecA, %vecB masks(%maskA, %maskB)
///                : vector<[4]xf32>, vector<[4]xf32>
///
/// Unmasked outerproducts can be directly replaced with the arm_sme op.
///
/// Example:
///
///   %result = vector.outerproduct %vecA, %vecB
///              : vector<[4]xf32>, vector<[4]xf32>
///
/// is converted to:
///
///   %result = arm_sme.outerproduct %vecA, %vecB
///              : vector<[4]xf32>, vector<[4]xf32>
///
struct VectorOuterProductToArmSMELowering
    : public OpRewritePattern<vector::OuterProductOp> {

  using OpRewritePattern<vector::OuterProductOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::OuterProductOp outerProductOp,
                                PatternRewriter &rewriter) const override {

    // We don't yet support lowering AXPY operations to SME. These could be
    // lowered by masking out all but the first element of the LHS.
    if (!isa<VectorType>(outerProductOp.getOperandTypeRHS()))
      return rewriter.notifyMatchFailure(outerProductOp,
                                         "AXPY operations not supported");

    if (!arm_sme::isValidSMETileVectorType(
            outerProductOp.getResultVectorType()))
      return rewriter.notifyMatchFailure(
          outerProductOp, "outer product does not fit into SME tile");

    auto kind = outerProductOp.getKind();
    if (kind != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(
          outerProductOp,
          "unsupported kind (lowering to SME only supports ADD at the moment)");

    Value lhsMask = {};
    Value rhsMask = {};
    Operation *rootOp = outerProductOp;
    auto loc = outerProductOp.getLoc();
    if (outerProductOp.isMasked()) {
      auto maskOp = outerProductOp.getMaskingOp();
      rewriter.setInsertionPoint(maskOp);
      rootOp = maskOp;
      auto operandMasks = decomposeResultMask(loc, maskOp.getMask(), rewriter);
      if (failed(operandMasks))
        return failure();
      std::tie(lhsMask, rhsMask) = *operandMasks;
    }

    rewriter.replaceOpWithNewOp<arm_sme::OuterProductOp>(
        rootOp, outerProductOp.getResultVectorType(), outerProductOp.getLhs(),
        outerProductOp.getRhs(), lhsMask, rhsMask, outerProductOp.getAcc());

    return success();
  }

  static FailureOr<std::pair<Value, Value>>
  decomposeResultMask(Location loc, Value mask, PatternRewriter &rewriter) {
    // Attempt to extract masks from vector.create_mask.
    // TODO: Add support for other mask sources.
    auto createMaskOp = mask.getDefiningOp<vector::CreateMaskOp>();
    if (!createMaskOp)
      return failure();

    auto maskType = createMaskOp.getVectorType();
    Value lhsMaskDim = createMaskOp.getOperand(0);
    Value rhsMaskDim = createMaskOp.getOperand(1);

    VectorType operandMaskType = VectorType::Builder(maskType).dropDim(0);
    Value lhsMask =
        rewriter.create<vector::CreateMaskOp>(loc, operandMaskType, lhsMaskDim);
    Value rhsMask =
        rewriter.create<vector::CreateMaskOp>(loc, operandMaskType, rhsMaskDim);

    return std::make_pair(lhsMask, rhsMask);
  }
};

/// Lower `vector.extract` using `arm_sme.move_tile_slice_to_vector`.
///
/// Example:
/// ```
/// %el = vector.extract %tile[%row, %col]: i32 from vector<[4]x[4]xi32>
/// ```
/// Becomes:
/// ```
/// %slice = arm_sme.move_tile_slice_to_vector %tile[%row]
///            : vector<[4]xi32> from vector<[4]x[4]xi32>
/// %el = vector.extract %slice[%col] : i32 from vector<[4]xi32>
/// ```
struct VectorExtractToArmSMELowering
    : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern<vector::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    VectorType sourceType = extractOp.getSourceVectorType();
    if (!arm_sme::isValidSMETileVectorType(sourceType))
      return failure();

    auto loc = extractOp.getLoc();
    auto position = extractOp.getMixedPosition();

    Value sourceVector = extractOp.getVector();

    // Extract entire vector. Should be handled by folder, but just to be safe.
    if (position.empty()) {
      rewriter.replaceOp(extractOp, sourceVector);
      return success();
    }

    Value sliceIndex = vector::getAsValues(rewriter, loc, position[0]).front();
    auto moveTileSliceToVector =
        rewriter.create<arm_sme::MoveTileSliceToVectorOp>(loc, sourceVector,
                                                          sliceIndex);

    if (position.size() == 1) {
      // Single index case: Extracts a 1D slice.
      rewriter.replaceOp(extractOp, moveTileSliceToVector);
      return success();
    }

    // Two indices case: Extracts a single element.
    assert(position.size() == 2);
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(
        extractOp, moveTileSliceToVector, position[1]);

    return success();
  }
};

/// Lower `vector.insert` using `arm_sme.move_vector_to_tile_slice` and
/// `arm_sme.move_tile_slice_to_vector`.
///
/// Example:
/// ```
/// %new_tile = vector.insert %el, %tile[%row, %col]
///                     : i32 into vector<[4]x[4]xi32>
/// ```
/// Becomes:
/// ```
/// %slice = arm_sme.move_tile_slice_to_vector %tile[%row]
///            : vector<[4]xi32> from vector<[4]x[4]xi32>
/// %new_slice = vector.insert %el, %slice[%col] : i32 into vector<[4]xi32>
/// %new_tile = arm_sme.move_vector_to_tile_slice %new_slice, %tile, %row
///               : vector<[4]xi32> into vector<[4]x[4]xi32>
/// ```
struct VectorInsertToArmSMELowering
    : public OpRewritePattern<vector::InsertOp> {
  using OpRewritePattern<vector::InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertOp insertOp,
                                PatternRewriter &rewriter) const override {
    VectorType resultType = insertOp.getResult().getType();

    if (!arm_sme::isValidSMETileVectorType(resultType))
      return failure();

    auto loc = insertOp.getLoc();
    auto position = insertOp.getMixedPosition();

    Value source = insertOp.getSource();

    // Overwrite entire vector with value. Should be handled by folder, but
    // just to be safe.
    if (position.empty()) {
      rewriter.replaceOp(insertOp, source);
      return success();
    }

    Value tileSlice = source;
    Value sliceIndex = vector::getAsValues(rewriter, loc, position[0]).front();
    if (position.size() == 2) {
      // Two indices case: Insert single element into tile.
      // We need to first extract the existing slice and update the element.
      tileSlice = rewriter.create<arm_sme::MoveTileSliceToVectorOp>(
          loc, insertOp.getDest(), sliceIndex);
      tileSlice = rewriter.create<vector::InsertOp>(loc, source, tileSlice,
                                                    position[1]);
    }

    // Insert the slice into the destination tile.
    rewriter.replaceOpWithNewOp<arm_sme::MoveVectorToTileSliceOp>(
        insertOp, tileSlice, insertOp.getDest(), sliceIndex);
    return success();
  }
};

/// Lowers `vector.print` of a tile into a loop over the rows of the tile,
/// extracting them via `arm_sme.move_tile_slice_to_vector`, then printing with
/// a 1D `vector.print`.
///
///  BEFORE:
///  ```mlir
///  vector.print %tile : vector<[4]x[4]xf32>
///  ```
///  AFTER:
///  ```mlir
///  %c0 = arith.constant 0 : index
///  %c1 = arith.constant 1 : index
///  %c4 = arith.constant 4 : index
///  %vscale = vector.vscale
///  %svl_s = arith.muli %c4, %vscale : index
///  scf.for %i = %c0 to %svl_s step %c1 {
///    %tile_slice = arm_sme.move_tile_slice_to_vector %tile[%i]
///                     : vector<[4]xf32> from vector<[4]x[4]xf32>
///    vector.print %tile_slice : vector<[4]xf32>
///  }
///  ```
struct VectorPrintToArmSMELowering : public OpRewritePattern<vector::PrintOp> {
  using OpRewritePattern<vector::PrintOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::PrintOp printOp,
                                PatternRewriter &rewriter) const override {
    if (!printOp.getSource())
      return failure();

    VectorType vectorType = dyn_cast<VectorType>(printOp.getPrintType());
    if (!vectorType || !arm_sme::isValidSMETileVectorType(vectorType))
      return failure();

    auto loc = printOp.getLoc();

    // Create a loop over the rows of the tile.
    auto vscale = rewriter.create<vector::VectorScaleOp>(loc);
    auto minTileRows =
        rewriter.create<arith::ConstantIndexOp>(loc, vectorType.getDimSize(0));
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto upperBound = rewriter.create<arith::MulIOp>(loc, minTileRows, vscale);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
    {
      // Loop body.
      rewriter.setInsertionPointToStart(forOp.getBody());
      // Extract the current row from the tile.
      Value rowIndex = forOp.getInductionVar();
      auto tileSlice = rewriter.create<arm_sme::MoveTileSliceToVectorOp>(
          loc, printOp.getSource(), rowIndex);
      // Print the row with a 1D vector.print.
      rewriter.create<vector::PrintOp>(loc, tileSlice,
                                       printOp.getPunctuation());
    }

    rewriter.eraseOp(printOp);
    return success();
  }
};

} // namespace

void mlir::populateVectorToArmSMEPatterns(RewritePatternSet &patterns,
                                          MLIRContext &ctx) {
  patterns
      .add<BroadcastOpToArmSMELowering, ConstantOpToArmSMELowering,
           SplatOpToArmSMELowering, TransferReadToArmSMELowering,
           TransferWriteToArmSMELowering, TransposeOpToArmSMELowering,
           VectorLoadToArmSMELowering, VectorStoreToArmSMELowering,
           VectorOuterProductToArmSMELowering, VectorExtractToArmSMELowering,
           VectorInsertToArmSMELowering, VectorPrintToArmSMELowering>(&ctx);
}
