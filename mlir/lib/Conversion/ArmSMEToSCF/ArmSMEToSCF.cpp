//===- ArmSMEToSCF.cpp - Convert ArmSME to SCF dialect ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of ArmSME operations to SCF.
//
//===----------------------------------------------------------------------===//
#include "mlir/Conversion/ArmSMEToSCF/ArmSMEToSCF.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTARMSMETOSCF
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// Adjusts `indices` as follows for a given tile slice and returns them in
/// `outIndices`:
///   rank 1: (indices[0] + (tileSliceIndex * tileSliceNumElts))
///   rank 2: (indices[0] + tileSliceIndex, indices[1])
void getMemrefIndices(ValueRange indices, unsigned rank, Value tileSliceIndex,
                      Value tileSliceNumElts,
                      SmallVectorImpl<Value> &outIndices, Location loc,
                      PatternRewriter &rewriter) {
  assert((rank == 1 || rank == 2) && "memref has unexpected rank!");

  auto tileSliceOffset = tileSliceIndex;
  if (rank == 1)
    tileSliceOffset =
        rewriter.create<arith::MulIOp>(loc, tileSliceOffset, tileSliceNumElts);

  auto baseIndexPlusTileSliceOffset =
      rewriter.create<arith::AddIOp>(loc, indices[0], tileSliceOffset);
  outIndices.push_back(baseIndexPlusTileSliceOffset);

  if (rank == 2)
    outIndices.push_back(indices[1]);
}

/// Lower `arm_sme.tile_load` to a loop over the tile slices and load each slice
/// using `arm_sme.load_tile_slice`.
///
///  BEFORE:
///  ```mlir
///  %tile = arm_sme.tile_load %src[%c0, %c0] :
///    memref<?x?xi32>, vector<[4]x[4]xi32>
///  ```
///
///  AFTER:
///  ```mlir
///  %ptrue_s = arith.constant dense<true> : vector<[4]xi1>
///  %tile_id = arm_sme.get_tile_id : i32
///  %tile = arm_sme.cast_tile_to_vector %tile_id : i32 to vector<[4]x[4]xi32>
///  %vscale = vector.vscale
///  %c0 = arith.constant 0 : index
///  %c1 = arith.constant 1 : index
///  %min_svl_s = arith.constant 4 : index
///  %svl_s = arith.muli %min_svl_s, %vscale : index
///  scf.for %tile_slice_idx = %c0 to %svl_s step %c1 {
///    %tile_update = arm_sme.load_tile_slice %src[%tile_slice_idx],
///      %ptrue_s, %tile, %tile_slice_idx
///        : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
///  }
///  ```
struct TileLoadOpConversion : public OpRewritePattern<arm_sme::TileLoadOp> {
  using OpRewritePattern<arm_sme::TileLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arm_sme::TileLoadOp tileLoadOp,
                                PatternRewriter &rewriter) const override {
    if (tileLoadOp.getMask())
      return rewriter.notifyMatchFailure(
          tileLoadOp, "op has mask, needs masked pattern(s)");

    OpBuilder::InsertionGuard g(rewriter);
    auto loc = tileLoadOp.getLoc();
    auto tileType = tileLoadOp.getVectorType();
    auto tileElementType = tileType.getElementType();
    unsigned tileElementWidth = tileElementType.getIntOrFloatBitWidth();

    // Create 'arm_sme.get_tile' op.
    auto tileId = rewriter.create<arm_sme::GetTileID>(
        loc, rewriter.getIntegerType(tileElementWidth));

    // Create `arm_sme.cast_tile_to_vector` to cast tile ID to a vector type to
    // use as input tile to 'arm_sme.load_tile_slice' ops.
    auto tile =
        rewriter.create<arm_sme::CastTileToVector>(loc, tileType, tileId);

    // Create a loop that loads each ZA tile slice from memory.
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto minTileSlices = rewriter.create<arith::ConstantIndexOp>(
        loc, arm_sme::getSMETileSliceMinNumElts(tileElementType));
    auto vscale =
        rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // This describes both the number of ZA tile slices and the number of
    // elements in a vector of SVL bits for a given element type (SVL_B, SVL_H,
    // ..., SVL_Q).
    auto numTileSlices =
        rewriter.create<arith::MulIOp>(loc, minTileSlices, vscale);
    auto forOp =
        rewriter.create<scf::ForOp>(loc, lowerBound, numTileSlices, step);

    rewriter.setInsertionPointToStart(forOp.getBody());

    // Create an 'all true' predicate for the tile slice.
    auto predicateType =
        VectorType::get(tileType.getDimSize(1), rewriter.getI1Type(), true);
    auto allTruePredicate = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(predicateType, true));

    // Create 'arm_sme.load_tile_slice' to load tile slice from memory into
    // tile.
    SmallVector<Value> memrefIndices;
    auto tileSliceIndex = forOp.getInductionVar();
    getMemrefIndices(tileLoadOp.getIndices(),
                     tileLoadOp.getMemRefType().getRank(), tileSliceIndex,
                     numTileSlices, memrefIndices, loc, rewriter);
    rewriter.create<arm_sme::LoadTileSliceOp>(
        loc, tileType, tileLoadOp.getBase(), allTruePredicate, tile,
        memrefIndices, tileSliceIndex, tileLoadOp.getLayout());

    rewriter.setInsertionPointAfter(forOp);

    // Replace 'arm_sme.tile_load' with the tile.
    rewriter.replaceOp(tileLoadOp, tile);

    return success();
  }
};

/// Lower `arm_sme.tile_load` with mask and pad of constant zero.
///
///  BEFORE:
///  ```mlir
///  %pad = arith.constant 0 : i32
///  %num_rows = arith.constant 2 : index
///  %num_cols = arith.constant 4 : index
///  %mask = vector.create_mask %num_rows, %num_cols : vector<[4]x[4]xi1>
///  %tile = arm_sme.tile_load %src[%c0, %c0], %pad, %mask :
///    memref<?x?xi32>, vector<[4]x[4]xi32>
///  ```
///
///  AFTER:
///  ```mlir
///  %c0 = arith.constant 0 : index
///  %c1 = arith.constant 1 : index
///  %tile = arm_sme.zero : vector<[4]x[4]xi32>
///  %num_cols = vector.create_mask %c4 : vector<[4]xi1>
///  scf.for %tile_slice_idx = %c0 to %num_rows step %c1 {
///    %tile_update = arm_sme.load_tile_slice
///      %src[%tile_slice_idx], %num_cols, %tile, %tile_slice_idx :
///      memref<?x?xi32>, vector<[1]xi32>, vector<[4]x[4]xi32>
///  }
///  ```
///
/// NOTE: Only mask of 'vector.create_mask' op is currently supported.
struct TileLoadOpWithMaskAndPadZeroConversion
    : public OpRewritePattern<arm_sme::TileLoadOp> {
  using OpRewritePattern<arm_sme::TileLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arm_sme::TileLoadOp tileLoadOp,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard g(rewriter);
    auto loc = tileLoadOp.getLoc();
    auto tileType = tileLoadOp.getVectorType();

    auto maskOp = tileLoadOp.getMask();
    if (!maskOp)
      return rewriter.notifyMatchFailure(
          tileLoadOp, "op has no mask, needs unmasked pattern");

    auto padOp = tileLoadOp.getPadding();
    assert(padOp && "expected padding when masking!");

    auto createMaskOp = maskOp.getDefiningOp<vector::CreateMaskOp>();
    if (!createMaskOp)
      return rewriter.notifyMatchFailure(
          tileLoadOp, "unsupported mask op, only 'vector.create_mask' is "
                      "currently supported");

    auto constPadOp = padOp.getDefiningOp<arith::ConstantOp>();
    if (!constPadOp || constPadOp.getValue() !=
                           rewriter.getZeroAttr(tileType.getElementType()))
      return rewriter.notifyMatchFailure(
          tileLoadOp, "op has non-zero pad, needs non-zero pad pattern");

    auto numRows = createMaskOp.getOperands()[0];
    auto numCols = createMaskOp.getOperands()[1];

    auto predicateType =
        VectorType::get(tileType.getDimSize(1), rewriter.getI1Type(), true);
    auto numColsOp =
        rewriter.create<vector::CreateMaskOp>(loc, predicateType, numCols);

    // Initialize tile with zero to satisfy padding. Inactive cols will be
    // zeroed anyway since the loads use zeroing predication. For inactive rows
    // however, no load will occur so these need to be zeroed.
    auto tile = rewriter.create<arm_sme::ZeroOp>(loc, tileType);

    // Create a loop to load the active tile slices from memory.
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto upperBound = numRows;
    auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

    rewriter.setInsertionPointToStart(forOp.getBody());

    // Create 'arm_sme.load_tile_slice' to load tile slice from memory into
    // tile.
    SmallVector<Value> memrefIndices;
    auto tileSliceIndex = forOp.getInductionVar();
    getMemrefIndices(tileLoadOp.getIndices(),
                     tileLoadOp.getMemRefType().getRank(), tileSliceIndex,
                     upperBound, memrefIndices, loc, rewriter);
    rewriter.create<arm_sme::LoadTileSliceOp>(
        loc, tileType, tileLoadOp.getBase(), numColsOp, tile, memrefIndices,
        tileSliceIndex, tileLoadOp.getLayout());

    rewriter.setInsertionPointAfter(forOp);

    // Replace 'arm_sme.tile_load' with the tile.
    rewriter.replaceOp(tileLoadOp, tile);

    return success();
  }
};

/// Lower `arm_sme.tile_load` with mask and non-zero pad.
///
///  BEFORE:
///  ```mlir
///  %pad = arith.constant 1 : i32
///  %num_rows = arith.constant 2 : index
///  %num_cols = arith.constant 4 : index
///  %mask = vector.create_mask %num_rows, %num_cols : vector<[4]x[4]xi1>
///  %tile = arm_sme.tile_load %src[%c0, %c0], %pad, %mask :
///    memref<?x?xi32>, vector<[4]x[4]xi32>
///  ```
///
///  AFTER:
///  ```mlir
///  %pad_1d = arith.constant dense<1> : vector<[4]xi32>
///  %num_rows = arith.constant 2 : index
///  %num_cols = arith.constant 4 : index
///  %tile_id = arm_sme.get_tile_id : i32
///  %tile = arm_sme.cast_tile_to_vector %tile_id : i32 to vector<[4]x[4]xi32>
///  %vscale = vector.vscale
///  %c0 = arith.constant 0 : index
///  %c1 = arith.constant 1 : index
///  %min_svl_s = arith.constant 4 : index
///  %svl_s = arith.muli %min_svl_s, %vscale : index
///  scf.for %tile_slice_idx = %c0 to %svl_s step %c1 {
///    %row_is_active = arith.cmpi ult %tile_slice_idx, %num_rows : index
///    %slice = scf.if %row_is_active -> vector<[4]xi32> {
///      %slice = vector.maskedload %base[%tile_slice_idx, %c0], %num_cols, %pad
///        : memref<?x?xi32>, vector<[4]xi1>,
///          vector<[4]xi32> into vector<[4]xi32>
///      scf.yield %slice : vector<[4]xi32>
///    } else {
///      scf.yield %pad_1d : vector<[4]xi32>
///    }
///    // Insert slice into tile
///    arm_sme.move_vector_to_tile_slice %slice, %tile, %tile_slice_idx
///      : vector<[4]xi32> into vector<[4]x[4]xi32>
///  }
///  ```
struct TileLoadOpWithMaskAndPadNonZeroConversion
    : public OpRewritePattern<arm_sme::TileLoadOp> {
  using OpRewritePattern<arm_sme::TileLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arm_sme::TileLoadOp tileLoadOp,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard g(rewriter);
    auto loc = tileLoadOp.getLoc();
    auto tileType = tileLoadOp.getVectorType();
    auto tileElementType = tileType.getElementType();
    unsigned tileElementWidth = tileElementType.getIntOrFloatBitWidth();

    auto maskOp = tileLoadOp.getMask();
    if (!maskOp)
      return rewriter.notifyMatchFailure(
          tileLoadOp, "op has no mask, needs unmasked pattern");

    auto padOp = tileLoadOp.getPadding();
    assert(padOp && "expected padding when masking!");

    auto createMaskOp = maskOp.getDefiningOp<vector::CreateMaskOp>();
    if (!createMaskOp)
      return rewriter.notifyMatchFailure(
          tileLoadOp, "unsupported mask op, only 'vector.create_mask' is "
                      "currently supported");

    auto constPadOp = padOp.getDefiningOp<arith::ConstantOp>();
    if (constPadOp &&
        constPadOp.getValue() == rewriter.getZeroAttr(tileElementType))
      return rewriter.notifyMatchFailure(
          tileLoadOp, "op has constant zero pad, needs zero pad pattern");

    auto numRows = createMaskOp.getOperands()[0];
    auto numCols = createMaskOp.getOperands()[1];

    VectorType tileSliceType = VectorType::Builder(tileType).dropDim(0);
    auto predicateType =
        VectorType::get(tileType.getDimSize(1), rewriter.getI1Type(), true);
    auto numColsOp =
        rewriter.create<vector::CreateMaskOp>(loc, predicateType, numCols);

    // Create 'arm_sme.get_tile' op.
    auto tileId = rewriter.create<arm_sme::GetTileID>(
        loc, rewriter.getIntegerType(tileElementWidth));

    // Create `arm_sme.cast_tile_to_vector` to cast tile ID to a vector type to
    // use as input tile to 'arm_sme.load_tile_slice' ops.
    auto tile =
        rewriter.create<arm_sme::CastTileToVector>(loc, tileType, tileId);

    // Create a loop that loads each ZA tile slice from memory.
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto minTileSlices = rewriter.create<arith::ConstantIndexOp>(
        loc, arm_sme::getSMETileSliceMinNumElts(tileElementType));
    auto vscale =
        rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto numTileSlices =
        rewriter.create<arith::MulIOp>(loc, minTileSlices, vscale);
    auto forOp =
        rewriter.create<scf::ForOp>(loc, lowerBound, numTileSlices, step);

    rewriter.setInsertionPointToStart(forOp.getBody());

    auto tileSliceIndex = forOp.getInductionVar();

    auto rowIsActive = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, tileSliceIndex, numRows);

    SmallVector<Value> memrefIndices;
    getMemrefIndices(tileLoadOp.getIndices(),
                     tileLoadOp.getMemRefType().getRank(), tileSliceIndex,
                     numTileSlices, memrefIndices, loc, rewriter);

    // Splat pad into 1-D vector matching type of tile slice.
    auto pad1DOp = rewriter.create<vector::SplatOp>(loc, tileSliceType, padOp);

    Operation *slice = rewriter.create<scf::IfOp>(
        loc, rowIsActive,
        [&](OpBuilder &b, Location loc) {
          // If the row is active, emit a masked load where the predicate is
          // 'numCols'. Pad is used for inactive elements, taken from
          // passthru.
          auto loadSlice = rewriter.create<vector::MaskedLoadOp>(
              loc, tileSliceType, tileLoadOp.getBase(), memrefIndices,
              numColsOp, /*passthru=*/pad1DOp);
          rewriter.create<scf::YieldOp>(loc, loadSlice->getResult(0));
        },
        [&](OpBuilder &b, Location loc) {
          // Inactive rows are filled with pad.
          rewriter.create<scf::YieldOp>(loc, pad1DOp.getResult());
        });

    // TODO: If the load is vertical the transpose can't be done in-flight with
    // a regular (SVE) maskedload. Propagate layout to
    // 'arm_sme.move_vector_to_tile_slice' below once it supports layout. This
    // is currently broken.

    // Create 'arm_sme.move_vector_to_tile_slice' to move slice into tile.
    rewriter.create<arm_sme::MoveVectorToTileSliceOp>(
        loc, tileType, slice->getResult(0), tile, tileSliceIndex,
        tileLoadOp.getLayout());

    rewriter.setInsertionPointAfter(forOp);

    // Replace 'arm_sme.tile_load' with the tile.
    rewriter.replaceOp(tileLoadOp, tile);

    return success();
  }
};

/// Lower `arm_sme.tile_store` to a loop over the tile slices and store each
/// slice using `arm_sme.store_tile_slice`.
///
///  BEFORE:
///  ```mlir
///  arm_sme.tile_store %tile, %dest[%c0, %c0] layout<vertical>
///    : memref<?x?xi32>, vector<[4]x[4]xi32
///  ```
///
///  AFTER:
///  ```mlir
///  %vscale = vector.vscale
///  %c0 = arith.constant 0 : index
///  %c1 = arith.constant 1 : index
///  %min_svl_s = arith.constant 4 : index
///  %svl_s = arith.muli %min_svl_s, %vscale : index
///  scf.for %tile_slice_idx = %c0 to %svl_s step %c1 {
///    arm_sme.store_tile_slice %tile, %tile_slice_idx, %dest[%tile_slice_idx],
///      layout<vertical> : memref<?x?xi32>, vector<[4]x[4]xi32>
///  }
///  ```
struct TileStoreOpConversion : public OpRewritePattern<arm_sme::TileStoreOp> {
  using OpRewritePattern<arm_sme::TileStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arm_sme::TileStoreOp tileStoreOp,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard g(rewriter);
    auto loc = tileStoreOp.getLoc();
    auto tileType = tileStoreOp.getVectorType();
    auto tileElementType = tileType.getElementType();

    auto predicateType =
        VectorType::get(tileType.getDimSize(1), rewriter.getI1Type(), true);

    Value maskCols;
    Value upperBound;
    auto maskOp = tileStoreOp.getMask();
    if (maskOp) {
      auto createMaskOp = maskOp.getDefiningOp<vector::CreateMaskOp>();
      if (!createMaskOp)
        return rewriter.notifyMatchFailure(
            tileStoreOp, "unsupported mask op, only 'vector.create_mask' is "
                         "currently supported");

      auto numRows = createMaskOp.getOperands()[0];
      auto numCols = createMaskOp.getOperands()[1];

      upperBound = numRows;
      maskCols =
          rewriter.create<vector::CreateMaskOp>(loc, predicateType, numCols);
    } else {
      // Store all tile slices if no mask.
      auto minTileSlices = rewriter.create<arith::ConstantIndexOp>(
          loc, arm_sme::getSMETileSliceMinNumElts(tileElementType));
      auto vscale =
          rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
      // This describes both the number of ZA tile slices and the number of
      // elements in a vector of SVL bits for a given element type (SVL_B,
      // SVL_H,
      // ..., SVL_Q).
      auto numTileSlices =
          rewriter.create<arith::MulIOp>(loc, minTileSlices, vscale);

      upperBound = numTileSlices;
      // Create an 'all true' predicate for the tile slice.
      maskCols = rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(predicateType, true));
    }

    // Create a loop that stores each (active) active ZA tile slice from memory.
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

    rewriter.setInsertionPointToStart(forOp.getBody());

    SmallVector<Value> memrefIndices;
    auto tileSliceIndex = forOp.getInductionVar();
    getMemrefIndices(tileStoreOp.getIndices(),
                     tileStoreOp.getMemRefType().getRank(), tileSliceIndex,
                     upperBound, memrefIndices, loc, rewriter);
    rewriter.replaceOpWithNewOp<arm_sme::StoreTileSliceOp>(
        tileStoreOp, tileStoreOp.getValueToStore(), tileSliceIndex, maskCols,
        tileStoreOp.getBase(), memrefIndices, tileStoreOp.getLayout());

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
struct TileVectorPrintOpConversion : public OpRewritePattern<vector::PrintOp> {
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

void mlir::populateArmSMEToSCFConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<TileLoadOpConversion, TileLoadOpWithMaskAndPadZeroConversion,
               TileLoadOpWithMaskAndPadNonZeroConversion, TileStoreOpConversion,
               TileVectorPrintOpConversion>(patterns.getContext());
}

namespace {

struct ConvertArmSMEToSCFPass
    : public impl::ConvertArmSMEToSCFBase<ConvertArmSMEToSCFPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    populateArmSMEToSCFConversionPatterns(patterns);
    target.addLegalDialect<arm_sme::ArmSMEDialect, vector::VectorDialect,
                           arith::ArithDialect, scf::SCFDialect>();
    target.addIllegalOp<arm_sme::TileLoadOp, arm_sme::TileStoreOp>();
    target.addDynamicallyLegalOp<vector::PrintOp>([](vector::PrintOp op) {
      if (!op.getSource())
        return true;
      VectorType vectorType = dyn_cast<VectorType>(op.getPrintType());
      return !vectorType || !arm_sme::isValidSMETileVectorType(vectorType);
    });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertArmSMEToSCFPass() {
  return std::make_unique<ConvertArmSMEToSCFPass>();
}
