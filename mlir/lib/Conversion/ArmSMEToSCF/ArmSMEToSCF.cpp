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
/// Returns adjusted (1-D or 2-D) `indices` for a tile slice as follows:
///   rank 1: (indices[0] + (tileSliceIndex * tileSliceNumElts))
///   rank 2: (indices[0] + tileSliceIndex, indices[1])
SmallVector<Value, 2> getMemrefIndices(ValueRange indices, unsigned rank,
                                       Value tileSliceIndex,
                                       Value tileSliceNumElts, Location loc,
                                       PatternRewriter &rewriter) {
  assert((rank == 1 || rank == 2) && "memref has unexpected rank!");
  SmallVector<Value, 2> outIndices;

  auto tileSliceOffset = tileSliceIndex;
  if (rank == 1)
    tileSliceOffset =
        rewriter.create<arith::MulIOp>(loc, tileSliceOffset, tileSliceNumElts);

  auto baseIndexPlusTileSliceOffset =
      rewriter.create<arith::AddIOp>(loc, indices[0], tileSliceOffset);
  outIndices.push_back(baseIndexPlusTileSliceOffset);

  if (rank == 2)
    outIndices.push_back(indices[1]);

  return outIndices;
}

/// Creates an scf.for for the load/store of an ArmSME tile.
FailureOr<scf::ForOp> createLoadStoreForOverTileSlices(
    PatternRewriter &rewriter, Location loc, VectorType tileType,
    ValueRange memrefIndices, int memrefRank, Value mask, Value initTile,
    function_ref<Value(/*index=*/Value, ValueRange, /*predicate=*/Value,
                       /*currentTile=*/Value)>
        makeLoopBody) {
  PatternRewriter::InsertionGuard guard(rewriter);

  auto minTileSlices = rewriter.create<arith::ConstantIndexOp>(
      loc, arm_sme::getSMETileSliceMinNumElts(tileType.getElementType()));
  auto vscale =
      rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
  auto predicateType =
      VectorType::get(tileType.getDimSize(1), rewriter.getI1Type(), true);

  // This describes both the number of ZA tile slices and the number of
  // elements in a vector of SVL bits for a given element type (SVL_B,
  // SVL_H, ..., SVL_Q).
  auto numTileSlices =
      rewriter.create<arith::MulIOp>(loc, minTileSlices, vscale);

  Value predicate;
  Value upperBound;
  if (mask) {
    auto createMaskOp = mask.getDefiningOp<vector::CreateMaskOp>();
    if (!createMaskOp)
      return rewriter.notifyMatchFailure(
          loc, "unsupported mask op, only 'vector.create_mask' is "
               "currently supported");

    auto maskDim0 = createMaskOp.getOperands()[0];
    auto maskDim1 = createMaskOp.getOperands()[1];

    // The upper bound of the loop must be clamped at `numTileSlices` as
    // `vector.create_mask` allows operands to be greater than the size of a
    // dimension.
    auto numRowI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), maskDim0);
    auto numTileSlicesI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), numTileSlices);
    auto upperBoundI64 =
        rewriter.create<arith::MinSIOp>(loc, numRowI64, numTileSlicesI64);
    upperBound = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), upperBoundI64);

    predicate =
        rewriter.create<vector::CreateMaskOp>(loc, predicateType, maskDim1);
  } else {
    upperBound = numTileSlices;
    // No mask. Create an 'all true' predicate for the tile slice.
    predicate = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(predicateType, true));
  }

  bool hasCarriedArgs = bool(initTile);
  auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step,
                                           hasCarriedArgs ? ValueRange{initTile}
                                                          : ValueRange{});

  rewriter.setInsertionPointToStart(forOp.getBody());
  Value tileSliceIndex = forOp.getInductionVar();

  auto adjustedIndices = getMemrefIndices(
      memrefIndices, memrefRank, tileSliceIndex, numTileSlices, loc, rewriter);
  auto nextTile = makeLoopBody(
      tileSliceIndex, adjustedIndices, predicate,
      /*currentTile=*/hasCarriedArgs ? forOp.getRegionIterArg(0) : Value{});

  assert(bool(nextTile) == hasCarriedArgs);
  if (nextTile)
    rewriter.create<scf::YieldOp>(loc, nextTile);

  return forOp;
}

FailureOr<scf::ForOp> createLoadStoreForOverTileSlices(
    PatternRewriter &rewriter, Location loc, VectorType tileType,
    ValueRange memrefIndices, int memrefRank, Value mask,
    function_ref<void(/*index=*/Value, ValueRange, /*predicate=*/Value)>
        makeLoopBody) {
  return createLoadStoreForOverTileSlices(
      rewriter, loc, tileType, memrefIndices, memrefRank, mask, Value{},
      [&](Value index, ValueRange adjustedIndices, Value predicate,
          Value) -> Value {
        makeLoopBody(index, adjustedIndices, predicate);
        return {};
      });
}

/// Lower `arm_sme.tile_load` without a mask, or with a mask and a zero pad.
///
///  With a mask:
///
///  BEFORE:
///  ```mlir
///  %pad = arith.constant 0 : i32
///  %mask = vector.create_mask %num_rows, %num_cols : vector<[4]x[4]xi1>
///  %tile = arm_sme.tile_load %src[%c0, %c0], %pad, %mask :
///    memref<?x?xi32>, vector<[4]x[4]xi32>
///  ```
///
///  AFTER:
///  ```mlir
///  %init_tile = arm_sme.zero : vector<[4]x[4]xi32>
///  %mask_cols = vector.create_mask %num_cols : vector<[4]xi1>
///  %loop_rows = arith.minsi %num_rows, %svl_s : index
///  %tile = scf.for %tile_slice_idx = %c0 to %loop_rows step %c1
///                iter_args(%iter_tile = %init_tile) -> (vector<[4]x[4]xi32>) {
///    %tile_update = arm_sme.load_tile_slice
///      %src[%tile_slice_idx], %num_cols, %iter_tile, %tile_slice_idx :
///      memref<?x?xi32>, vector<[1]xi32>, vector<[4]x[4]xi32>
///    scf.yield %tile_update : vector<[4]x[4]xi32>
///  }
///  ```
///
/// Without a mask the lowering is pretty much identical. The only difference is
/// %mask_cols becomes an all-true mask, and %loop_rows becomes %svl_s.
///
/// NOTE: Only mask of 'vector.create_mask' op is currently supported.
struct TileLoadOpConversion : public OpRewritePattern<arm_sme::TileLoadOp> {
  using OpRewritePattern<arm_sme::TileLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arm_sme::TileLoadOp tileLoadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = tileLoadOp.getLoc();
    auto tileType = tileLoadOp.getVectorType();
    auto mask = tileLoadOp.getMask();

    Value initTile;
    if (mask) {
      auto padOp = tileLoadOp.getPadding();
      assert(padOp && "expected padding when masking!");

      auto constPadOp = padOp.getDefiningOp<arith::ConstantOp>();
      if (!constPadOp || constPadOp.getValue() !=
                             rewriter.getZeroAttr(tileType.getElementType()))
        return rewriter.notifyMatchFailure(
            tileLoadOp, "op has non-zero pad, needs non-zero pad pattern");

      // Initialize tile with zero to satisfy padding. Inactive cols will be
      // zeroed anyway since the loads use zeroing predication. For inactive
      // rows however, no load will occur so these need to be zeroed.
      initTile = tileLoadOp.createOpAndForwardTileId<arm_sme::ZeroOp>(
          rewriter, loc, tileType);
    } else {
      // Allocate a new SME tile.
      initTile = tileLoadOp.createOpAndForwardTileId<arm_sme::GetTileOp>(
          rewriter, loc, tileType);
    }

    // Create a loop to load the active tile slices from memory.
    auto forOp = createLoadStoreForOverTileSlices(
        rewriter, loc, tileType, tileLoadOp.getIndices(),
        tileLoadOp.getMemRefType().getRank(), mask, initTile,
        [&](Value tileSliceIndex, ValueRange memrefIndices, Value predicate,
            Value currentTile) -> Value {
          // Create 'arm_sme.load_tile_slice' to load tile slice from memory
          // into tile.
          return tileLoadOp.createOpAndForwardTileId<arm_sme::LoadTileSliceOp>(
              rewriter, loc, tileType, tileLoadOp.getBase(), predicate,
              currentTile, memrefIndices, tileSliceIndex,
              tileLoadOp.getLayout());
        });

    if (failed(forOp))
      return forOp;

    // Replace 'arm_sme.tile_load' with the result.
    rewriter.replaceOp(tileLoadOp, forOp->getResult(0));

    return success();
  }
};

/// Lower `arm_sme.tile_load` with mask and non-zero pad.
///
///  BEFORE:
///  ```mlir
///  %mask = vector.create_mask %num_rows, %num_cols : vector<[4]x[4]xi1>
///  %tile = arm_sme.tile_load %src[%c0, %c0], %pad, %mask :
///    memref<?x?xi32>, vector<[4]x[4]xi32>
///  ```
///
///  AFTER:
///  ```mlir
///  ...
///  %pad_1d = vector.splat %pad : vector<[4]xi32>
///  %tile = scf.for %tile_slice_idx = %c0 to %svl_s step %c1
///                iter_args(%iter_tile = %init_tile) -> (vector<[4]x[4]xi32>) {
///    ...
///    %mask_1d = vector.create_mask <combined_mask> : vector<[4]xi1>
///    %slice = vector.maskedload %base[%tile_slice_idx, %c0], %mask_1d, %pad_1d
///      : memref<?x?xi32>, vector<[4]xi1>,
///        vector<[4]xi32> into vector<[4]xi32>
///    // Insert slice into tile
///    %tile_update = arm_sme.move_vector_to_tile_slice
///      %slice, %iter_tile, %tile_slice_idx :
///      vector<[4]xi32> into vector<[4]x[4]xi32>
///    scf.yield %tile_update : vector<[4]x[4]xi32>
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

    auto numColsI32 = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getI32Type(), numCols);

    // Allocate a new SME tile.
    auto initTile = tileLoadOp.createOpAndForwardTileId<arm_sme::GetTileOp>(
        rewriter, loc, tileType);

    // Create a loop that loads each ZA tile slice from memory.
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto minTileSlices = rewriter.create<arith::ConstantIndexOp>(
        loc, arm_sme::getSMETileSliceMinNumElts(tileElementType));
    auto vscale =
        rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto numTileSlices =
        rewriter.create<arith::MulIOp>(loc, minTileSlices, vscale);
    auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, numTileSlices,
                                             step, ValueRange{initTile});

    rewriter.setInsertionPointToStart(forOp.getBody());

    auto tileSliceIndex = forOp.getInductionVar();
    auto currentTile = forOp.getRegionIterArg(0);

    // Combine masks.
    auto rowIsActive = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, tileSliceIndex, numRows);
    auto rowIsActiveI32 = rewriter.create<arith::ExtSIOp>(
        loc, rewriter.getI32Type(), rowIsActive);
    auto mask = rewriter.create<arith::AndIOp>(loc, rowIsActiveI32, numColsI32);
    auto maskIndex =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), mask);
    auto predicateType =
        VectorType::get(tileType.getDimSize(1), rewriter.getI1Type(), true);
    auto maskOp1D = rewriter.create<vector::CreateMaskOp>(
        loc, predicateType, maskIndex.getResult());

    auto memrefIndices = getMemrefIndices(
        tileLoadOp.getIndices(), tileLoadOp.getMemRefType().getRank(),
        tileSliceIndex, numTileSlices, loc, rewriter);

    // Splat pad into 1-D vector matching type of tile slice.
    VectorType tileSliceType = VectorType::Builder(tileType).dropDim(0);
    auto pad1DOp = rewriter.create<vector::SplatOp>(loc, tileSliceType, padOp);

    auto loadSlice = rewriter.create<vector::MaskedLoadOp>(
        loc, tileSliceType, tileLoadOp.getBase(), memrefIndices, maskOp1D,
        /*passthru=*/pad1DOp);

    // Create 'arm_sme.move_vector_to_tile_slice' to move slice into tile.
    auto moveSlice =
        tileLoadOp.createOpAndForwardTileId<arm_sme::MoveVectorToTileSliceOp>(
            rewriter, loc, tileType, loadSlice->getResult(0), currentTile,
            tileSliceIndex, tileLoadOp.getLayout());
    rewriter.create<scf::YieldOp>(loc, moveSlice.getResult());

    rewriter.setInsertionPointAfter(forOp);

    // Replace 'arm_sme.tile_load' with the result.
    rewriter.replaceOp(tileLoadOp, forOp.getResult(0));

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
    // Create a loop that stores each active ZA tile slice from memory.
    return createLoadStoreForOverTileSlices(
        rewriter, tileStoreOp.getLoc(), tileStoreOp.getVectorType(),
        tileStoreOp.getIndices(), tileStoreOp.getMemRefType().getRank(),
        tileStoreOp.getMask(),
        [&](Value tileSliceIndex, ValueRange memrefIndices, Value predicate) {
          tileStoreOp.replaceWithAndForwardTileId<arm_sme::StoreTileSliceOp>(
              rewriter, tileStoreOp.getValueToStore(), tileSliceIndex,
              predicate, tileStoreOp.getBase(), memrefIndices,
              tileStoreOp.getLayout());
        });
  }
};

} // namespace

void mlir::populateArmSMEToSCFConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<TileLoadOpConversion, TileLoadOpWithMaskAndPadNonZeroConversion,
               TileStoreOpConversion>(patterns.getContext());
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
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertArmSMEToSCFPass() {
  return std::make_unique<ConvertArmSMEToSCFPass>();
}
