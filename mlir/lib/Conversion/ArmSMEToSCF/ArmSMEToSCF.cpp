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
///  %tile_id = arm_sme.get_tile_id : i32
///  %tile = arm_sme.cast_tile_to_vector %tile_id : i32 to vector<[4]x[4]xi32>
///  %vscale = vector.vscale
///  %c0 = arith.constant 0 : index
///  %c1 = arith.constant 1 : index
///  %min_svl_s = arith.constant 4 : index
///  %svl_s = arith.muli %min_svl_s, %vscale : index
///  scf.for %tile_slice_idx = %c0 to %svl_s step %c1 {
///    %tile_update = arm_sme.load_tile_slice %src[%tile_slice_idx],
///      %tile, %tile_slice_idx : memref<?x?xi32>, vector<[4]x[4]xi32>
///  }
///  ```
struct TileLoadOpConversion : public OpRewritePattern<arm_sme::TileLoadOp> {
  using OpRewritePattern<arm_sme::TileLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arm_sme::TileLoadOp tileLoadOp,
                                PatternRewriter &rewriter) const override {
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

    // Create 'arm_sme.load_tile_slice' to load tile slice from memory into
    // tile.
    SmallVector<Value> memrefIndices;
    auto tileSliceIndex = forOp.getInductionVar();
    getMemrefIndices(tileLoadOp.getIndices(),
                     tileLoadOp.getMemRefType().getRank(), tileSliceIndex,
                     numTileSlices, memrefIndices, loc, rewriter);
    rewriter.create<arm_sme::LoadTileSliceOp>(
        loc, tileType, tileLoadOp.getBase(), tile, memrefIndices,
        tileSliceIndex, tileLoadOp.getLayout());

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

    // Create a loop that stores each ZA tile slice from memory.
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

    SmallVector<Value> memrefIndices;
    auto tileSliceIndex = forOp.getInductionVar();
    getMemrefIndices(tileStoreOp.getIndices(),
                     tileStoreOp.getMemRefType().getRank(), tileSliceIndex,
                     numTileSlices, memrefIndices, loc, rewriter);
    rewriter.replaceOpWithNewOp<arm_sme::StoreTileSliceOp>(
        tileStoreOp, tileStoreOp.getValueToStore(), tileSliceIndex,
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
  patterns.add<TileLoadOpConversion, TileStoreOpConversion,
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
