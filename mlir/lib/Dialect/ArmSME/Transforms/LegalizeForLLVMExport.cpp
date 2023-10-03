//===- LegalizeForLLVMExport.cpp - Prepare ArmSME for LLVM translation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Transforms.h"
#include "mlir/Dialect/ArmSME/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::arm_sme;

namespace {
/// Insert 'llvm.aarch64.sme.za.enable' intrinsic at the start of 'func.func'
/// ops to enable the ZA storage array.
struct EnableZAPattern : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&op.front());
    rewriter.create<arm_sme::aarch64_sme_za_enable>(op->getLoc());
    rewriter.updateRootInPlace(op, [] {});
    return success();
  }
};

/// Insert 'llvm.aarch64.sme.za.disable' intrinsic before 'func.return' ops to
/// disable the ZA storage array.
struct DisableZAPattern : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    rewriter.create<arm_sme::aarch64_sme_za_disable>(op->getLoc());
    rewriter.updateRootInPlace(op, [] {});
    return success();
  }
};

/// Lower 'arm_sme.zero' to SME intrinsics.
///
///  BEFORE:
///  ```mlir
///     %v = arm_sme.zero : vector<[4]x[4]xi32>
///  ```
///
///  AFTER:
///  ```mlir
///     %tile_id = arm_sme.get_tile_id : i32
///     %zero_mask = arith.shli %c17_i32, %tile_id : i32
///     "arm_sme.intr.zero"(%zero_mask) : (i32) -> ()
///     %v = arm_sme.cast_tile_to_vector %tile_id : i32 to vector<[4]x[4]xi32>
///  ```
///
///  The 'arm_sme.cast_tile_to_vector' (which models the return) and the
///  'arith.shli' (which generates the mask) will be folded away after tile
///  allocation and canonization.
struct ZeroOpConversion : public ConvertOpToLLVMPattern<ZeroOp> {
  using ConvertOpToLLVMPattern<ZeroOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ZeroOp zero, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = zero.getLoc();

    unsigned tileElementWidth =
        zero.getVectorType().getElementType().getIntOrFloatBitWidth();

    // Get Tile ID for the `zero` intrinsic.
    auto tileId = rewriter.create<arm_sme::GetTileID>(
        loc, rewriter.getIntegerType(tileElementWidth));

    // Get the base mask for tile based on the element size.
    // The base mask is just the mask to zero the first tile (of a size).
    // These masks are derived from:
    // https://developer.arm.com/documentation/ddi0602/2022-06/SME-Instructions/ZERO--Zero-a-list-of-64-bit-element-ZA-tiles-
    auto baseMaskForSize = [&] {
      switch (tileElementWidth) {
      case 8:
        // Zeroing the 8-bit ZA0.B tile is equivalent to zeroing all eight
        // 64-bit element tiles named ZA0.D to ZA7.D.
        return 0b1111'1111;
      case 16:
        // Zeroing the 16-bit ZA0.H tile is equivalent to zeroing 64-bit element
        // tiles named ZA0.D, ZA2.D, ZA4.D, and ZA6.D.
        // Shift this left once for ZA1.H.
        return 0b0101'0101;
      case 32:
        // Zeroing the 32-bit ZA0.S tile is equivalent to zeroing 64-bit
        // element tiles named ZA0.D and ZA4.D.
        // Shift left by 1, 2, or 3 respectively for ZA1.S, ZA2.S, ZA3.S.
        return 0b0001'0001;
      case 64:
        // Zeroing one of the a 64-bit tiles ZA0.D to ZA7.D just requires
        // setting the bit for that tile.
        return 0b0000'0001;
      default:
        llvm_unreachable("bad element size");
      }
    }();
    auto maskType = rewriter.getI32Type();
    auto baseMask = rewriter.create<arith::ConstantOp>(
        loc, maskType, rewriter.getIntegerAttr(maskType, baseMaskForSize));

    // The actual mask is just the base mask shifted by the tile ID.
    // This will be folded to a constant after tile allocation.
    //
    // The shift is just derived from the layout of the tiles, and that the tile
    // ID is the index of the tile. For example, looking at the 32-bit ZAx.S
    // tiles:
    //
    // ZA0.S = ZA0.D and ZA4.D
    //  * Tile ID -> 0
    //  * Mask    -> 00010001 = (00010001 << 0)
    // ZA1.S = ZA1.D and ZA5.D
    //  * Tile ID -> 1
    //  * Mask    -> 00100010 = (00010001 << 1)
    // ZA2.S = ZA2.D and ZA6.D
    //  * Tile ID -> 2
    //  * Mask    -> 01000100 = (00010001 << 2)
    // ZA3.S = ZA3.D and ZA7.D
    //  * Tile ID -> 3
    //  * Mask    -> 10001000 = (00010001 << 3)
    //
    // This holds for all tile sizes.
    auto tileMask = rewriter.create<arith::ShLIOp>(
        loc, baseMask, castTileIDToI32(tileId, loc, rewriter));
    rewriter.create<arm_sme::aarch64_sme_zero>(loc, tileMask);

    // Create `CastTileToVectorOp` to use as the output.
    rewriter.replaceOpWithNewOp<arm_sme::CastTileToVector>(zero, zero.getType(),
                                                           tileId);

    return success();
  }
};

/// Lower `arm_sme.load_tile_slice` to SME intrinsics.
struct LoadTileSliceToArmSMELowering
    : public ConvertOpToLLVMPattern<arm_sme::LoadTileSliceOp> {
  using ConvertOpToLLVMPattern<
      arm_sme::LoadTileSliceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arm_sme::LoadTileSliceOp loadTileSliceOp,
                  arm_sme::LoadTileSliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = loadTileSliceOp.getLoc();
    auto tileType = loadTileSliceOp.getVectorType();
    auto tileElementType = tileType.getElementType();
    unsigned tileElementWidth = tileElementType.getIntOrFloatBitWidth();

    // Create 'arm_sme.cast_vector_to_tile' to get a tile ID for the tile being
    // loaded to.
    auto tile = rewriter.create<arm_sme::CastVectorToTile>(
        loc, rewriter.getIntegerType(tileElementWidth),
        loadTileSliceOp.getTile());

    Value ptr = this->getStridedElementPtr(loc, loadTileSliceOp.getMemRefType(),
                                           adaptor.getBase(),
                                           adaptor.getIndices(), rewriter);

    auto tileSlice = loadTileSliceOp.getTileSliceIndex();

    // Cast tile slice to i32 for intrinsic.
    auto tileSliceI32 = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getI32Type(), tileSlice);

    // Create all active predicate mask.
    auto one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI1Type(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    auto predTy = VectorType::get(tileType.getShape()[0], rewriter.getI1Type(),
                                  /*scalableDims=*/{true});
    auto allActiveMask = rewriter.create<vector::SplatOp>(loc, predTy, one);

    auto tileI32 = castTileIDToI32(tile, loc, rewriter);
    arm_sme::TileSliceLayout layout = loadTileSliceOp.getLayout();

    // Create 'arm_sme.intr.ld1*.(horiz|vert)' intrinsic to load ZA tile slice.
    if (layout == arm_sme::TileSliceLayout::Horizontal) {
      switch (tileElementWidth) {
      default:
        llvm_unreachable("unexpected element type!");
      case 8:
        rewriter.create<arm_sme::aarch64_sme_ld1b_horiz>(
            loc, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      case 16:
        rewriter.create<arm_sme::aarch64_sme_ld1h_horiz>(
            loc, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      case 32:
        rewriter.create<arm_sme::aarch64_sme_ld1w_horiz>(
            loc, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      case 64:
        rewriter.create<arm_sme::aarch64_sme_ld1d_horiz>(
            loc, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      case 128:
        rewriter.create<arm_sme::aarch64_sme_ld1q_horiz>(
            loc, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      }
    } else {
      switch (tileElementWidth) {
      default:
        llvm_unreachable("unexpected element type!");
      case 8:
        rewriter.create<arm_sme::aarch64_sme_ld1b_vert>(loc, allActiveMask, ptr,
                                                        tileI32, tileSliceI32);
        break;
      case 16:
        rewriter.create<arm_sme::aarch64_sme_ld1h_vert>(loc, allActiveMask, ptr,
                                                        tileI32, tileSliceI32);
        break;
      case 32:
        rewriter.create<arm_sme::aarch64_sme_ld1w_vert>(loc, allActiveMask, ptr,
                                                        tileI32, tileSliceI32);
        break;
      case 64:
        rewriter.create<arm_sme::aarch64_sme_ld1d_vert>(loc, allActiveMask, ptr,
                                                        tileI32, tileSliceI32);
        break;
      case 128:
        rewriter.create<arm_sme::aarch64_sme_ld1q_vert>(loc, allActiveMask, ptr,
                                                        tileI32, tileSliceI32);
        break;
      }
    }

    // The load intrinsics have no result, replace 'arm_sme.tile_load' with
    // 'arm_sme.cast_tile_to_vector' to preserve dataflow.
    rewriter.replaceOpWithNewOp<arm_sme::CastTileToVector>(loadTileSliceOp,
                                                           tileType, tile);

    return success();
  }
};

/// Lower for `arm_sme.store_tile_slice` to SME intrinsics.
struct StoreTileSliceToArmSMELowering
    : public ConvertOpToLLVMPattern<arm_sme::StoreTileSliceOp> {
  using ConvertOpToLLVMPattern<
      arm_sme::StoreTileSliceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arm_sme::StoreTileSliceOp storeTileSliceOp,
                  arm_sme::StoreTileSliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = storeTileSliceOp.getLoc();
    auto tileType = storeTileSliceOp.getVectorType();
    auto tileElementType = tileType.getElementType();
    unsigned tileElementWidth = tileElementType.getIntOrFloatBitWidth();

    // Create 'arm_sme.cast_vector_to_tile' to get a tile ID for the vector
    // being stored.
    auto tile = rewriter.create<arm_sme::CastVectorToTile>(
        loc, rewriter.getIntegerType(tileElementWidth),
        storeTileSliceOp.getTile());

    // Create 'arm_sme.intr.st1*.horiz' intrinsic to store ZA tile slice.
    Value ptr = this->getStridedElementPtr(
        loc, storeTileSliceOp.getMemRefType(), adaptor.getBase(),
        adaptor.getIndices(), rewriter);

    auto tileSlice = storeTileSliceOp.getTileSliceIndex();

    // Cast tile slice to i32 for intrinsic.
    auto tileSliceI32 = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getI32Type(), tileSlice);

    // Create all active predicate mask.
    auto one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI1Type(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    auto predTy = VectorType::get(tileType.getShape()[0], rewriter.getI1Type(),
                                  /*scalableDims=*/{true});
    auto allActiveMask = rewriter.create<vector::SplatOp>(loc, predTy, one);

    Value tileI32 = castTileIDToI32(tile, loc, rewriter);
    arm_sme::TileSliceLayout layout = storeTileSliceOp.getLayout();

    if (layout == arm_sme::TileSliceLayout::Horizontal) {
      switch (tileElementWidth) {
      default:
        llvm_unreachable("unexpected element type!");
      case 8:
        rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_st1b_horiz>(
            storeTileSliceOp, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      case 16:
        rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_st1h_horiz>(
            storeTileSliceOp, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      case 32:
        rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_st1w_horiz>(
            storeTileSliceOp, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      case 64:
        rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_st1d_horiz>(
            storeTileSliceOp, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      case 128:
        rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_st1q_horiz>(
            storeTileSliceOp, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      }
    } else {
      switch (tileElementWidth) {
      default:
        llvm_unreachable("unexpected element type!");
      case 8:
        rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_st1b_vert>(
            storeTileSliceOp, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      case 16:
        rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_st1h_vert>(
            storeTileSliceOp, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      case 32:
        rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_st1w_vert>(
            storeTileSliceOp, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      case 64:
        rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_st1d_vert>(
            storeTileSliceOp, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      case 128:
        rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_st1q_vert>(
            storeTileSliceOp, allActiveMask, ptr, tileI32, tileSliceI32);
        break;
      }
    }

    return success();
  }
};

/// Lower `arm_sme.move_vector_to_tile_slice` to SME intrinsics. Only horizontal
/// tile slices are currently supported.
struct MoveVectorToTileSliceToArmSMELowering
    : public ConvertOpToLLVMPattern<arm_sme::MoveVectorToTileSliceOp> {
  using ConvertOpToLLVMPattern<
      arm_sme::MoveVectorToTileSliceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arm_sme::MoveVectorToTileSliceOp moveVectorToTileSliceOp,
                  arm_sme::MoveVectorToTileSliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = moveVectorToTileSliceOp.getLoc();
    auto tileType = moveVectorToTileSliceOp.getTileType();
    auto tileElementType = tileType.getElementType();
    unsigned tileElementWidth = tileElementType.getIntOrFloatBitWidth();

    // Create 'arm_sme.cast_vector_to_tile' to get a tile ID for the tile being
    // loaded to.
    auto tile = rewriter.create<arm_sme::CastVectorToTile>(
        loc, rewriter.getIntegerType(tileElementWidth),
        moveVectorToTileSliceOp.getTile());

    auto tileSlice = moveVectorToTileSliceOp.getTileSliceIndex();

    // Cast tile slice from index to i32 for intrinsic.
    auto tileSliceI32 = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getI32Type(), tileSlice);

    // Create all active predicate mask.
    auto one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI1Type(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    auto predTy = VectorType::get(tileType.getShape()[0], rewriter.getI1Type(),
                                  /*scalableDims=*/{true});
    auto allActiveMask = rewriter.create<vector::SplatOp>(loc, predTy, one);

    auto tileI32 = castTileIDToI32(tile, loc, rewriter);

    // Create 'arm_sme.intr.write.horiz' to write vector to tile slice.
    rewriter.create<arm_sme::aarch64_sme_write_horiz>(
        loc, tileI32, tileSliceI32, allActiveMask,
        moveVectorToTileSliceOp.getVector());

    // Intrinsic has no result, replace 'arm_sme.move_vector_to_tile_slice' with
    // 'arm_sme.cast_tile_to_vector' to preserve dataflow.
    rewriter.replaceOpWithNewOp<arm_sme::CastTileToVector>(
        moveVectorToTileSliceOp, tileType, tile);

    return success();
  }
};

/// Lower `arm_sme.move_tile_slice_to_vector` to SME intrinsics. Only horizontal
/// tile slices are currently supported.
struct MoveTileSliceToVectorArmSMELowering
    : public ConvertOpToLLVMPattern<arm_sme::MoveTileSliceToVectorOp> {
  using ConvertOpToLLVMPattern<
      arm_sme::MoveTileSliceToVectorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arm_sme::MoveTileSliceToVectorOp moveTileSliceToVector,
                  OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = moveTileSliceToVector.getLoc();
    auto sliceType = moveTileSliceToVector.getSliceType();
    auto tile = moveTileSliceToVector.getTile();
    auto sliceIndex = moveTileSliceToVector.getTileSliceIndex();

    // Cast tile to i32 tile ID.
    auto tileId = rewriter.create<arm_sme::CastVectorToTile>(loc, tile);
    Value tileIdI32 = castTileIDToI32(tileId, loc, rewriter);

    // Create an 'all true' predicate for the tile slice.
    auto predicateType = sliceType.cloneWith({}, rewriter.getI1Type());
    auto allTruePredicate = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(predicateType, true));

    // Zero destination/fallback for tile slice extraction.
    auto zeroVector = rewriter.create<arith::ConstantOp>(
        loc, sliceType, rewriter.getZeroAttr(sliceType));

    // Cast tile slice from index to i32 for intrinsic.
    auto sliceIndexI32 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), sliceIndex);

    // Create 'arm_sme.intr.read.horiz' to extract the tile slice.
    rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_read_horiz>(
        moveTileSliceToVector, sliceType, zeroVector, allTruePredicate,
        tileIdI32, sliceIndexI32);

    return success();
  }
};

/// Lower `vector.outerproduct` to SME MOPA intrinsics.
///
/// Example:
///
///   %0 = vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>}
///     : vector<[4]xf32>, vector<[4]xf32>
///
/// is converted to:
///
///   "arm_sme.intr.mopa"(%tile_id, %ptrue_s, %ptrue_s, %lhs, %rhs)
///     : (i32, vector<[4]xi1>, vector<[4]xi1>, vector<[4]xf32>,
///        vector<[4]xf32>) -> ()
///
/// Currently only supports FMOPA and BFMOPA (non-widening).
struct VectorOuterProductToArmSMELowering
    : public ConvertOpToLLVMPattern<vector::OuterProductOp> {
  using ConvertOpToLLVMPattern<vector::OuterProductOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::OuterProductOp outerProductOp,
                  vector::OuterProductOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto isSupportedType = [](VectorType vectorType) {
      // TODO: the FP outer product instruction variants are predicated on
      // different features [1]:
      //
      // * FMOPA (non-widening)
      //   * half-precision   - +sme2p1,+sme-f16f16
      //   * single-precision - +sme
      //   * double-precision - +sme-f64f64
      // * BFMOPA
      //   * half-precision   - +sme2p1,+b16b16
      //
      // It should be possible to control lowering based on target features.
      // [1] https://developer.arm.com/downloads/-/exploration-tools/feature-names-for-a-profile
      if ((vectorType.getRank() != 2) || !vectorType.allDimsScalable())
        return false;

      auto elementType = vectorType.getElementType();

      if (!elementType.isF16() && !elementType.isBF16() &&
          !elementType.isF32() && !elementType.isF64())
        return false;

      unsigned minNumElts = arm_sme::MinStreamingVectorLengthInBits /
                            vectorType.getElementTypeBitWidth();
      if (vectorType.getShape() != ArrayRef<int64_t>({minNumElts, minNumElts}))
        return false;

      return true;
    };

    auto resultVectorType = outerProductOp.getResultVectorType();
    if (!isSupportedType(resultVectorType))
      return outerProductOp.emitError("unsupported type");

    vector::CombiningKind kind = outerProductOp.getKind();
    if (kind != vector::CombiningKind::ADD)
      // TODO: support subtract.
      return outerProductOp.emitError("unsupported kind");

    auto maskableOp =
        cast<vector::MaskableOpInterface>(outerProductOp.getOperation());
    if (maskableOp.isMasked())
      // TODO: support masking.
      return outerProductOp.emitError("masking is currently unsupported");

    if (!isa<VectorType>(outerProductOp.getOperandTypeRHS()))
      // AXPY operation not suited for SME.
      return failure();

    auto loc = outerProductOp.getLoc();

    Value acc = outerProductOp.getAcc();
    if (!acc)
      // Initalize accumulator with zero.
      acc = rewriter.create<arm_sme::ZeroOp>(loc, resultVectorType);

    unsigned elementWidth = resultVectorType.getElementTypeBitWidth();
    auto tileId = rewriter.create<arm_sme::CastVectorToTile>(
        loc, rewriter.getIntegerType(elementWidth), acc);

    // Create all active predicate mask.
    auto one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI1Type(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    auto predTy =
        VectorType::get(resultVectorType.getShape()[0], rewriter.getI1Type(),
                        /*scalableDims=*/{true});
    auto allActiveMask = rewriter.create<vector::SplatOp>(loc, predTy, one);

    auto tileI32 = castTileIDToI32(tileId, loc, rewriter);

    // Create 'arm_sme.intr.mopa' outer product intrinsic.
    rewriter.create<arm_sme::aarch64_sme_mopa>(
        loc, tileI32, allActiveMask, allActiveMask, outerProductOp.getLhs(),
        outerProductOp.getRhs());

    // Create `CastTileToVectorOp` to use as the output.
    rewriter.replaceOpWithNewOp<arm_sme::CastTileToVector>(
        outerProductOp, resultVectorType, tileId);

    return success();
  }
};

} // namespace

void mlir::configureArmSMELegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalOp<
      scf::ForOp, scf::YieldOp, arm_sme::CastTileToVector,
      arm_sme::CastVectorToTile, arm_sme::aarch64_sme_zero,
      arm_sme::aarch64_sme_str, arm_sme::aarch64_sme_ld1b_horiz,
      arm_sme::aarch64_sme_ld1h_horiz, arm_sme::aarch64_sme_ld1w_horiz,
      arm_sme::aarch64_sme_ld1d_horiz, arm_sme::aarch64_sme_ld1q_horiz,
      arm_sme::aarch64_sme_st1b_horiz, arm_sme::aarch64_sme_st1h_horiz,
      arm_sme::aarch64_sme_st1w_horiz, arm_sme::aarch64_sme_st1d_horiz,
      arm_sme::aarch64_sme_st1q_horiz, arm_sme::aarch64_sme_ld1b_vert,
      arm_sme::aarch64_sme_ld1h_vert, arm_sme::aarch64_sme_ld1w_vert,
      arm_sme::aarch64_sme_ld1d_vert, arm_sme::aarch64_sme_ld1q_vert,
      arm_sme::aarch64_sme_st1b_vert, arm_sme::aarch64_sme_st1h_vert,
      arm_sme::aarch64_sme_st1w_vert, arm_sme::aarch64_sme_st1d_vert,
      arm_sme::aarch64_sme_st1q_vert, arm_sme::aarch64_sme_read_horiz,
      arm_sme::aarch64_sme_write_horiz, arm_sme::aarch64_sme_mopa,
      arm_sme::aarch64_sme_za_enable, arm_sme::aarch64_sme_za_disable>();
  target.addLegalOp<GetTileID>();
  target.addIllegalOp<vector::OuterProductOp>();

  // Mark 'func.func' ops as legal if either:
  //   1. no 'arm_za' function attribute is present.
  //   2. the 'arm_za' function attribute is present and the first op in the
  //      function is an 'arm_sme::aarch64_sme_za_enable' intrinsic.
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp funcOp) {
    if (funcOp.isDeclaration())
      return true;
    auto firstOp = funcOp.getBody().front().begin();
    return !funcOp->hasAttr("arm_za") ||
           isa<arm_sme::aarch64_sme_za_enable>(firstOp);
  });

  // Mark 'func.return' ops as legal if either:
  //   1. no 'arm_za' function attribute is present.
  //   2. the 'arm_za' function attribute is present and there's a preceding
  //      'arm_sme::aarch64_sme_za_disable' intrinsic.
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp returnOp) {
    bool hasDisableZA = false;
    auto funcOp = returnOp->getParentOp();
    funcOp->walk<WalkOrder::PreOrder>(
        [&](arm_sme::aarch64_sme_za_disable op) { hasDisableZA = true; });
    return !funcOp->hasAttr("arm_za") || hasDisableZA;
  });
}

void mlir::populateArmSMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<DisableZAPattern, EnableZAPattern>(patterns.getContext());
  patterns.add<
      LoadTileSliceToArmSMELowering, MoveTileSliceToVectorArmSMELowering,
      MoveVectorToTileSliceToArmSMELowering, StoreTileSliceToArmSMELowering,
      VectorOuterProductToArmSMELowering, ZeroOpConversion>(converter);
}
