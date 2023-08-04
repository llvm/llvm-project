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

static constexpr unsigned kZeroZAMask = 255;

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

/// Lower 'arm_sme.zero'. Use 'arm_sme.cast_tile_to_vector' to model the return
/// value. The latter is a nop, which should be folded away (e.g. during
/// canonicalisation).
///
///  BEFORE:
///  ```mlir
///     %0 = arm_sme.zero : vector<[16]x[16]xi8>
///  ```
///
///  AFTER:
///  ```mlir
///     %1 = arm_sme.get_tile_id : i8
///     %2 = arm_sme.cast_tile_to_vector %1 : i8 to vector<[16]x[16]xi8>
///     "arm_sme.intr.zero"(%c255_i32) : (i32) -> ()
///  ```
struct ZeroOpConversion : public ConvertOpToLLVMPattern<ZeroOp> {
  using ConvertOpToLLVMPattern<ZeroOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ZeroOp zero, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = zero.getLoc();

    // Get Tile ID for the `zero` intrinsic.
    // TODO: Map this to a valid `mask` for the `zero` intrinsic.
    auto tileId = rewriter.create<arm_sme::GetTileID>(
        loc, zero.getVectorType().getElementType());

    // Create 'arm_sme.intr.zero' intrinsic to zero ZA.
    // FIXME: Replace the hard-coded mask with a valid value based
    // on `tileId`.
    auto mask = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(kZeroZAMask));
    rewriter.create<arm_sme::aarch64_sme_zero>(loc, mask);

    // Create `CastTileToVectorOp` to use it as the output
    rewriter.replaceOpWithNewOp<arm_sme::CastTileToVector>(zero, zero.getType(),
                                                           tileId);

    return success();
  }
};

/// Extends or truncates `tile`, which should be an `arm_sme::GetTileID` or
/// `arm_sme::CastVectorToTile` op returning an 8/16/32/64/128-bit scalar
/// integer, to an i32 that can be passed as the `tile` parameter to the SME
/// intrinsics. Or returns `tile` if already i32.
Value castTileIDToI32(Value tile, Location loc,
                      ConversionPatternRewriter &rewriter) {
  assert((isa<arm_sme::GetTileID, arm_sme::CastVectorToTile>(
             tile.getDefiningOp())) &&
         "expected ArmSME GetTileID or CastVectorToTile op!");
  unsigned tileElementWidth = tile.getType().getIntOrFloatBitWidth();
  if (tileElementWidth < 32)
    return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), tile);
  if (tileElementWidth > 32)
    return rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), tile);
  return tile;
}

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
    // Create 'arm_sme.intr.ld1*.horiz' intrinsic to load ZA tile slice.
    switch (tileElementWidth) {
    default:
      llvm_unreachable("unexpected element type!");
    case 8:
      rewriter.create<arm_sme::aarch64_sme_ld1b_horiz>(loc, allActiveMask, ptr,
                                                       tileI32, tileSliceI32);
      break;
    case 16:
      rewriter.create<arm_sme::aarch64_sme_ld1h_horiz>(loc, allActiveMask, ptr,
                                                       tileI32, tileSliceI32);
      break;
    case 32:
      rewriter.create<arm_sme::aarch64_sme_ld1w_horiz>(loc, allActiveMask, ptr,
                                                       tileI32, tileSliceI32);
      break;
    case 64:
      rewriter.create<arm_sme::aarch64_sme_ld1d_horiz>(loc, allActiveMask, ptr,
                                                       tileI32, tileSliceI32);
      break;
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
    }

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
      arm_sme::aarch64_sme_ld1d_horiz, arm_sme::aarch64_sme_st1b_horiz,
      arm_sme::aarch64_sme_st1h_horiz, arm_sme::aarch64_sme_st1w_horiz,
      arm_sme::aarch64_sme_st1d_horiz, arm_sme::aarch64_sme_za_enable,
      arm_sme::aarch64_sme_za_disable>();
  target.addLegalOp<GetTileID>();

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
  patterns.add<EnableZAPattern, DisableZAPattern>(patterns.getContext());
  patterns.add<ZeroOpConversion, StoreTileSliceToArmSMELowering,
               LoadTileSliceToArmSMELowering>(converter);
}
