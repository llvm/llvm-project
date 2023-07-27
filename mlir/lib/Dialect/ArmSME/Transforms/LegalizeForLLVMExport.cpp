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

/// Returns the following
/// * for rank 2 memrefs `tileSliceIndex`, since `getStridedElementPtr` does
///   the arithmetic.
/// * for rank 1 memrefs `tileSliceIndex * tileSliceNumElts`, adjusting the
///   index by the number of elements in a vector of SVL bits.
/// * otherwise throws an unreachable error.
Value getTileSlicePtrIndex(unsigned rank, Value tileSliceIndex,
                           Value tileSliceNumElts, Location loc,
                           ConversionPatternRewriter &rewriter) {
  assert((rank == 1 || rank == 2) && "memref has unexpected rank!");

  auto tileSliceIndexI64 = rewriter.create<arith::IndexCastUIOp>(
      loc, rewriter.getI64Type(), tileSliceIndex);

  if (rank == 1) {
    auto tileSliceNumEltsI64 = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getI64Type(), tileSliceNumElts);
    return rewriter.create<arith::MulIOp>(loc, tileSliceIndexI64,
                                          tileSliceNumEltsI64);
  }

  if (rank == 2)
    return tileSliceIndexI64;

  llvm_unreachable("memref has unexpected rank!");
}

/// Conversion pattern for `arm_sme.tile_load` to SME intrinsics.
///
/// Lower `arm_sme.tile_load` to a loop over the rows of ZA and load each row
/// using `arm_sme.intr.ld1*.horiz`.
///
///  BEFORE:
///  ```mlir
///  %tile = arm_sme.tile_load %base[%c0, %c0] :
///     memref<?x?xi32>, vector<[4]x[4]xi32>
///  ```
///
///  AFTER:
///  ```mlir
///  %tile_id = arm_sme.get_tile_id : i32
///  %vscale = vector.vscale
///  %c0 = arith.constant 0 : index
///  %c1 = arith.constant 1 : index
///  %min_svl_s = arith.constant 4 : index
///  %svl_s = arith.muli %min_svl_s, %vscale : index
///  scf.for %tile_slice = %c0 to %svl_s step %c1 {
///    // (...)
///    "arm_sme.intr.ld1w.horiz"(%ptrue_s, %ptr, %tile_id, %tile_slice) :
///         (vector<[4]xi1>, !llvm.ptr, i32, i32) -> ()
///  }
///  %tile = arm_sme.cast_tile_to_vector %tile_id : i32 to vector<[4]x[4]xi32>
///  ```
struct TileLoadToArmSMELowering
    : public ConvertOpToLLVMPattern<arm_sme::TileLoadOp> {
  using ConvertOpToLLVMPattern<arm_sme::TileLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arm_sme::TileLoadOp tileLoadOp,
                  arm_sme::TileLoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = tileLoadOp.getLoc();
    auto tileType = tileLoadOp.getVectorType();
    auto tileElementType = tileType.getElementType();
    unsigned tileElementWidth = tileElementType.getIntOrFloatBitWidth();

    // Create 'arm_sme.get_tile_id' op.
    auto tile = rewriter.create<arm_sme::GetTileID>(
        loc, rewriter.getIntegerType(tileElementWidth));

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

    // Create 'arm_sme.intr.ld1*.horiz' intrinsic to load ZA tile slice.
    auto memRefType = tileLoadOp.getMemRefType();
    auto tileSlice = forOp.getInductionVar();
    // TODO: The 'indices' argument for the 'base' memref is currently ignored,
    // 'tileSliceIndex' should be added to 'indices[0]'.
    Value tileSliceIndex = getTileSlicePtrIndex(memRefType.getRank(), tileSlice,
                                                numTileSlices, loc, rewriter);
    Value ptr = this->getStridedElementPtr(loc, memRefType, adaptor.getBase(),
                                           {tileSliceIndex}, rewriter);

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

    rewriter.setInsertionPointAfter(forOp);

    // The load intrinsics have no result, replace 'arm_sme.tile_load' with
    // 'arm_sme.cast_tile_to_vector' to preserve dataflow.
    rewriter.replaceOpWithNewOp<arm_sme::CastTileToVector>(tileLoadOp, tileType,
                                                           tile);

    return success();
  }
};

/// Conversion pattern for `arm_sme.tile_store` to SME intrinsics.
///
/// Lower `arm_sme.tile_store` to a loop over the rows of ZA and store each row
/// using `arm_sme.intr.st1*.horiz`.
///
///  BEFORE:
///  ```mlir
///  arm_sme.tile_store %value, %base[%c0, %c0] : memref<?x?xi32>,
///     vector<[4]x[4]xi32
///  ```
///
///  AFTER:
///  ```mlir
///  %tile_id = arm_sme.cast_vector_to_tile %tile : vector<[4]x[4]xi32> to i32
///  %vscale = vector.vscale
///  %c0 = arith.constant 0 : index
///  %c1 = arith.constant 1 : index
///  %min_svl_s = arith.constant 4 : index
///  %svl_s = arith.muli %min_svl_s, %vscale : index
///  scf.for %tile_slice = %c0 to %svl_s step %c1 {
///    // (...)
///    "arm_sme.intr.st1w.horiz"(%ptrue_s, %ptr, %tile_id, %tile_slice) :
///         (vector<[4]xi1>, !llvm.ptr, i32, i32) -> ()
///  }
///  ```
struct TileStoreToArmSMELowering
    : public ConvertOpToLLVMPattern<arm_sme::TileStoreOp> {
  using ConvertOpToLLVMPattern<arm_sme::TileStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arm_sme::TileStoreOp tileStoreOp,
                  arm_sme::TileStoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = tileStoreOp.getLoc();
    auto tileType = tileStoreOp.getVectorType();
    auto tileElementType = tileType.getElementType();
    unsigned tileElementWidth = tileElementType.getIntOrFloatBitWidth();

    // Create 'arm_sme.cast_vector_to_tile' to get a tile ID for the vector
    // being stored.
    auto tile = rewriter.create<arm_sme::CastVectorToTile>(
        loc, rewriter.getIntegerType(tileElementWidth),
        tileStoreOp.getValueToStore());

    // Create a loop that stores each ZA tile slice to memory.
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

    // Create 'arm_sme.intr.st1*.horiz' intrinsic to store ZA tile slice.
    auto memRefType = tileStoreOp.getMemRefType();
    auto tileSlice = forOp.getInductionVar();
    // TODO: The 'indices' argument for the 'base' memref is currently ignored,
    // 'tileSliceIndex' should be added to 'indices[0]'.
    Value tileSliceIndex = getTileSlicePtrIndex(memRefType.getRank(), tileSlice,
                                                numTileSlices, loc, rewriter);
    Value ptr = this->getStridedElementPtr(loc, memRefType, adaptor.getBase(),
                                           {tileSliceIndex}, rewriter);

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
          tileStoreOp, allActiveMask, ptr, tileI32, tileSliceI32);
      break;
    case 16:
      rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_st1h_horiz>(
          tileStoreOp, allActiveMask, ptr, tileI32, tileSliceI32);
      break;
    case 32:
      rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_st1w_horiz>(
          tileStoreOp, allActiveMask, ptr, tileI32, tileSliceI32);
      break;
    case 64:
      rewriter.replaceOpWithNewOp<arm_sme::aarch64_sme_st1d_horiz>(
          tileStoreOp, allActiveMask, ptr, tileI32, tileSliceI32);
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
  patterns.add<ZeroOpConversion, TileLoadToArmSMELowering,
               TileStoreToArmSMELowering>(converter);
}
