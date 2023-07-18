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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::arm_sme;

static constexpr unsigned kMinNumElts = 16;
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
} // namespace

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

/// Lower 'arm_sme.store_tile' to a loop over the rows of ZA and store each row
/// using 'arm_sme.intr.str'.
///
///  BEFORE:
///  ```mlir
///     arm_sme.tile_store %arg0[%c0, %c0], %0 : memref<?x?xi8>,
///     vector<[16]x[16]xi8
///  ```
///
///  AFTER:
///  ```mlir
///      %vscale = "llvm.intr.vscale"() : () -> index
///      %c0 = arith.constant 0 : index
///      %c1 = arith.constant 1 : index
///      %c16 = arith.constant 16 : index
///      %vec_size = arith.muli %c16, %vscale : index
///      scf.for %row_idx = %c0 to %vec_size step %c1 {
///        // (...)
///        "arm_sme.intr.str"(%row_idx, %addr) : (i32, !llvm.ptr) -> ()
///  ```
struct TileStoreOpConversion : public ConvertOpToLLVMPattern<TileStoreOp> {
  using ConvertOpToLLVMPattern<TileStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TileStoreOp store, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = store.getLoc();

    // Create loop that iterates from 0 to SVLB-1 inclusive (the number of
    // vectors in ZA) and stores each ZA vector to memory.
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto minElems = rewriter.create<arith::ConstantIndexOp>(loc, kMinNumElts);
    auto vscale =
        rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto upperBound = rewriter.create<arith::MulIOp>(loc, minElems, vscale);
    auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
    rewriter.setInsertionPointToStart(forOp.getBody());

    // Create 'arm_sme.intr.str' intrinsic to store ZA vector.
    auto vnumI64 = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getI64Type(), forOp.getInductionVar());
    auto offset =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0);
    Value ptr =
        getStridedElementPtr(loc, store.getMemRefType(), adaptor.getBase(),
                             ValueRange{vnumI64, offset}, rewriter);
    auto vnumI32 = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getI32Type(), forOp.getInductionVar());
    rewriter.create<arm_sme::aarch64_sme_str>(loc, vnumI32, ptr);

    rewriter.eraseOp(store);
    return success();
  }
};

void mlir::configureArmSMELegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalOp<scf::ForOp, scf::YieldOp, arm_sme::CastTileToVector,
                    arm_sme::CastVectorToTile, arm_sme::aarch64_sme_zero,
                    arm_sme::aarch64_sme_str, arm_sme::aarch64_sme_za_enable,
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
  patterns.add<TileStoreOpConversion, ZeroOpConversion>(converter);
}
