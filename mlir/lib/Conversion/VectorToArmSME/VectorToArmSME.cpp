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

static constexpr unsigned kMinNumElts = 16;

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
    auto vType = dyn_cast<VectorType>(constantOp.getType());
    if (!vType)
      return failure();
    if (vType.getRank() != 2)
      return failure();
    if (vType.getShape() != ArrayRef<int64_t>({kMinNumElts, kMinNumElts}))
      return failure();
    if (vType.getElementType() != rewriter.getI8Type())
      return failure();
    if (vType.getScalableDims().size() != 2)
      return failure();

    auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValueAttr());
    if (!denseAttr || !isSplatZero(vType.getElementType(), denseAttr))
      return failure();

    rewriter.replaceOpWithNewOp<arm_sme::ZeroOp>(constantOp, vType);

    return success();
  }
};

} // namespace

void mlir::populateVectorToArmSMEPatterns(RewritePatternSet &patterns,
                                          MLIRContext &ctx) {
  patterns.add<TransferWriteToArmSMELowering, VectorLoadToArmSMELowering,
               VectorStoreToArmSMELowering, ConstantOpToArmSMELowering>(&ctx);
}
