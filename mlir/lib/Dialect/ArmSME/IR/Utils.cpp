//===- Utils.cpp - Utilities to support the ArmSME dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the ArmSME dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/Utils/Utils.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"

namespace mlir::arm_sme {

unsigned getSMETileSliceMinNumElts(Type type) {
  assert(isValidSMETileElementType(type) && "invalid tile type!");
  return MinStreamingVectorLengthInBits / type.getIntOrFloatBitWidth();
}

bool isValidSMETileElementType(Type type) {
  return type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
         type.isInteger(64) || type.isInteger(128) || type.isF16() ||
         type.isBF16() || type.isF32() || type.isF64() || type.isF128();
}

bool isValidSMETileVectorType(VectorType vType) {
  if ((vType.getRank() != 2) || !vType.allDimsScalable())
    return false;

  auto elemType = vType.getElementType();
  if (!isValidSMETileElementType(elemType))
    return false;

  unsigned minNumElts = getSMETileSliceMinNumElts(elemType);
  if (vType.getShape() != ArrayRef<int64_t>({minNumElts, minNumElts}))
    return false;

  return true;
}

std::optional<ArmSMETileType> getSMETileType(VectorType type) {
  if (!isValidSMETileVectorType(type))
    return {};
  switch (type.getElementTypeBitWidth()) {
  case 8:
    return ArmSMETileType::ZAB;
  case 16:
    return ArmSMETileType::ZAH;
  case 32:
    return ArmSMETileType::ZAS;
  case 64:
    return ArmSMETileType::ZAD;
  case 128:
    return ArmSMETileType::ZAQ;
  default:
    llvm_unreachable("unknown SME tile type");
  }
}

LogicalResult verifyOperationHasValidTileId(Operation *op) {
  auto tileOp = llvm::dyn_cast<ArmSMETileOpInterface>(op);
  if (!tileOp)
    return success(); // Not a tile op (no need to check).
  auto tileId = tileOp.getTileId();
  if (!tileId)
    return success(); // Not having a tile ID (yet) is okay.
  if (!tileId.getType().isSignlessInteger(32))
    return tileOp.emitOpError("tile ID should be a 32-bit signless integer");
  return success();
}

scf::ForOp createLoopOverTileSlices(
    PatternRewriter &rewriter, Location loc, Value initTile,
    std::function<Value(OpBuilder &, Location, Value, Value)> makeLoopBody) {
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
  Value nextTile =
      makeLoopBody(rewriter, loc, /*tileSliceIndex=*/forOp.getInductionVar(),
                   /*currentTile=*/forOp.getRegionIterArg(0));
  rewriter.create<scf::YieldOp>(loc, nextTile);
  return forOp;
}

bool isMultipleOfSMETileVectorType(VectorType vType) {
  if (vType.getRank() != 2 || !vType.allDimsScalable())
    return false;

  auto elementType = vType.getElementType();
  if (!isValidSMETileElementType(elementType))
    return false;

  unsigned minNumElts = getSMETileSliceMinNumElts(elementType);

  int64_t vectorRows = vType.getDimSize(0);
  int64_t vectorCols = vType.getDimSize(1);

  return (vectorRows > minNumElts || vectorCols > minNumElts) &&
         vectorRows % minNumElts == 0 && vectorCols % minNumElts == 0;
}

VectorType getSMETileTypeForElement(Type elementType) {
  unsigned minNumElts = getSMETileSliceMinNumElts(elementType);
  return VectorType::get({minNumElts, minNumElts}, elementType, {true, true});
}

} // namespace mlir::arm_sme
