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

namespace mlir::arm_sme {

unsigned getSizeInBytes(TypeSize type) {
  switch (type) {
  case arm_sme::TypeSize::Byte:
    return 1;
  case arm_sme::TypeSize::Half:
    return 2;
  case arm_sme::TypeSize::Word:
    return 4;
  case arm_sme::TypeSize::Double:
    return 8;
  }
  llvm_unreachable("unknown type size");
  return 0;
}

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
  auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto minTileSlices = arith::ConstantIndexOp::create(
      rewriter, loc, llvm::cast<VectorType>(initTile.getType()).getDimSize(0));
  auto vscale =
      vector::VectorScaleOp::create(rewriter, loc, rewriter.getIndexType());
  auto lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto numTileSlices =
      arith::MulIOp::create(rewriter, loc, minTileSlices, vscale);
  auto forOp = scf::ForOp::create(rewriter, loc, lowerBound, numTileSlices,
                                  step, ValueRange{initTile});
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value nextTile =
      makeLoopBody(rewriter, loc, /*tileSliceIndex=*/forOp.getInductionVar(),
                   /*currentTile=*/forOp.getRegionIterArg(0));
  scf::YieldOp::create(rewriter, loc, nextTile);
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

void eraseTriviallyDeadTileOps(IRRewriter &rewriter,
                               FunctionOpInterface function) {
  SmallVector<Operation *> worklist;
  function->walk([&](Operation *op) {
    auto armSMEOp = dyn_cast<arm_sme::ArmSMETileOpInterface>(op);
    if (armSMEOp && isOpTriviallyDead(armSMEOp))
      worklist.push_back(armSMEOp);
  });
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!isOpTriviallyDead(op))
      continue;
    for (Value value : op->getOperands()) {
      if (auto armSMEOp = value.getDefiningOp<arm_sme::ArmSMETileOpInterface>())
        worklist.push_back(armSMEOp);
    }
    rewriter.eraseOp(op);
  }
}

bool isTriviallyCloneableTileOp(arm_sme::ArmSMETileOpInterface tileOp) {
  return tileOp && tileOp->getNumResults() == 1 &&
         tileOp->getNumOperands() == 0 && isPure(tileOp);
}

bool hasTileResult(arm_sme::ArmSMETileOpInterface tileOp) {
  for (Value result : tileOp->getResults()) {
    if (arm_sme::isValidSMETileVectorType(result.getType()))
      return true;
  }
  return false;
}

OpOperand *getTileOpOperand(arm_sme::ArmSMETileOpInterface tileOp) {
  if (!tileOp)
    return nullptr;
  auto isTileOperandType = [](OpOperand &operand) {
    return arm_sme::isValidSMETileVectorType(operand.get().getType());
  };
  assert(llvm::count_if(tileOp->getOpOperands(), isTileOperandType) <= 1 &&
         "expected at most one tile operand");
  OpOperand *tileOperand =
      llvm::find_if(tileOp->getOpOperands(), isTileOperandType);
  if (tileOperand == tileOp->getOpOperands().end())
    return nullptr;
  return tileOperand;
}

bool isTileTypeGreaterOrEqual(ArmSMETileType typeA, ArmSMETileType typeB) {
  // Note: This is <= due to how tile types are numbered in ArmSMEOps.td.
  return static_cast<unsigned>(typeA) <= static_cast<unsigned>(typeB);
}

} // namespace mlir::arm_sme
