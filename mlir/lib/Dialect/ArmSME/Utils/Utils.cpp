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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"

using namespace mlir;
using namespace mlir::arm_sme;

unsigned mlir::arm_sme::getSMETileSliceMinNumElts(Type type) {
  assert(isValidSMETileElementType(type) && "invalid tile type!");
  return MinStreamingVectorLengthInBits / type.getIntOrFloatBitWidth();
}

bool mlir::arm_sme::isValidSMETileElementType(Type type) {
  return type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
         type.isInteger(64) || type.isInteger(128) || type.isF16() ||
         type.isBF16() || type.isF32() || type.isF64() || type.isF128();
}

bool mlir::arm_sme::isValidSMETileVectorType(VectorType vType) {
  if ((vType.getRank() != 2) || !vType.allDimsScalable())
    return false;

  auto elemType = vType.getElementType();
  if (!isValidSMETileElementType(elemType))
    return false;

  unsigned minNumElts = arm_sme::getSMETileSliceMinNumElts(elemType);
  if (vType.getShape() != ArrayRef<int64_t>({minNumElts, minNumElts}))
    return false;

  return true;
}

Value mlir::arm_sme::castTileIDToI32(Value tile, Location loc,
                                     RewriterBase &rewriter) {
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
