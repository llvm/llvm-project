//===- ArmSMEDialect.cpp - MLIR ArmSME dialect implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ArmSME dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace mlir;
using namespace mlir::arm_sme;

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/IR/ArmSMEDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/ArmSME/IR/ArmSME.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/ArmSME/IR/ArmSMETypes.cpp.inc"

void ArmSMEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/ArmSME/IR/ArmSME.cpp.inc"
      >();
}

// cast_vector_to_tile(cast_tile_to_vector(tile_id)) -> tile_id
LogicalResult CastVectorToTile::canonicalize(CastVectorToTile op,
                                             PatternRewriter &rewriter) {
  if (auto castTileToVectorOp = op.getVector().getDefiningOp<CastTileToVector>()) {
    op.replaceAllUsesWith(castTileToVectorOp.getTileId());
    return success();
  }
  return failure();
}

// cast_tile_to_vector(cast_vector_to_tile(tile)) -> tile
LogicalResult CastTileToVector::canonicalize(CastTileToVector op,
                                             PatternRewriter &rewriter) {
  if (auto castVectorToTileOp = op.getTileId().getDefiningOp<CastVectorToTile>()) {
    op.replaceAllUsesWith(castVectorToTileOp.getVector());
    return success();
  }
  return failure();
}
