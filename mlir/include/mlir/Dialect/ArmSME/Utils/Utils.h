//===- Utils.h - General ArmSME transformation utilities --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various utilities for the ArmSME
// dialect. These are not passes by themselves but are used either by passes,
// optimization sequences, or in turn by other transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMSME_UTILS_UTILS_H_
#define MLIR_DIALECT_ARMSME_UTILS_UTILS_H_

#include "mlir/Dialect/ArmSME/IR/ArmSMEEnums.h"
#include "mlir/Dialect/ArmSME/IR/ArmSMEOpInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include <optional>

namespace mlir {
class Location;
class PatternRewriter;
class Value;
} // namespace mlir

namespace mlir::arm_sme {

constexpr unsigned MinStreamingVectorLengthInBits = 128;

/// Return minimum number of elements for the given element `type` in
/// a vector of SVL bits.
unsigned getSMETileSliceMinNumElts(Type type);

/// Returns true if `type` is a valid element type for an SME tile or false
/// otherwise.
bool isValidSMETileElementType(Type type);

/// Returns true if `vType` is a valid vector type for an SME tile or false
/// otherwise.
bool isValidSMETileVectorType(VectorType vType);

inline bool isValidSMETileVectorType(Type type) {
  auto vType = dyn_cast<VectorType>(type);
  return vType && isValidSMETileVectorType(vType);
}

/// Returns the type of SME tile this vector type corresponds to, or none if the
/// vector type does not fit within an SME tile.
std::optional<ArmSMETileType> getSMETileType(VectorType);

/// Verifies the tile ID (if set) on this tile operation is valid.
LogicalResult verifyOperationHasValidTileId(Operation *);

/// Generates a for loop over ZA tile slices where the induction variable is
/// the tile slice index and each iteration yields a new tile. Loop body is
/// built via `makeLoopBody`, which returns the next tile value.
scf::ForOp createLoopOverTileSlices(
    PatternRewriter &rewriter, Location loc, Value initTile,
    std::function<Value(OpBuilder &, Location, Value, Value)> makeLoopBody);

/// Returns true if `vType` is a multiple of an SME tile size. Returns false if
/// the `vType` exactly matches the size of an SME tile.
bool isMultipleOfSMETileVectorType(VectorType vType);

/// Creates a vector type for the SME tile of `elementType`.
VectorType getSMETileTypeForElement(Type elementType);

/// Erase trivially dead tile ops from a function.
void eraseTriviallyDeadTileOps(IRRewriter &rewriter,
                               FunctionOpInterface function);

/// Returns true if `tileOp` is trivially cloneable. A tile operation is
/// trivially cloneable if:
///
///  1. It has no operands (and only a single tile result)
///  2. It is 'Pure'
///
/// This ensures that the cloned operation will not share any dependencies with
/// the original operation (which could also need to be considered), and that
/// inserting the cloned operation at a different point in the program won't
/// change the semantics of the program (as it has no side effects).
bool isTriviallyCloneableTileOp(arm_sme::ArmSMETileOpInterface tileOp);

/// Returns true if `tileOp` produces a tile result.
bool hasTileResult(arm_sme::ArmSMETileOpInterface tileOp);

/// Returns the tile `OpOperand` for this `tileOp` (or null).
OpOperand *getTileOpOperand(arm_sme::ArmSMETileOpInterface tileOp);

/// Returns true `typeA` is >= (in terms of bytes) than `typeB`.
bool isTileTypeGreaterOrEqual(ArmSMETileType typeA, ArmSMETileType typeB);

} // namespace mlir::arm_sme

#endif // MLIR_DIALECT_ARMSME_UTILS_UTILS_H_
