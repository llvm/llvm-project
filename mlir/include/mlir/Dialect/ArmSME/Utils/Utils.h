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
#include "mlir/IR/BuiltinTypes.h"
#include <optional>

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

/// Returns the type of SME tile this vector type corresponds to, or none if the
/// vector type does not fit within an SME tile.
std::optional<ArmSMETileType> getSMETileType(VectorType);

/// Verifies the tile ID (if set) on this tile operation is valid.
LogicalResult verifyOperationHasValidTileId(Operation *);

} // namespace mlir::arm_sme

#endif // MLIR_DIALECT_ARMSME_UTILS_UTILS_H_
