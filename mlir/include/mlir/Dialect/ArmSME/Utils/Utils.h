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

#include "mlir/Dialect/ArmSME/IR/ArmSME.h"

namespace mlir {
namespace arm_sme {

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

/// Extends or truncates `tile`, which should be an `arm_sme::GetTileID` or
/// `arm_sme::CastVectorToTile` op returning an 8/16/32/64/128-bit scalar
/// integer, to an i32 that can be passed as the `tile` parameter to the SME
/// intrinsics. Or returns `tile` if already i32.
Value castTileIDToI32(Value tile, Location loc, RewriterBase &rewriter);

} // namespace arm_sme
} // namespace mlir

#endif // MLIR_DIALECT_ARMSME_UTILS_UTILS_H_
