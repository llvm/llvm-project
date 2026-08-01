//===- OpenACCUtilsReduction.h - OpenACC reduction utilities ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utility functions for OpenACC reductions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACCUTILSREDUCTION_H_
#define MLIR_DIALECT_OPENACC_OPENACCUTILSREDUCTION_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCParMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace acc {

/// Returns the parallel dimensions that participate in \p op's combine step.
///
/// Used when lowering reductions to determine which GPU parallelism levels must
/// be synchronized before combining partial results.
SmallVector<GPUParallelDimAttr>
getReductionCombineParDims(ReductionCombineOp op);

/// Returns the parallel dimensions that participate in \p op's combine step.
///
/// Prefers dimensions from an `acc.reduction_accumulate` user of the source
/// variable; otherwise falls back to \p op's `acc.par_dims` attribute.
SmallVector<GPUParallelDimAttr>
getReductionCombineParDims(ReductionCombineRegionOp op);

/// Maps an `arith` atomic RMW kind to the corresponding acc reduction operator.
ReductionOperator translateAtomicRMWKind(arith::AtomicRMWKind kind);

/// Maps an acc reduction operator to the `arith` atomic RMW kind for \p type.
///
/// Returns `std::nullopt` when \p redOp is not supported for \p type.
std::optional<arith::AtomicRMWKind>
translateACCReductionOperator(ReductionOperator redOp, Type type);

/// Creates the identity (neutral) value for a reduction of \p type and \p kind.
///
/// When \p useOnlyFiniteValue is true, floating-point identities avoid
/// non-finite sentinel values where applicable.
Value createIdentityValue(OpBuilder &b, Location loc, Type type,
                          arith::AtomicRMWKind kind,
                          bool useOnlyFiniteValue = true);

/// Combines two reduction partial values using the operator for \p kind.
Value generateReductionOp(OpBuilder &b, Location loc, Value lhs, Value rhs,
                          arith::AtomicRMWKind kind);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCUTILSREDUCTION_H_
