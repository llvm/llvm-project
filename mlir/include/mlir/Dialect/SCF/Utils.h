//===- Utils.h - SCF dialect utilities --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various SCF utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_UTILS_H_
#define MLIR_DIALECT_SCF_UTILS_H_

namespace mlir {
class OpBuilder;
class ValueRange;

namespace scf {
class ForOp;
class ParallelOp;
} // end namespace scf

/// Create a clone of `loop` with `newIterOperands` added as new initialization
/// values and `newYieldedValues` added as new yielded values. The returned
/// ForOp has `newYieldedValues.size()` new result values.  The `loop` induction
/// variable and `newIterOperands` are remapped to the new induction variable
/// and the new entry block arguments respectively.
///
/// Additionally, if `replaceLoopResults` is true, all uses of
/// `loop.getResults()` are replaced with the first `loop.getNumResults()`
/// return values respectively. This additional replacement is provided as a
/// convenience to update the consumers of `loop`, in the case e.g. when `loop`
/// is soon to be deleted.
///
/// Return the cloned loop.
///
/// This convenience function is useful to factorize common mechanisms related
/// to hoisting roundtrips to memory into yields. It does not perform any
/// legality checks.
///
/// Prerequisite: `newYieldedValues.size() == newYieldedValues.size()`.
scf::ForOp cloneWithNewYields(OpBuilder &b, scf::ForOp loop,
                              ValueRange newIterOperands,
                              ValueRange newYieldedValues,
                              bool replaceLoopResults = true);

} // end namespace mlir
#endif // MLIR_DIALECT_SCF_UTILS_H_
