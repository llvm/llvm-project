//===- OpenACCUtilsLoop.h - OpenACC Loop Utilities --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for converting OpenACC loop operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACCUTILSLOOP_H_
#define MLIR_DIALECT_OPENACC_OPENACCUTILSLOOP_H_

namespace mlir {
class OpBuilder;
namespace scf {
class ForOp;
class ParallelOp;
class ExecuteRegionOp;
} // namespace scf
namespace acc {
class LoopOp;

/// Convert a structured acc.loop to scf.for.
/// The loop arguments are converted to index type. If enableCollapse is true,
/// nested loops are collapsed into a single loop.
/// @param loopOp The acc.loop operation to convert (must not be unstructured)
/// @param enableCollapse Whether to collapse nested loops into one
/// @return The created scf.for operation or nullptr on creation error.
///         An InFlightDiagnostic is emitted on creation error.
scf::ForOp convertACCLoopToSCFFor(LoopOp loopOp, bool enableCollapse);

/// Convert acc.loop to scf.parallel.
/// The loop induction variables are converted to index types.
/// @param loopOp The acc.loop operation to convert
/// @param builder OpBuilder for creating operations
/// @return The created scf.parallel operation or nullptr on creation error.
///         An InFlightDiagnostic is emitted on creation error.
scf::ParallelOp convertACCLoopToSCFParallel(LoopOp loopOp, OpBuilder &builder);

/// Convert an unstructured acc.loop to scf.execute_region.
/// @param loopOp The acc.loop operation to convert (must be unstructured)
/// @param builder OpBuilder for creating operations
/// @return The created scf.execute_region operation or nullptr on creation
///         error. An InFlightDiagnostic is emitted on creation error.
scf::ExecuteRegionOp
convertUnstructuredACCLoopToSCFExecuteRegion(LoopOp loopOp, OpBuilder &builder);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCUTILSLOOP_H_
