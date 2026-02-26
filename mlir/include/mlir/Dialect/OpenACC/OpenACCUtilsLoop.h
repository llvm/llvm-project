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
class IRMapping;
class Location;
class Region;
class RewriterBase;
namespace scf {
class ForOp;
class ParallelOp;
class ExecuteRegionOp;
} // namespace scf
namespace acc {
class LoopOp;

/// Wrap a multi-block region in an scf.execute_region.
/// Clones the given region into a new scf.execute_region, replacing
/// acc.yield/acc.terminator with scf.yield. Use this to convert unstructured
/// control flow (e.g. multiple blocks with branches) into a single SCF region.
/// @param region The region to wrap (cloned into the execute_region; not
/// modified).
/// @param mapping IR mapping for the clone; updated with block and value
/// mappings.
/// @param loc Location for the created execute_region op.
/// @param rewriter RewriterBase for creating and erasing operations.
/// @return The created scf.execute_region operation, or nullptr if the region
///         has an acc.yield with operands (results not yet supported).
scf::ExecuteRegionOp
wrapMultiBlockRegionWithSCFExecuteRegion(Region &region, IRMapping &mapping,
                                         Location loc, RewriterBase &rewriter);
/// Convert a structured acc.loop to scf.for.
/// The loop arguments are converted to index type. If enableCollapse is true,
/// nested loops are collapsed into a single loop.
/// @param loopOp The acc.loop operation to convert (must not be unstructured)
/// @param rewriter RewriterBase for creating operations
/// @param enableCollapse Whether to collapse nested loops into one
/// @return The created scf.for operation or nullptr on creation error.
///         An InFlightDiagnostic is emitted on creation error.
scf::ForOp convertACCLoopToSCFFor(LoopOp loopOp, RewriterBase &rewriter,
                                  bool enableCollapse);

/// Convert acc.loop to scf.parallel.
/// The loop induction variables are converted to index types.
/// @param loopOp The acc.loop operation to convert
/// @param rewriter RewriterBase for creating and erasing operations
/// @return The created scf.parallel operation or nullptr on creation error.
///         An InFlightDiagnostic is emitted on creation error.
scf::ParallelOp convertACCLoopToSCFParallel(LoopOp loopOp,
                                            RewriterBase &rewriter);

/// Convert an unstructured acc.loop to scf.execute_region.
/// @param loopOp The acc.loop operation to convert (must be unstructured)
/// @param rewriter RewriterBase for creating and erasing operations
/// @return The created scf.execute_region operation or nullptr on creation
///         error. An InFlightDiagnostic is emitted on creation error.
scf::ExecuteRegionOp
convertUnstructuredACCLoopToSCFExecuteRegion(LoopOp loopOp,
                                             RewriterBase &rewriter);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCUTILSLOOP_H_
