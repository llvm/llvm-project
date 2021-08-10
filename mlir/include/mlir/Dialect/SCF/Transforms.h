//===- Transforms.h - SCF dialect transformation utilities ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines transformations on SCF operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_TRANSFORMS_H_
#define MLIR_DIALECT_SCF_TRANSFORMS_H_

#include "llvm/ADT/ArrayRef.h"

namespace mlir {

class ConversionTarget;
struct LogicalResult;
class MLIRContext;
class Region;
class RewriterBase;
class TypeConverter;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;
class Operation;

namespace scf {

class IfOp;
class ForOp;
class ParallelOp;
class ForOp;

/// Fuses all adjacent scf.parallel operations with identical bounds and step
/// into one scf.parallel operations. Uses a naive aliasing and dependency
/// analysis.
void naivelyFuseParallelOps(Region &region);

/// Rewrite a for loop with bounds/step that potentially do not divide evenly
/// into a for loop where the step divides the iteration space evenly, followed
/// by an scf.if for the last (partial) iteration (if any). This transformation
/// is called "loop peeling".
///
/// Other patterns can simplify/canonicalize operations in the body of the loop
/// and the scf.if. This is beneficial for a wide range of transformations such
/// as vectorization or loop tiling.
///
/// E.g., assuming a lower bound of 0 (for illustration purposes):
/// ```
/// scf.for %iv = %c0 to %ub step %c4 {
///   (loop body)
/// }
/// ```
/// is rewritten into the following pseudo IR:
/// ```
/// %newUb = %ub - (%ub mod %c4)
/// scf.for %iv = %c0 to %newUb step %c4 {
///   (loop body)
/// }
/// scf.if %newUb < %ub {
///   (loop body)
/// }
/// ```
///
/// This function rewrites the given scf.for loop in-place and creates a new
/// scf.if operation (returned via `ifOp`) for the last iteration.
///
/// TODO: Simplify affine.min ops inside the new loop/if statement.
LogicalResult peelForLoop(RewriterBase &b, ForOp forOp, scf::IfOp &ifOp);

/// Tile a parallel loop of the form
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                             step (%arg4, %arg5)
///
/// into
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                             step (%arg4*tileSize[0],
///                                                   %arg5*tileSize[1])
///     scf.parallel (%j0, %j1) = (0, 0) to (min(tileSize[0], %arg2-%j0)
///                                           min(tileSize[1], %arg3-%j1))
///                                        step (%arg4, %arg5)
/// The old loop is replaced with the new one.
///
/// The function returns the resulting ParallelOps, i.e. {outer_loop_op,
/// inner_loop_op}.
std::pair<ParallelOp, ParallelOp>
tileParallelLoop(ParallelOp op, llvm::ArrayRef<int64_t> tileSizes);

/// Populates patterns for SCF structural type conversions and sets up the
/// provided ConversionTarget with the appropriate legality configuration for
/// the ops to get converted properly.
///
/// A "structural" type conversion is one where the underlying ops are
/// completely agnostic to the actual types involved and simply need to update
/// their types. An example of this is scf.if -- the scf.if op and the
/// corresponding scf.yield ops need to update their types accordingly to the
/// TypeConverter, but otherwise don't care what type conversions are happening.
void populateSCFStructuralTypeConversionsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target);

/// Options to dictate how loops should be pipelined.
struct PipeliningOption {
  /// Lambda returning all the operation in the forOp, with their stage, in the
  /// order picked for the pipelined loop.
  using GetScheduleFnType = std::function<void(
      scf::ForOp, std::vector<std::pair<Operation *, unsigned>> &)>;
  GetScheduleFnType getScheduleFn;
  // TODO: add option to decide if the prologue/epilogue should be peeled.
};

/// Populate patterns for SCF software pipelining transformation.
/// This transformation generates the pipelined loop and doesn't do any
/// assumptions on the schedule dictated by the option structure.
/// Software pipelining is usually done in two part. The first part of
/// pipelining is to schedule the loop and assign a stage and cycle to each
/// operations. This is highly dependent on the target and is implemented as an
/// heuristic based on operation latencies, and other hardware characteristics.
/// The second part is to take the schedule and generate the pipelined loop as
/// well as the prologue and epilogue. It is independent of the target.
/// This pattern only implement the second part.
/// For example if we break a loop into 3 stages named S0, S1, S2 we would
/// generate the following code with the number in parenthesis the iteration
/// index:
/// S0(0)                        // Prologue
/// S0(1) S1(0)                  // Prologue
/// scf.for %I = %C0 to %N - 2 {
///  S0(I+2) S1(I+1) S2(I)       // Pipelined kernel
/// }
/// S1(N) S2(N-1)                // Epilogue
/// S2(N)                        // Epilogue
void populateSCFLoopPipeliningPatterns(RewritePatternSet &patterns,
                                       const PipeliningOption &options);

} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMS_H_
