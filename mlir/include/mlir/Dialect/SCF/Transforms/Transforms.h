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

#ifndef MLIR_DIALECT_SCF_TRANSFORMS_TRANSFORMS_H_
#define MLIR_DIALECT_SCF_TRANSFORMS_TRANSFORMS_H_

#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {

struct LogicalResult;
class Region;
class RewriterBase;
class Operation;
class Value;

namespace scf {

class IfOp;
class ForOp;
class ParallelOp;

/// Fuses all adjacent scf.parallel operations with identical bounds and step
/// into one scf.parallel operations. Uses a naive aliasing and dependency
/// analysis.
void naivelyFuseParallelOps(Region &region);

/// Rewrite a for loop with bounds/step that potentially do not divide evenly
/// into a for loop where the step divides the iteration space evenly, followed
/// by another scf.for for the last (partial) iteration (if any; returned via
/// `partialIteration`). This transformation is called "loop peeling".
///
/// This transformation is beneficial for a wide range of transformations such
/// as vectorization or loop tiling: It enables additional canonicalizations
/// inside the peeled loop body such as rewriting masked loads into unmaked
/// loads.
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
/// scf.for %iv2 = %newUb to %ub {
///   (loop body)
/// }
/// ```
///
/// After loop peeling, this function tries to simplify affine.min and
/// affine.max ops in the body of the peeled loop and in the body of the partial
/// iteration loop, taking advantage of the fact that the peeled loop has only
/// "full" iterations. This simplification is expected to enable further
/// canonicalization opportunities through other patterns.
///
/// The return value indicates whether the loop was rewritten or not. Loops are
/// not rewritten if:
/// * Loop step size is 1 or
/// * Loop bounds and step size are static, and step already divides the
///   iteration space evenly.
///
/// Note: This function rewrites the given scf.for loop in-place and creates a
/// new scf.for operation for the last iteration. It replaces all uses of the
/// unpeeled loop with the results of the newly generated scf.for.
LogicalResult peelForLoopAndSimplifyBounds(RewriterBase &rewriter, ForOp forOp,
                                           scf::ForOp &partialIteration);

/// Peel the first iteration out of the scf.for loop. If there is only one
/// iteration, return the original loop.
LogicalResult peelForLoopFirstIteration(RewriterBase &rewriter, ForOp forOp,
                                        scf::ForOp &partialIteration);

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
tileParallelLoop(ParallelOp op, llvm::ArrayRef<int64_t> tileSizes,
                 bool noMinMaxBounds);

/// Options to dictate how loops should be pipelined.
struct PipeliningOption {
  /// Lambda returning all the operation in the forOp, with their stage, in the
  /// order picked for the pipelined loop.
  using GetScheduleFnType = std::function<void(
      scf::ForOp, std::vector<std::pair<Operation *, unsigned>> &)>;
  GetScheduleFnType getScheduleFn = nullptr;
  enum class PipelinerPart {
    Prologue,
    Kernel,
    Epilogue,
  };
  /// Lambda called by the pipeliner to allow the user to annotate the IR while
  /// it is generated.
  /// The callback passes the operation created along with the part of the
  /// pipeline and the iteration index. The iteration index is always 0 for the
  /// kernel. For the prologue and epilogue, it corresponds to the iteration
  /// peeled out of the loop in the range [0, maxStage[.
  using AnnotationlFnType =
      std::function<void(Operation *, PipelinerPart, unsigned)>;
  AnnotationlFnType annotateFn = nullptr;

  /// Control whether the epilogue should be peeled out of the loop or
  /// operations should be predicated to skip the early stages in the last loop
  /// iterations. If the epilogue is predicated; the user needs to provide a
  /// lambda to generate the predicated version of operations.
  bool peelEpilogue = true;

  /// Control whether the transformation checks that the number of iterations is
  /// greater or equal to the number of stages and skip the transformation if
  /// this is not the case. If the loop is dynamic and this is set to true and
  /// the loop bounds are not static the pipeliner will have to predicate
  /// operations in the the prologue/epilogue.
  bool supportDynamicLoops = false;

  // Callback to predicate operations when the prologue or epilogue are not
  // peeled. This takes the original operation, an i1 predicate value and the
  // pattern rewriter. It is expected to replace the given operation with
  // the predicated equivalent and return it, or return nullptr if the
  // predication is impossible. In the latter case, pipelining will fail and
  // may leave IR in a partially transformed state.
  using PredicateOpFn =
      std::function<Operation *(RewriterBase &, Operation *, Value)>;
  PredicateOpFn predicateFn = nullptr;

  // TODO: add option to decide if the prologue should be peeled.
};

/// Generate a pipelined version of the scf.for loop based on the schedule given
/// as option. This applies the mechanical transformation of changing the loop
/// and generating the prologue/epilogue for the pipelining and doesn't make any
/// decision regarding the schedule.
/// Based on the options the loop is split into several stages.
/// The transformation assumes that the scheduling given by user is valid.
/// For example if we break a loop into 3 stages named S0, S1, S2 we would
/// generate the following code with the number in parenthesis as the iteration
/// index:
///
///   S0(0)                        // Prologue
///   S0(1) S1(0)                  // Prologue
///   scf.for %I = %C0 to %N - 2 {
///     S0(I+2) S1(I+1) S2(I)       // Pipelined kernel
///   }
///   S1(N) S2(N-1)                // Epilogue
///   S2(N)                        // Epilogue
///
/// If `modifiedIR` is provided, it will be set to a value that indicates
/// whether pipelining modified the IR before failing, signaling to the caller
/// whether they can proceed with different transformations.
FailureOr<ForOp> pipelineForLoop(RewriterBase &rewriter, ForOp forOp,
                                 const PipeliningOption &options,
                                 bool *modifiedIR = nullptr);

} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMS_TRANSFORMS_H_
