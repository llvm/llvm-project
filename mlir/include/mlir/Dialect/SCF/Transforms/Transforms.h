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

class AffineMap;
class ConversionTarget;
struct LogicalResult;
class MLIRContext;
class Region;
class RewriterBase;
class TypeConverter;
class RewritePatternSet;
class Operation;
class Value;
class ValueRange;
class PatternRewriter;

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

/// Populates the provided pattern set with patterns that do 1:N type
/// conversions on (some) SCF ops. This is intended to be used with
/// applyPartialOneToNConversion.
void populateSCFStructuralOneToNTypeConversions(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns);

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

  // Lamdba to predicate operations when the prologue or epilogue are not
  // peeled. This takes the original operation, an i1 predicate value and the
  // pattern rewriter.
  using PredicateOpFn =
      std::function<Operation *(RewriterBase &, Operation *, Value)>;
  PredicateOpFn predicateFn = nullptr;

  // TODO: add option to decide if the prologue should be peeled.
};

/// Populate patterns for SCF software pipelining transformation. See the
/// ForLoopPipeliningPattern for the transformation details.
void populateSCFLoopPipeliningPatterns(RewritePatternSet &patterns,
                                       const PipeliningOption &options);

/// Populate patterns for canonicalizing operations inside SCF loop bodies.
/// At the moment, only affine.min/max computations with iteration variables,
/// loop bounds and loop steps are canonicalized.
void populateSCFForLoopCanonicalizationPatterns(RewritePatternSet &patterns);

} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMS_TRANSFORMS_H_
