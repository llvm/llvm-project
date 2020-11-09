//===- Utils.h - Utilities to support the Linalg dialect --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_UTILS_H_
#define MLIR_DIALECT_LINALG_UTILS_H_

#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "llvm/ADT/SetVector.h"

using mlir::edsc::intrinsics::AffineIndexedValue;
using mlir::edsc::intrinsics::StdIndexedValue;

namespace mlir {
class AffineExpr;
class AffineForOp;
class AffineMap;
class OperationFolder;
class PatternRewriter;

namespace linalg {
class LinalgDependenceGraph;

/// A struct containing the Linalg producer before and after fusion.
/// When operating on tensors, `fusedProducer` may feed into a `tensor_cast` op
/// before the consumer Linalg op, until enough canonicalizations have applied.
struct FusionInfo {
  LinalgOp originalProducer;
  LinalgOp fusedProducer;
};

/// A struct containing common matchers over linalg op's region.
struct RegionMatcher {
  enum class BinaryOpKind {
    IAdd,
  };

  /// Matches the given linalg op if its body is performing binary operation on
  /// int or float scalar values and returns the binary op kind.
  ///
  /// The linalg op's region is expected to be
  /// ```
  /// {
  ///   ^bb(%a: <scalar-type>, %b: <scalar-type>):
  ///     %0 = <binary-op> %a, %b: <scalar-type>
  ///     linalg.yield %0: <scalar-type>
  /// }
  /// ```
  static Optional<BinaryOpKind> matchAsScalarBinaryOp(GenericOp op);
};

/// Checks if an iterator_type attribute is parallel.
bool isParallelIteratorType(Attribute attr);

/// Checks if an iterator_type attribute is parallel.
bool isReductionIteratorType(Attribute attr);

/// Checks if an iterator_type attribute is parallel.
bool isWindowIteratorType(Attribute attr);

/// Checks whether the specific `producer` is the last write to exactly the
/// whole `consumedView`. This checks structural dominance, that the dependence
/// is a RAW without any interleaved write to any piece of `consumedView`.
bool isProducerLastWriteOfView(const LinalgDependenceGraph &graph,
                               LinalgOp consumer, Value consumedView,
                               LinalgOp producer);

/// Checks whether fusing the specific `producer` of the `consumedView` is
/// feasible. This checks `producer` is the last write of `consumedView` and
/// that no interleaved dependence would be violated (RAW, WAR or WAW).
bool isFusableInto(const LinalgDependenceGraph &graph, LinalgOp consumer,
                   Value consumedView, LinalgOp producer);

/// Fuses producer into consumer if the producer is structurally feasible and
/// the fusion would not violate dependencies.
/// Implements the fusion part of the "tileAndFuse on buffers"
/// transformation and thus requires the `consumerdIdx`^th operand of `consumer`
/// to be a `subview` op (generally obtained by applying the tiling
/// transformation).
Optional<FusionInfo> fuseProducerOfBuffer(OpBuilder &b, LinalgOp consumer,
                                          unsigned consumerIdx,
                                          const LinalgDependenceGraph &graph);
/// Tensor counterpart of `fuseProducerOfBuffer`.
/// This implements the fusion part of the "tileAndFuse on tensors"
/// transformation and thus requires the `consumerdIdx`^th operand of `consumer`
/// to be the result of a `subtensor` op (generally obtained by applying the
/// tiling transformation).
Optional<FusionInfo> fuseProducerOfTensor(OpBuilder &b, LinalgOp consumer,
                                          unsigned consumerIdx);

/// Fuse linalg operation on tensors, with the producer of the operand at
/// position `consumerIdx` of the consumer.
Optional<SmallVector<Value, 1>> fuseTensorOps(PatternRewriter &rewriter,
                                              Operation *consumer,
                                              unsigned consumerIdx);

/// Returns the linearized list of all shape dimensions in a `linalgOp`.
/// Applying the inverse, concatenated loopToOperandRangeMaps to this list
/// allows the derivation of loop ranges for any linalgOp.
SmallVector<Value, 8> getShape(OpBuilder &builder, LinalgOp linalgOp);
template <typename ConcreteOpTy>
SmallVector<Value, 8> getShape(OpBuilder &builder, ConcreteOpTy linalgOp) {
  return getShape(builder, cast<linalg::LinalgOp>(linalgOp.getOperation()));
}

/// Returns the loop ranges of the `linalgOp`. Applies the inverse of the
/// concatenated indexing maps to the result of `getShape`. Returns None if
/// the bounds computation fails.
Optional<SmallVector<Value, 4>> getLoopRanges(OpBuilder &builder,
                                              LinalgOp linalgOp);

/// Returns the values obtained by applying `map` to the list of values.
SmallVector<Value, 4> applyMapToValues(OpBuilder &b, Location loc,
                                       AffineMap map, ValueRange values);

/// Apply the permutation defined by `permutation` to `inVec`.
/// Element `i` in `inVec` is mapped to location `j = permutation[i]`.
/// E.g.: for an input vector `inVec = ['a', 'b', 'c']` and a permutation vector
/// `permutation = [2, 0, 1]`, this function leaves `inVec = ['c', 'a', 'b']`.
template <typename T, unsigned N>
void applyPermutationToVector(SmallVector<T, N> &inVec,
                              ArrayRef<unsigned> permutation) {
  SmallVector<T, N> auxVec(inVec.size());
  for (unsigned i = 0; i < permutation.size(); ++i)
    auxVec[i] = inVec[permutation[i]];
  inVec = auxVec;
}

/// Scheme used to distribute loops to processors.
enum class DistributionMethod {
  /// Cyclic distribution where no assumption is made about the dynamic
  /// relationship between number of processors and number of iterations of the
  /// distributed loop. Distributes the following loop
  ///
  /// scf.parallel (%iv) = (%lb) to (%ub) step (%step)
  ///
  /// to
  ///
  /// scf.parallel(%iv)= (%lb + %procId * %step) to (%ub) step (%step * %nprocs)
  Cyclic = 0,

  /// Cyclic distribution where the number of processors can be assumed to be
  /// more than or equal to the number of iterations of the distributed loop. In
  /// such cases, a simple in-bounds check is enough (instead of materializing a
  /// loop). Distributes the following loop
  ///
  /// scf.parallel (%iv) = (%lb) to (%ub) step (%step)
  ///
  /// to
  ///
  /// %iv = %lb + %procId * %step
  /// %cond = cmpi "slt", %iv, %ub
  /// scf.if %cond {
  ///   ...
  /// }
  CyclicNumProcsGeNumIters = 1,

  /// Cyclic distribution where the number of processors can be assumed to be
  ///  equal to the number of iterations of the distributed loop. In such cases,
  ///  no bounds check is needed. Distributes the following loop
  ///
  /// scf.parallel (%iv) = (%lb) to (%ub) step (%step)
  ///
  /// to
  ///
  /// %iv = %lb + %procId * %step
  CyclicNumProcsEqNumIters = 2
};

/// Callback function type used to get processor ID, and number of processors
/// used for distribution for all parallel loops generated.
struct ProcInfo {
  Value procId;
  Value nprocs;
};
using ProcInfoCallBackFn = std::function<SmallVector<ProcInfo, 2>(
    OpBuilder &b, Location loc, ArrayRef<Range> parallelLoopRanges)>;

/// Options that allow distribution of loops generated in Linalg transforms to
/// processors while generating the loops.
struct LinalgLoopDistributionOptions {
  /// Callback function that returns the Values for processor ID (`procId`), and
  /// number of processors (`nprocs`) used to execute the parallel loops. The
  /// number of `{procId, nprocs}` pairs returned must be equal to the number of
  /// `parallelLoopRanges` passed into the callback, which in-turn is same as
  /// the number of parallel loops for which the `distributionMethod` is
  /// specified below.
  ProcInfoCallBackFn procInfo;
  /// Specification of how to distribute the `scf.parallel` loops that are
  /// generated. As the `scf.parallel` loop is generated, the elements of this
  /// vector is used (from left to right) and the specified distribution is
  /// applied. If the vector is less than the number of `scf.parallel` loops
  /// generated, then no distribution is applied.
  SmallVector<DistributionMethod, 0> distributionMethod = {};
};

/// Utility class used to generate nested loops with ranges described by
/// `loopRanges` and loop type described by the `iteratorTypes`. `bodyBuilderFn`
/// is used to generate the body of the innermost loop. It is passed a range
/// of loop induction variables.
template <typename LoopTy>
struct GenerateLoopNest {
  using IndexedValueTy =
      typename std::conditional<std::is_same<LoopTy, AffineForOp>::value,
                                AffineIndexedValue, StdIndexedValue>::type;

  static void
  doit(ArrayRef<Range> loopRanges, ValueRange iterArgInitValues,
       ArrayRef<Attribute> iteratorTypes,
       function_ref<scf::ValueVector(ValueRange, ValueRange)> bodyBuilderFn,
       Optional<LinalgLoopDistributionOptions> = None);
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_UTILS_H_
