//===- Utils.h - Utilities to support the Linalg dialect --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_UTILS_H_
#define MLIR_DIALECT_LINALG_UTILS_H_

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"

#include "llvm/ADT/SetVector.h"

namespace mlir {
class AffineExpr;
class AffineMap;
class OperationFolder;

namespace linalg {
class LinalgDependenceGraph;

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
/// When non-null, the optional pointer `folder` is used to call into the
/// `createAndFold` builder method. If `folder` is null, the regular `create`
/// method is called.
Optional<FusionInfo> fuseProducerOf(OpBuilder &b, LinalgOp consumer,
                                    unsigned consumerIdx,
                                    const LinalgDependenceGraph &graph,
                                    OperationFolder *folder = nullptr);

/// Fuse linalg operation on tensors, where the result of the producer is used
/// as the operand of the consumer at position `consumerIdx`.
Optional<LinalgOp> fuseTensorOps(OpBuilder &b, LinalgOp producer,
                                 LinalgOp consumer, unsigned consumerIdx,
                                 OperationFolder *folder = nullptr);

/// Returns the linearized list of all view dimensions in a linalgOp. Applying
/// the inverse, concatenated loopToOperandRangeMaps to this list allows the
/// derivation of loop ranges for any linalgOp.
template <typename ConcreteOp>
SmallVector<Value, 8> getViewSizes(OpBuilder &builder, ConcreteOp linalgOp) {
  auto loc = linalgOp.getLoc();
  SmallVector<Value, 8> res;
  for (auto v : linalgOp.getInputsAndOutputBuffers()) {
    MemRefType t = v.getType().template cast<MemRefType>();
    for (unsigned i = 0; i < t.getRank(); ++i)
      res.push_back(builder.create<DimOp>(loc, v, i));
  }
  return res;
}

/// Returns the values obtained by applying `map` to the list of values.
/// When non-null, the optional pointer `folder` is used to call into the
/// `createAndFold` builder method. If `folder` is null, the regular `create`
/// method is called.
SmallVector<Value, 4> applyMapToValues(OpBuilder &b, Location loc,
                                       AffineMap map, ArrayRef<Value> values,
                                       OperationFolder *folder = nullptr);

struct TiledLinalgOp {
  LinalgOp op;
  SmallVector<Operation *, 8> loops;
};

/// Performs standalone tiling of a single LinalgOp by `tileSizes`.
/// and permute the loop nest according to `permutation`
/// The permutation is expressed as a list of integers that specify
/// the new ordering of the loop nest. The length of `permutation`
/// must be equal to the length of `tileSizes`.
/// E.g. the permutation `(i,j,k) -> (j,k,i)` will be expressed with
/// `permutation = [1,2,0]`. All values in `permutation` must be
/// integers, in the range 0..`tileSizes.size()` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation). An empty list
/// states for the identity permutation.
/// Returns a struct containing the tiled loops in the specified order
/// and the cloned op if successful, llvm::None otherwise.
/// When non-null, the optional pointer `folder` is used to call into the
/// `createAndFold` builder method. If `folder` is null, the regular `create`
/// method is called.
Optional<TiledLinalgOp> tileLinalgOp(OpBuilder &b, LinalgOp op,
                                     ArrayRef<Value> tileSizes,
                                     ArrayRef<unsigned> permutation = {},
                                     OperationFolder *folder = nullptr);
Optional<TiledLinalgOp> tileLinalgOpToParallelLoops(
    OpBuilder &b, LinalgOp op, ArrayRef<Value> tileSizes,
    ArrayRef<unsigned> permutation = {}, OperationFolder *folder = nullptr);

/// Performs standalone tiling of a single LinalgOp by constant `tileSizes`.
/// and permute the loop nest according to `permutation`
/// The permutation is expressed as a list of integers that specify
/// the new ordering of the loop nest. The length of `permutation`
/// must be equal to the length of `tileSizes`.
/// E.g. the permutation `(i,j,k) -> (j,k,i)` will be expressed with
/// `permutation = [1,2,0]`. All values in `permutation` must be
/// integers, in the range 0..`tileSizes.size()` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation). An empty list
/// states for the identity permutation.
/// Returns a struct containing the tiled loops in the specified order
/// and the cloned op if successful, llvm::None otherwise.
/// When non-null, the optional pointer `folder` is used to call into the
/// `createAndFold` builder method. If `folder` is null, the regular `create`
/// method is called.
Optional<TiledLinalgOp> tileLinalgOp(OpBuilder &b, LinalgOp op,
                                     ArrayRef<int64_t> tileSizes,
                                     ArrayRef<unsigned> permutation = {},
                                     OperationFolder *folder = nullptr);
Optional<TiledLinalgOp> tileLinalgOpToParallelLoops(
    OpBuilder &b, LinalgOp op, ArrayRef<int64_t> tileSizes,
    ArrayRef<unsigned> permutation = {}, OperationFolder *folder = nullptr);

template <typename... Args>
Optional<TiledLinalgOp> tileLinalgOperation(OpBuilder &b, Operation *op,
                                            Args... args) {
  return tileLinalgOp(b, cast<LinalgOp>(op), args...);
}

struct PromotionInfo {
  Value buffer;
  Value fullLocalView;
  Value partialLocalView;
};

/// Promotes the `subViews` into a new buffer allocated at the insertion point
/// `b`. For now, promotion occurs in 3 steps:
///   1. Create a new buffer for a full tile (i.e. not clipped at the boundary).
///   2. Take a full view on the buffer and `linalg.fill` it with zeros (use
///      float zero for now).
///   3. Take a partial slice of the full view in step 2. and copy into it.
/// Infers statically sized buffers from subViews unless `dynamicBuffers` is
/// true.
///
/// Returns a list of PromotionInfo which hold the promoted buffer and the
/// full and partial views indexing into the buffer.
SmallVector<PromotionInfo, 8>
promoteSubViews(OpBuilder &b, Location loc, ArrayRef<Value> subViews,
                bool dynamicBuffers = false, OperationFolder *folder = nullptr);

/// Returns all the operands of `linalgOp` that are not views.
/// Asserts that these operands are value types to allow transformations like
/// tiling to just use the values when cloning `linalgOp`.
SmallVector<Value, 4> getAssumedNonViewOperands(LinalgOp linalgOp);

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

/// Prepares the SubView promotion later performed by `promoteSubViews`
/// (where most of the transformation happens). It arranges the new
/// operands for `LinalgOp op` and deallocates the new buffer(s)
/// It is the entry point for declarative transformation
/// Returns the cloned `LinalgOp` with the new operands
LinalgOp promoteSubViewOperands(OpBuilder &b, LinalgOp op,
                                llvm::SetVector<Value> subViews,
                                bool dynamicBuffers = false,
                                OperationFolder *folder = nullptr);

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_UTILS_H_
