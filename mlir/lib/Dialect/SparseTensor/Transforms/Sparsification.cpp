//===- Sparsification.cpp - Implementation of sparsification --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements converting sparse tensor types to actual sparse code.
//
//===----------------------------------------------------------------------===//

#include "CodegenEnv.h"
#include "CodegenUtils.h"
#include "LoopEmitter.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TensorEncoding.h"
#include "llvm/ADT/SmallBitVector.h"
#include <optional>

using namespace mlir;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

namespace {

/// Iteration graph sorting.
enum SortMask {
  kSparseOnly = 0x0,
  kIncludeDense = 0x1,
  kIncludeUndef = 0x2,
  kIncludeAll = 0x3
};

/// A helper class that visits an affine expression and tries to find an
/// AffineDimExpr to which the corresponding iterator from a GenericOp matches
/// the desired iterator type.
class AffineDimFinder : public AffineExprVisitor<AffineDimFinder> {
public:
  explicit AffineDimFinder(linalg::GenericOp op)
      : iterTypes(op.getIteratorTypesArray()) {}
  void visitDimExpr(AffineDimExpr expr) {
    if (pickedDim == nullptr || pickIterType == iterTypes[expr.getPosition()]) {
      pickedDim = expr;
    }
  }

  /// Set the desired iterator type that we want to pick.
  void setPickedIterType(utils::IteratorType iterType) {
    pickIterType = iterType;
  }

  /// Get the desired AffineDimExpr.
  AffineDimExpr getDimExpr() const { return pickedDim.cast<AffineDimExpr>(); }

private:
  /// The picked AffineDimExpr after visit.
  AffineExpr pickedDim;
  /// The iterator type that we want.
  utils::IteratorType pickIterType;
  /// The mapping between dim=>iterator type.
  SmallVector<utils::IteratorType> iterTypes;
};

} // namespace

//===----------------------------------------------------------------------===//
// Sparse compiler analysis methods.
//===----------------------------------------------------------------------===//

/// Determines if affine expression is invariant.
static bool isInvariantAffine(AffineExpr a, ArrayRef<unsigned> loopStack,
                              unsigned ldx, bool &atLevel) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    if (idx == ldx) {
      atLevel = true;
      // Must be invariant if we are at the level.
      return true;
    }
    bool isInvariant = false;
    for (unsigned loop : loopStack) {
      isInvariant = (loop == idx);
      if (isInvariant)
        break;
    }
    return isInvariant;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return isInvariantAffine(binOp.getLHS(), loopStack, ldx, atLevel) &&
           isInvariantAffine(binOp.getRHS(), loopStack, ldx, atLevel);
  }
  default: {
    assert(a.isa<AffineConstantExpr>());
    return true;
  }
  }
}

/// Determines if affine expression is invariant.
static bool isInvariantAffine(CodegenEnv &env, AffineExpr a, unsigned ldx,
                              bool &atLevel) {
  return isInvariantAffine(a, env.getLoopCurStack(), ldx, atLevel);
}

/// Helper method to construct a permuted dimension ordering
/// that adheres to the given topological sort.
static AffineMap permute(CodegenEnv &env, AffineMap m) {
  assert(m.getNumDims() + env.merger().getNumFilterLoops() ==
             env.topSortSize() &&
         "size mismatch");
  // Construct the inverse of `m`; to avoid the asymptotic complexity
  // of calling `m.getPermutedPosition` repeatedly.
  SmallVector<unsigned> perm;
  unsigned numResults = m.getNumResults();
  BitVector worklist(numResults, true);
  unsigned loopDepth = 1;

  // Construct the permutation.
  while (worklist.any() && loopDepth <= env.topSortSize()) {
    unsigned preSize = perm.size();
    for (auto dim : worklist.set_bits()) {
      bool atLevel = false;
      if (m.getResult(dim).isa<AffineConstantExpr>() ||
          (isInvariantAffine(m.getResult(dim),
                             env.getTopSortSlice(0, loopDepth),
                             env.topSortAt(loopDepth - 1), atLevel) &&
           atLevel)) {
        // If the matching affine is constant expression or just become
        // invariant. We can visit the dimension now without breaking the
        // topSort constraint.
        perm.push_back(dim);
      }
    }

    // Removes resolved dimension.
    for (unsigned i = preSize, e = perm.size(); i < e; i++)
      worklist.reset(perm[i]);

    // Tries to entering the next loop level.
    loopDepth += 1;
  }

  assert(perm.size() == numResults);
  return AffineMap::getPermutationMap(perm, env.op().getContext());
}

/// Helper method to inspect affine expressions. Rejects cases where the
/// same index is used more than once. Also rejects compound affine
/// expressions in sparse dimensions.
/// filterIdx stores the current filter loop idx should be used for the next
/// compound affine sparse level, and it will be incremented by one when
/// used.
static bool findAffine(Merger &merger, unsigned tensor, unsigned dim,
                       AffineExpr a, DimLevelType dlt, unsigned &filterLdx,
                       bool setLvlFormat = true) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    if (!isUndefDLT(merger.getDimLevelType(tensor, idx)))
      return false; // used more than once

    if (setLvlFormat)
      merger.setDimAndDimLevelType(tensor, idx, dim, dlt);
    return true;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::Constant: {
    if (!isDenseDLT(dlt) && setLvlFormat) {
      assert(isUndefDLT(merger.getDimLevelType(tensor, filterLdx)));
      // Use a filter loop for sparse affine expression.
      merger.setDimAndDimLevelType(tensor, filterLdx++, dim, dlt);
    }

    if (auto binOp = a.dyn_cast<AffineBinaryOpExpr>()) {
      // We do not set dim level format for affine expresssion like d0 + d1 on
      // either loop index at d0 or d1.
      // We continue the recursion merely to check whether current affine is
      // admissible or not.
      return findAffine(merger, tensor, dim, binOp.getLHS(), dlt, filterLdx,
                        false) &&
             findAffine(merger, tensor, dim, binOp.getRHS(), dlt, filterLdx,
                        false);
    }
    // Falls through when it is a constant Affine
    return true;
  }
  default:
    return false;
  }
}

/// Get the total number of compound affine expressions in affineMap that are
/// attached to the given tensor. For the following inputs:
///
/// affineMap = (d0, d1, d2) => (d0 + d1, d2)
/// tensor = ["compressed", "compressed"]
///
/// Returns 1 (because the first level is compressed and its corresponding
/// affineMap is d0 + d1)
static unsigned getNumCompoundAffineOnSparseDims(AffineMap affineMap,
                                                 Value tensor) {
  unsigned num = 0;
  const auto enc = getSparseTensorEncoding(tensor.getType());
  if (enc) {
    const ArrayRef<AffineExpr> exps = affineMap.getResults();
    const Level lvlRank = enc.getLvlRank();
    assert(static_cast<Level>(exps.size()) == lvlRank);
    for (Level l = 0; l < lvlRank; l++) {
      // FIXME: `toOrigDim` is deprecated.
      const Dimension d = toOrigDim(enc, l);
      // FIXME: there's some dim/lvl confusion here; since `d` isn't
      // guaranteed to be in bounds (for non-permutations).
      if (!exps[d].isa<AffineDimExpr>() && !enc.isDenseLvl(l))
        num++;
    }
  }
  return num;
}

/// Get the total number of compound affine expressions attached on a sparse
/// level in the given GenericOp.
static unsigned getNumCompoundAffineOnSparseDims(linalg::GenericOp op) {
  unsigned num = 0;
  for (OpOperand &t : op->getOpOperands())
    num += getNumCompoundAffineOnSparseDims(op.getMatchingIndexingMap(&t),
                                            t.get());
  return num;
}

static bool hasCompoundAffineOnSparseOut(linalg::GenericOp op) {
  OpOperand *out = op.getDpsInitOperand(0);
  if (getSparseTensorType(out->get()).isAllDense())
    return false;
  return getNumCompoundAffineOnSparseDims(op.getMatchingIndexingMap(out),
                                          out->get());
}

/// Helper method to inspect sparse encodings in the tensor types.
/// Fills the per-dimension sparsity information for all tensors.
/// Returns true if the sparse annotations and affine subscript
/// expressions of all tensors are admissible. Returns false if
/// no annotations are found or inadmissible constructs occur.
static bool findSparseAnnotations(CodegenEnv &env) {
  bool annotated = false;
  unsigned filterLdx = env.merger().getFilterLoopStartingIdx();
  for (OpOperand &t : env.op()->getOpOperands()) {
    const auto map = env.op().getMatchingIndexingMap(&t);
    const auto enc = getSparseTensorEncoding(t.get().getType());
    if (enc)
      annotated = true;
    const Level lvlRank = map.getNumResults();
    assert(!enc || lvlRank == enc.getLvlRank());
    assert(env.op().getRank(&t) == lvlRank);
    for (Level l = 0; l < lvlRank; l++) {
      const unsigned tensor = t.getOperandNumber();
      // FIXME: `toOrigDim` is deprecated.
      const AffineExpr a = map.getResult(toOrigDim(enc, l));
      if (!findAffine(env.merger(), tensor, l, a, enc.getLvlType(l), filterLdx))
        return false; // inadmissible affine expression
    }
  }
  assert(filterLdx == env.merger().getNumLoops());
  return annotated;
}

/// A helper to compute a topological sort. O(n^2) time complexity
/// as we use adj matrix for the graph.
/// The sorted result will put the first Reduction iterator to the
/// latest possible index.
static bool topSortOptimal(CodegenEnv &env, unsigned n,
                           ArrayRef<utils::IteratorType> iteratorTypes,
                           std::vector<unsigned> &inDegree,
                           std::vector<std::vector<bool>> &adjM) {
  std::vector<unsigned> redIt;    // reduce iterator with 0 degree
  std::vector<unsigned> parIt;    // parallel iterator with 0 degree
  std::vector<unsigned> filterIt; // filter loop with 0 degree
  for (unsigned i = 0; i < n; i++) {
    if (inDegree[i] == 0) {
      if (env.merger().isFilterLoop(i))
        filterIt.push_back(i);
      else if (linalg::isReductionIterator(iteratorTypes[i]))
        redIt.push_back(i);
      else
        parIt.push_back(i);
    }
  }

  while (!redIt.empty() || !parIt.empty() || !filterIt.empty()) {
    // We always choose in order of filter loop -> parallel loop -> reduction
    // loop because
    // 1. Putting reduction loop early might make the loop sequence
    // inadmissible.
    // 2. Filter loops should be put as early as possible for better
    // performance, since only one (if any) iteration will carry the
    // computation. E.g., for (1 to N)
    //  for (1 to M)
    //    for (1 to K)
    //      if (xxx)
    //        O(X) computation  => O(NMK+NMX) time complexity
    //
    // By putting the filter loop one level up, we got
    //
    // for (1 to N)
    //  for (1 to K)
    //    if (xxx)
    //      for (1 to M)
    //        O(X) computation  => O(NK+NMX) time complexity
    auto &it = !filterIt.empty() ? filterIt : (!parIt.empty() ? parIt : redIt);
    auto src = it.back();
    env.topSortPushBack(src);
    it.pop_back();
    // Update in-degree, and push 0-degree node into worklist.
    for (unsigned dst = 0; dst < n; dst++) {
      if (adjM[src][dst] && --inDegree[dst] == 0) {
        if (env.merger().isFilterLoop(dst))
          filterIt.push_back(dst);
        else if (linalg::isReductionIterator(iteratorTypes[dst]))
          redIt.push_back(dst);
        else
          parIt.push_back(dst);
      }
    }
  }
  return env.topSortSize() == n;
}

/// Helper method to add all constraints from the indices in one affine
/// expression before all indices in the other affine expression. For
/// example i0+i1 < i2+i3+1 yields i0<i2, i0<i3, i1<i2, and i1<i3.
/// The affine expression `a` is empty iff `fidx` have a value, leading to
/// b = (i0 + i1) < fidx => i0 < fidx, i1 < fidx.
/// The affine expression `b` is empty iff `tidx` have a value, leading to
/// tidx < a = (i0 + i1) => tidx < i0, tidx < i1.
static void addAffineOrderings(std::vector<std::vector<bool>> &adjM,
                               std::vector<unsigned> &inDegree, AffineExpr a,
                               AffineExpr b, std::optional<unsigned> fidx,
                               std::optional<unsigned> tidx) {
  if (!a && !b) {
    // Recursion leaf.
    assert(fidx && tidx);
    unsigned f = *fidx, t = *tidx;
    if (!adjM[f][t]) {
      adjM[f][t] = true;
      inDegree[t]++;
    }
    return;
  }
  // Picks an affine expression and expand (recurse into) it.
  auto toExpand = a ? a : b;
  switch (toExpand.getKind()) {
  case AffineExprKind::DimId: {
    auto idx = toExpand.cast<AffineDimExpr>().getPosition();
    if (toExpand == a)
      addAffineOrderings(adjM, inDegree, AffineExpr(), b, idx, tidx);
    else // toExpand == b
      addAffineOrderings(adjM, inDegree, a, AffineExpr(), fidx, idx);
    break;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    auto binOp = toExpand.cast<AffineBinaryOpExpr>();
    if (toExpand == a) {
      addAffineOrderings(adjM, inDegree, binOp.getLHS(), b, fidx, tidx);
      addAffineOrderings(adjM, inDegree, binOp.getRHS(), b, fidx, tidx);
    } else {
      addAffineOrderings(adjM, inDegree, a, binOp.getLHS(), fidx, tidx);
      addAffineOrderings(adjM, inDegree, a, binOp.getRHS(), fidx, tidx);
    }
    break;
  }
  default:
    break;
  }
}

static void tryLoosenAffineDenseConstraints(linalg::GenericOp op,
                                            std::optional<unsigned> &fldx,
                                            AffineExpr &fa,
                                            std::optional<unsigned> &tldx,
                                            AffineExpr &ta) {
  // We use a heuristic here to only pick one dim expression from each
  // compound affine expression to establish the order between two dense
  // dimensions.
  if (!tldx) {
    AffineDimFinder finder(op);
    // NOTE: The ordering can only be loosen when the destination level is
    // dense (when !tldx), for [dense, sparse] -> (d0 + d1, d2), we still
    // require both d0 < d2 and d1 < d2 to ensure correct ordering (i.e.,
    // no ordering like d0->d2->d1).
    // TODO: this is obviously a sub optimal solution.
    if (!fldx && !fa.isa<AffineConstantExpr>()) {
      // Heuristic: we prefer parallel loop for lhs to reduce the chance
      // we add reduce < parallel ordering.
      finder.setPickedIterType(utils::IteratorType::parallel);
      finder.walkPostOrder(fa);
      fa = finder.getDimExpr();
      fldx = finder.getDimExpr().getPosition();
    }
    if (!ta.isa<AffineConstantExpr>()) {
      // Heuristic: we prefer reduction loop for rhs to reduce the chance
      // addint reduce < parallel ordering.
      finder.setPickedIterType(utils::IteratorType::reduction);
      finder.walkPostOrder(ta);
      ta = finder.getDimExpr();
      tldx = finder.getDimExpr().getPosition();
    }
  }
}

/// Computes a topologically sorted iteration graph for the linalg
/// operation. Ensures all tensors are visited in natural index order. This
/// is essential for sparse storage formats since these only support access
/// along fixed dimensions. Even for dense storage formats, however, the
/// natural index order yields innermost unit-stride access with better
/// spatial locality.
static bool computeIterationGraph(CodegenEnv &env, unsigned mask,
                                  OpOperand *skip = nullptr) {
  // Set up an n x n from/to adjacency matrix of the iteration graph
  // for the implicit loop indices i_0 .. i_n-1.
  const unsigned n = env.merger().getNumLoops();
  std::vector<std::vector<bool>> adjM(n, std::vector<bool>(n, false));
  std::vector<unsigned> inDegree(n, 0); // in-degree of each node.
  const auto iteratorTypes = env.op().getIteratorTypesArray();
  // Iterate over the indexing maps of every tensor in the tensor expression.
  for (OpOperand &t : env.op()->getOpOperands()) {
    // Get map and encoding.
    const auto map = env.op().getMatchingIndexingMap(&t);
    const auto enc = getSparseTensorEncoding(t.get().getType());
    assert(map.getNumDims() + getNumCompoundAffineOnSparseDims(env.op()) == n);
    // Skip dense tensor constraints when not requested.
    if (!(mask & SortMask::kIncludeDense) && !enc)
      continue;
    // Each tensor expression and optional dimension ordering (row-major
    // by default) puts an ordering constraint on the loop indices. For
    // example, the tensor expresion A_ijk forces the ordering i < j < k
    // on the loop indices if no explicit dimension ordering is given.
    const Level lvlRank = map.getNumResults();
    assert(!enc || lvlRank == enc.getLvlRank());
    for (Level l = 0; l < lvlRank; l++) {
      // FIXME: `toOrigDim` is deprecated.
      AffineExpr ta = map.getResult(toOrigDim(enc, l));
      std::optional<unsigned> tldx =
          env.merger().getLoopIdx(t.getOperandNumber(), l);

      // Filter loops should be constructed after all the dependent loops,
      // i.e., d0 + d1 < filter_loop(d0 + d1)
      if (tldx && env.merger().isFilterLoop(*tldx)) {
        assert(!ta.isa<AffineDimExpr>() && !isDenseDLT(enc.getLvlType(l)));
        addAffineOrderings(adjM, inDegree, ta, AffineExpr(), std::nullopt,
                           tldx);
        // Now that the ordering of affine expression is captured by filter
        // loop idx, we only need to ensure the affine ordering against filter
        // loop. Thus, we reset the affine express to nil here to mark it as
        // resolved.
        ta = AffineExpr();
      }

      // Skip tensor during cycle resolution, though order between filter loop
      // and dependent loops need to be guaranteed unconditionally.
      if (&t == skip)
        continue;

      if (l > 0) {
        // FIXME: `toOrigDim` is deprecated.
        AffineExpr fa = map.getResult(toOrigDim(enc, l - 1));
        std::optional<unsigned> fldx =
            env.merger().getLoopIdx(t.getOperandNumber(), l - 1);

        // Applying order constraints on every pair of dimExpr between two
        // compound affine expressions can sometime too strict:
        // E.g, for [dense, dense] -> (d0 + d1, d2 + d3).
        // It is totally fine to have loop sequence d0->d2->d1->d3 instead of
        // requiring d0 < d2, d1 < d2, d0 < d3, d1 < d3.
        if (!(mask & SortMask::kIncludeDense))
          tryLoosenAffineDenseConstraints(env.op(), fldx, fa, tldx, ta);

        // (d0 + d1) < (d2 + d3), or
        // filter_loop_d-1 < (d2 + d3), or
        // (d0 + d1) < filter_loop_d, or
        // filter_loop_d-1 < filter_loop_d depending on whether fa/ta is reset
        // above.
        addAffineOrderings(adjM, inDegree, fa, ta, fldx, tldx);
      }
    }
    // Push unrelated loops into sparse iteration space, so these
    // will be skipped more often.
    if (mask & SortMask::kIncludeUndef) {
      unsigned tensor = t.getOperandNumber();
      for (unsigned i = 0; i < n; i++) {
        if (isCompressedDLT(env.dlt(tensor, i)) ||
            isSingletonDLT(env.dlt(tensor, i))) {
          for (unsigned j = 0; j < n; j++)
            if (isUndefDLT(env.dlt(tensor, j))) {
              adjM[i][j] = true;
              inDegree[j]++;
            }
        } else {
          assert(isDenseDLT(env.dlt(tensor, i)) ||
                 isUndefDLT(env.dlt(tensor, i)));
        }
      }
    }
  }
  // Topologically sort the iteration graph to determine loop order.
  // Report failure for a cyclic iteration graph.
  env.topSortClear(n);
  return topSortOptimal(env, n, iteratorTypes, inDegree, adjM);
}

//===----------------------------------------------------------------------===//
// Sparse compiler synthesis methods (statements and expressions).
//===----------------------------------------------------------------------===//

/// Local bufferization of all dense and sparse data structures.
static void genBuffers(CodegenEnv &env, OpBuilder &builder) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  assert(op.getNumOperands() == op.getNumDpsInputs() + 1);

  env.emitter().initializeLoopEmit(
      builder, loc,
      /// Generates buffer for the output tensor.
      /// Note that all sparse kernels assume that when all elements are written
      /// to (viz. x(i) = y(i) * z(i)), the output buffer is already initialized
      /// to all zeroes and only nonzeroes values are computed and written out.
      /// For updates (viz. x(i) += y(i) * z(i)), only nonzeroes values are used
      /// for the updates and no assumption on the original contents of the
      /// output buffer is necessary.
      [&op](OpBuilder &builder, Location loc, Value memref,
            Value tensor) -> Value {
        // Must not be a sparse tensor.
        assert(!getSparseTensorEncoding(tensor.getType()));
        // Two output tensor references should point to the same object.
        OpOperand *lhs = op.getDpsInitOperand(0);
        assert(lhs->get() == tensor);
        // An output tensor can simply materialize from the buffer of the tensor
        // that appears in the outs() clause. For updates, this has the
        // advantage that only the nonzero value are involved in the
        // computation, keeping the operation O(nnz). In all other cases, we are
        // forced to zero out the buffer to enforce the assumption above, which
        // may negatively impact running complexity (viz. O(n^2 + nnz) vs.
        // O(nnz) for matrices).
        // TODO: use better analysis to avoid zeroing out the buffer?
        bool isInit = op.isInitTensor(lhs);
        Value init = memref;
        if (!isInit) {
          Value zero = constantZero(builder, loc,
                                    getElementTypeOrSelf(tensor.getType()));
          builder.create<linalg::FillOp>(loc, ValueRange{zero},
                                         ValueRange{init});
        }
        return init;
      });
}

/// Generates index for load/store on sparse tensor.
static Value genIndex(CodegenEnv &env, OpOperand *t) {
  auto map = env.op().getMatchingIndexingMap(t);
  const auto stt = getSparseTensorType(t->get());
  const Level lvlRank = stt.getLvlRank();
  assert(static_cast<Level>(map.getNumResults()) == lvlRank);
  // FIXME: `toOrigDim` is deprecated.
  AffineExpr a = map.getResult(toOrigDim(stt.getEncoding(), lvlRank - 1));
  assert(a.getKind() == AffineExprKind::DimId);
  unsigned idx = a.cast<AffineDimExpr>().getPosition();
  return env.getLoopIdxValue(idx);
}

/// Generates subscript for load/store on a dense or sparse tensor.
static Value genSubscript(CodegenEnv &env, OpBuilder &builder, OpOperand *t,
                          SmallVectorImpl<Value> &args) {
  linalg::GenericOp op = env.op();
  unsigned tensor = t->getOperandNumber();
  auto map = op.getMatchingIndexingMap(t);
  const auto stt = getSparseTensorType(t->get());
  if (stt.hasEncoding()) {
    Value pidx = env.emitter().getPidxs()[tensor].back();
    assert(pidx);
    args.push_back(pidx); // position index
  } else {
    const Level lvlRank = stt.getLvlRank();
    assert(static_cast<Level>(map.getNumResults()) == lvlRank);
    for (Level l = 0; l < lvlRank; l++) {
      AffineExpr a = map.getResult(l);
      args.push_back(env.emitter().genAffine(builder, a, op.getLoc()));
    }
  }
  return env.emitter().getValBuffer()[tensor];
}

/// Generates insertion code to implement dynamic tensor load.
static Value genInsertionLoad(CodegenEnv &env, OpBuilder &builder,
                              OpOperand *t) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  // Direct lexicographic index order, tensor loads as zero.
  if (!env.isExpand()) {
    Type tp = getElementTypeOrSelf(t->get().getType());
    return constantZero(builder, loc, tp);
  }
  // Load from expanded access pattern.
  Value index = genIndex(env, t);
  return builder.create<memref::LoadOp>(loc, env.getExpandValues(), index);
}

/// Generates insertion code to implement dynamic tensor load for reduction.
static Value genInsertionLoadReduce(CodegenEnv &env, OpBuilder &builder,
                                    OpOperand *t) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  Value identity = env.getCustomRedId();
  // Direct lexicographic index order, tensor loads as identity.
  if (!env.isExpand())
    return identity;
  // Load from expanded access pattern if filled, identity otherwise.
  Value values = env.getExpandValues();
  Value filled = env.getExpandFilled();
  Value index = genIndex(env, t);
  Value isFilled = builder.create<memref::LoadOp>(loc, filled, index);
  Value valAtIndex = builder.create<memref::LoadOp>(loc, values, index);
  return builder.create<arith::SelectOp>(loc, isFilled, valAtIndex, identity);
}

/// Generates insertion code to implement dynamic tensor store.
static void genInsertionStore(CodegenEnv &env, OpBuilder &builder, OpOperand *t,
                              Value rhs) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  // Direct insertion in lexicographic index order.
  if (!env.isExpand()) {
    unsigned rank = op.getRank(t);
    SmallVector<Value> indices;
    for (unsigned i = 0; i < rank; i++) {
      assert(env.emitter().getLoopIV(i));
      indices.push_back(env.emitter().getLoopIV(i));
    }
    Value chain = env.getInsertionChain();
    if (!env.getValidLexInsert()) {
      env.updateInsertionChain(
          builder.create<InsertOp>(loc, rhs, chain, indices));
    } else {
      // Generates runtime check for a valid lex during reduction,
      // to avoid inserting the identity value for empty reductions.
      //   if (validLexInsert) then
      //     insert(rhs) into chain
      //     return updated chain
      //   else
      //     return unmodified chain
      scf::IfOp ifValidLexInsert = builder.create<scf::IfOp>(
          loc, chain.getType(), env.getValidLexInsert(),
          /*else=*/true);
      // True branch.
      builder.setInsertionPointToStart(ifValidLexInsert.thenBlock());
      Value res = builder.create<InsertOp>(loc, rhs, chain, indices);
      builder.create<scf::YieldOp>(loc, res);
      // False branch.
      builder.setInsertionPointToStart(ifValidLexInsert.elseBlock());
      builder.create<scf::YieldOp>(loc, chain);
      // Value assignment.
      builder.setInsertionPointAfter(ifValidLexInsert);
      env.updateInsertionChain(ifValidLexInsert.getResult(0));
    }
    return;
  }
  // Generates insertion code along expanded access pattern.
  //   if (!expFilled[i]) then
  //     expFilled[i] = true
  //     expAdded[inserts++] = i
  //   endif
  //   values[i] = rhs
  Value values = env.getExpandValues();
  Value filled = env.getExpandFilled();
  Value added = env.getExpandAdded();
  Value count = env.getExpandCount();
  Value index = genIndex(env, t);
  Value fval = constantI1(builder, loc, false);
  Value tval = constantI1(builder, loc, true);
  // If statement.
  Value isFilled = builder.create<memref::LoadOp>(loc, filled, index);
  Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             isFilled, fval);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, builder.getIndexType(), cond,
                                             /*else=*/true);
  // True branch.
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  builder.create<memref::StoreOp>(loc, tval, filled, index);
  builder.create<memref::StoreOp>(loc, index, added, count);
  Value one = constantIndex(builder, loc, 1);
  Value add = builder.create<arith::AddIOp>(loc, count, one);
  builder.create<scf::YieldOp>(loc, add);
  // False branch.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, count);
  builder.setInsertionPointAfter(ifOp);
  // Value assignment.
  env.updateExpandCount(ifOp.getResult(0));
  builder.create<memref::StoreOp>(loc, rhs, values, index);
}

/// Generates a load on a dense or sparse tensor.
static Value genTensorLoad(CodegenEnv &env, OpBuilder &builder, unsigned exp) {
  // Test if the load was hoisted to a higher loop nest.
  Value val = env.exp(exp).val;
  if (val)
    return val;

  // Load during insertion.
  linalg::GenericOp op = env.op();
  OpOperand *t = &op->getOpOperand(env.exp(exp).tensor);
  if (env.isSparseOutput(t)) {
    if (env.isCustomReduc())
      return genInsertionLoadReduce(env, builder, t);
    return genInsertionLoad(env, builder, t);
  }
  // Actual load.
  SmallVector<Value> args;
  Value ptr = genSubscript(env, builder, t, args);
  return builder.create<memref::LoadOp>(op.getLoc(), ptr, args);
}

/// Generates a store on a dense or sparse tensor.
static void genTensorStore(CodegenEnv &env, OpBuilder &builder, unsigned exp,
                           Value rhs) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  // Test if this is a scalarized reduction.
  if (env.isReduc()) {
    env.updateReduc(rhs);
    return;
  }
  // Store during insertion.
  OpOperand *t = op.getDpsInitOperand(0);
  if (env.isSparseOutput(t)) {
    if (!rhs) {
      // Only unary and binary are allowed to return uninitialized rhs
      // to indicate missing output.
      assert(env.exp(exp).kind == kUnary || env.exp(exp).kind == kBinary);
    } else if (env.exp(exp).kind == kSelect) {
      // Select operation insertion.
      Value chain = env.getInsertionChain();
      scf::IfOp ifOp =
          builder.create<scf::IfOp>(loc, chain.getType(), rhs, /*else=*/true);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      // Existing value was preserved to be used here.
      assert(env.exp(exp).val);
      Value v0 = env.exp(exp).val;
      genInsertionStore(env, builder, t, v0);
      env.exp(exp).val = Value();
      // Yield modified insertion chain along true branch.
      Value mchain = env.getInsertionChain();
      builder.create<scf::YieldOp>(op.getLoc(), mchain);
      // Yield original insertion chain along false branch.
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      builder.create<scf::YieldOp>(loc, chain);
      // Done with if statement.
      env.updateInsertionChain(ifOp->getResult(0));
      builder.setInsertionPointAfter(ifOp);
    } else {
      genInsertionStore(env, builder, t, rhs);
    }
    return;
  }
  // Actual store.
  SmallVector<Value> args;
  Value ptr = genSubscript(env, builder, t, args);
  builder.create<memref::StoreOp>(loc, rhs, ptr, args);
}

/// Generates an invariant value.
inline static Value genInvariantValue(CodegenEnv &env, unsigned exp) {
  return env.exp(exp).val;
}

/// Generates an index value.
inline static Value genIndexValue(CodegenEnv &env, unsigned idx) {
  return env.getLoopIdxValue(idx);
}

/// Semi-ring branches are simply inlined by the sparse compiler. Prior
/// analysis has verified that all computations are "local" to the inlined
/// branch or otherwise invariantly defined outside the loop nest, with the
/// exception of index computations, which need to be relinked to actual
/// inlined cloned code.
static Value relinkBranch(CodegenEnv &env, RewriterBase &rewriter, Block *block,
                          Value e, unsigned ldx) {
  if (Operation *def = e.getDefiningOp()) {
    if (auto indexOp = dyn_cast<linalg::IndexOp>(def))
      return genIndexValue(env, indexOp.getDim());
    if (def->getBlock() == block) {
      for (unsigned i = 0, n = def->getNumOperands(); i < n; i++)
        def->setOperand(
            i, relinkBranch(env, rewriter, block, def->getOperand(i), ldx));
    }
  }
  return e;
}

/// Recursively generates tensor expression.
static Value genExp(CodegenEnv &env, RewriterBase &rewriter, unsigned exp,
                    unsigned ldx) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();

  if (exp == -1u)
    return Value();
  if (env.exp(exp).kind == Kind::kTensor)
    return genTensorLoad(env, rewriter, exp);
  if (env.exp(exp).kind == Kind::kInvariant)
    return genInvariantValue(env, exp);
  if (env.exp(exp).kind == Kind::kIndex)
    return genIndexValue(env, env.exp(exp).index);

  if (env.exp(exp).kind == Kind::kReduce)
    env.startCustomReduc(exp); // enter custom

  Value v0 = genExp(env, rewriter, env.exp(exp).children.e0, ldx);
  Value v1 = genExp(env, rewriter, env.exp(exp).children.e1, ldx);
  Value ee = env.merger().buildExp(rewriter, loc, exp, v0, v1);
  if (ee && (env.exp(exp).kind == Kind::kUnary ||
             env.exp(exp).kind == Kind::kBinary ||
             env.exp(exp).kind == Kind::kBinaryBranch ||
             env.exp(exp).kind == Kind::kReduce ||
             env.exp(exp).kind == Kind::kSelect))
    ee = relinkBranch(env, rewriter, ee.getParentBlock(), ee, ldx);

  if (env.exp(exp).kind == Kind::kReduce)
    env.endCustomReduc(); // exit custom

  if (env.exp(exp).kind == kSelect) {
    assert(!env.exp(exp).val);
    env.exp(exp).val = v0; // Preserve value for later use.
  }

  return ee;
}

/// Hoists loop invariant tensor loads for which indices have been exhausted.
static void genInvariants(CodegenEnv &env, OpBuilder &builder, unsigned exp,
                          unsigned ldx, bool atStart) {
  if (exp == -1u)
    return;
  if (env.exp(exp).kind == Kind::kTensor) {
    // Inspect tensor indices.
    bool atLevel = ldx == -1u;
    linalg::GenericOp op = env.op();
    OpOperand &t = op->getOpOperand(env.exp(exp).tensor);
    auto map = op.getMatchingIndexingMap(&t);
    const auto stt = getSparseTensorType(t.get());
    const Level lvlRank = stt.getLvlRank();
    assert(static_cast<Level>(map.getNumResults()) == lvlRank);
    for (Level l = 0; l < lvlRank; l++) {
      // FIXME: `toOrigDim` is deprecated.
      AffineExpr a = map.getResult(toOrigDim(stt.getEncoding(), l));
      std::optional<unsigned> sldx =
          env.merger().getLoopIdx(t.getOperandNumber(), l);
      if (sldx && env.merger().isFilterLoop(*sldx)) {
        if (!env.getLoopIdxValue(*sldx))
          // The filter loops has not been constructed.
          return;
        if (*sldx == ldx)
          atLevel = true;
      } else if (!isInvariantAffine(env, a, ldx, atLevel))
        return; // still in play
    }
    // All exhausted at this level (atLevel denotes exactly at this level).
    if (!atLevel)
      return;
    OpOperand *lhs = op.getDpsInitOperand(0);
    if (lhs == &t) {
      // Start or end a scalarized reduction.
      if (atStart) {
        Value load = env.isCustomReduc() ? env.getCustomRedId()
                                         : genTensorLoad(env, builder, exp);
        env.startReduc(exp, load);
        if (env.hasSparseOutput())
          env.setValidLexInsert(constantI1(builder, env.op().getLoc(), false));
      } else {
        genTensorStore(env, builder, exp, env.endReduc());
        env.clearValidLexInsert();
      }
    } else {
      // Start or end loop invariant hoisting of a tensor load.
      env.exp(exp).val = atStart ? genTensorLoad(env, builder, exp) : Value();
    }
  } else if (env.exp(exp).kind != Kind::kInvariant &&
             env.exp(exp).kind != Kind::kIndex) {
    // Traverse into the binary operations. Note that we only hoist
    // tensor loads, since subsequent MLIR/LLVM passes know how to
    // deal with all other kinds of derived loop invariants.
    if (env.exp(exp).kind == Kind::kReduce)
      env.startCustomReduc(exp); // enter custom
    unsigned e0 = env.exp(exp).children.e0;
    unsigned e1 = env.exp(exp).children.e1;
    genInvariants(env, builder, e0, ldx, atStart);
    genInvariants(env, builder, e1, ldx, atStart);
    if (env.exp(exp).kind == Kind::kReduce)
      env.endCustomReduc(); // exit custom
  }
}

/// Generates an expanded access pattern in innermost dimension.
static void genExpand(CodegenEnv &env, OpBuilder &builder, unsigned at,
                      bool atStart) {
  linalg::GenericOp op = env.op();
  OpOperand *lhs = op.getDpsInitOperand(0);
  if (!env.atExpandLevel(lhs, op.getRank(lhs), at))
    return; // not needed at this level
  assert(!env.isReduc());
  // Generate start or end of an expanded access pattern. Note that because
  // an expension does not rely on the ongoing contents of the sparse storage
  // scheme, we can use the original tensor as incoming SSA value (which
  // simplifies codegen a bit). If expansion on the actual contents is ever
  // needed, we will need to use the SSA value in the insertion chain instead.
  Value tensor = lhs->get();
  Location loc = op.getLoc();
  if (atStart) {
    auto dynShape = {ShapedType::kDynamic};
    Type etp = tensor.getType().cast<ShapedType>().getElementType();
    Type t1 = MemRefType::get(dynShape, etp);
    Type t2 = MemRefType::get(dynShape, builder.getI1Type());
    Type t3 = MemRefType::get(dynShape, builder.getIndexType());
    Type t4 = builder.getIndexType();
    auto r = builder.create<ExpandOp>(loc, TypeRange({t1, t2, t3, t4}), tensor);
    assert(r.getNumResults() == 4);
    env.startExpand(r.getResult(0), r.getResult(1), r.getResult(2),
                    r.getResult(3));
  } else {
    SmallVector<Value> indices;
    for (unsigned i = 0; i < at; i++)
      indices.push_back(env.emitter().getLoopIV(i));
    Value values = env.getExpandValues();
    Value filled = env.getExpandFilled();
    Value added = env.getExpandAdded();
    Value count = env.getExpandCount();
    Value chain = env.getInsertionChain();
    Value compress = builder.create<CompressOp>(loc, values, filled, added,
                                                count, chain, indices);
    env.updateInsertionChain(compress);
    env.endExpand();
  }
}

/// Returns parallelization strategy. Any implicit loop in the Linalg
/// operation that is marked "parallel" is a candidate. Whether it is actually
/// converted to a parallel operation depends on the requested strategy.
static bool isParallelFor(CodegenEnv &env, bool isOuter, bool isSparse) {
  // Reject parallelization of sparse output.
  if (env.hasSparseOutput())
    return false;
  // Parallel loops on tensor expansion can cause data races.
  if (env.isExpand())
    return false;
  // Inspect strategy.
  switch (env.options().parallelizationStrategy) {
  case SparseParallelizationStrategy::kNone:
    return false;
  case SparseParallelizationStrategy::kDenseOuterLoop:
    return isOuter && !isSparse;
  case SparseParallelizationStrategy::kAnyStorageOuterLoop:
    return isOuter;
  case SparseParallelizationStrategy::kDenseAnyLoop:
    return !isSparse;
  case SparseParallelizationStrategy::kAnyStorageAnyLoop:
    return true;
  }
  llvm_unreachable("unexpected parallelization strategy");
}

/// Generates a for-loop on a single index.
static Operation *genFor(CodegenEnv &env, OpBuilder &builder, bool isOuter,
                         bool isInner, unsigned idx, ArrayRef<size_t> tids,
                         ArrayRef<size_t> dims) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  auto iteratorTypes = op.getIteratorTypesArray();
  bool isSparse = llvm::any_of(tids, [idx, &env](size_t tid) {
    return isCompressedDLT(env.dlt(tid, idx)) ||
           isSingletonDLT(env.dlt(tid, idx));
  });

  bool isParallel = isParallelFor(env, isOuter, isSparse);

  Operation *loop = *env.genLoopBoundary([&](MutableArrayRef<Value> reduc) {
    if (env.merger().isFilterLoop(idx)) {
      size_t tid = tids.front(), dim = dims.front();
      // tids/dims must only have one value because filter loops only
      // corresponding to the one and only sparse tensor level.
      assert(isSparse && tids.size() == 1 && dims.size() == 1);
      OpOperand *t = &op->getOpOperand(tid);
      auto enc = getSparseTensorEncoding(t->get().getType());
      // Retrieves the affine expression for the filter loop.
      // FIXME: `toOrigDim` is deprecated.
      AffineExpr a =
          op.getMatchingIndexingMap(t).getResult(toOrigDim(enc, dim));
      return env.emitter().enterFilterLoopOverTensorAtDim(builder, loc, tid,
                                                          dim, a, reduc);
    }
    return env.emitter().enterLoopOverTensorAtDim(builder, loc, tids, dims,
                                                  reduc, isParallel);
  });
  assert(loop);
  return loop;
}

/// Emit a while-loop for co-iteration over multiple indices.
static Operation *genWhile(CodegenEnv &env, OpBuilder &builder, unsigned idx,
                           bool needsUniv, ArrayRef<size_t> tids,
                           ArrayRef<size_t> dims) {
  Operation *loop = *env.genLoopBoundary([&](MutableArrayRef<Value> reduc) {
    // Construct the while-loop with a parameter for each
    // index.
    return env.emitter().enterCoIterationOverTensorsAtDims(
        builder, env.op().getLoc(), tids, dims, needsUniv, reduc);
  });
  assert(loop);
  return loop;
}

/// Generates a for-loop or a while-loop, depending on whether it implements
/// singleton iteration or co-iteration over the given conjunction.
static Operation *genLoop(CodegenEnv &env, OpBuilder &builder, unsigned at,
                          bool needsUniv, ArrayRef<size_t> tids,
                          ArrayRef<size_t> dims, bool isFor) {
  assert(tids.size() == dims.size());
  unsigned idx = env.topSortAt(at);
  if (isFor) {
    bool isOuter = at == 0;
    bool isInner = at == env.topSortSize() - 1;
    return genFor(env, builder, isOuter, isInner, idx, tids, dims);
  }
  return genWhile(env, builder, idx, needsUniv, tids, dims);
}

/// Generates the induction structure for a while-loop.
static void finalizeWhileOp(CodegenEnv &env, OpBuilder &builder, unsigned idx,
                            bool needsUniv, BitVector &induction,
                            scf::WhileOp whileOp) {
  Location loc = env.op().getLoc();
  // Finalize each else branch of all if statements.
  if (env.isReduc() || env.isExpand() || env.getInsertionChain()) {
    while (auto ifOp = dyn_cast_or_null<scf::IfOp>(
               builder.getInsertionBlock()->getParentOp())) {
      unsigned y = 0;
      SmallVector<Value> yields;
      if (env.isReduc()) {
        yields.push_back(env.getReduc());
        env.updateReduc(ifOp.getResult(y++));
        if (env.getValidLexInsert()) {
          yields.push_back(env.getValidLexInsert());
          env.setValidLexInsert(ifOp.getResult(y++));
        }
      }
      if (env.isExpand()) {
        yields.push_back(env.getExpandCount());
        env.updateExpandCount(ifOp->getResult(y++));
      }
      if (env.getInsertionChain()) {
        yields.push_back(env.getInsertionChain());
        env.updateInsertionChain(ifOp->getResult(y++));
      }
      assert(y == yields.size());
      builder.create<scf::YieldOp>(loc, yields);
      builder.setInsertionPointAfter(ifOp);
    }
  }
  builder.setInsertionPointToEnd(&whileOp.getAfter().front());
}

/// Generates a single if-statement within a while-loop.
static scf::IfOp genIf(CodegenEnv &env, OpBuilder &builder, unsigned idx,
                       BitVector &conditions) {
  Location loc = env.op().getLoc();
  SmallVector<Type> types;
  Value cond;
  for (unsigned b = 0, be = conditions.size(); b < be; b++) {
    if (!conditions[b])
      continue;
    unsigned tensor = env.merger().tensor(b);
    assert(idx == env.merger().index(b));
    Value clause;
    if (isCompressedDLT(env.dlt(b)) || isSingletonDLT(env.dlt(b))) {
      auto dim = *env.merger().getDimNum(tensor, idx);
      Value op1 = env.emitter().getCoord()[tensor][dim];
      Value op2 = env.getLoopIdxValue(idx);
      clause = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, op1,
                                             op2);
    } else {
      assert(isDenseDLT(env.merger().getDimLevelType(b)) ||
             isUndefDLT(env.merger().getDimLevelType(b)));
      clause = constantI1(builder, loc, true);
    }
    cond = cond ? builder.create<arith::AndIOp>(loc, cond, clause) : clause;
  }
  if (env.isReduc()) {
    types.push_back(env.getReduc().getType());
    if (env.getValidLexInsert())
      types.push_back(env.getValidLexInsert().getType());
  }
  if (env.isExpand())
    types.push_back(builder.getIndexType());
  if (env.getInsertionChain())
    types.push_back(env.getInsertionChain().getType());
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, types, cond, /*else=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return ifOp;
}

/// Generates end of true branch of if-statement within a while-loop.
static void endIf(CodegenEnv &env, OpBuilder &builder, scf::IfOp ifOp,
                  Operation *loop, Value redInput, Value cntInput,
                  Value insInput) {
  SmallVector<Value> operands;
  if (env.isReduc()) {
    operands.push_back(env.getReduc());
    env.updateReduc(redInput);
    if (env.getValidLexInsert())
      // Any overlapping indices during a reduction creates a valid lex insert.
      operands.push_back(constantI1(builder, env.op().getLoc(), true));
  }
  if (env.isExpand()) {
    operands.push_back(env.getExpandCount());
    env.updateExpandCount(cntInput);
  }
  if (env.getInsertionChain()) {
    operands.push_back(env.getInsertionChain());
    env.updateInsertionChain(insInput);
  }
  if (!operands.empty())
    builder.create<scf::YieldOp>(env.op().getLoc(), operands);
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
}

//===----------------------------------------------------------------------===//
// Sparse compiler synthesis methods (loop sequence).
//===----------------------------------------------------------------------===//

/// Starts a loop sequence at given level. Returns true if
/// the universal loop index must be maintained at this level.
static bool startLoopSeq(CodegenEnv &env, OpBuilder &builder, unsigned exp,
                         unsigned at, unsigned idx, unsigned ldx,
                         unsigned lts) {
  assert(!env.getLoopIdxValue(idx));
  // Emit invariants at this loop sequence level.
  genInvariants(env, builder, exp, ldx, /*atStart=*/true);
  // Emit access pattern expansion for sparse tensor output.
  genExpand(env, builder, at, /*atStart=*/true);
  // Emit further intitialization at this loop sequence level.
  unsigned l0 = env.set(lts)[0];
  bool needsUniv = false;

  SmallVector<size_t> tids;
  SmallVector<size_t> dims;
  env.merger().foreachTidDimPairInBits(
      env.lat(l0).bits, [&](unsigned b, unsigned tid,
                            std::optional<unsigned> dim, DimLevelType dlt) {
        assert(env.merger().index(b) == idx);
        if (isDenseDLT(dlt) || isUndefDLT(dlt)) {
          needsUniv = true;
        } else {
          // sparse/singleton dim levels.
          tids.push_back(tid);
          dims.push_back(*dim);
        }
      });

  env.emitter().enterNewLoopSeq(builder, env.op().getLoc(), tids, dims);

  // Maintain the universal index only if it is actually
  // consumed by a subsequent lattice point.
  if (needsUniv) {
    unsigned lsize = env.set(lts).size();
    for (unsigned i = 1; i < lsize; i++) {
      unsigned li = env.set(lts)[i];
      if (!env.merger().hasAnySparse(env.lat(li).simple))
        return true;
    }
  }
  return false;
}

static void genConstantDenseAddressFromLevel(CodegenEnv &env,
                                             OpBuilder &builder, unsigned tid,
                                             Level lvl) {
  // TODO: Handle affine expression on output tensor.
  linalg::GenericOp op = env.op();
  assert(tid < op.getNumDpsInputs());
  OpOperand *input = op.getDpsInputOperands()[tid];
  ArrayRef<AffineExpr> affines = op.getMatchingIndexingMap(input).getResults();
  const auto enc = getSparseTensorEncoding(input->get().getType());
  if (enc) {
    const Level lvlRank = enc.getLvlRank();
    assert(affines.size() == static_cast<size_t>(lvlRank));
    for (Level l = lvl; l < lvlRank; l++) {
      // FIXME: `toOrigDim` is deprecated.
      AffineExpr affine = affines[toOrigDim(enc, l)];
      if (enc.isDenseLvl(l) && affine.isa<AffineConstantExpr>())
        env.emitter().genDenseAffineAddressAtCurLevel(
            builder, op.getLoc(), input->getOperandNumber(), l, affine);
      else
        return; // break on first non-dense non-constant level
    }
  }
}

static void genInitConstantDenseAddress(CodegenEnv &env,
                                        RewriterBase &rewriter) {
  // We can generate address for constant affine expression before any loops
  // starting from the first level as they do not depend on any thing.
  // E.g., [Dense, Dense, Sparse] -> (1, 2, d0), the addresses for the first two
  // levels can be determined before loops.
  for (unsigned tid = 0, e = env.op().getNumDpsInputs(); tid < e; tid++)
    genConstantDenseAddressFromLevel(env, rewriter, tid, 0);
}

/// Return true if the lattices bit can be iterated by a for loop.
static bool translateBitsToTidDimPairs(
    CodegenEnv &env, unsigned li, unsigned idx, SmallVectorImpl<size_t> &tids,
    SmallVectorImpl<size_t> &dims, SmallVectorImpl<size_t> &affineTids,
    SmallVectorImpl<size_t> &affineDims, SmallVectorImpl<AffineExpr> &exps) {
  const BitVector &all = env.lat(li).bits;
  const BitVector &simple = env.lat(li).simple;

  unsigned numloopCond = 0;
  // Converts bits to array + dim pair
  env.merger().foreachTidDimPairInBits(
      all, [&, idx](unsigned b, unsigned tid, std::optional<unsigned> dim,
                    DimLevelType dlt) {
        if (simple.test(b)) {
          if (isUndefDLT(dlt)) {
            // An undefined dlt in the lattices, we probably mean to iterate
            // based on the dim of output tensor.
            // E.g., this could be a synthetic tensor (for invariants and sparse
            // output tensor).
            // out[i][j] = invariant; or a broadcast
            // out[i][j] = in[i] (j is undef for input)
            tid = env.merger().getOutTensorID();
            dim = env.merger().getDimNum(tid, idx);
            // Skips invalid dim (e.g., when this is a zero ranked tensor).
            if (!dim)
              return;
          }
          tids.push_back(tid);
          dims.push_back(*dim);
          numloopCond++;
        } else if (isDenseDLT(dlt)) {
          tids.push_back(tid);
          dims.push_back(*dim);
        } else {
          assert(isUndefDLT(dlt));
          linalg::GenericOp op = env.op();
          if (tid >= op.getNumDpsInputs())
            // We only handle affine expression on input tensors (for now).
            return;
          OpOperand *operand = &op->getOpOperand(tid);
          const auto stt = getSparseTensorType(operand->get());
          // Non-annotated dense tensors requires no special handling.
          if (!stt.hasEncoding())
            return;

          ArrayRef<AffineExpr> affines =
              op.getMatchingIndexingMap(operand).getResults();
          const Level lvlRank = stt.getLvlRank();
          assert(affines.size() == static_cast<size_t>(lvlRank));
          for (Level l = 0; l < lvlRank; l++) {
            // FIXME: `toOrigDim` is deprecated.
            AffineExpr exp = affines[toOrigDim(stt.getEncoding(), l)];
            // Skip simple affine expression and non dense dimensions (which has
            // it own filter loop).
            if (exp.isa<AffineDimExpr>() || !stt.isDenseLvl(l))
              continue;

            // Constant affine expression are handled in genLoop
            if (!exp.isa<AffineConstantExpr>()) {
              bool atLevel = false;
              if (isInvariantAffine(env, exp, idx, atLevel) && atLevel) {
                // If the compound affine is invariant and we are right at the
                // level. We need to generate the address according to the
                // affine expression. This is also the best place we can do it
                // to avoid putting it inside inner loops.
                // NOTE: It assumes that the levels of the input tensor are
                // initialized in order (and it is also currently guaranteed by
                // computeIterationGraph), another more admissible approach
                // might be accepting out-of-order access between consecutive
                // dense levels.
                affineTids.push_back(tid);
                affineDims.push_back(l);
                exps.push_back(exp);
              }
            }
          }
        }
      });

  if (isDenseDLT(env.dlt(env.merger().getOutTensorID(), idx))) {
    // Note that we generate dense indices of the output tensor
    // unconditionally, since they may not appear in the lattice, but may be
    // needed for linearized env.
    auto dim = *env.merger().getDimNum(env.merger().getOutTensorID(), idx);
    tids.push_back(env.merger().getOutTensorID());
    dims.push_back(dim);
  }

  assert(numloopCond > 0);
  // If we just need to one loop conditions, the loop can be generated by a for
  // loop.
  return numloopCond == 1;
}

/// Starts a single loop in current sequence.
static Operation *startLoop(CodegenEnv &env, OpBuilder &builder, unsigned at,
                            unsigned li, bool needsUniv) {
  // The set of tensors + dims to generate loops on
  SmallVector<size_t> tids, dims;
  // The set of dense tensors with non-trivial affine expression that just
  // becomes invariant and the address shall now be generated at the current
  // level.
  SmallVector<size_t> affineTids, affineDims;
  SmallVector<AffineExpr> affines;
  bool isFor = translateBitsToTidDimPairs(
      env, li, env.topSortAt(at), tids, dims, affineTids, affineDims, affines);

  // Emit the for/while-loop control.
  Operation *loop = genLoop(env, builder, at, needsUniv, tids, dims, isFor);
  for (auto [tid, dim, exp] : llvm::zip(affineTids, affineDims, affines)) {
    env.emitter().genDenseAffineAddressAtCurLevel(builder, env.op().getLoc(),
                                                  tid, dim, exp);
  }

  // Until now, we have entered every <tid, dim> pair in {cond, extra,
  // affine}Tids/Dims. The addresses of the upcoming levels which are dependent
  // on constant affines expression may now be determined.
  auto allTids = llvm::concat<size_t>(tids, affineTids);
  auto allDims = llvm::concat<size_t>(dims, affineDims);
  for (auto [tid, dim] : llvm::zip(allTids, allDims)) {
    if (tid != env.merger().getOutTensorID())
      genConstantDenseAddressFromLevel(env, builder, tid, dim + 1);
  }

  return loop;
}

/// Ends a single loop in current sequence. Returns new values for needsUniv.
static bool endLoop(CodegenEnv &env, RewriterBase &rewriter, Operation *loop,
                    unsigned idx, unsigned li, bool needsUniv) {
  // End a while-loop.
  if (auto whileOp = dyn_cast<scf::WhileOp>(loop)) {
    finalizeWhileOp(env, rewriter, idx, needsUniv, env.lat(li).bits, whileOp);
  } else if (auto forOp = dyn_cast<scf::ForOp>(loop)) {
    // Any iteration of a reduction for-loop creates a valid lex insert.
    if (env.isReduc() && env.getValidLexInsert())
      env.setValidLexInsert(constantI1(rewriter, env.op().getLoc(), true));
  } else {
    needsUniv = false;
  }

  env.genLoopBoundary([&](MutableArrayRef<Value> reduc) {
    env.emitter().exitCurrentLoop(rewriter, env.op().getLoc(), reduc);
    return std::nullopt;
  });

  return needsUniv;
}

/// Ends a loop sequence at given level.
static void endLoopSeq(CodegenEnv &env, OpBuilder &builder, unsigned exp,
                       unsigned at, unsigned idx, unsigned ldx) {
  assert(env.getLoopIdxValue(idx) == nullptr);
  env.emitter().exitCurrentLoopSeq();
  // Unmark bookkeeping of invariants and loop index.
  genInvariants(env, builder, exp, ldx, /*atStart=*/false);
  // Finalize access pattern expansion for sparse tensor output.
  genExpand(env, builder, at, /*atStart=*/false);
}

/// Recursively generates code while computing iteration lattices in order
/// to manage the complexity of implementing co-iteration over unions
/// and intersections of sparse iterations spaces.
static void genStmt(CodegenEnv &env, RewriterBase &rewriter, unsigned exp,
                    unsigned at) {
  // At each leaf, assign remaining tensor (sub)expression to output tensor.
  if (at == env.topSortSize()) {
    unsigned ldx = env.topSortAt(at - 1);
    Value rhs = genExp(env, rewriter, exp, ldx);
    genTensorStore(env, rewriter, exp, rhs);
    return;
  }

  // Construct iteration lattices for current loop index, with L0 at top.
  unsigned idx = env.topSortAt(at);
  unsigned ldx = at == 0 ? -1u : env.topSortAt(at - 1);
  unsigned lts = env.merger().optimizeSet(env.merger().buildLattices(exp, idx));

  // TODO: sort
  // TODO: dedup

  // Start a loop sequence.
  bool needsUniv = startLoopSeq(env, rewriter, exp, at, idx, ldx, lts);

  // Emit a loop for every lattice point L0 >= Li in this loop sequence.
  unsigned lsize = env.set(lts).size();
  for (unsigned i = 0; i < lsize; i++) {
    // Start a loop.
    unsigned li = env.set(lts)[i];
    Operation *loop = startLoop(env, rewriter, at, li, needsUniv);

    // Visit all lattices points with Li >= Lj to generate the
    // loop-body, possibly with if statements for coiteration.
    Value redInput = env.getReduc();
    Value cntInput = env.getExpandCount();
    Value insInput = env.getInsertionChain();
    bool isWhile = dyn_cast<scf::WhileOp>(loop) != nullptr;
    for (unsigned j = 0; j < lsize; j++) {
      unsigned lj = env.set(lts)[j];
      unsigned ej = env.lat(lj).exp;
      if (li == lj || env.merger().latGT(li, lj)) {
        // Recurse into body of each branch.
        if (isWhile) {
          scf::IfOp ifOp = genIf(env, rewriter, idx, env.lat(lj).simple);
          genStmt(env, rewriter, ej, at + 1);
          endIf(env, rewriter, ifOp, loop, redInput, cntInput, insInput);
        } else {
          genStmt(env, rewriter, ej, at + 1);
        }
      }
    }

    // End a loop.
    needsUniv = endLoop(env, rewriter, loop, idx, li, needsUniv);
  }

  // End a loop sequence.
  endLoopSeq(env, rewriter, exp, at, idx, ldx);
}

/// Converts the result computed by the sparse kernel into the required form.
static void genResult(CodegenEnv &env, RewriterBase &rewriter) {
  linalg::GenericOp op = env.op();
  OpOperand *lhs = op.getDpsInitOperand(0);
  Value tensor = lhs->get();
  Type resType = tensor.getType();
  if (getSparseTensorEncoding(resType)) {
    // The sparse tensor rematerializes from the original sparse tensor's
    // underlying sparse storage format. For an insertion chain, the
    // tensor materializes from the chain with 'hasInserts' enabled.
    bool hasInserts = false;
    if (Value chain = env.getInsertionChain()) {
      hasInserts = true;
      tensor = chain;
    }
    rewriter.replaceOpWithNewOp<LoadOp>(op, resType, tensor, hasInserts);
  } else {
    // To rematerialize an non-annotated tensor, simply load it
    // from the bufferized value.
    Value val = env.emitter().getValBuffer().back(); // value array
    rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, resType, val);
  }
}

//===----------------------------------------------------------------------===//
// Sparse compiler rewriting methods.
//===----------------------------------------------------------------------===//

namespace {

/// Sparse rewriting rule for generic Lingalg operation.
struct GenericOpSparsifier : public OpRewritePattern<linalg::GenericOp> {
public:
  GenericOpSparsifier(MLIRContext *context, SparsificationOptions o)
      : OpRewritePattern<linalg::GenericOp>(context), options(o) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Only accept single output operations without affine index on sparse
    // output.
    if (op.getNumDpsInits() != 1 || hasCompoundAffineOnSparseOut(op))
      return failure();

    // Sets up a code generation environment.
    unsigned numTensors = op->getNumOperands();
    unsigned numLoops = op.getNumLoops();
    unsigned numFilterLoops = getNumCompoundAffineOnSparseDims(op);
    CodegenEnv env(op, options, numTensors, numLoops, numFilterLoops);

    // Detects sparse annotations and translates the per-dimension sparsity
    // information for all tensors to loop indices in the kernel.
    if (!findSparseAnnotations(env))
      return failure();

    // Constructs the tensor expressions tree from `op`, returns failure if the
    // tree can not be built or the tensor expression is inadmissible.
    if (failed(env.initTensorExp()))
      return failure();

    // Computes a topologically sorted iteration graph to ensure tensors
    // are visited in natural index order. Gradually relaxes the considered
    // constraints until an acyclic iteration graph results, such that sparse
    // code generation can proceed. As a last resort, an attempt is made
    // to resolve cycles by inserting a conversion.
    bool isAdmissible = false;
    bool hasCycle = true;

    // An const list of all masks that we used for interation graph
    // computation. Must be ordered from more strict to less strict.
    const auto allMask = {SortMask::kIncludeAll, SortMask::kIncludeUndef,
                          SortMask::kIncludeDense, SortMask::kSparseOnly};
    for (auto mask : allMask) {
      if (computeIterationGraph(env, mask)) {
        hasCycle = false;
        if (env.isAdmissibleTopoOrder()) {
          isAdmissible = true;
          break;
        }
        // else try a set of less strict constraints.
      }
    }
    if (hasCycle)
      return resolveCycle(env, rewriter); // one last shot
    if (!isAdmissible)
      return failure(); // inadmissible expression, reject

    // Recursively generates code if admissible.
    env.startEmit();
    genBuffers(env, rewriter);
    genInitConstantDenseAddress(env, rewriter);
    genStmt(env, rewriter, env.getTensorExp(), 0);
    genResult(env, rewriter);
    return success();
  }

private:
  // Last resort cycle resolution.
  LogicalResult resolveCycle(CodegenEnv &env, PatternRewriter &rewriter) const {
    // Compute topological sort while leaving out every
    // sparse input tensor in succession until an acylic
    // iteration graph results.
    for (OpOperand *t : env.op().getDpsInputOperands()) {
      unsigned tensor = t->getOperandNumber();
      Value tval = t->get();
      auto srcEnc = getSparseTensorEncoding(tval.getType());
      if (!srcEnc || !computeIterationGraph(env, SortMask::kSparseOnly, t))
        continue;
      // Found an input tensor that resolves the cycle by inserting a
      // conversion into a sparse tensor that adheres to the iteration
      // graph order. Also releases the temporary sparse tensor.
      //
      // TODO: investigate fusing the conversion with computation,
      //       especially if it is a direct yield!
      //
      auto srcTp = getRankedTensorType(tval);
      auto dstEnc = SparseTensorEncodingAttr::get(
          getContext(), srcEnc.getDimLevelType(),
          permute(env, env.op().getMatchingIndexingMap(t)), // new order
          srcEnc.getHigherOrdering(), srcEnc.getPointerBitWidth(),
          srcEnc.getIndexBitWidth());
      auto dstTp = RankedTensorType::get(srcTp.getShape(),
                                         srcTp.getElementType(), dstEnc);
      auto convert = rewriter.create<ConvertOp>(tval.getLoc(), dstTp, tval);
      env.op()->setOperand(tensor, convert);
      rewriter.setInsertionPointAfter(env.op());
      rewriter.create<bufferization::DeallocTensorOp>(tval.getLoc(), convert);
      return success();
    }
    // Cannot be resolved with a single conversion.
    // TODO: convert more than one?
    return failure();
  }

  /// Options to control sparse code generation.
  SparsificationOptions options;
};

} // namespace

/// Populates the given patterns list with rewriting rules required for
/// the sparsification of linear algebra operations.
void mlir::populateSparsificationPatterns(
    RewritePatternSet &patterns, const SparsificationOptions &options) {
  patterns.add<GenericOpSparsifier>(patterns.getContext(), options);
}
