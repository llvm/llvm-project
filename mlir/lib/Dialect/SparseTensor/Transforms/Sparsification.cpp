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

#include "CodegenUtils.h"

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
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TensorEncoding.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
// Declarations of data structures.
//===----------------------------------------------------------------------===//

namespace {

// Iteration graph sorting.
enum SortMask {
  kSparseOnly = 0x0,
  kIncludeDense = 0x1,
  kIncludeUndef = 0x2,
  kIncludeAll = 0x3
};

// Reduction kinds.
enum Reduction { kNoReduc, kSum, kProduct, kAnd, kOr, kXor, kCustom };

// Code generation.
struct CodeGen {
  CodeGen(SparsificationOptions o, MLIRContext *context, ValueRange tensors,
          unsigned numTensors, unsigned numLoops, OpOperand *op, unsigned nest,
          std::vector<unsigned> &ts)
      : options(o),
        loopEmitter(
            tensors,
            StringAttr::get(context, linalg::GenericOp::getOperationName()),
            /*hasOutput=*/true,
            /*isSparseOut=*/op != nullptr, ts),
        sparseOut(op), outerParNest(nest), topSort(ts) {
    if (op)
      insChain = op->get();
  }
  /// Sparsification options.
  SparsificationOptions options;
  /// Loop emitter helper class.
  SparseTensorLoopEmitter loopEmitter;
  /// Current reduction, updated during code generation. When indices of a
  /// reduction are exhausted, all inner loops can use a scalarized reduction.
  unsigned redExp = -1u;
  Value redVal;
  Reduction redKind = kNoReduc;
  unsigned redCustom = -1u;
  // Sparse tensor as output. Implemented either through direct injective
  // insertion in lexicographic index order or through access pattern expansion
  // in the innermost loop nest (`expValues` through `expCount`).
  OpOperand *sparseOut;
  unsigned outerParNest;
  Value insChain; // bookkeeping for insertion chain
  Value expValues;
  Value expFilled;
  Value expAdded;
  Value expCount;
  // Topsort (reference should remain in scope).
  std::vector<unsigned> &topSort;

  ArrayRef<unsigned> getLoopCurStack() const {
    ArrayRef<unsigned> topSortRef = topSort;
    return topSortRef.slice(0, loopEmitter.getCurrentDepth());
  }

  Value getLoopIdxValue(size_t loopIdx) const {
    for (unsigned lv = 0; lv < topSort.size(); lv++)
      if (topSort[lv] == loopIdx)
        return loopEmitter.getLoopIV(lv);

    llvm_unreachable("invalid loop index");
  }
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
static bool isInvariantAffine(const CodeGen &codegen, AffineExpr a,
                              unsigned ldx, bool &atLevel) {
  return isInvariantAffine(a, codegen.getLoopCurStack(), ldx, atLevel);
}

/// Helper method to construct a permuted dimension ordering
/// that adheres to the given topological sort.
static AffineMap permute(const Merger &merger, MLIRContext *context,
                         AffineMap m, ArrayRef<unsigned> topSort) {
  assert(m.getNumDims() + merger.getNumFilterLoops() == topSort.size() &&
         "TopoSort/AffineMap size mismatch");
  // Construct the inverse of `m`; to avoid the asymptotic complexity
  // of calling `m.getPermutedPosition` repeatedly.
  SmallVector<unsigned> perm;
  unsigned numResults = m.getNumResults();
  BitVector worklist(numResults, true);
  unsigned loopDepth = 1;

  // Construct the permutation.
  while (worklist.any() && loopDepth <= topSort.size()) {
    unsigned preSize = perm.size();
    for (auto dim : worklist.set_bits()) {
      bool atLevel = false;
      if (m.getResult(dim).isa<AffineConstantExpr>() ||
          (isInvariantAffine(m.getResult(dim), topSort.slice(0, loopDepth),
                             topSort[loopDepth - 1], atLevel) &&
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
  return AffineMap::getPermutationMap(perm, context);
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
  auto enc = getSparseTensorEncoding(tensor.getType());
  if (enc) {
    ArrayRef<AffineExpr> exps = affineMap.getResults();
    for (unsigned rank = 0; rank < exps.size(); rank++) {
      auto aidx = toOrigDim(enc, rank);
      auto affine = exps[aidx];
      if (!affine.isa<AffineDimExpr>())
        if (!isDenseDLT(getDimLevelType(enc, rank)))
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

/// Helper method to inspect sparse encodings in the tensor types.
/// Fills the per-dimension sparsity information for all tensors.
/// Returns true if the sparse annotations and affine subscript
/// expressions of all tensors are admissible. Returns false if
/// no annotations are found or inadmissible constructs occur.
static bool findSparseAnnotations(Merger &merger, linalg::GenericOp op) {
  bool annotated = false;
  unsigned filterLdx = merger.getFilterLoopStartingIdx();
  for (OpOperand &t : op->getOpOperands()) {
    auto map = op.getMatchingIndexingMap(&t);
    auto enc = getSparseTensorEncoding(t.get().getType());
    if (enc)
      annotated = true;
    assert(map.getNumResults() == op.getRank(&t));

    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      unsigned tensor = t.getOperandNumber();
      AffineExpr a = map.getResult(toOrigDim(enc, d));
      if (!findAffine(merger, tensor, d, a, getDimLevelType(enc, d), filterLdx))
        return false; // inadmissible affine expression
    }
  }
  assert(filterLdx == merger.getNumLoops());
  return annotated;
}

/// A helper to compute a topological sort. O(n^2) time complexity
/// as we use adj matrix for the graph.
/// The sorted result will put the first Reduction iterator to the
/// latest possible index.
static bool topSortOptimal(unsigned n,
                           ArrayRef<utils::IteratorType> iteratorTypes,
                           const Merger &merger, std::vector<unsigned> &topSort,
                           std::vector<unsigned> &inDegree,
                           std::vector<std::vector<bool>> &adjM) {
  std::vector<unsigned> redIt;    // reduce iterator with 0 degree
  std::vector<unsigned> parIt;    // parallel iterator with 0 degree
  std::vector<unsigned> filterIt; // filter loop with 0 degree
  for (unsigned i = 0; i < n; i++) {
    if (inDegree[i] == 0) {
      if (merger.isFilterLoop(i))
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
    topSort.push_back(src);
    it.pop_back();
    // Update in-degree, and push 0-degree node into worklist.
    for (unsigned dst = 0; dst < n; dst++) {
      if (adjM[src][dst] && --inDegree[dst] == 0) {
        if (merger.isFilterLoop(dst))
          filterIt.push_back(dst);
        else if (linalg::isReductionIterator(iteratorTypes[dst]))
          redIt.push_back(dst);
        else
          parIt.push_back(dst);
      }
    }
  }
  return topSort.size() == n;
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
                               AffineExpr b, Optional<unsigned> fidx,
                               Optional<unsigned> tidx) {
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
                                            Optional<unsigned> &fldx,
                                            AffineExpr &fa,
                                            Optional<unsigned> &tldx,
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
static bool computeIterationGraph(Merger &merger, linalg::GenericOp op,
                                  std::vector<unsigned> &topSort, unsigned mask,
                                  OpOperand *skip = nullptr) {
  // Set up an n x n from/to adjacency matrix of the iteration graph
  // for the implicit loop indices i_0 .. i_n-1.
  unsigned n = merger.getNumLoops();
  std::vector<std::vector<bool>> adjM(n, std::vector<bool>(n, false));
  std::vector<unsigned> inDegree(n, 0); // in-degree of each node.
  auto iteratorTypes = op.getIteratorTypesArray();
  // Iterate over the indexing maps of every tensor in the tensor expression.
  for (OpOperand &t : op->getOpOperands()) {
    // Get map and encoding.
    auto map = op.getMatchingIndexingMap(&t);
    auto enc = getSparseTensorEncoding(t.get().getType());
    assert(map.getNumDims() + getNumCompoundAffineOnSparseDims(op) == n);
    // Skip dense tensor constraints when not requested.
    if (!(mask & SortMask::kIncludeDense) && !enc)
      continue;
    // Each tensor expression and optional dimension ordering (row-major
    // by default) puts an ordering constraint on the loop indices. For
    // example, the tensor expresion A_ijk forces the ordering i < j < k
    // on the loop indices if no explicit dimension ordering is given.
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      AffineExpr ta = map.getResult(toOrigDim(enc, d));
      Optional<unsigned> tldx = merger.getLoopIdx(t.getOperandNumber(), d);

      // Filter loops should be constructed after all the dependent loops,
      // i.e., d0 + d1 < filter_loop(d0 + d1)
      if (tldx && merger.isFilterLoop(tldx.value())) {
        assert(!ta.isa<AffineDimExpr>() &&
               !isDenseDLT(getDimLevelType(enc, d)));
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

      if (d > 0) {
        AffineExpr fa = map.getResult(toOrigDim(enc, d - 1));
        Optional<unsigned> fldx =
            merger.getLoopIdx(t.getOperandNumber(), d - 1);

        // Applying order constraints on every pair of dimExpr between two
        // compound affine expressions can sometime too strict:
        // E.g, for [dense, dense] -> (d0 + d1, d2 + d3).
        // It is totally fine to have loop sequence d0->d2->d1->d3 instead of
        // requiring d0 < d2, d1 < d2, d0 < d3, d1 < d3.
        if (!(mask & SortMask::kIncludeDense))
          tryLoosenAffineDenseConstraints(op, fldx, fa, tldx, ta);

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
      for (unsigned i = 0; i < n; i++)
        if (isCompressedDLT(merger.getDimLevelType(tensor, i)) ||
            isSingletonDLT(merger.getDimLevelType(tensor, i))) {
          for (unsigned j = 0; j < n; j++)
            if (isUndefDLT(merger.getDimLevelType(tensor, j))) {
              adjM[i][j] = true;
              inDegree[j]++;
            }
        } else {
          assert(isDenseDLT(merger.getDimLevelType(tensor, i)) ||
                 isUndefDLT(merger.getDimLevelType(tensor, i)));
        }
    }
  }
  // Topologically sort the iteration graph to determine loop order.
  // Report failure for a cyclic iteration graph.
  topSort.clear();
  topSort.reserve(n);
  return topSortOptimal(n, iteratorTypes, merger, topSort, inDegree, adjM);
}

/// Returns true if tensor materializes uninitialized into the computation.
static bool isMaterializing(Value val) {
  return val.getDefiningOp<tensor::EmptyOp>() ||
         val.getDefiningOp<bufferization::AllocTensorOp>();
}

/// Returns true when the tensor expression is admissible for codegen.
/// Since all sparse input tensors are admissible, we just need to check
/// whether the out tensor in the tensor expression codegen is admissible.
/// Sets `sparseOut` to the tensor and `outerParNest` to the outer injective
/// nesting depth when a "truly dynamic" sparse tensor output occurs.
static bool isAdmissibleTensorExp(Merger &merger, linalg::GenericOp op,
                                  std::vector<unsigned> &topSort, unsigned exp,
                                  OpOperand **sparseOut,
                                  unsigned &outerParNest) {
  // We reject any expression that makes a reduction from `-outTensor`, as those
  // expression create dependency between the current iteration (i) and the
  // previous iteration (i-1). It would then require iterating over the whole
  // coordinate space, which prevent us from exploiting sparsity for faster
  // code.
  for (utils::IteratorType it : op.getIteratorTypesArray()) {
    if (it == utils::IteratorType::reduction) {
      if (merger.hasNegateOnOut(exp))
        return false;
      break;
    }
  }

  OpOperand *lhs = op.getDpsInitOperand(0);
  unsigned tensor = lhs->getOperandNumber();
  auto enc = getSparseTensorEncoding(lhs->get().getType());
  // An non-annotated output tensor is assumed dense, and becomes a random
  // access n-dim memref. Admissible since insertions cannot occur.
  if (!enc)
    return true;
  // An all-dense annotated "sparse" output tensor becomes a linearized random
  // access 1-dim memref. Also admissible since insertions cannot occur.
  bool allDense = true;
  unsigned numLoops = merger.getNumLoops(); // numNativeLoops + numFilterLoops
  for (unsigned i = 0; i < merger.getNumLoops(); i++)
    if (isCompressedDLT(merger.getDimLevelType(tensor, i)) ||
        isSingletonDLT(merger.getDimLevelType(tensor, i))) {
      allDense = false;
      break;
    } else {
      assert(isDenseDLT(merger.getDimLevelType(tensor, i)) ||
             isUndefDLT(merger.getDimLevelType(tensor, i)));
    }
  if (allDense)
    return true;

  // TODO: support compound affine expression on sparse output.
  if (getNumCompoundAffineOnSparseDims(op.getMatchingIndexingMap(lhs),
                                       lhs->get()) != 0)
    return false;

  // A tensor expression with a sparse output tensor that changes its values
  // but not its nonzero structure, an operation called "simply dynamic" in
  // [Bik96,Ch9], is also admissible without special codegen.
  if (merger.isSingleCondition(tensor, exp))
    return true;

  // Accept "truly dynamic" if the output tensor materializes uninitialized
  // into the computation and insertions occur in lexicographic index order.
  if (isMaterializing(lhs->get())) {
    auto iteratorTypes = op.getIteratorTypesArray();
    unsigned nest = 0;
    for (unsigned i = 0; i < numLoops; i++) {
      if (!merger.isFilterLoop(topSort[i])) {
        // We only count non-filter loops as filter loops should be considered
        // as a special type of parallel loops.
        if (linalg::isReductionIterator(iteratorTypes[topSort[i]]))
          break; // terminate at first reduction
        nest++;
      }
    }
    // Determine admissible dynamic insertion situations:
    // (1) fully injective, since there are no reductions,
    // (2) admissible 1-d expansion in innermost dimension.
    if (nest >= op.getRank(lhs) - 1) {
      *sparseOut = lhs;
      outerParNest = nest;
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Sparse compiler synthesis methods (reductions).
//===----------------------------------------------------------------------===//

/// Maps operation to reduction.
static Reduction getReduction(Kind kind) {
  switch (kind) {
  case Kind::kAddF:
  case Kind::kAddC:
  case Kind::kAddI:
  case Kind::kSubF:
  case Kind::kSubC:
  case Kind::kSubI:
    return kSum;
  case Kind::kMulF:
  case Kind::kMulC:
  case Kind::kMulI:
    return kProduct;
  case Kind::kAndI:
    return kAnd;
  case Kind::kOrI:
    return kOr;
  case Kind::kXorI:
    return kXor;
  case Kind::kReduce:
    return kCustom;
  default:
    llvm_unreachable("unexpected reduction operator");
  }
}

/// Updates scalarized reduction value.
static void updateReduc(Merger &merger, CodeGen &codegen, Value reduc) {
  assert(codegen.redKind != kNoReduc);
  codegen.redVal = merger.exp(codegen.redExp).val = reduc;
}

/// Extracts identity from custom reduce.
static Value getCustomRedId(Operation *op) {
  return dyn_cast<sparse_tensor::ReduceOp>(op).getIdentity();
}

//===----------------------------------------------------------------------===//
// Sparse compiler synthesis methods (statements and expressions).
//===----------------------------------------------------------------------===//

/// Generates loop boundary statements (entering/exiting loops). The function
/// passes and updates the reduction value.
static Optional<Operation *> genLoopBoundary(
    CodeGen &codegen, Merger &merger,
    function_ref<Optional<Operation *>(MutableArrayRef<Value> reduc)>
        callback) {
  SmallVector<Value> reduc;
  if (codegen.redVal)
    reduc.push_back(codegen.redVal);
  if (codegen.expValues)
    reduc.push_back(codegen.expCount);
  if (codegen.insChain)
    reduc.push_back(codegen.insChain);

  auto r = callback(reduc);

  // Callback should do in-place update on reduction value vector.
  unsigned i = 0;
  if (codegen.redVal)
    updateReduc(merger, codegen, reduc[i++]);
  if (codegen.expValues)
    codegen.expCount = reduc[i++];
  if (codegen.insChain)
    codegen.insChain = reduc[i];

  return r;
}

/// Local bufferization of all dense and sparse data structures.
static void genBuffers(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                       linalg::GenericOp op) {
  Location loc = op.getLoc();
  assert(op.getNumOperands() == op.getNumDpsInputs() + 1);

  codegen.loopEmitter.initializeLoopEmit(
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
        OpOperand *lhs = op.getDpsInitOperand(0);
        // Two output tensors references should pointed to the same object.
        assert(lhs->get() == tensor);
        bool isInit = op.isInitTensor(lhs);
        // An output tensor can simply materialize from the buffer of the tensor
        // that appears in the outs() clause. For updates, this has the
        // advantage that only the nonzero value are involved in the
        // computation, keeping the operation O(nnz). In all other cases, we are
        // forced to zero out the buffer to enforce the assumption above, which
        // may negatively impact running complexity (viz. O(n^2 + nnz) vs.
        // O(nnz) for matrices).
        // TODO: use better analysis to avoid zeroing out the buffer?
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
static Value genIndex(CodeGen &codegen, linalg::GenericOp op, OpOperand *t) {
  auto map = op.getMatchingIndexingMap(t);
  auto enc = getSparseTensorEncoding(t->get().getType());
  AffineExpr a = map.getResult(toOrigDim(enc, map.getNumResults() - 1));
  assert(a.getKind() == AffineExprKind::DimId);
  unsigned idx = a.cast<AffineDimExpr>().getPosition();
  return codegen.getLoopIdxValue(idx);
}

/// Generates subscript for load/store on a dense or sparse tensor.
static Value genSubscript(CodeGen &codegen, OpBuilder &builder,
                          linalg::GenericOp op, OpOperand *t,
                          SmallVectorImpl<Value> &args) {
  unsigned tensor = t->getOperandNumber();
  auto map = op.getMatchingIndexingMap(t);
  auto enc = getSparseTensorEncoding(t->get().getType());
  unsigned rank = map.getNumResults();
  if (enc) {
    Value pidx = codegen.loopEmitter.getPidxs()[tensor].back();
    assert(pidx);
    args.push_back(pidx); // position index
  } else {
    for (unsigned d = 0; d < rank; d++) {
      AffineExpr a = map.getResult(d);
      args.push_back(codegen.loopEmitter.genAffine(builder, a, op.getLoc()));
    }
  }
  return codegen.loopEmitter.getValBuffer()[tensor];
}

/// Generates insertion code to implement dynamic tensor load.
static Value genInsertionLoad(CodeGen &codegen, OpBuilder &builder,
                              linalg::GenericOp op, OpOperand *t) {
  Location loc = op.getLoc();
  // Direct lexicographic index order, tensor loads as zero.
  if (!codegen.expValues) {
    Type tp = getElementTypeOrSelf(t->get().getType());
    return constantZero(builder, loc, tp);
  }
  // Load from expanded access pattern.
  Value index = genIndex(codegen, op, t);
  return builder.create<memref::LoadOp>(loc, codegen.expValues, index);
}

/// Generates insertion code to implement dynamic tensor load for reduction.
static Value genInsertionLoadReduce(Merger &merger, CodeGen &codegen,
                                    OpBuilder &builder, linalg::GenericOp op,
                                    OpOperand *t) {
  Location loc = op.getLoc();
  Value identity = getCustomRedId(merger.exp(codegen.redCustom).op);
  // Direct lexicographic index order, tensor loads as identity.
  if (!codegen.expValues) {
    return identity;
  }
  // Load from expanded access pattern if filled, identity otherwise.
  Value index = genIndex(codegen, op, t);
  Value isFilled =
      builder.create<memref::LoadOp>(loc, codegen.expFilled, index);
  Value valAtIndex =
      builder.create<memref::LoadOp>(loc, codegen.expValues, index);
  return builder.create<arith::SelectOp>(loc, isFilled, valAtIndex, identity);
}

/// Generates insertion code to implement dynamic tensor store.
static void genInsertionStore(CodeGen &codegen, OpBuilder &builder,
                              linalg::GenericOp op, OpOperand *t, Value rhs) {
  Location loc = op.getLoc();
  // Direct insertion in lexicographic index order.
  if (!codegen.expValues) {
    unsigned rank = op.getRank(t);
    SmallVector<Value> indices;
    for (unsigned i = 0; i < rank; i++) {
      assert(codegen.loopEmitter.getLoopIV(i));
      indices.push_back(codegen.loopEmitter.getLoopIV(i));
    }
    codegen.insChain =
        builder.create<InsertOp>(loc, rhs, codegen.insChain, indices);
    return;
  }
  // Generates insertion code along expanded access pattern.
  //   if (!expFilled[i]) then
  //     expFilled[i] = true
  //     expAdded[inserts++] = i
  //   endif
  //   values[i] = rhs
  Value index = genIndex(codegen, op, t);
  Value fval = constantI1(builder, loc, false);
  Value tval = constantI1(builder, loc, true);
  // If statement.
  Value filled = builder.create<memref::LoadOp>(loc, codegen.expFilled, index);
  Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             filled, fval);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, builder.getIndexType(), cond,
                                             /*else=*/true);
  // True branch.
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  builder.create<memref::StoreOp>(loc, tval, codegen.expFilled, index);
  builder.create<memref::StoreOp>(loc, index, codegen.expAdded,
                                  codegen.expCount);
  Value one = constantIndex(builder, loc, 1);
  Value add = builder.create<arith::AddIOp>(loc, codegen.expCount, one);
  builder.create<scf::YieldOp>(loc, add);
  // False branch.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, codegen.expCount);
  builder.setInsertionPointAfter(ifOp);
  // Value assignment.
  codegen.expCount = ifOp.getResult(0);
  builder.create<memref::StoreOp>(loc, rhs, codegen.expValues, index);
}

/// Generates a load on a dense or sparse tensor.
static Value genTensorLoad(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                           linalg::GenericOp op, unsigned exp) {
  // Test if the load was hoisted to a higher loop nest.
  Value val = merger.exp(exp).val;
  if (val)
    return val;

  // Load during insertion.
  OpOperand &t = op->getOpOperand(merger.exp(exp).tensor);
  if (&t == codegen.sparseOut) {
    if (codegen.redCustom != -1u)
      return genInsertionLoadReduce(merger, codegen, builder, op, &t);
    return genInsertionLoad(codegen, builder, op, &t);
  }
  // Actual load.
  SmallVector<Value> args;
  Value ptr = genSubscript(codegen, builder, op, &t, args);
  return builder.create<memref::LoadOp>(op.getLoc(), ptr, args);
}

/// Generates a store on a dense or sparse tensor.
static void genTensorStore(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                           linalg::GenericOp op, unsigned exp, Value rhs) {
  Location loc = op.getLoc();
  // Test if this is a scalarized reduction.
  if (codegen.redVal) {
    updateReduc(merger, codegen, rhs);
    return;
  }
  // Store during insertion.
  OpOperand *t = op.getDpsInitOperand(0);
  if (t == codegen.sparseOut) {
    if (!rhs) {
      // Only unary and binary are allowed to return uninitialized rhs
      // to indicate missing output.
      assert(merger.exp(exp).kind == kUnary || merger.exp(exp).kind == kBinary);
    } else if (merger.exp(exp).kind == kSelect) {
      // Select operation insertion.
      Value insChain = codegen.insChain;
      assert(insChain);
      scf::IfOp ifOp = builder.create<scf::IfOp>(loc, insChain.getType(), rhs,
                                                 /*else=*/true);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      // Existing value was preserved to be used here.
      assert(merger.exp(exp).val);
      Value v0 = merger.exp(exp).val;
      genInsertionStore(codegen, builder, op, t, v0);
      merger.exp(exp).val = Value();
      // Yield modified insertion chain along true branch.
      builder.create<scf::YieldOp>(op.getLoc(), codegen.insChain);
      // Yield original insertion chain along false branch.
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      builder.create<scf::YieldOp>(loc, insChain);
      // Done with if statement.
      codegen.insChain = ifOp->getResult(0);
      builder.setInsertionPointAfter(ifOp);
    } else {
      genInsertionStore(codegen, builder, op, t, rhs);
    }
    return;
  }
  // Actual store.
  SmallVector<Value> args;
  Value ptr = genSubscript(codegen, builder, op, t, args);
  builder.create<memref::StoreOp>(loc, rhs, ptr, args);
}

/// Generates an invariant value.
inline static Value genInvariantValue(Merger &merger, CodeGen &codegen,
                                      OpBuilder &builder, unsigned exp) {
  return merger.exp(exp).val;
}

/// Generates an index value.
inline static Value genIndexValue(CodeGen &codegen, OpBuilder &builder,
                                  unsigned idx) {
  return codegen.getLoopIdxValue(idx);
}

/// Semi-ring branches are simply inlined by the sparse compiler. Prior
/// analysis has verified that all computations are "local" to the inlined
/// branch or otherwise invariantly defined outside the loop nest, with the
/// exception of index computations, which need to be relinked to actual
/// inlined cloned code.
static Value relinkBranch(CodeGen &codegen, RewriterBase &rewriter,
                          Block *block, Value e, unsigned ldx) {
  if (Operation *def = e.getDefiningOp()) {
    if (auto indexOp = dyn_cast<linalg::IndexOp>(def))
      return genIndexValue(codegen, rewriter, indexOp.getDim());
    if (def->getBlock() == block) {
      for (unsigned i = 0, n = def->getNumOperands(); i < n; i++)
        def->setOperand(
            i, relinkBranch(codegen, rewriter, block, def->getOperand(i), ldx));
    }
  }
  return e;
}

/// Recursively generates tensor expression.
static Value genExp(Merger &merger, CodeGen &codegen, RewriterBase &rewriter,
                    linalg::GenericOp op, unsigned exp, unsigned ldx) {
  Location loc = op.getLoc();
  if (exp == -1u)
    return Value();
  if (merger.exp(exp).kind == Kind::kTensor)
    return genTensorLoad(merger, codegen, rewriter, op, exp);
  if (merger.exp(exp).kind == Kind::kInvariant)
    return genInvariantValue(merger, codegen, rewriter, exp);
  if (merger.exp(exp).kind == Kind::kIndex)
    return genIndexValue(codegen, rewriter, merger.exp(exp).index);

  if (merger.exp(exp).kind == Kind::kReduce) {
    // Make custom reduction identity accessible for expanded access pattern.
    assert(codegen.redCustom == -1u);
    codegen.redCustom = exp;
  }

  Value v0 =
      genExp(merger, codegen, rewriter, op, merger.exp(exp).children.e0, ldx);
  Value v1 =
      genExp(merger, codegen, rewriter, op, merger.exp(exp).children.e1, ldx);
  Value ee = merger.buildExp(rewriter, loc, exp, v0, v1);
  if (ee && (merger.exp(exp).kind == Kind::kUnary ||
             merger.exp(exp).kind == Kind::kBinary ||
             merger.exp(exp).kind == Kind::kBinaryBranch ||
             merger.exp(exp).kind == Kind::kReduce ||
             merger.exp(exp).kind == Kind::kSelect))
    ee = relinkBranch(codegen, rewriter, ee.getParentBlock(), ee, ldx);

  if (merger.exp(exp).kind == kSelect) {
    assert(!merger.exp(exp).val);
    merger.exp(exp).val = v0; // Preserve value for later use.
  }

  if (merger.exp(exp).kind == Kind::kReduce) {
    assert(codegen.redCustom != -1u);
    codegen.redCustom = -1u;
  }

  return ee;
}

/// Hoists loop invariant tensor loads for which indices have been exhausted.
static void genInvariants(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                          linalg::GenericOp op, unsigned exp, unsigned ldx,
                          bool atStart, unsigned last = -1u) {
  if (exp == -1u)
    return;
  if (merger.exp(exp).kind == Kind::kTensor) {
    // Inspect tensor indices.
    bool atLevel = ldx == -1u;
    OpOperand &t = op->getOpOperand(merger.exp(exp).tensor);
    auto map = op.getMatchingIndexingMap(&t);
    auto enc = getSparseTensorEncoding(t.get().getType());
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      AffineExpr a = map.getResult(toOrigDim(enc, d));
      Optional<unsigned> sldx = merger.getLoopIdx(t.getOperandNumber(), d);
      if (sldx && merger.isFilterLoop(sldx.value())) {
        if (!codegen.getLoopIdxValue(sldx.value()))
          // The filter loops has not been constructed.
          return;
        if (sldx.value() == ldx)
          atLevel = true;
      } else if (!isInvariantAffine(codegen, a, ldx, atLevel))
        return; // still in play
    }
    // All exhausted at this level (atLevel denotes exactly at this level).
    if (!atLevel)
      return;
    OpOperand *lhs = op.getDpsInitOperand(0);
    if (lhs == &t) {
      // Start or end a scalarized reduction
      if (atStart) {
        Kind kind = merger.exp(last).kind;
        Value load = kind == Kind::kReduce
                         ? getCustomRedId(merger.exp(last).op)
                         : genTensorLoad(merger, codegen, builder, op, exp);
        codegen.redKind = getReduction(kind);
        codegen.redExp = exp;
        updateReduc(merger, codegen, load);
      } else {
        Value redVal = codegen.redVal;
        updateReduc(merger, codegen, Value());
        codegen.redExp = -1u;
        codegen.redKind = kNoReduc;
        genTensorStore(merger, codegen, builder, op, exp, redVal);
      }
    } else {
      // Start or end loop invariant hoisting of a tensor load.
      merger.exp(exp).val =
          atStart ? genTensorLoad(merger, codegen, builder, op, exp) : Value();
    }
  } else if (merger.exp(exp).kind != Kind::kInvariant &&
             merger.exp(exp).kind != Kind::kIndex) {
    // Traverse into the binary operations. Note that we only hoist
    // tensor loads, since subsequent MLIR/LLVM passes know how to
    // deal with all other kinds of derived loop invariants.
    unsigned e0 = merger.exp(exp).children.e0;
    unsigned e1 = merger.exp(exp).children.e1;
    genInvariants(merger, codegen, builder, op, e0, ldx, atStart, exp);
    genInvariants(merger, codegen, builder, op, e1, ldx, atStart, exp);
  }
}

/// Generates an expanded access pattern in innermost dimension.
static void genExpansion(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                         linalg::GenericOp op, unsigned at, bool atStart) {
  OpOperand *lhs = codegen.sparseOut;
  if (!lhs || codegen.outerParNest != op.getRank(lhs) - 1 ||
      at != codegen.outerParNest)
    return; // not needed at this level
  assert(codegen.redVal == nullptr);
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
    auto res =
        builder.create<ExpandOp>(loc, TypeRange({t1, t2, t3, t4}), tensor);
    assert(res.getNumResults() == 4);
    assert(!codegen.expValues);
    codegen.expValues = res.getResult(0);
    codegen.expFilled = res.getResult(1);
    codegen.expAdded = res.getResult(2);
    codegen.expCount = res.getResult(3);
  } else {
    assert(codegen.expValues);
    SmallVector<Value> indices;
    for (unsigned i = 0; i < at; i++) {
      assert(codegen.loopEmitter.getLoopIV(i));
      indices.push_back(codegen.loopEmitter.getLoopIV(i));
    }
    codegen.insChain = builder.create<CompressOp>(
        loc, codegen.expValues, codegen.expFilled, codegen.expAdded,
        codegen.expCount, codegen.insChain, indices);
    codegen.expValues = codegen.expFilled = codegen.expAdded =
        codegen.expCount = Value();
  }
}

/// Returns parallelization strategy. Any implicit loop in the Linalg
/// operation that is marked "parallel" is a candidate. Whether it is actually
/// converted to a parallel operation depends on the requested strategy.
static bool isParallelFor(CodeGen &codegen, bool isOuter, bool isSparse) {
  // Reject parallelization of sparse output.
  if (codegen.sparseOut)
    return false;
  // Parallel loops on tensor expansion can cause data races.
  if (codegen.expCount)
    return false;
  // Inspect strategy.
  switch (codegen.options.parallelizationStrategy) {
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
static Operation *genFor(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                         linalg::GenericOp op, bool isOuter, bool isInner,
                         unsigned idx, size_t tid, size_t dim,
                         ArrayRef<size_t> extraTids,
                         ArrayRef<size_t> extraDims) {
  Location loc = op.getLoc();
  bool isSparse = isCompressedDLT(merger.getDimLevelType(tid, idx)) ||
                  isSingletonDLT(merger.getDimLevelType(tid, idx));
  bool isParallel = isParallelFor(codegen, isOuter, isSparse);

  Operation *loop =
      genLoopBoundary(codegen, merger, [&](MutableArrayRef<Value> reduc) {
        if (merger.isFilterLoop(idx)) {
          // extraTids/extraDims must be empty because filter loops only
          // corresponding to the one and only sparse tensor level.
          assert(isSparse && extraTids.empty() && extraDims.empty());
          OpOperand *t = &op->getOpOperand(tid);
          auto enc = getSparseTensorEncoding(t->get().getType());
          // Retrieves the affine expression for the filter loop.
          AffineExpr a =
              op.getMatchingIndexingMap(t).getResult(toOrigDim(enc, dim));
          return codegen.loopEmitter.enterFilterLoopOverTensorAtDim(
              builder, loc, tid, dim, a, reduc);
        }
        return codegen.loopEmitter.enterLoopOverTensorAtDim(
            builder, loc, tid, dim, reduc, isParallel, extraTids, extraDims);
      }).value();
  assert(loop);
  return loop;
}

/// Emit a while-loop for co-iteration over multiple indices.
static Operation *genWhile(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                           linalg::GenericOp op, unsigned idx, bool needsUniv,
                           ArrayRef<size_t> condTids, ArrayRef<size_t> condDims,
                           ArrayRef<size_t> extraTids,
                           ArrayRef<size_t> extraDims) {

  Operation *loop =
      genLoopBoundary(codegen, merger, [&](MutableArrayRef<Value> reduc) {
        // Construct the while-loop with a parameter for each index.
        return codegen.loopEmitter.enterCoIterationOverTensorsAtDims(
            builder, op.getLoc(), condTids, condDims, needsUniv, reduc,
            extraTids, extraDims);
      }).value();
  assert(loop);
  return loop;
}

/// Generates a for-loop or a while-loop, depending on whether it implements
/// singleton iteration or co-iteration over the given conjunction.
static Operation *genLoop(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                          linalg::GenericOp op, unsigned at, bool needsUniv,
                          ArrayRef<size_t> condTids, ArrayRef<size_t> condDims,
                          ArrayRef<size_t> extraTids,
                          ArrayRef<size_t> extraDims) {
  assert(condTids.size() == condDims.size());
  assert(extraTids.size() == extraDims.size());
  unsigned idx = codegen.topSort[at];
  if (condTids.size() == 1) {
    bool isOuter = at == 0;
    bool isInner = at == codegen.topSort.size() - 1;
    return genFor(merger, codegen, builder, op, isOuter, isInner, idx,
                  condTids.front(), condDims.front(), extraTids, extraDims);
  }
  return genWhile(merger, codegen, builder, op, idx, needsUniv, condTids,
                  condDims, extraTids, extraDims);
}

/// Generates the induction structure for a while-loop.
static void finalizeWhileOp(Merger &merger, CodeGen &codegen,
                            OpBuilder &builder, linalg::GenericOp op,
                            unsigned idx, bool needsUniv, BitVector &induction,
                            scf::WhileOp whileOp) {
  Location loc = op.getLoc();
  // Finalize each else branch of all if statements.
  if (codegen.redVal || codegen.expValues || codegen.insChain) {
    while (auto ifOp = dyn_cast_or_null<scf::IfOp>(
               builder.getInsertionBlock()->getParentOp())) {
      unsigned y = 0;
      SmallVector<Value> yields;
      if (codegen.redVal) {
        yields.push_back(codegen.redVal);
        updateReduc(merger, codegen, ifOp.getResult(y++));
      }
      if (codegen.expValues) {
        yields.push_back(codegen.expCount);
        codegen.expCount = ifOp->getResult(y++);
      }
      if (codegen.insChain) {
        yields.push_back(codegen.insChain);
        codegen.insChain = ifOp->getResult(y++);
      }
      assert(y == yields.size());
      builder.create<scf::YieldOp>(loc, yields);
      builder.setInsertionPointAfter(ifOp);
    }
  }
  builder.setInsertionPointToEnd(&whileOp.getAfter().front());
}

/// Generates a single if-statement within a while-loop.
static scf::IfOp genIf(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                       linalg::GenericOp op, unsigned idx,
                       BitVector &conditions) {
  Location loc = op.getLoc();
  SmallVector<Type> types;
  Value cond;
  for (unsigned b = 0, be = conditions.size(); b < be; b++) {
    if (!conditions[b])
      continue;
    unsigned tensor = merger.tensor(b);
    assert(idx == merger.index(b));
    Value clause;
    if (isCompressedDLT(merger.getDimLevelType(b)) ||
        isSingletonDLT(merger.getDimLevelType(b))) {
      auto dim = merger.getDimNum(tensor, idx).value();
      Value op1 = codegen.loopEmitter.getCoord()[tensor][dim];
      Value op2 = codegen.getLoopIdxValue(idx);
      clause = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, op1,
                                             op2);
    } else {
      assert(isDenseDLT(merger.getDimLevelType(b)) ||
             isUndefDLT(merger.getDimLevelType(b)));
      clause = constantI1(builder, loc, true);
    }
    cond = cond ? builder.create<arith::AndIOp>(loc, cond, clause) : clause;
  }
  if (codegen.redVal)
    types.push_back(codegen.redVal.getType());
  if (codegen.expValues)
    types.push_back(builder.getIndexType());
  if (codegen.insChain)
    types.push_back(codegen.insChain.getType());
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, types, cond, /*else=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return ifOp;
}

/// Generates end of true branch of if-statement within a while-loop.
static void endIf(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                  linalg::GenericOp op, scf::IfOp ifOp, Operation *loop,
                  Value redInput, Value cntInput, Value insInput) {
  SmallVector<Value> operands;
  if (codegen.redVal) {
    operands.push_back(codegen.redVal);
    updateReduc(merger, codegen, redInput);
  }
  if (codegen.expValues) {
    operands.push_back(codegen.expCount);
    codegen.expCount = cntInput;
  }
  if (codegen.insChain) {
    operands.push_back(codegen.insChain);
    codegen.insChain = insInput;
  }
  if (!operands.empty())
    builder.create<scf::YieldOp>(op.getLoc(), operands);
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
}

//===----------------------------------------------------------------------===//
// Sparse compiler synthesis methods (loop sequence).
//===----------------------------------------------------------------------===//

/// Starts a loop sequence at given level. Returns true if
/// the universal loop index must be maintained at this level.
static bool startLoopSeq(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                         linalg::GenericOp op, unsigned exp, unsigned at,
                         unsigned idx, unsigned ldx, unsigned lts) {
  assert(!codegen.getLoopIdxValue(idx));
  // Emit invariants at this loop sequence level.
  genInvariants(merger, codegen, builder, op, exp, ldx, /*atStart=*/true);
  // Emit access pattern expansion for sparse tensor output.
  genExpansion(merger, codegen, builder, op, at, /*atStart=*/true);
  // Emit further intitialization at this loop sequence level.
  unsigned l0 = merger.set(lts)[0];
  bool needsUniv = false;

  SmallVector<size_t> tids;
  SmallVector<size_t> dims;
  merger.foreachTidDimPairInBits(
      merger.lat(l0).bits,
      [&](unsigned b, unsigned tid, Optional<unsigned> dim, DimLevelType dlt) {
        assert(merger.index(b) == idx);
        if (isDenseDLT(dlt) || isUndefDLT(dlt)) {
          needsUniv = true;
        } else {
          // sparse/singleton dim levels.
          tids.push_back(tid);
          dims.push_back(dim.value());
        }
      });

  codegen.loopEmitter.enterNewLoopSeq(builder, op.getLoc(), tids, dims);

  // Maintain the universal index only if it is actually
  // consumed by a subsequent lattice point.
  if (needsUniv) {
    unsigned lsize = merger.set(lts).size();
    for (unsigned i = 1; i < lsize; i++) {
      unsigned li = merger.set(lts)[i];
      if (!merger.hasAnySparse(merger.lat(li).simple))
        return true;
    }
  }
  return false;
}

static void genConstantDenseAddressFromLevel(CodeGen &codegen,
                                             OpBuilder &builder,
                                             linalg::GenericOp op, unsigned tid,
                                             unsigned lvl) {
  // TODO: Handle affine expression on output tensor.
  assert(tid < op.getNumDpsInputs());

  OpOperand *input = op.getDpsInputOperands()[tid];
  ArrayRef<AffineExpr> affines = op.getMatchingIndexingMap(input).getResults();
  auto enc = getSparseTensorEncoding(input->get().getType());
  if (enc) {
    for (unsigned i = lvl, e = affines.size(); i < e; i++) {
      AffineExpr affine = affines[toOrigDim(enc, i)];
      if (isDenseDLT(getDimLevelType(enc, i)) &&
          affine.isa<AffineConstantExpr>()) {
        codegen.loopEmitter.genDenseAffineAddressAtCurLevel(
            builder, op.getLoc(), input->getOperandNumber(), i, affine);
      } else {
        // Breaks on first non-dense non-constant level.
        return;
      }
    }
  }
}

static void genInitConstantDenseAddress(CodeGen &codegen,
                                        RewriterBase &rewriter,
                                        linalg::GenericOp op) {
  // We can generates address for constant affine expression before any loops
  // starting from the first level as they do not depend on any thing.
  // E.g., [Dense, Dense, Sparse] -> (1, 2, d0), the addresses for the first two
  // levels can be determined before loops.
  for (unsigned tid = 0, e = op.getNumDpsInputs(); tid < e; tid++)
    genConstantDenseAddressFromLevel(codegen, rewriter, op, tid, 0);
}

static void translateBitsToTidDimPairs(
    Merger &merger, CodeGen &codegen, linalg::GenericOp op, unsigned li,
    unsigned idx, SmallVectorImpl<size_t> &condTids,
    SmallVectorImpl<size_t> &condDims, SmallVectorImpl<size_t> &extraTids,
    SmallVectorImpl<size_t> &extraDims, SmallVectorImpl<size_t> &affineTids,
    SmallVectorImpl<size_t> &affineDims, SmallVectorImpl<AffineExpr> &exps) {

  const BitVector &all = merger.lat(li).bits;
  const BitVector &simple = merger.lat(li).simple;

  // Converts bits to array + dim pair
  merger.foreachTidDimPairInBits(all, [&, idx](unsigned b, unsigned tid,
                                               Optional<unsigned> dim,
                                               DimLevelType dlt) {
    if (simple.test(b)) {
      if (isUndefDLT(dlt)) {
        // An undefined dlt in the lattices, we probably mean to iterate based
        // on the dim of output tensor.
        // E.g., this could be a synthetic tensor (for invariants and sparse
        // output tensor).
        // out[i][j] = invariant; or a broadcast
        // out[i][j] = in[i] (j is undef for input)
        tid = merger.getOutTensorID();
        dim = merger.getDimNum(tid, idx);
        // Skips invalid dim (e.g., when this is a zero ranked tensor).
        if (!dim)
          return;
      }
      condTids.push_back(tid);
      condDims.push_back(dim.value());
    } else if (isDenseDLT(dlt)) {
      // TODO: get rid of extraTids and extraDims.
      extraTids.push_back(tid);
      extraDims.push_back(dim.value());
    } else {
      assert(isUndefDLT(dlt));
      if (tid >= op.getNumDpsInputs())
        // We only handle affine expression on input tensors (for now).
        return;
      OpOperand *operand = &op->getOpOperand(tid);
      auto enc = getSparseTensorEncoding(operand->get().getType());
      // Non-annotated dense tensors requires no special handling.
      if (!enc)
        return;

      ArrayRef<AffineExpr> affines =
          op.getMatchingIndexingMap(operand).getResults();
      assert(affines.size() == enc.getDimLevelType().size());
      for (unsigned i = 0, e = affines.size(); i < e; i++) {
        AffineExpr exp = affines[toOrigDim(enc, i)];
        // Skip simple affine expression and non dense dimensions (which has
        // it own filter loop).
        if (exp.isa<AffineDimExpr>() || !isDenseDLT(getDimLevelType(enc, i)))
          continue;

        // Constant affine expression are handled in genLoop
        if (!exp.isa<AffineConstantExpr>()) {
          bool atLevel = false;
          if (isInvariantAffine(codegen, exp, idx, atLevel) && atLevel) {
            // If the compound affine is invariant and we are right at the
            // level. We need to generate the address according to the affine
            // expression. This is also the best place we can do it to avoid
            // putting it inside inner loops.
            // NOTE: It assumes that the levels of the input tensor are
            // initialized in order (and it is also currently guaranteed by
            // computeIterationGraph), another more admissible approach might be
            // accepting out-of-order access between consecutive dense levels.
            affineTids.push_back(tid);
            affineDims.push_back(i);
            exps.push_back(exp);
          }
        }
      }
    }
  });

  if (isDenseDLT(merger.getDimLevelType(merger.getOutTensorID(), idx))) {
    // Note that we generate dense indices of the output tensor
    // unconditionally, since they may not appear in the lattice, but may be
    // needed for linearized codegen.
    auto dim = merger.getDimNum(merger.getOutTensorID(), idx).value();
    extraTids.push_back(merger.getOutTensorID());
    extraDims.push_back(dim);
  }
}

/// Starts a single loop in current sequence.
static Operation *startLoop(Merger &merger, CodeGen &codegen,
                            OpBuilder &builder, linalg::GenericOp op,
                            unsigned at, unsigned li, bool needsUniv) {
  // The set of tensors + dims to generate loops on
  SmallVector<size_t> condTids, condDims;
  // The set of (dense) tensors that is optimized from condition, yet still
  // need extra locals to iterate on them.
  SmallVector<size_t> extraTids, extraDims;
  // The set of dense tensors with non-trivial affine expression that just
  // becomes invariant and the address shall now be generated at the current
  // level.
  SmallVector<size_t> affineTids, affineDims;
  SmallVector<AffineExpr> affines;

  translateBitsToTidDimPairs(merger, codegen, op, li, codegen.topSort[at],
                             condTids, condDims, extraTids, extraDims,
                             affineTids, affineDims, affines);
  // Emit the for/while-loop control.
  Operation *loop = genLoop(merger, codegen, builder, op, at, needsUniv,
                            condTids, condDims, extraTids, extraDims);

  for (auto [tid, dim, exp] : llvm::zip(affineTids, affineDims, affines)) {
    codegen.loopEmitter.genDenseAffineAddressAtCurLevel(builder, op.getLoc(),
                                                        tid, dim, exp);
  }

  // Until now, we have entered every <tid, dim> pair in {cond, extra,
  // affine}Tids/Dims. The addresses of the upcoming levels which are dependent
  // on constant affines expression may now be determined.
  auto allTids = llvm::concat<size_t>(condTids, extraTids, affineTids);
  auto allDims = llvm::concat<size_t>(condDims, extraDims, affineDims);
  for (auto [tid, dim] : llvm::zip(allTids, allDims)) {
    if (tid != merger.getOutTensorID())
      genConstantDenseAddressFromLevel(codegen, builder, op, tid, dim + 1);
  }

  return loop;
}

/// Ends a single loop in current sequence. Returns new values for needsUniv.
static bool endLoop(Merger &merger, CodeGen &codegen, RewriterBase &rewriter,
                    linalg::GenericOp op, Operation *loop, unsigned idx,
                    unsigned li, bool needsUniv) {
  // End a while-loop.
  if (auto whileOp = dyn_cast<scf::WhileOp>(loop)) {
    finalizeWhileOp(merger, codegen, rewriter, op, idx, needsUniv,
                    merger.lat(li).bits, whileOp);
  } else {
    needsUniv = false;
  }

  genLoopBoundary(codegen, merger, [&](MutableArrayRef<Value> reduc) {
    codegen.loopEmitter.exitCurrentLoop(rewriter, op.getLoc(), reduc);
    return std::nullopt;
  });

  return needsUniv;
}

/// Ends a loop sequence at given level.
static void endLoopSeq(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                       linalg::GenericOp op, unsigned exp, unsigned at,
                       unsigned idx, unsigned ldx) {
  assert(codegen.getLoopIdxValue(idx) == nullptr);
  codegen.loopEmitter.exitCurrentLoopSeq();
  // Unmark bookkeeping of invariants and loop index.
  genInvariants(merger, codegen, builder, op, exp, ldx, /*atStart=*/false);
  // Finalize access pattern expansion for sparse tensor output.
  genExpansion(merger, codegen, builder, op, at, /*atStart=*/false);
}

/// Recursively generates code while computing iteration lattices in order
/// to manage the complexity of implementing co-iteration over unions
/// and intersections of sparse iterations spaces.
static void genStmt(Merger &merger, CodeGen &codegen, RewriterBase &rewriter,
                    linalg::GenericOp op, unsigned exp, unsigned at) {
  // At each leaf, assign remaining tensor (sub)expression to output tensor.
  if (at == codegen.topSort.size()) {
    unsigned ldx = codegen.topSort[at - 1];
    Value rhs = genExp(merger, codegen, rewriter, op, exp, ldx);
    genTensorStore(merger, codegen, rewriter, op, exp, rhs);
    return;
  }

  // Construct iteration lattices for current loop index, with L0 at top.
  unsigned idx = codegen.topSort[at];
  unsigned ldx = at == 0 ? -1u : codegen.topSort[at - 1];
  unsigned lts = merger.optimizeSet(merger.buildLattices(exp, idx));

  // TODO: sort
  // TODO: dedup

  // Start a loop sequence.
  bool needsUniv =
      startLoopSeq(merger, codegen, rewriter, op, exp, at, idx, ldx, lts);

  // Emit a loop for every lattice point L0 >= Li in this loop sequence.
  unsigned lsize = merger.set(lts).size();
  for (unsigned i = 0; i < lsize; i++) {
    // Start a loop.
    unsigned li = merger.set(lts)[i];
    Operation *loop =
        startLoop(merger, codegen, rewriter, op, at, li, needsUniv);

    // Visit all lattices points with Li >= Lj to generate the
    // loop-body, possibly with if statements for coiteration.
    Value redInput = codegen.redVal;
    Value cntInput = codegen.expCount;
    Value insInput = codegen.insChain;
    bool isWhile = dyn_cast<scf::WhileOp>(loop) != nullptr;
    for (unsigned j = 0; j < lsize; j++) {
      unsigned lj = merger.set(lts)[j];
      unsigned ej = merger.lat(lj).exp;
      if (li == lj || merger.latGT(li, lj)) {
        // Recurse into body of each branch.
        if (isWhile) {
          scf::IfOp ifOp =
              genIf(merger, codegen, rewriter, op, idx, merger.lat(lj).simple);
          genStmt(merger, codegen, rewriter, op, ej, at + 1);
          endIf(merger, codegen, rewriter, op, ifOp, loop, redInput, cntInput,
                insInput);
        } else {
          genStmt(merger, codegen, rewriter, op, ej, at + 1);
        }
      }
    }

    // End a loop.
    needsUniv =
        endLoop(merger, codegen, rewriter, op, loop, idx, li, needsUniv);
  }

  // End a loop sequence.
  endLoopSeq(merger, codegen, rewriter, op, exp, at, idx, ldx);
}

/// Converts the result computed by the sparse kernel into the required form.
static void genResult(Merger &merger, CodeGen &codegen, RewriterBase &rewriter,
                      linalg::GenericOp op) {
  OpOperand *lhs = op.getDpsInitOperand(0);
  Value tensor = lhs->get();
  Type resType = tensor.getType();
  if (getSparseTensorEncoding(resType)) {
    // The sparse tensor rematerializes from the original sparse tensor's
    // underlying sparse storage format. For an insertion chain, the
    // tensor materializes from the chain with 'hasInserts' enabled.
    bool hasInserts = codegen.sparseOut == lhs;
    if (hasInserts)
      tensor = codegen.insChain;
    rewriter.replaceOpWithNewOp<LoadOp>(op, resType, tensor, hasInserts);
  } else {
    // To rematerialize an non-annotated tensor, simply load it
    // from the bufferized value.
    Value val = codegen.loopEmitter.getValBuffer().back(); // value array
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
    // Detects sparse annotations and translate the per-dimension sparsity
    // information for all tensors to loop indices in the kernel.
    if (op.getNumDpsInits() != 1)
      return failure();
    unsigned numTensors = op->getNumOperands();
    unsigned numLoops = op.getNumLoops();
    unsigned numFilterLoops = getNumCompoundAffineOnSparseDims(op);
    Merger merger(numTensors, numLoops, numFilterLoops);
    if (!findSparseAnnotations(merger, op))
      return failure();

    // Builds the tensor expression for the Linalg operation in SSA form.
    Optional<unsigned> optExp = merger.buildTensorExpFromLinalg(op);
    if (!optExp.has_value())
      return failure();

    unsigned exp = optExp.value();
    OpOperand *sparseOut = nullptr;
    unsigned outerParNest = 0;
    // Computes a topologically sorted iteration graph to ensure tensors
    // are visited in natural index order. Gradually relaxes the considered
    // constraints until an acyclic iteration graph results, such that sparse
    // code generation can proceed. As a last resort, an attempt is made
    // to resolve cycles by inserting a conversion.
    std::vector<unsigned> topSort;
    // Whether the current GenericOp is admissible.
    bool isAdmissible = false;
    bool hasCycle = true;
    // An const list of all masks that we used for interation graph
    // computation. Must be ordered from strict -> loose.
    const auto allMask = {SortMask::kIncludeAll, SortMask::kIncludeUndef,
                          SortMask::kIncludeDense, SortMask::kSparseOnly};
    for (auto mask : allMask)
      if (computeIterationGraph(merger, op, topSort, mask)) {
        hasCycle = false;
        if (isAdmissibleTensorExp(merger, op, topSort, exp, &sparseOut,
                                  outerParNest)) {
          isAdmissible = true;
          break;
        }
        // else try a set of less strict constraints.
      }

    if (hasCycle)
      // Give it one last shot to resolve the cycle.
      return resolveCycle(merger, rewriter, op);
    if (!isAdmissible)
      // Inadmissible expression, reject.
      return failure();

    merger.setHasSparseOut(sparseOut != nullptr);

    SmallVector<Value> tensors;
    for (OpOperand &t : op->getOpOperands())
      tensors.push_back(t.get());

    // Recursively generates code if admissible.
    CodeGen codegen(options, op.getContext(), tensors, numTensors, numLoops,
                    sparseOut, outerParNest, topSort);
    genBuffers(merger, codegen, rewriter, op);
    genInitConstantDenseAddress(codegen, rewriter, op);
    genStmt(merger, codegen, rewriter, op, exp, 0);
    genResult(merger, codegen, rewriter, op);
    return success();
  }

private:
  // Last resort cycle resolution.
  LogicalResult resolveCycle(Merger &merger, PatternRewriter &rewriter,
                             linalg::GenericOp op) const {
    // Compute topological sort while leaving out every
    // sparse input tensor in succession until an acylic
    // iteration graph results.
    std::vector<unsigned> topSort;
    for (OpOperand *t : op.getDpsInputOperands()) {
      unsigned tensor = t->getOperandNumber();
      Value tval = t->get();
      auto srcEnc = getSparseTensorEncoding(tval.getType());
      if (!srcEnc ||
          !computeIterationGraph(merger, op, topSort, SortMask::kSparseOnly, t))
        continue;
      // Found an input tensor that resolves the cycle by inserting a
      // conversion into a sparse tensor that adheres to the iteration
      // graph order. Also releases the temporary sparse tensor.
      //
      // TODO: investigate fusing the conversion with computation,
      //       especially if it is a direct yield!
      //
      auto srcTp = tval.getType().cast<RankedTensorType>();
      auto dstEnc = SparseTensorEncodingAttr::get(
          op->getContext(), srcEnc.getDimLevelType(),
          permute(merger, getContext(), op.getMatchingIndexingMap(t),
                  topSort), // new order
          srcEnc.getHigherOrdering(), srcEnc.getPointerBitWidth(),
          srcEnc.getIndexBitWidth());
      auto dstTp = RankedTensorType::get(srcTp.getShape(),
                                         srcTp.getElementType(), dstEnc);
      auto convert = rewriter.create<ConvertOp>(tval.getLoc(), dstTp, tval);
      op->setOperand(tensor, convert);
      rewriter.setInsertionPointAfter(op);
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
