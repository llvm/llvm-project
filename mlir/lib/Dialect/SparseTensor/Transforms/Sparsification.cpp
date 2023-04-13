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
enum class SortMask : unsigned {
  // The individual mask bits.
  kIncludeDenseOutput = 0x1, // b001
  kIncludeDenseInput = 0x2,  // b010
  kIncludeUndef = 0x4,       // b100
  // The subsets of mask bits.
  kIncludeAll = 0x7,   // b111
  kIncludeDense = 0x3, // b011
  kSparseOnly = 0x0,   // b000
};

inline static bool includesAny(SortMask mask1, SortMask mask2) {
  return static_cast<unsigned>(mask1) & static_cast<unsigned>(mask2);
}

inline static bool includesDenseInput(SortMask mask) {
  return includesAny(mask, SortMask::kIncludeDenseInput);
}

inline static bool includesDenseOutput(SortMask mask) {
  return includesAny(mask, SortMask::kIncludeDenseOutput);
}

inline static bool includesDense(SortMask mask) {
  return includesAny(mask, SortMask::kIncludeDense);
}

inline static bool includesUndef(SortMask mask) {
  return includesAny(mask, SortMask::kIncludeUndef);
}

/// A helper class that visits an affine expression and tries to find an
/// AffineDimExpr to which the corresponding iterator from a GenericOp matches
/// the desired iterator type.
class AffineDimFinder : public AffineExprVisitor<AffineDimFinder> {
public:
  explicit AffineDimFinder(linalg::GenericOp op)
      : iterTypes(op.getIteratorTypes()) {}

  // Overrides method from AffineExprVisitor.
  void visitDimExpr(AffineDimExpr expr) {
    if (pickedDim == nullptr ||
        pickIterType == iterTypes[expr.getPosition()]
                            .cast<linalg::IteratorTypeAttr>()
                            .getValue()) {
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
  /// The picked AffineDimExpr after visit.  This must be stored as
  /// `AffineExpr` rather than `AffineDimExpr`, because the latter
  /// doesn't have a default ctor.
  AffineExpr pickedDim;
  /// The iterator type that we want.
  utils::IteratorType pickIterType;
  /// The mapping between dim=>iterator type.
  ArrayAttr iterTypes;
};

// Flattens an affine expression into a list of AffineDimExprs.
struct AffineDimCollector : public AffineExprVisitor<AffineDimCollector> {
  // Overrides method from AffineExprVisitor.
  void visitDimExpr(AffineDimExpr expr) { dims.push_back(expr); }
  SmallVector<AffineDimExpr> dims;
};

} // namespace

//===----------------------------------------------------------------------===//
// Sparse compiler analysis methods.
//===----------------------------------------------------------------------===//

// TODO: the "idx"-vs-"ldx" naming convention is not self-explanatory,
// and those letters are too easy to confuse visually.  We should switch
// to a more self-explanatory naming convention like "curLoop"-vs-"prevLoop"
// (assuming that's the actual meaning behind the "idx"-vs-"ldx" convention).

/// Determines if affine expression is invariant.
static bool isInvariantAffine(AffineExpr a, ArrayRef<LoopId> loopStack,
                              LoopId ldx, bool &isAtLoop) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    const LoopId i = a.cast<AffineDimExpr>().getPosition();
    if (i == ldx) {
      isAtLoop = true;
      // Must be invariant if we are at the given loop.
      return true;
    }
    bool isInvariant = false;
    for (LoopId l : loopStack) {
      isInvariant = (l == i);
      if (isInvariant)
        break;
    }
    return isInvariant;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return isInvariantAffine(binOp.getLHS(), loopStack, ldx, isAtLoop) &&
           isInvariantAffine(binOp.getRHS(), loopStack, ldx, isAtLoop);
  }
  default: {
    assert(a.isa<AffineConstantExpr>());
    return true;
  }
  }
}

/// Determines if affine expression is invariant.
static bool isInvariantAffine(CodegenEnv &env, AffineExpr a, LoopId ldx,
                              bool &isAtLoop) {
  return isInvariantAffine(a, env.getCurrentLoopStack(), ldx, isAtLoop);
}

/// Helper method to construct a permuted dimension ordering
/// that adheres to the given topological sort.
//
// FIXME: does the above actually mean "dimensions", or should it say
// "level ordering"?  The same dim/lvl confusion applies to all the code
// and comments in the definition below.
static AffineMap permute(CodegenEnv &env, AffineMap m) {
  assert(m.getNumDims() + env.merger().getNumFilterLoops() ==
             env.topSortSize() &&
         "size mismatch");
  // Construct the inverse of `m`; to avoid the asymptotic complexity
  // of calling `m.getPermutedPosition` repeatedly.
  //
  // The variable `perm` must use `unsigned` rather than `Dimension`/`Level`,
  // because that's what `AffineMap::getPermutationMap` requires.
  // TODO: however, `perm` should be renamed to make clear what exactly
  // it's storing a permutation of.
  SmallVector<unsigned> perm;
  const unsigned numResults = m.getNumResults();
  BitVector worklist(numResults, true);
  LoopOrd loopDepth = 1;

  // Construct the permutation.
  while (worklist.any() && loopDepth <= env.topSortSize()) {
    const unsigned preSize = perm.size();
    for (unsigned dim : worklist.set_bits()) {
      bool isAtLoop = false;
      if (m.getResult(dim).isa<AffineConstantExpr>() ||
          (isInvariantAffine(m.getResult(dim), env.getLoopStackUpTo(loopDepth),
                             env.topSortAt(loopDepth - 1), isAtLoop) &&
           isAtLoop)) {
        // If the matching affine is constant expression or just become
        // invariant. We can visit the dimension now without breaking the
        // topSort constraint.
        perm.push_back(dim);
      }
    }

    // Removes resolved dimension.
    for (unsigned i = preSize, e = perm.size(); i < e; i++)
      worklist.reset(perm[i]);

    // Try entering the next loop in the stack.
    loopDepth++;
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
static bool findAffine(Merger &merger, TensorId tid, Level lvl, AffineExpr a,
                       DimLevelType dlt, LoopId &filterLdx,
                       bool setLvlFormat = true) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    const LoopId idx = merger.makeLoopId(a.cast<AffineDimExpr>().getPosition());
    if (!isUndefDLT(merger.getDimLevelType(tid, idx)))
      return false; // used more than once

    if (setLvlFormat)
      merger.setLevelAndType(tid, idx, lvl, dlt);
    return true;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::Constant: {
    if (!isDenseDLT(dlt) && setLvlFormat) {
      assert(isUndefDLT(merger.getDimLevelType(tid, filterLdx)));
      // Use a filter loop for sparse affine expression.
      merger.setLevelAndType(tid, filterLdx, lvl, dlt);
      ++filterLdx;
    }

    if (auto binOp = a.dyn_cast<AffineBinaryOpExpr>()) {
      // We do not set dim level format for affine expresssion like d0 + d1 on
      // either loop index at d0 or d1.
      // We continue the recursion merely to check whether current affine is
      // admissible or not.
      return findAffine(merger, tid, lvl, binOp.getLHS(), dlt, filterLdx,
                        false) &&
             findAffine(merger, tid, lvl, binOp.getRHS(), dlt, filterLdx,
                        false);
    }
    // Falls through when it is a constant Affine
    return true;
  }
  default:
    return false;
  }
}

/// Helper method to inspect affine expressions for index variable reduction
/// based codegen. It finds the dependent index set for all tensor levels in the
/// current expression we are generating.
///
/// For example, when handling A[i+j][j+k], we build the two way mapping in
/// merger between (tensor, level) pairs and their dependent index variable set:
/// A_0 <=> [i, j] and A_1 <=> [j, k]
///
/// It rejects cases (returns false)
/// 1st, when the same index is used more than once, e.g., A[i+j][i]
/// 2nd, when multiplication is used in the non-trivial index expression.
/// 3rd, when a constant operand is used in the non-trivial index expression.
///
/// TODO: constant should be easy to handle.
static bool findDepIdxSet(Merger &merger, TensorId tensor, Level lvl,
                          AffineExpr a, DimLevelType dlt,
                          bool isSubExp = false) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    const LoopId ldx = merger.makeLoopId(a.cast<AffineDimExpr>().getPosition());
    if (!isUndefDLT(merger.getDimLevelType(tensor, ldx)))
      return false; // used more than once, e.g., A[i][i]

    // TODO: Generalizes the following two cases. A[i] (with trivial index
    // expression) can be treated as a special affine index expression. We do
    // not necessarily need to differentiate them.
    if (!isSubExp)
      merger.setLevelAndType(tensor, ldx, lvl, dlt);

    if (isSubExp) {
      // The current loops appears in more than one affine expressions on the
      // same tensor. We can not handle this case. e.g., A[i+j][i+k], `i` is
      // used twice.
      if (merger.hasDependentLvl(ldx, tensor)) {
        // TODO: This can be supported by coiterate slices if the loop idx is
        // appeared on affine index for different tensor, or take slice on
        // mulitple dimensions when it is on the same tensor.
        // E.g.,
        // `d0 + d1` for indexing t0[lvl0] and `d0 + d2` for indexing t1[lvl0]
        // d0_1 = getNextSliceOffset t0 along lvl0
        // d0_2 = getNextSliceOffset t1 along lvl0
        // if d0_1 == d0_2 then d0 = d0_1 = d0_1
        // else increase min(d0_1, d0_2).
        return false;
      }
      merger.setLoopDependentTensorLevel(ldx, tensor, lvl, dlt);
    }
    return true;
  }
  case AffineExprKind::Constant:
  case AffineExprKind::Mul:
    // TODO: Support Mul and Constant AffineExp for slice-based codegen
    return false;
  case AffineExprKind::Add: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return findDepIdxSet(merger, tensor, lvl, binOp.getLHS(), dlt, true) &&
           findDepIdxSet(merger, tensor, lvl, binOp.getRHS(), dlt, true);
  }
  default:
    return false;
  }
}

/// Get the total number of compound affine expressions in the
/// `getMatchingIndexingMap` for the given tensor.  For the following inputs:
///
/// map = (d0, d1, d2) => (d0 + d1, d2)
/// lvlTypes = ["compressed", "compressed"]
///
/// Returns 1 (because the first level is compressed and its corresponding
/// indexing-expression is `d0 + d1`)
static unsigned getNumNonTrivialIdxExpOnSparseLvls(AffineMap map,
                                                   Value tensor) {
  // The `tensor` is not guaranted to have `RankedTensorType`, therefore
  // we can't use `getRankedTensorType`/`getSparseTensorType` here.
  // However, we don't need to handle `StorageSpecifierType`, so we
  // can use `SparseTensorType` once we guard against non-tensors.
  const auto rtp = tensor.getType().dyn_cast<RankedTensorType>();
  if (!rtp)
    return 0;
  const SparseTensorType stt(rtp);

  // FIXME: There's some dim/lvl confusion here.  The previous version of
  // the code asserted that there are `lvlRank`-many expressions, but then
  // the `exprs[d]` expression assumes there are in fact `dimRank`-many
  // expressions.  Even though `ArrayRef::operator[]` will check for OOB,
  // the mismatch between the assertion and the usage belies that this code
  // cannot support non-permutations.
  //
  // Elsewhere in this file the maps returned by
  // `linalg::GenericOp::getMatchingIndexingMap` are inconsistent about
  // whether they're expected to have `lvlRank`-many or `dimRank`-many
  // expressions (cf., `genSubscript` vs `findSparseAnnotations`);
  // so those are no help in determining which is actually intended.
  //
  // For now we work around this problem by asserting the two ranks agree.
  const Dimension dimRank = stt.getDimRank();
  const Level lvlRank = stt.getLvlRank();
  assert(dimRank == lvlRank && "Non-permutations not currently supported");
  const auto exprs = map.getResults();
  assert(static_cast<Dimension>(exprs.size()) == dimRank &&
         "AffineMap does not have dimension-rank many results");
  (void)dimRank;
  unsigned num = 0;
  for (Level l = 0; l < lvlRank; l++) {
    // FIXME: `toOrigDim` is deprecated.
    const Dimension d = toOrigDim(stt.getEncoding(), l);
    if (!exprs[d].isa<AffineDimExpr>() && !stt.isDenseLvl(l))
      num++;
  }
  return num;
}

/// Get the total number of sparse levels with compound affine
/// expressions, summed over all operands of the `GenericOp`.
static unsigned getNumNonTrivialIdxExpOnSparseLvls(linalg::GenericOp op) {
  unsigned num = 0;
  for (OpOperand &t : op->getOpOperands())
    num += getNumNonTrivialIdxExpOnSparseLvls(op.getMatchingIndexingMap(&t),
                                              t.get());
  return num;
}

static bool hasNonTrivialAffineOnSparseOut(linalg::GenericOp op) {
  OpOperand *out = op.getDpsInitOperand(0);
  if (getSparseTensorType(out->get()).isAllDense())
    return false;
  return getNumNonTrivialIdxExpOnSparseLvls(op.getMatchingIndexingMap(out),
                                            out->get());
}

/// Helper method to inspect sparse encodings in the tensor types.
/// Fills the per-dimension sparsity information for all tensors.
/// Returns true if the sparse annotations and affine subscript
/// expressions of all tensors are admissible. Returns false if
/// no annotations are found or inadmissible constructs occur.
/// We currently support two different ways to handle non-trivial index
/// expression on sparse tensors, and they accept different affine expressions.
/// When using filter-loop-based approach, it accept (almost) arbitrary affine
/// index expression on sparse tensor but it is much less efficient, and will be
/// gradually removed from the codebase.
/// When using dependent index reducton-based approach, it currently only
/// supports affine addition index expression.
static bool findSparseAnnotations(CodegenEnv &env, bool idxReducBased) {
  bool annotated = false;
  // `filterLdx` may be mutated by `findAffine`.
  LoopId filterLdx = env.merger().getStartingFilterLoopId();
  for (OpOperand &t : env.op()->getOpOperands()) {
    const TensorId tid = env.makeTensorId(t.getOperandNumber());
    const auto map = env.op().getMatchingIndexingMap(&t);
    const auto enc = getSparseTensorEncoding(t.get().getType());
    if (enc)
      annotated = true;

    const Level lvlRank = map.getNumResults();
    assert(!enc || lvlRank == enc.getLvlRank());
    assert(static_cast<Level>(env.op().getRank(&t)) == lvlRank);

    // We only need to do index reduction if there is at least one non-trivial
    // index expression on sparse levels.
    // If all non-trivial index expression is on dense levels, we can
    // efficiently rely on the random access to locate the element.
    bool needIdxReduc =
        enc && getNumNonTrivialIdxExpOnSparseLvls(map, t.get()) != 0;
    // If then current tensor being inspected requires affine index, it need
    // to be sliced.
    for (Level l = 0; l < lvlRank; l++) {
      // FIXME: `toOrigDim` is deprecated.
      const AffineExpr a = map.getResult(toOrigDim(enc, l));
      const DimLevelType dlt = enc.getLvlType(l);
      if (idxReducBased && needIdxReduc) {
        if (!findDepIdxSet(env.merger(), tid, l, a, dlt))
          return false; // inadmissible affine expression
      } else {
        if (!findAffine(env.merger(), tid, l, a, dlt, filterLdx))
          return false; // inadmissible affine expression
      }
    }
  }
  assert(filterLdx == env.merger().getNumLoops());
  return annotated;
}

/// A helper to compute a topological sort. O(n^2) time complexity
/// as we use adj matrix for the graph.
/// The sorted result will put the first Reduction iterator to the
/// latest possible `LoopOrd`.
///
/// The `inDegree` is indexed by `LoopId`, and the `adjM` is indexed by
/// `(LoopId,LoopId)`.
static bool topSortOptimal(CodegenEnv &env,
                           ArrayRef<utils::IteratorType> iteratorTypes,
                           std::vector<unsigned> &inDegree,
                           std::vector<std::vector<bool>> &adjM) {
  std::vector<LoopId> redIt;    // reduce iterator with 0 degree
  std::vector<LoopId> parIt;    // parallel iterator with 0 degree
  std::vector<LoopId> filterIt; // filter loop with 0 degree
  const LoopId numLoops = env.merger().getNumLoops();
  for (LoopId i = 0; i < numLoops; i++) {
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
    for (LoopId dst = 0; dst < numLoops; dst++) {
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
  return env.topSortSize() == numLoops;
}

/// Helper method to add all constraints from the indices in one affine
/// expression before all indices in the other affine expression. For
/// example i0+i1 < i2+i3+1 yields i0<i2, i0<i3, i1<i2, and i1<i3.
/// The affine expression `a` is empty iff `fidx` have a value, leading to
/// b = (i0 + i1) < fidx => i0 < fidx, i1 < fidx.
/// The affine expression `b` is empty iff `tidx` have a value, leading to
/// tidx < a = (i0 + i1) => tidx < i0, tidx < i1.
///
/// The `inDegree` is indexed by `LoopId`, and the `adjM` is indexed by
/// `(LoopId,LoopId)`.
static void addAffineOrderings(std::vector<std::vector<bool>> &adjM,
                               std::vector<unsigned> &inDegree, AffineExpr a,
                               AffineExpr b, std::optional<LoopId> fidx,
                               std::optional<LoopId> tidx) {
  if (!a && !b) {
    // Recursion leaf.
    assert(fidx && tidx);
    const LoopId f = *fidx, t = *tidx;
    if (!adjM[f][t]) {
      adjM[f][t] = true;
      inDegree[t]++;
    }
    return;
  }
  // Picks an affine expression and expand (recurse into) it.
  const auto toExpand = a ? a : b;
  switch (toExpand.getKind()) {
  case AffineExprKind::DimId: {
    const std::optional<LoopId> idx{
        toExpand.cast<AffineDimExpr>().getPosition()};
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

static void tryRelaxAffineConstraints(linalg::GenericOp op,
                                      std::optional<LoopId> &fldx,
                                      AffineExpr &fa,
                                      std::optional<LoopId> &tldx,
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
      // adding reduce < parallel ordering.
      finder.setPickedIterType(utils::IteratorType::reduction);
      finder.walkPostOrder(ta);
      ta = finder.getDimExpr();
      tldx = finder.getDimExpr().getPosition();
    }
  }
}

static void addFilterLoopBasedConstraints(CodegenEnv &env, OpOperand &t,
                                          OpOperand *skip, SortMask mask,
                                          std::vector<std::vector<bool>> &adjM,
                                          std::vector<unsigned> &inDegree) {
  // Get map, encoding, and tensor-identifier.
  const auto map = env.op().getMatchingIndexingMap(&t);
  const auto enc = getSparseTensorEncoding(t.get().getType());
  const TensorId tid = env.makeTensorId(t.getOperandNumber());

  // Each tensor expression and optional dimension ordering (row-major
  // by default) puts an ordering constraint on the loop indices. For
  // example, the tensor expresion A_ijk forces the ordering i < j < k
  // on the loop indices if no explicit dimension ordering is given.
  const Level lvlRank = map.getNumResults();
  assert(!enc || lvlRank == enc.getLvlRank());
  for (Level lvl = 0; lvl < lvlRank; lvl++) {
    // FIXME: `toOrigDim` is deprecated.
    AffineExpr ta = map.getResult(toOrigDim(enc, lvl));
    std::optional<LoopId> tldx = env.merger().getLoopId(tid, lvl);
    // Filter loops should be constructed after all the dependent loops,
    // i.e., d0 + d1 < filter_loop(d0 + d1)
    if (tldx && env.merger().isFilterLoop(*tldx)) {
      assert(!ta.isa<AffineDimExpr>() &&
             !isDenseDLT(enc.getDimLevelType()[lvl]));
      addAffineOrderings(adjM, inDegree, ta, AffineExpr(), std::nullopt, tldx);
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

    if (lvl > 0) {
      // FIXME: `toOrigDim` is deprecated.
      AffineExpr fa = map.getResult(toOrigDim(enc, lvl - 1));
      std::optional<LoopId> fldx = env.merger().getLoopId(tid, lvl - 1);

      // Applying order constraints on every pair of dimExpr between two
      // compound affine expressions can sometime too strict:
      // E.g, for [dense, dense] -> (d0 + d1, d2 + d3).
      // It is totally fine to have loop sequence d0->d2->d1->d3 instead of
      // requiring d0 < d2, d1 < d2, d0 < d3, d1 < d3.
      // We also relax the affine constraint when use slice-based algorithm
      // as there is no filter loop for affine index on sparse dimension.
      // TODO: do we really need the condition?
      if (!includesDense(mask))
        tryRelaxAffineConstraints(env.op(), fldx, fa, tldx, ta);

      // (d0 + d1) < (d2 + d3), or
      // filter_loop_d-1 < (d2 + d3), or
      // (d0 + d1) < filter_loop_d, or
      // filter_loop_d-1 < filter_loop_d depending on whether fa/ta is reset
      // above.
      addAffineOrderings(adjM, inDegree, fa, ta, fldx, tldx);
    }
  }
}

static void addSliceBasedConstraints(CodegenEnv &env, OpOperand &t,
                                     OpOperand *skip, SortMask mask,
                                     std::vector<std::vector<bool>> &adjM,
                                     std::vector<unsigned> &inDegree) {
  // Get map and encoding.
  const auto map = env.op().getMatchingIndexingMap(&t);
  const auto enc = getSparseTensorEncoding(t.get().getType());

  // No special treatment for simple indices.
  if (getNumNonTrivialIdxExpOnSparseLvls(map, t.get()) == 0)
    return addFilterLoopBasedConstraints(env, t, skip, mask, adjM, inDegree);

  // Skip tensor during cycle resolution, though order between filter loop
  // and dependent loops need to be guaranteed unconditionally.
  if (&t == skip)
    return;

  AffineDimFinder finder(env.op());
  finder.setPickedIterType(utils::IteratorType::reduction);
  // To compute iteration graph for tensor[d0 + d1 + d3, d4 + d5 + d6],
  // we requires there exist d_x \in {d0, d1, d3} and d_y \in {d4, d5, d6},
  // and d_x > d_y && {d0, d1, d3} - d_x > {d4, d5, d6} - d_y
  const Level lvlRank = map.getNumResults();
  assert(!enc || lvlRank == enc.getLvlRank());
  for (Level lvl = 1; lvl < lvlRank; lvl++) {
    // FIXME: `toOrigDim` is deprecated.
    const AffineExpr fa = map.getResult(toOrigDim(enc, lvl - 1));
    const AffineExpr ta = map.getResult(toOrigDim(enc, lvl));

    // This is a heuristic, we pick an abitrary reduction loop from lhs and
    // rhs and use them as d_x and d_y.
    finder.walkPostOrder(fa);
    const AffineDimExpr fexp = finder.getDimExpr();
    const LoopId fldx = env.makeLoopId(fexp.getPosition());

    finder.walkPostOrder(ta);
    const AffineDimExpr texp = finder.getDimExpr();
    const LoopId tldx = env.makeLoopId(texp.getPosition());

    // d_x > d_y
    if (!adjM[fldx][tldx]) {
      adjM[fldx][tldx] = true;
      inDegree[tldx]++;
    }

    AffineDimCollector fCollector;
    fCollector.walkPostOrder(fa);
    AffineDimCollector tCollector;
    tCollector.walkPostOrder(ta);

    // make sure dx and dy is the last;
    for (auto fd : fCollector.dims) {
      const LoopId f = env.makeLoopId(fd.getPosition());
      if (f == fldx)
        continue;
      if (!adjM[f][fldx]) {
        adjM[f][fldx] = true;
        inDegree[fldx]++;
      }
    }
    for (auto td : tCollector.dims) {
      const LoopId t = env.makeLoopId(td.getPosition());
      if (t == tldx)
        continue;
      if (!adjM[t][tldx]) {
        adjM[t][tldx] = true;
        inDegree[tldx]++;
      }
    }
    // Since we only support affine addition, the order between two dim
    // expression does not really matters.
    // {d0, d1, d3} - d_x > {d4, d5, d6} - d_y
    // This is to ensure that the affine expressions are reduced in sparse
    // tensor level ordering.
    // TODO: this ordering could probably be loosen if we support out-of-order
    // reduction.
    // TODO: the evaluation order need to be ensure to
    // support affine multiplication.
    for (auto fd : fCollector.dims) {
      const LoopId f = env.makeLoopId(fd.getPosition());
      if (f == fldx) // skip d_x
        continue;

      for (auto td : tCollector.dims) {
        const LoopId t = env.makeLoopId(td.getPosition());
        if (t == tldx) // skip d_y
          continue;
        if (!adjM[f][t]) {
          adjM[f][t] = true;
          inDegree[t]++;
        }
      }
    }
  }
}

/// Computes a topologically sorted iteration graph for the linalg operation.
/// Ensures all tensors are visited in natural index order. This is
/// essential for sparse storage formats since these only support access
/// along fixed dimensions. Even for dense storage formats, however, the natural
/// index order yields innermost unit-stride access with better spatial
/// locality.
static bool computeIterationGraph(CodegenEnv &env, SortMask mask,
                                  OpOperand *skip, bool idxReducBased = false) {
  // Set up an n x n from/to adjacency matrix of the iteration graph
  // for the implicit loop indices i_0 .. i_n-1.
  const unsigned numLoops = env.merger().getNumLoops();
  std::vector<std::vector<bool>> adjM(numLoops,
                                      std::vector<bool>(numLoops, false));
  std::vector<unsigned> inDegree(numLoops, 0); // in-degree of each node.
  const auto iteratorTypes = env.op().getIteratorTypesArray();
  // Iterate over the indexing maps of every tensor in the tensor expression.
  for (OpOperand &t : env.op()->getOpOperands()) {
    // Get map and encoding.
    const auto enc = getSparseTensorEncoding(t.get().getType());
    // Skips dense inputs/outputs when not requested.
    const bool isDenseInput = !enc && env.op().isDpsInput(&t);
    const bool isDenseOutput = !enc && !isDenseInput;
    if ((isDenseInput && !includesDenseInput(mask)) ||
        (isDenseOutput && !includesDenseOutput(mask)))
      continue;

    // Push unrelated loops into sparse iteration space, so these
    // will be skipped more often.
    // TODO: Do we really need this?
    if (includesUndef(mask)) {
      const TensorId tid = env.makeTensorId(t.getOperandNumber());
      for (LoopId i = 0; i < numLoops; i++) {
        const auto dltI = env.dlt(tid, i);
        if (isCompressedDLT(dltI) || isSingletonDLT(dltI)) {
          for (LoopId j = 0; j < numLoops; j++)
            if (isUndefDLT(env.dlt(tid, j))) {
              adjM[i][j] = true;
              inDegree[j]++;
            }
        } else {
          assert(isDenseDLT(dltI) || isUndefDLT(dltI));
        }
      }
    }
    // Push unrelated loops into sparse iteration space, so these
    // will be skipped more often.
    if (idxReducBased)
      addSliceBasedConstraints(env, t, skip, mask, adjM, inDegree);
    else
      addFilterLoopBasedConstraints(env, t, skip, mask, adjM, inDegree);
  }
  // Topologically sort the iteration graph to determine loop order.
  // Report failure for a cyclic iteration graph.
  env.topSortClear(numLoops);
  return topSortOptimal(env, iteratorTypes, inDegree, adjM);
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
// FIXME: It's not entirely clear what "index" means here (i.e., is it
// a "coordinate", or "Ldx", or what).  So the function should be renamed
// and/or the documentation expanded in order to clarify.
static Value genIndex(CodegenEnv &env, OpOperand *t) {
  const auto map = env.op().getMatchingIndexingMap(t);
  const auto stt = getSparseTensorType(t->get());
  const Level lvlRank = stt.getLvlRank();
  assert(static_cast<Level>(map.getNumResults()) == lvlRank);
  // FIXME: `toOrigDim` is deprecated.
  // FIXME: above we asserted that there are `lvlRank` many results,
  // but this is assuming there are in fact `dimRank` many results instead.
  const AffineExpr a = map.getResult(toOrigDim(stt.getEncoding(), lvlRank - 1));
  assert(a.getKind() == AffineExprKind::DimId);
  const LoopId idx = env.makeLoopId(a.cast<AffineDimExpr>().getPosition());
  return env.getLoopVar(idx);
}

/// Generates subscript for load/store on a dense or sparse tensor.
static Value genSubscript(CodegenEnv &env, OpBuilder &builder, OpOperand *t,
                          SmallVectorImpl<Value> &args) {
  const Location loc = env.op().getLoc();
  const TensorId tid = env.makeTensorId(t->getOperandNumber());
  const auto map = env.op().getMatchingIndexingMap(t);
  const auto stt = getSparseTensorType(t->get());
  if (stt.hasEncoding()) {
    // For sparse tensors we only push the last-level's position onto `args`.
    const auto pos = env.emitter().getPosits()[tid].back();
    assert(pos);
    args.push_back(pos);
  } else {
    // For dense tensors we push all level's coordinates onto `args`.
    const Level lvlRank = stt.getLvlRank();
    assert(static_cast<Level>(map.getNumResults()) == lvlRank);
    for (Level l = 0; l < lvlRank; l++) {
      const auto lvlExpr = map.getResult(l);
      const auto lvlCrd = env.emitter().genAffine(builder, loc, lvlExpr);
      args.push_back(lvlCrd);
    }
  }
  return env.emitter().getValBuffer()[tid];
}

/// Generates insertion code to implement dynamic tensor load.
static Value genInsertionLoad(CodegenEnv &env, OpBuilder &builder,
                              OpOperand *t) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  // Direct lexicographic coordinate order, tensor loads as zero.
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
  // Direct lexicographic coordinate order, tensor loads as identity.
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
  // Direct insertion in lexicographic coordinate order.
  if (!env.isExpand()) {
    const LoopOrd numLoops = op.getRank(t);
    // TODO: rewrite this to use `env.emitter().getLoopIVs(ivs)`
    // instead.  We just need to either assert that `numLoops ==
    // env.emitter().getCurrentDepth()`, or else update the `getLoopIVs`
    // method to take an optional parameter to restrict to a smaller depth.
    SmallVector<Value> ivs;
    ivs.reserve(numLoops);
    for (LoopOrd n = 0; n < numLoops; n++) {
      const auto iv = env.emitter().getLoopIV(n);
      assert(iv);
      ivs.push_back(iv);
    }
    Value chain = env.getInsertionChain();
    if (!env.getValidLexInsert()) {
      env.updateInsertionChain(builder.create<InsertOp>(loc, rhs, chain, ivs));
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
      Value res = builder.create<InsertOp>(loc, rhs, chain, ivs);
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
static Value genTensorLoad(CodegenEnv &env, OpBuilder &builder, ExprId exp) {
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
static void genTensorStore(CodegenEnv &env, OpBuilder &builder, ExprId exp,
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
      assert(env.exp(exp).kind == TensorExp::Kind::kUnary ||
             env.exp(exp).kind == TensorExp::Kind::kBinary);
    } else if (env.exp(exp).kind == TensorExp::Kind::kSelect) {
      // Select operation insertion.
      Value chain = env.getInsertionChain();
      scf::IfOp ifOp =
          builder.create<scf::IfOp>(loc, chain.getType(), rhs, /*else=*/true);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      // Existing value was preserved to be used here.
      assert(env.exp(exp).val);
      Value v0 = env.exp(exp).val;
      genInsertionStore(env, builder, t, v0);
      env.merger().clearExprValue(exp);
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
inline static Value genInvariantValue(CodegenEnv &env, ExprId exp) {
  return env.exp(exp).val;
}

/// Semi-ring branches are simply inlined by the sparse compiler. Prior
/// analysis has verified that all computations are "local" to the inlined
/// branch or otherwise invariantly defined outside the loop nest, with the
/// exception of index computations, which need to be relinked to actual
/// inlined cloned code.
static Value relinkBranch(CodegenEnv &env, RewriterBase &rewriter, Block *block,
                          Value e, LoopId ldx) {
  if (Operation *def = e.getDefiningOp()) {
    if (auto indexOp = dyn_cast<linalg::IndexOp>(def))
      return env.getLoopVar(env.makeLoopId(indexOp.getDim()));
    if (def->getBlock() == block) {
      for (unsigned i = 0, n = def->getNumOperands(); i < n; i++) {
        rewriter.updateRootInPlace(def, [&]() {
          def->setOperand(
              i, relinkBranch(env, rewriter, block, def->getOperand(i), ldx));
        });
      }
    }
  }
  return e;
}

/// Recursively generates tensor expression.
static Value genExp(CodegenEnv &env, RewriterBase &rewriter, ExprId e,
                    LoopId ldx) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();

  if (e == ::mlir::sparse_tensor::detail::kInvalidId)
    return Value();
  const TensorExp &exp = env.exp(e);
  const auto kind = exp.kind;
  if (kind == TensorExp::Kind::kTensor)
    return genTensorLoad(env, rewriter, e);
  if (kind == TensorExp::Kind::kInvariant)
    return genInvariantValue(env, e);
  if (kind == TensorExp::Kind::kLoopVar)
    return env.getLoopVar(exp.loop);

  if (kind == TensorExp::Kind::kReduce)
    env.startCustomReduc(e); // enter custom

  Value v0 = genExp(env, rewriter, exp.children.e0, ldx);
  Value v1 = genExp(env, rewriter, exp.children.e1, ldx);
  Value ee = env.merger().buildExp(rewriter, loc, e, v0, v1);
  if (ee &&
      (kind == TensorExp::Kind::kUnary || kind == TensorExp::Kind::kBinary ||
       kind == TensorExp::Kind::kBinaryBranch ||
       kind == TensorExp::Kind::kReduce || kind == TensorExp::Kind::kSelect))
    ee = relinkBranch(env, rewriter, ee.getParentBlock(), ee, ldx);

  if (kind == TensorExp::Kind::kReduce)
    env.endCustomReduc(); // exit custom

  if (kind == TensorExp::Kind::kSelect)
    env.merger().setExprValue(e, v0); // Preserve value for later use.

  return ee;
}

/// Hoists loop invariant tensor loads for which indices have been exhausted.
static void genInvariants(CodegenEnv &env, OpBuilder &builder, ExprId exp,
                          LoopId ldx, bool atStart) {
  if (exp == ::mlir::sparse_tensor::detail::kInvalidId)
    return;
  if (env.exp(exp).kind == TensorExp::Kind::kTensor) {
    // Inspect tensor indices.
    bool isAtLoop = ldx == ::mlir::sparse_tensor::detail::kInvalidId;
    linalg::GenericOp op = env.op();
    OpOperand &t = op->getOpOperand(env.exp(exp).tensor);
    const auto map = op.getMatchingIndexingMap(&t);
    const auto stt = getSparseTensorType(t.get());
    const Level lvlRank = stt.getLvlRank();
    assert(static_cast<Level>(map.getNumResults()) == lvlRank);
    for (Level l = 0; l < lvlRank; l++) {
      // FIXME: `toOrigDim` is deprecated.
      // FIXME: above we asserted that there are `lvlRank` many results,
      // but this is assuming there are in fact `dimRank` many results instead.
      const AffineExpr a = map.getResult(toOrigDim(stt.getEncoding(), l));
      const auto sldx =
          env.merger().getLoopId(env.makeTensorId(t.getOperandNumber()), l);
      if (sldx && env.merger().isFilterLoop(*sldx)) {
        if (!env.getLoopVar(*sldx))
          // The filter loops has not been constructed.
          return;
        if (*sldx == ldx)
          isAtLoop = true;
      } else if (!isInvariantAffine(env, a, ldx, isAtLoop))
        return; // still in play
    }
    // All exhausted at this level (isAtLoop denotes exactly at this LoopId).
    if (!isAtLoop)
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
      if (atStart)
        env.merger().setExprValue(exp, genTensorLoad(env, builder, exp));
      else
        env.merger().clearExprValue(exp);
    }
  } else if (env.exp(exp).kind != TensorExp::Kind::kInvariant &&
             env.exp(exp).kind != TensorExp::Kind::kLoopVar) {
    // Traverse into the binary operations. Note that we only hoist
    // tensor loads, since subsequent MLIR/LLVM passes know how to
    // deal with all other kinds of derived loop invariants.
    if (env.exp(exp).kind == TensorExp::Kind::kReduce)
      env.startCustomReduc(exp); // enter custom
    const ExprId e0 = env.exp(exp).children.e0;
    const ExprId e1 = env.exp(exp).children.e1;
    genInvariants(env, builder, e0, ldx, atStart);
    genInvariants(env, builder, e1, ldx, atStart);
    if (env.exp(exp).kind == TensorExp::Kind::kReduce)
      env.endCustomReduc(); // exit custom
  }
}

/// Generates an expanded access pattern in innermost dimension.
static void genExpand(CodegenEnv &env, OpBuilder &builder, LoopOrd at,
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
    for (LoopOrd i = 0; i < at; i++)
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
                         bool isInner, LoopId ldx, ArrayRef<TensorId> tids,
                         ArrayRef<Level> lvls) {
  linalg::GenericOp op = env.op();
  Location loc = op.getLoc();
  auto iteratorTypes = op.getIteratorTypesArray();
  bool isSparse = llvm::any_of(tids, [ldx, &env](TensorId tid) {
    const auto dlt = env.dlt(tid, ldx);
    return isCompressedDLT(dlt) || isSingletonDLT(dlt);
  });

  bool isParallel = isParallelFor(env, isOuter, isSparse);

  Operation *loop = *env.genLoopBoundary([&](MutableArrayRef<Value> reduc) {
    if (env.merger().isFilterLoop(ldx)) {
      const TensorId tid = tids.front();
      const Level lvl = lvls.front();
      // tids/lvls must only have one value because filter loops only
      // corresponding to the one and only sparse tensor level.
      assert(isSparse && tids.size() == 1 && lvls.size() == 1);
      OpOperand *t = &op->getOpOperand(tid);
      auto enc = getSparseTensorEncoding(t->get().getType());
      // Retrieves the affine expression for the filter loop.
      // FIXME: `toOrigDim` is deprecated.
      AffineExpr a =
          op.getMatchingIndexingMap(t).getResult(toOrigDim(enc, lvl));
      return env.emitter().enterFilterLoopOverTensorAtLvl(builder, loc, tid,
                                                          lvl, a, reduc);
    }
    return env.emitter().enterLoopOverTensorAtLvl(builder, loc, tids, lvls,
                                                  reduc, isParallel);
  });
  assert(loop);
  return loop;
}

/// Emit a while-loop for co-iteration over multiple indices.
static Operation *genWhile(CodegenEnv &env, OpBuilder &builder, LoopId idx,
                           bool needsUniv, ArrayRef<TensorId> tids,
                           ArrayRef<Level> lvls) {
  Operation *loop = *env.genLoopBoundary([&](MutableArrayRef<Value> reduc) {
    // Construct the while-loop with a parameter for each
    // index.
    return env.emitter().enterCoIterationOverTensorsAtLvls(
        builder, env.op().getLoc(), tids, lvls, needsUniv, reduc);
  });
  assert(loop);
  return loop;
}

/// Generates a for-loop or a while-loop, depending on whether it implements
/// singleton iteration or co-iteration over the given conjunction.
static Operation *genLoop(CodegenEnv &env, OpBuilder &builder, LoopOrd at,
                          bool needsUniv, ArrayRef<TensorId> tids,
                          ArrayRef<Level> lvls, bool isFor) {
  assert(tids.size() == lvls.size());
  const LoopId idx = env.topSortAt(at);
  if (isFor) {
    bool isOuter = at == 0;
    bool isInner = at == env.topSortSize() - 1;
    return genFor(env, builder, isOuter, isInner, idx, tids, lvls);
  }
  return genWhile(env, builder, idx, needsUniv, tids, lvls);
}

/// Generates the induction structure for a while-loop.
static void finalizeWhileOp(CodegenEnv &env, OpBuilder &builder, LoopId idx,
                            bool needsUniv, scf::WhileOp whileOp) {
  Location loc = env.op().getLoc();
  // Finalize each else branch of all if statements.
  if (env.isReduc() || env.isExpand() || env.getInsertionChain()) {
    while (auto ifOp = dyn_cast_or_null<scf::IfOp>(
               builder.getInsertionBlock()->getParentOp())) {
      // Break on IfOp for slicing filtering.
      if (ifOp->getAttr(LoopEmitter::getLoopEmitterLoopAttrName()) ==
          StringAttr::get(ifOp->getContext(), "slice"))
        break;

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
static scf::IfOp genIf(CodegenEnv &env, OpBuilder &builder, LoopId ldx,
                       LatPointId p) {
  Location loc = env.op().getLoc();
  SmallVector<Type> types;
  Value cond;
  env.merger().foreachTensorLoopId(
      p, /*simple=*/true,
      [&](TensorLoopId b, TensorId tid, std::optional<Level> lvl,
          DimLevelType dlt, bool /*unused*/) {
        assert(ldx == env.merger().loop(b));
        Value clause;
        if (isCompressedDLT(dlt) || isSingletonDLT(dlt)) {
          assert(lvl.has_value());
          const Value crd = env.emitter().getCoords()[tid][*lvl];
          const Value lvar = env.getLoopVar(ldx);
          clause = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 crd, lvar);
        } else {
          assert(isDenseDLT(dlt) || isUndefDLT(dlt));
          clause = constantI1(builder, loc, true);
        }
        cond = cond ? builder.create<arith::AndIOp>(loc, cond, clause) : clause;
      });
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
static bool startLoopSeq(CodegenEnv &env, OpBuilder &builder, ExprId exp,
                         LoopOrd at, LoopId idx, LoopId ldx, LatSetId lts) {
  assert(!env.getLoopVar(idx));
  // Emit invariants at this loop sequence level.
  genInvariants(env, builder, exp, ldx, /*atStart=*/true);
  // Emit access pattern expansion for sparse tensor output.
  genExpand(env, builder, at, /*atStart=*/true);
  // Emit further intitialization at this loop sequence level.
  const LatPointId l0 = env.set(lts)[0];
  bool needsUniv = false;

  SmallVector<TensorId> tids;
  SmallVector<Level> lvls;
  env.merger().foreachTensorLoopId(l0, [&](TensorLoopId b, TensorId tid,
                                           std::optional<Level> lvl,
                                           DimLevelType dlt, bool isIdxReduc) {
    assert(env.merger().loop(b) == idx);
    // FIXME: Dense index reduction can reuse the universal index as well.
    if (!isIdxReduc && (isDenseDLT(dlt) || isUndefDLT(dlt))) {
      needsUniv = true;
    } else {
      // sparse/singleton levels.
      tids.push_back(tid);
      lvls.push_back(*lvl);
    }
  });

  env.emitter().enterNewLoopSeq(builder, env.op().getLoc(), tids, lvls);

  // Maintain the universal index only if it is actually
  // consumed by a subsequent lattice point.
  if (needsUniv) {
    for (const LatPointId li : env.set(lts).drop_front())
      if (!env.merger().hasAnySparse(env.lat(li).simple))
        return true;
  }
  return false;
}

static void genConstantDenseAddressFromLevel(CodegenEnv &env,
                                             OpBuilder &builder, TensorId tid,
                                             Level startLvl) {
  // TODO: Handle affine expression on output tensor.
  linalg::GenericOp op = env.op();
  assert(tid < op.getNumDpsInputs());
  OpOperand *input = op.getDpsInputOperands()[tid];
  const auto lvlExprs = op.getMatchingIndexingMap(input).getResults();
  const auto enc = getSparseTensorEncoding(input->get().getType());
  if (enc) {
    const Location loc = op.getLoc();
    const TensorId tid = env.makeTensorId(input->getOperandNumber());
    const Level lvlRank = enc.getLvlRank();
    assert(lvlExprs.size() == static_cast<size_t>(lvlRank));
    // FIXME: there is dim/lvl confusion here
    for (Level l = startLvl; l < lvlRank; l++) {
      // FIXME: `toOrigDim` is deprecated.
      AffineExpr lvlExpr = lvlExprs[toOrigDim(enc, l)];
      if (enc.isDenseLvl(l) && lvlExpr.isa<AffineConstantExpr>())
        env.emitter().genDenseAffineAddress(builder, loc, tid, l, lvlExpr);
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
  for (TensorId tid = 0, e = env.op().getNumDpsInputs(); tid < e; tid++)
    genConstantDenseAddressFromLevel(env, rewriter, tid, 0);
}

/// Return true if the lattices bit can be iterated by a for loop.
static bool translateBitsToTidLvlPairs(
    CodegenEnv &env, LatPointId li, LoopId ldx, SmallVectorImpl<TensorId> &tids,
    SmallVectorImpl<Level> &lvls, SmallVectorImpl<TensorId> &affineTids,
    SmallVectorImpl<Level> &affineLvls, SmallVectorImpl<AffineExpr> &exps) {
  const BitVector &simple = env.lat(li).simple;
  const TensorId outTid = env.merger().getOutTensorID();
  const std::optional<Level> outLvl = env.merger().getLvl(outTid, ldx);

  unsigned numloopCond = 0;
  bool hasNonUnique = false;

  env.merger().foreachTensorLoopId(
      li, [&, ldx](TensorLoopId b, TensorId tid, std::optional<Level> lvl,
                   DimLevelType dlt, bool isIdxReduc) {
        if (simple[b]) {
          if (isIdxReduc) {
            tids.push_back(tid);
            lvls.push_back(*lvl);
            numloopCond++;
            return;
          }
          if (isUndefDLT(dlt)) {
            // An undefined dlt in the lattices, we probably mean to
            // iterate based on the level of output tensor.  E.g., this
            // could be a synthetic tensor (for invariants and sparse
            // output tensor).
            // out[i][j] = invariant; or a broadcast
            // out[i][j] = in[i] (j is undef for input)
            tid = outTid;
            lvl = outLvl;
            // Skips invalid lvl (e.g., when this is a zero ranked tensor).
            if (!lvl)
              return;
          }
          hasNonUnique = !isUniqueDLT(dlt) || hasNonUnique;
          tids.push_back(tid);
          lvls.push_back(*lvl);
          numloopCond++;
        } else if (isDenseDLT(dlt)) {
          tids.push_back(tid);
          lvls.push_back(*lvl);
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
            // Skip simple affine expression and non-dense levels (which
            // have their own filter loop).
            if (exp.isa<AffineDimExpr>() || !stt.isDenseLvl(l))
              continue;

            // Constant affine expression are handled in genLoop
            if (!exp.isa<AffineConstantExpr>()) {
              bool isAtLoop = false;
              if (isInvariantAffine(env, exp, ldx, isAtLoop) && isAtLoop) {
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
                affineLvls.push_back(l);
                exps.push_back(exp);
              }
            }
          }
        }
      });

  if (isDenseDLT(env.dlt(outTid, ldx))) {
    // Note that we generate dense indices of the output tensor
    // unconditionally, since they may not appear in the lattice, but may be
    // needed for linearized env.
    tids.push_back(outTid);
    lvls.push_back(*outLvl);
  }

  assert(numloopCond > 0);
  // If we just need to one loop conditions and the conditions is not imposed on
  // non-unique level, the loop can be generated by a for loop.
  return numloopCond == 1 && !hasNonUnique;
}

/// Starts a single loop in current sequence.
static std::pair<Operation *, bool> startLoop(CodegenEnv &env,
                                              OpBuilder &builder, LoopOrd at,
                                              LatPointId li, bool needsUniv) {
  // The set of tensors + lvls to generate loops on
  SmallVector<TensorId> tids, affineTids;
  SmallVector<Level> lvls, affineLvls;
  // The set of dense tensors with non-trivial affine expression that just
  // becomes invariant and the address shall now be generated at the current
  // level.
  SmallVector<AffineExpr> affines;
  bool isSingleCond = translateBitsToTidLvlPairs(
      env, li, env.topSortAt(at), tids, lvls, affineTids, affineLvls, affines);

  // Emit the for/while-loop control.
  Operation *loop =
      genLoop(env, builder, at, needsUniv, tids, lvls, isSingleCond);
  Location loc = env.op().getLoc();
  for (auto [tid, lvl, exp] : llvm::zip(affineTids, affineLvls, affines)) {
    env.emitter().genDenseAffineAddress(builder, loc, tid, lvl, exp);
  }

  // Until now, we have entered every <tid, lvl> pair in {cond, extra,
  // affine}Tids/Lvls. The addresses of the upcoming levels which are dependent
  // on constant affines expression may now be determined.
  auto allTids = llvm::concat<TensorId>(tids, affineTids);
  auto allLvls = llvm::concat<Level>(lvls, affineLvls);
  for (auto [tid, lvl] : llvm::zip(allTids, allLvls)) {
    if (tid != env.merger().getOutTensorID())
      genConstantDenseAddressFromLevel(env, builder, tid, lvl + 1);
  }

  return std::make_pair(loop, isSingleCond);
}

/// Ends a single loop in current sequence. Returns new values for needsUniv.
static bool endLoop(CodegenEnv &env, RewriterBase &rewriter, Operation *loop,
                    LoopId idx, LatPointId li, bool needsUniv,
                    bool isSingleCond) {

  if (isSingleCond) {
    // Either a for-loop or a while-loop that iterates over a slice.
    // Any iteration creates a valid lex insert.
    if (env.isReduc() && env.getValidLexInsert())
      env.setValidLexInsert(constantI1(rewriter, env.op().getLoc(), true));
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(loop)) {
    // End a while-loop.
    finalizeWhileOp(env, rewriter, idx, needsUniv, whileOp);
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
  assert(!env.getLoopVar(idx));
  env.emitter().exitCurrentLoopSeq(builder, env.op().getLoc());
  // Unmark bookkeeping of invariants and loop index.
  genInvariants(env, builder, exp, ldx, /*atStart=*/false);
  // Finalize access pattern expansion for sparse tensor output.
  genExpand(env, builder, at, /*atStart=*/false);
}

/// Recursively generates code while computing iteration lattices in order
/// to manage the complexity of implementing co-iteration over unions
/// and intersections of sparse iterations spaces.
static void genStmt(CodegenEnv &env, RewriterBase &rewriter, ExprId exp,
                    LoopOrd at) {
  // At each leaf, assign remaining tensor (sub)expression to output tensor.
  if (at == env.topSortSize()) {
    const LoopId ldx = env.topSortAt(at - 1);
    Value rhs = genExp(env, rewriter, exp, ldx);
    genTensorStore(env, rewriter, exp, rhs);
    return;
  }

  // Construct iteration lattices for current loop index, with L0 at top.
  const LoopId idx = env.topSortAt(at);
  const LoopId ldx = at == 0 ? ::mlir::sparse_tensor::detail::kInvalidId
                             : env.topSortAt(at - 1);
  const LatSetId lts =
      env.merger().optimizeSet(env.merger().buildLattices(exp, idx));

  // Start a loop sequence.
  bool needsUniv = startLoopSeq(env, rewriter, exp, at, idx, ldx, lts);

  // Emit a loop for every lattice point L0 >= Li in this loop sequence.
  //
  // NOTE: We cannot change this to `for (const LatPointId li : env.set(lts))`
  // because the loop body causes data-movement which invalidates
  // the iterator.
  const unsigned lsize = env.set(lts).size();
  for (unsigned i = 0; i < lsize; i++) {
    const LatPointId li = env.set(lts)[i];
    // Start a loop.
    auto [loop, isSingleCond] = startLoop(env, rewriter, at, li, needsUniv);

    // Visit all lattices points with Li >= Lj to generate the
    // loop-body, possibly with if statements for coiteration.
    Value redInput = env.getReduc();
    Value cntInput = env.getExpandCount();
    Value insInput = env.getInsertionChain();
    // NOTE: We cannot change this to `for (const LatPointId lj : env.set(lts))`
    // because the loop body causes data-movement which invalidates the
    // iterator.
    for (unsigned j = 0; j < lsize; j++) {
      const LatPointId lj = env.set(lts)[j];
      const ExprId ej = env.lat(lj).exp;
      if (li == lj || env.merger().latGT(li, lj)) {
        // Recurse into body of each branch.
        if (!isSingleCond) {
          scf::IfOp ifOp = genIf(env, rewriter, idx, lj);
          genStmt(env, rewriter, ej, at + 1);
          endIf(env, rewriter, ifOp, loop, redInput, cntInput, insInput);
        } else {
          genStmt(env, rewriter, ej, at + 1);
        }
      }
    }

    // End a loop.
    needsUniv = endLoop(env, rewriter, loop, idx, li, needsUniv, isSingleCond);
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
    if (op.getNumDpsInits() != 1 || hasNonTrivialAffineOnSparseOut(op))
      return failure();

    // Sets up a code generation environment.
    const unsigned numTensors = op->getNumOperands();
    const unsigned numLoops = op.getNumLoops();
    const unsigned numFilterLoops = getNumNonTrivialIdxExpOnSparseLvls(op);
    // TODO: we should probably always use slice-based codegen whenever
    // possible, we can even intermix slice-based and filter-loop based codegen.
    bool idxReducBased = options.enableIndexReduction && numFilterLoops != 0;

    // If we have indexing map like (d0) -> (0, d0), there might be more
    // levels then loops because of the constant index, that means we can not
    // use numLoops as the upper bound for ranks of all tensors.
    // TODO: Constant indices are currently not support on sparse tensor, but
    // are allowed in non-annotated dense tensor. Support it, it would be
    // required for sparse tensor slice rank reducing too.
    Level maxLvlRank = 0;
    for (auto operand : op.getOperands()) {
      if (auto rtp = operand.getType().dyn_cast<RankedTensorType>()) {
        maxLvlRank = std::max(maxLvlRank, SparseTensorType(rtp).getLvlRank());
      }
    }

    // If we uses slice based algorithm for affine index, we do not need filter
    // loop.
    CodegenEnv env(op, options, numTensors, numLoops,
                   /*numFilterLoops=*/idxReducBased ? 0 : numFilterLoops,
                   maxLvlRank);

    // Detects sparse annotations and translates the per-level sparsity
    // information for all tensors to loop indices in the kernel.
    if (!findSparseAnnotations(env, idxReducBased))
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
    // Ideally (though might not be guaranteed), the eariler a constraint mask
    // can be satisfied, the faster the generated kernel will be.
    const auto allMasks = {
        SortMask::kIncludeAll,        SortMask::kIncludeDense,
        SortMask::kIncludeDenseInput, SortMask::kIncludeDenseOutput,
        SortMask::kIncludeUndef,      SortMask::kSparseOnly};
    for (const SortMask mask : allMasks) {
      if (computeIterationGraph(env, mask, nullptr, idxReducBased)) {
        hasCycle = false;
        if (env.isAdmissibleTopoOrder()) {
          isAdmissible = true;
          break;
        }
        // else try a set of less strict constraints.
      }
    }
    if (hasCycle) {
      return idxReducBased
                 ? failure() // TODO: should cycle be resolved differently?
                 : resolveCycle(env, rewriter); // one last shot
    }

    if (!isAdmissible)
      return failure(); // inadmissible expression, reject

    // Recursively generates code if admissible.
    env.startEmit();
    genBuffers(env, rewriter);
    // TODO: Constant affine expression should be handled differently when using
    // slice-based codegen, it does not matter now becasue we already reject the
    // constant expression at a earlier stage.
    genInitConstantDenseAddress(env, rewriter);
    genStmt(env, rewriter, env.getExprId(), 0);
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
      const TensorId tid = env.makeTensorId(t->getOperandNumber());
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
          srcEnc.getHigherOrdering(), srcEnc.getPosWidth(),
          srcEnc.getCrdWidth());
      auto dstTp = RankedTensorType::get(srcTp.getShape(),
                                         srcTp.getElementType(), dstEnc);
      auto convert = rewriter.create<ConvertOp>(tval.getLoc(), dstTp, tval);
      rewriter.updateRootInPlace(env.op(),
                                 [&]() { env.op()->setOperand(tid, convert); });
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
