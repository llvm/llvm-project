//===- SparseReinterpretMap.cpp - reinterpret sparse tensor maps ----------===/
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils/CodegenUtils.h"
#include "Utils/IterationGraphSorter.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

//===----------------------------------------------------------------------===//
// File Local Helper classes.
//===----------------------------------------------------------------------===//

// CRTP to help implementing a rewriter that demaps all its inputs.
template <typename SubClass, typename SourceOp>
struct DemapInsRewriter : public OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Demaps non-trivial inputs.
    bool changed = false;
    SmallVector<Value> deMappedIns(op->getOperands());
    for (Value &in : deMappedIns) {
      if (auto stt = tryGetSparseTensorType(in); stt && !stt->isIdentity()) {
        in = rewriter.create<ReinterpretMapOp>(loc, stt->getDemappedType(), in);
        changed = true;
      }
    }

    // CRTP call.
    OpAdaptor adaptor(deMappedIns, op);
    LogicalResult status =
        static_cast<const SubClass *>(this)->rewriteOp(op, adaptor, rewriter);
    return changed ? success() : status;
  }
};

// Flattens an affine expression into a list of AffineDimExprs.
struct AffineDimCollector : public AffineExprVisitor<AffineDimCollector> {
  explicit AffineDimCollector(unsigned dimNum) : dims(dimNum){};
  void visitDimExpr(AffineDimExpr expr) { dims.set(expr.getPosition()); }
  BitVector dims;
};

// Flattens an affine expression into a list of AffineDimExprs.
struct AffineExprAdmissibleVisitor
    : public AffineExprVisitor<AffineExprAdmissibleVisitor> {
  explicit AffineExprAdmissibleVisitor(bool isOutput)
      : admissible(true), isOutput(isOutput){};

  // We only allow AffineDimExpr on output.
  void visitAddExpr(AffineBinaryOpExpr expr) {
    if (isOutput)
      admissible = false;
  }
  void visitMulExpr(AffineBinaryOpExpr expr) {
    if (isOutput)
      admissible = false;
  }

  // We disallow mod, floor div and ceil div  on inputs.
  void visitModExpr(AffineBinaryOpExpr expr) { admissible = false; }
  void visitFloorDivExpr(AffineBinaryOpExpr expr) { admissible = false; }
  void visitCeilDivExpr(AffineBinaryOpExpr expr) { admissible = false; }
  operator bool() { return admissible; }

private:
  bool admissible;
  bool isOutput;
};

// The first BitVector stores levels where inadmissible exprs are used.
// The second BitVector stores the AffineDimExp that are used by the
// inadmissible expressions.
using InadmissInfo = std::pair<BitVector, BitVector>;

} // namespace

//===----------------------------------------------------------------------===//
// File Local Helper methods.
//===----------------------------------------------------------------------===//

// Collects the inadmissible affine expression imposed on levels.
static InadmissInfo collectInadmissInfo(AffineMap map, bool isOutput) {
  auto ret = std::make_pair(BitVector(map.getNumResults()),
                            BitVector(map.getNumDims()));
  AffineDimCollector collector(map.getNumDims());
  for (unsigned lvl = 0, e = map.getNumResults(); lvl < e; lvl++) {
    AffineExprAdmissibleVisitor admissible(isOutput);
    admissible.walkPostOrder(map.getResult(lvl));
    if (!admissible) {
      // Record the inadmissible level.
      ret.first.set(lvl);
      // Record the AffineDimExpr that is used in the inadmissible expr.
      collector.walkPostOrder(map.getResult(lvl));
    }
  }
  ret.second = collector.dims;
  return ret;
}

// Builds the AffineMap to replace the idx in idxMap to lvl such that all tht
// inadmissible affine expressions can be eliminated.
// For example, we can rewrite
// idxMap = (d0, d1) -> (d0 floordiv 2, d1 floordiv 3, d0 mod 2, d1 mod 3)
// to
// idxMap = (l0, l1, l2, l3) -> (l0, l1, l2, l3)
// by composing inverse(idxMap), that is
// inverse(idxMap) . idxMap = (l0, l1, l2, l3) -> (l0 * 2 + l2, l1 * 3 + l3)
//                         -> ((l0 * 2 + l2) floordiv 2,
//                             (l1 * 3 + l3) floordiv 3,
//                             (l0 * 2 + l2) mod 2,
//                             (l1 * 3 + l3) mod 3) = (l0, l1, l2, l3)
//
// This function builds the inverse(idxMap) that replace every dimensions used
// in `info` to levels, and updates the iterator type array `itTps` for the new
// index variable introduced.
//
// Note that the returned affine map does not retain the order of the input
// affine map. Instead, it always uses the first `info.inAdlvls.count()` for the
// replaced levels, and remaining ones for unused dimensions.
// For example, to handle
// idxMap = (d0, d1) -> (d0, d1 floordiv 4, d2 mod 4)
// which is a typical map for block_2to4. The function returns:
// inverse(idxMap) = (l0, l1, d0) -> (d0, l0 * 4 + l1)
// in which, (l0, l1) together replaces `d1`, yet they appear
// before `d0` in the resulting affine map.
// The index (loop) order can later be canonicalized by a topo sort.
static AffineMap
genReplaceDimToLvlMap(const InadmissInfo &info, AffineMap idxMap,
                      SmallVector<utils::IteratorType> &itTps) {
  MLIRContext *ctx = idxMap.getContext();
  auto [inAdLvls, usedDims] = info;
  // Note that idxMap does not equal to dim2Lvl map, it is computed by
  // composing idx2Dim(dim2Lvl). They are only equal when idx2Dim is an
  // ID map.
  // TODO: we might fail here, in those case we should really return
  // failure instead of assertion error.
  auto lvl2Idx = inferLvlToDim(idxMap, ctx);

  assert(lvl2Idx.getNumResults() <= idxMap.getNumDims());
  if (lvl2Idx.getNumResults() != idxMap.getNumDims()) {
    // This could happen when some dimensions are projected.
    // E.g., idx2Lvl = (*i*, j, k) -> (j, k)
    //   ==> lvl2Idx = (j, k) -> (j, k)
    // In this case, we append the unused dimesion at the end.
    //   ==> lvl2Idx = (j, k, *i*) -> (*i*, j, k)
    SmallVector<AffineExpr> results;
    AffineDimCollector usedInLvl(idxMap.getNumDims());
    for (auto e : idxMap.getResults())
      usedInLvl.walkPostOrder(e);

    unsigned curUsedDimID = 0;
    unsigned curUnusedDimID = lvl2Idx.getNumDims();

    BitVector unused = usedInLvl.dims.flip();
    for (unsigned i = 0; i < idxMap.getNumDims(); i++) {
      if (unused.test(i))
        results.push_back(getAffineDimExpr(curUnusedDimID++, ctx));
      else
        results.push_back(lvl2Idx.getResult(curUsedDimID++));
    }
    lvl2Idx =
        AffineMap::get(lvl2Idx.getNumDims() + unused.count(), 0, results, ctx);
  }
  assert(lvl2Idx.getNumResults() == idxMap.getNumDims());

  // We do not need to replace the DimExpr that is not used in inadmissible
  // level expressions. We use the first inAdLvl.count() dim to represent the
  // replaced level, the remainings are reserved for unchanged ones.
  // Note that results from the inverse map computed previously does not follow
  // the convention we used, and we need to fix the mismatch below.
  unsigned curRepID = 0;
  unsigned curOriID = inAdLvls.count();
  SmallVector<AffineExpr> results;
  SmallVector<AffineExpr> dimRep(idxMap.getNumResults(), AffineExpr());
  SmallVector<utils::IteratorType> transItTps;

  for (unsigned l : inAdLvls.set_bits()) {
    // By our convention, the inadmissible level `l` always appears in the
    // leading part (accumulated by curRepID) of the affine map's parameter
    // list. Record the mapping so that we can replace all the uses of `l` to
    // the correct position after the translation.
    dimRep[l] = getAffineDimExpr(curRepID++, ctx);
    // A new index variable is introduced for the inadmissible level, inherit
    // the iterator type. E.g., if l0 = d0 floordiv 2, the
    // iterator type of l0 equals to the iterator type of d0.
    AffineExpr lvlExp = idxMap.getResult(l);
    AffineDimCollector collector(idxMap.getNumDims());
    collector.walkPostOrder(lvlExp);
    // We assumes a level can only be derived from one dimension.
    assert(collector.dims.count() == 1);
    transItTps.push_back(itTps[collector.dims.find_first()]);
  }

  for (unsigned d = 0, e = idxMap.getNumDims(); d < e; d++) {
    if (usedDims.test(d)) {
      // The dimension is used in some of the inadmissible levels, and it need
      // to be inversed. Get the inversion from the inverse map, and fix the
      // mismatch captured by the above loop.
      results.push_back(lvl2Idx.getResult(d).replaceDims(dimRep));
    } else {
      // The dimension is not used in any of the inadmissible levels, and it
      // does not need to be inversed. Fix the mismatch by mapping it to the
      // trailing part of the affine map (accumulated by curOriID).
      results.push_back(getAffineDimExpr(curOriID++, ctx));
      transItTps.push_back(itTps[d]);
    }
  }
  unsigned numDim = idxMap.getNumDims() - usedDims.count() + inAdLvls.count();
  // Update iterator type.
  itTps.assign(transItTps.begin(), transItTps.end());
  return AffineMap::get(numDim, 0, results, ctx);
}

// Translates the index map in the linalg::GenericOp from idx->dim map to
// idx->lvl map. Returns failure if the index map can not be translated to an
// admissible form.
// Returns the translated index map array and the iterator type array.
static std::optional<std::pair<ArrayAttr, ArrayAttr>>
translateMap(linalg::GenericOp op, PatternRewriter &rewriter) {
  // idxMap is a idx2dim map before reinterpretation.
  MLIRContext *ctx = op.getContext();
  SmallVector<AffineMap> idxMapArray = op.getIndexingMapsArray();
  SmallVector<utils::IteratorType> itTps = op.getIteratorTypesArray();
  for (unsigned i = 0, e = idxMapArray.size(); i < e; i++) {
    Value tensor = op->getOpOperand(i).get();
    auto stt = tryGetSparseTensorType(tensor);
    if (stt && !stt->isIdentity()) {
      AffineMap dim2Lvl = stt->getDimToLvl();
      // By composing the idx2dim(dim2lvl), we got a idx2lvl Map
      idxMapArray[i] = dim2Lvl.compose(idxMapArray[i]);
    }
  }

  // A naive way to handle common constant expressions that arise during dim2lvl
  // translation.
  auto populateCstMapping = [ctx](DenseMap<AffineExpr, AffineExpr> &cstMapping,
                                  unsigned pos, int64_t lvlSz) {
    if (!ShapedType::isDynamic(lvlSz)) {
      auto c0 = getAffineConstantExpr(0, ctx);
      auto lvlExp = getAffineDimExpr(pos, ctx);
      auto szExp = getAffineConstantExpr(lvlSz, ctx);

      // lvl floordiv lvlSz = 0
      auto divExp =
          getAffineBinaryOpExpr(AffineExprKind::FloorDiv, lvlExp, szExp);
      cstMapping.try_emplace(divExp, c0);

      // lvl mod lvlSz = lvl
      auto modExp = getAffineBinaryOpExpr(AffineExprKind::Mod, lvlExp, szExp);
      cstMapping.try_emplace(modExp, lvlExp);
    }
  };

  unsigned boundedNum = 0;
  // A fixed-point algorithm.
  bool changed = true;
  while (changed) {
    changed = false;
    for (OpOperand &operand : op->getOpOperands()) {
      auto stt = tryGetSparseTensorType(operand.get());
      // Skip on dense operands.
      if (!stt || !stt->getEncoding())
        continue;

      unsigned tid = operand.getOperandNumber();
      bool isOutput = &operand == op.getDpsInitOperand(0);
      AffineMap idxMap = idxMapArray[tid];
      InadmissInfo inAdInfo = collectInadmissInfo(idxMap, isOutput);
      auto [inAdLvls, dimExprs] = inAdInfo;
      for (unsigned d : dimExprs.set_bits()) {
        // The first `boundedNum` used in the AffineMap is introduced to
        // resolve previous inadmissible expressions. We can not replace them
        // as it might bring back the inadmissible expressions.
        if (d < boundedNum)
          return std::nullopt;
      }

      if (inAdLvls.count() != 0) {
        // Naive constant progagation, should be sufficient to handle block
        // sparsity in our cases.
        SmallVector<int64_t> lvlShape = stt->getLvlShape();
        DenseMap<AffineExpr, AffineExpr> cstMapping;
        unsigned position = 0;
        for (unsigned lvl : inAdLvls.set_bits()) {
          int64_t lvlSz = lvlShape[lvl];
          populateCstMapping(cstMapping, position, lvlSz);
          position++;
        }

        AffineMap lvl2Idx = genReplaceDimToLvlMap(inAdInfo, idxMap, itTps);
        // Compose the lvl2Idx Map to all AffineIdxMap to eliminate
        // inadmissible expressions.
        for (unsigned tid = 0, e = idxMapArray.size(); tid < e; tid++) {
          AffineMap transMap = idxMapArray[tid].compose(lvl2Idx);
          idxMapArray[tid] = transMap.replace(
              cstMapping, /*numResultDims=*/transMap.getNumDims(),
              /*numResultSyms=*/0);
        }
        changed = true;
        boundedNum += inAdLvls.count();
      }
    }
  };

  SmallVector<Attribute> iterAttr =
      llvm::map_to_vector(itTps, [ctx](auto itTp) -> Attribute {
        return linalg::IteratorTypeAttr::get(ctx, itTp);
      });

  return std::make_pair(rewriter.getAffineMapArrayAttr(idxMapArray),
                        rewriter.getArrayAttr(iterAttr));
}

// Generates a "de"mapping reinterpretation of the map.
static Value genDemap(OpBuilder &builder, SparseTensorEncodingAttr enc,
                      Value val) {
  return builder.create<ReinterpretMapOp>(val.getLoc(), enc.withoutDimToLvl(),
                                          val);
}

// Generates a "re"mapping reinterpretation of the map.
static Value genRemap(OpBuilder &builder, SparseTensorEncodingAttr enc,
                      Value val) {
  return builder.create<ReinterpretMapOp>(val.getLoc(), enc, val);
}

static SmallVector<Value> remapValueRange(OpBuilder &rewriter, TypeRange types,
                                          ValueRange outs) {
  SmallVector<Value> ret(outs);
  assert(outs.size() == types.size());
  for (auto [r, t] : llvm::zip(ret, types))
    if (r.getType() != t)
      r = rewriter.create<ReinterpretMapOp>(r.getLoc(), t, r);
  return ret;
}

namespace {

//===----------------------------------------------------------------------===//
// Rewriting rules for linalg generic ops.
//===----------------------------------------------------------------------===//

/// Sparse rewriting rule for the generic `linalg` operation.
struct GenericOpReinterpretMap
    : public DemapInsRewriter<GenericOpReinterpretMap, linalg::GenericOp> {
public:
  using DemapInsRewriter::DemapInsRewriter;
  LogicalResult rewriteOp(linalg::GenericOp linalgOp, OpAdaptor adaptor,
                          PatternRewriter &rewriter) const {
    // Only rewrite single output operations with pure (sparse) tensor
    // semantics.
    if (linalgOp.getNumDpsInits() != 1 || !linalgOp.hasPureTensorSemantics() ||
        !hasAnySparseOperandOrResult(linalgOp) ||
        !hasAnyNonIdentityOperandsOrResults(linalgOp))
      return failure();

    // Try translating the index map.
    auto transMap = translateMap(linalgOp, rewriter);
    if (!transMap)
      return rewriter.notifyMatchFailure(
          linalgOp, "the sparse kernel can not be sparsified.");

    // On success, replace update the linalg operands and maps in place.
    Value res = linalgOp.getResult(0);
    auto stt = tryGetSparseTensorType(res);
    auto [idxMap, itTp] = *transMap;

    rewriter.startOpModification(linalgOp);
    linalgOp.setIndexingMapsAttr(idxMap);
    linalgOp.setIteratorTypesAttr(itTp);
    // Use demapped arguments.
    linalgOp.getInputsMutable().assign(adaptor.getInputs());
    linalgOp.getDpsInitsMutable().assign(adaptor.getOutputs());
    res.setType(adaptor.getOutputs()[0].getType());
    rewriter.finalizeOpModification(linalgOp);

    rewriter.setInsertionPointAfter(linalgOp);
    if (stt && stt->hasEncoding()) {
      Value t = genRemap(rewriter, stt->getEncoding(), res);
      rewriter.replaceAllUsesExcept(res, t, t.getDefiningOp());
    }
    return success();
  }
};

struct GenericOpScheduler : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (linalgOp.getNumDpsInits() != 1 || !linalgOp.hasPureTensorSemantics() ||
        hasAnyNonIdentityOperandsOrResults(linalgOp) || // need demap first
        !hasAnySparseOperandOrResult(linalgOp)) {
      return failure();
    }

    const StringRef sorted = "sorted";
    if (linalgOp->hasAttr(sorted))
      return failure();

    auto scheduler = IterationGraphSorter::fromGenericOp(linalgOp);
    bool isAdmissible = false;
    AffineMap order;
    // A const list of all masks that we used for iteration graph
    // computation. Must be ordered from more strict to less strict.
    // Ideally (though might not be guaranteed), the earlier a constraint mask
    // can be satisfied, the faster the generated kernel will be.
    const auto allMasks = {SortMask::kIncludeAll, SortMask::kIncludeDense,
                           SortMask::kIncludeDenseInput,
                           SortMask::kIncludeDenseOutput,
                           SortMask::kSparseOnly};
    for (const SortMask mask : allMasks) {
      order = scheduler.sort(mask);
      if (order) {
        if (isAdmissibleOrder(linalgOp, order)) {
          isAdmissible = true;
          break;
        }
        // else try a set of less strict constraints.
      }
    }

    if (!order) {
      // Cycles detected.
      if (failed(resolveCycle(scheduler, linalgOp, rewriter))) {
        return rewriter.notifyMatchFailure(
            linalgOp, "the sparse kernel can not be scheduled: loop detected.");
      }
      return success();
    }

    if (!isAdmissible) {
      return rewriter.notifyMatchFailure(
          linalgOp, "the sparse kernel can not be scheduled.");
    }

    // Marks the GenericOp to avoid recursive matching.
    rewriter.modifyOpInPlace(linalgOp, [&]() {
      linalgOp->setAttr(sorted, rewriter.getBoolAttr(true));
    });

    // Already sorted.
    if (order.isIdentity())
      return success();

    assert(order.isPermutation());
    // `order` is orignial loop -> sorted loop map
    ArrayAttr preItTypes = linalgOp.getIteratorTypesAttr();
    SmallVector<Attribute> curItTypes;
    curItTypes.reserve(preItTypes.size());
    for (AffineExpr expr : order.getResults()) {
      unsigned loopID = llvm::cast<AffineDimExpr>(expr).getPosition();
      curItTypes.push_back(preItTypes[loopID]);
    }

    // Inverse `order` to get sorted loop -> original loop map
    order = inversePermutation(order);
    SmallVector<AffineMap> idxMaps = linalgOp.getIndexingMapsArray();
    for (AffineMap &idxMap : idxMaps)
      idxMap = idxMap.compose(order); // sorted loop -> lvl map

    rewriter.startOpModification(linalgOp);
    linalgOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(idxMaps));
    linalgOp.setIteratorTypesAttr(rewriter.getArrayAttr(curItTypes));
    rewriter.finalizeOpModification(linalgOp);

    return success();
  }

private:
  /// Whether the loop order is admissible by sparsification.
  static bool isAdmissibleOrder(linalg::GenericOp linalgOp, AffineMap order) {
    if (!hasAnySparseResult(linalgOp))
      return true;

    OpOperand *lhs = linalgOp.getDpsInitOperand(0);
    unsigned nest = 0;
    const auto iteratorTypes = linalgOp.getIteratorTypesArray();
    for (const AffineExpr l : order.getResults()) {
      unsigned loopId = llvm::cast<AffineDimExpr>(l).getPosition();
      auto itTp =
          linalgOp.getIteratorTypes()[loopId].cast<linalg::IteratorTypeAttr>();
      if (linalg::isReductionIterator(itTp.getValue()))
        break; // terminate at first reduction
      nest++;
    }
    // Determine admissible dynamic insertion situations:
    // (1) fully injective, since there are no reductions,
    // (2) admissible 1-d expansion in innermost dimension.
    return static_cast<int64_t>(nest) >= linalgOp.getRank(lhs) - 1;
  };

  // Last resort cycle resolution.
  static LogicalResult resolveCycle(IterationGraphSorter &scheduler,
                                    linalg::LinalgOp linalgOp,
                                    PatternRewriter &rewriter) {
    // Compute topological sort while leaving out every sparse input tensor in
    // succession until an acylic iteration graph results.
    for (OpOperand *t : linalgOp.getDpsInputOperands()) {
      Value tval = t->get();
      auto srcEnc = getSparseTensorEncoding(tval.getType());
      // The constraints introduced by compound index expression are
      // complicated. Skip them.
      AffineMap idxMap = linalgOp.getMatchingIndexingMap(t);
      bool hasCompExpr = llvm::any_of(idxMap.getResults(), [](AffineExpr exp) {
        return !llvm::isa<AffineDimExpr>(exp);
      });
      if (!srcEnc || hasCompExpr)
        continue;

      // Try scheduling loop without constraints from `tval`.
      AffineMap order = scheduler.sort(SortMask::kSparseOnly, tval);
      if (!order) // still cyclic
        continue;

      // Found an input tensor that resolves the cycle by inserting a
      // conversion into a sparse tensor that adheres to the iteration
      // graph order.
      auto stt = getSparseTensorType(tval);
      assert(stt.isIdentity());
      order = inversePermutation(order);
      // sorted loop -> lvl map.
      idxMap = idxMap.compose(order);

      // Found a permutation such that the results in `idxMap` is sorted.
      // For example,
      //  (d0, d1, d2, d3) -> (d2, d1, d0)
      // loops are scheduled in order of d0->d1->d2->d3, to resolve the cycle,
      // we find a permutation, perm(d2, d1, d0) -> (d0, d1, d2), such that the
      // transposed tensor's levels are visited in the same order as the loop
      // scheduling order.
      SmallVector<std::pair<unsigned, unsigned>> lvlSeq;
      for (AffineExpr expr : idxMap.getResults()) {
        unsigned lvl = llvm::cast<AffineDimExpr>(expr).getPosition();
        lvlSeq.push_back(std::make_pair(lvl, lvlSeq.size()));
      }
      std::sort(lvlSeq.begin(), lvlSeq.end(), [](auto &lhs, auto &rhs) -> bool {
        return lhs.first < rhs.first;
      });
      SmallVector<unsigned> perm =
          llvm::to_vector(llvm::make_second_range(lvlSeq));
      auto dimToLvl = AffineMap::getPermutationMap(perm, linalgOp.getContext());
      // The result of the idxMap must be unsorted.
      assert(!dimToLvl.isIdentity());

      // Inserting the transpose
      rewriter.setInsertionPoint(linalgOp);
      RankedTensorType dstTp = stt.withDimToLvl(dimToLvl).getRankedTensorType();
      Value dst = rewriter.create<ConvertOp>(tval.getLoc(), dstTp, tval);
      rewriter.modifyOpInPlace(linalgOp, [&]() {
        linalgOp->setOperand(t->getOperandNumber(), dst);
      });
      return success();
    }
    // Cannot be resolved with a single conversion.
    // TODO: convert more than one?
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Reinterpret Map Rewriters for operations other than linalg.generics
//===----------------------------------------------------------------------===//

template <typename AllocOp>
struct TensorAllocDemapper : public OpRewritePattern<AllocOp> {
  using OpRewritePattern<AllocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AllocOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasAnyNonIdentityOperandsOrResults(op))
      return failure();

    Location loc = op.getLoc();
    auto stt = getSparseTensorType(op.getResult());

    SmallVector<Value> maxDimCrds;
    maxDimCrds.reserve(stt.getDimRank());
    ValueRange dynSz = op.getDynamicSizes();
    for (int64_t dimSz : stt.getDimShape()) {
      if (ShapedType::isDynamic(dimSz)) {
        Value maxCrd = rewriter.create<arith::SubIOp>(
            loc, dynSz.front(), constantIndex(rewriter, loc, 1));
        maxDimCrds.push_back(maxCrd);
        dynSz = dynSz.drop_front();
      } else {
        maxDimCrds.push_back(constantIndex(rewriter, loc, dimSz - 1));
      }
    }

    ValueRange maxLvlCrds = stt.translateCrds(rewriter, loc, maxDimCrds,
                                              CrdTransDirectionKind::dim2lvl);
    auto lvlShape = stt.getLvlShape();
    SmallVector<Value> dynLvlSzs;
    for (unsigned i = 0, e = lvlShape.size(); i < e; i++) {
      if (ShapedType::isDynamic(lvlShape[i])) {
        Value sz = rewriter.create<arith::AddIOp>(
            loc, maxLvlCrds[i], constantIndex(rewriter, loc, 1));
        dynLvlSzs.push_back(sz);
      }
    }

    assert(dynSz.empty()); // should have consumed all.
    rewriter.startOpModification(op);
    op->setOperands(dynLvlSzs);
    op.getResult().setType(stt.getDemappedType());
    rewriter.finalizeOpModification(op);
    rewriter.setInsertionPointAfter(op);

    Value t = genRemap(rewriter, stt.getEncoding(), op.getResult());
    rewriter.replaceAllUsesExcept(op.getResult(), t, t.getDefiningOp());
    return success();
  }
};

struct TensorInsertDemapper
    : public DemapInsRewriter<TensorInsertDemapper, tensor::InsertOp> {
  using DemapInsRewriter::DemapInsRewriter;
  LogicalResult rewriteOp(tensor::InsertOp op, OpAdaptor adaptor,
                          PatternRewriter &rewriter) const {
    if (!hasAnySparseResult(op))
      return failure();

    Location loc = op.getLoc();
    auto stt = getSparseTensorType(op.getResult());
    ValueRange lvlCrd = stt.translateCrds(rewriter, loc, op.getIndices(),
                                          CrdTransDirectionKind::dim2lvl);
    auto insertOp = rewriter.create<sparse_tensor::InsertOp>(
        loc, op.getScalar(), adaptor.getDest(), lvlCrd);

    Value out = genRemap(rewriter, stt.getEncoding(), insertOp.getResult());
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct SparseAssembleDemapper : public OpRewritePattern<AssembleOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AssembleOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasAnyNonIdentityOperandsOrResults(op))
      return failure();

    assert(hasAnySparseResult(op));
    auto stt = getSparseTensorType(op.getResult());
    rewriter.modifyOpInPlace(
        op, [&op, &stt]() { op.getResult().setType(stt.getDemappedType()); });
    rewriter.setInsertionPointAfter(op);
    Value out = genRemap(rewriter, stt.getEncoding(), op.getResult());
    rewriter.replaceAllUsesExcept(op, out, out.getDefiningOp());
    return success();
  }
};

struct SparseDisassembleDemapper
    : public DemapInsRewriter<SparseDisassembleDemapper, DisassembleOp> {
  using DemapInsRewriter::DemapInsRewriter;
  LogicalResult rewriteOp(DisassembleOp op, OpAdaptor adaptor,
                          PatternRewriter &rewriter) const {
    if (!hasAnyNonIdentityOperandsOrResults(op))
      return failure();

    assert(hasAnySparseOperandOrResult(op));
    rewriter.modifyOpInPlace(op, [&op, &adaptor]() {
      op.getTensorMutable().assign(adaptor.getTensor());
    });
    return success();
  }
};

struct ForeachOpDemapper
    : public DemapInsRewriter<ForeachOpDemapper, ForeachOp> {
  using DemapInsRewriter::DemapInsRewriter;
  LogicalResult rewriteOp(ForeachOp op, OpAdaptor adaptor,
                          PatternRewriter &rewriter) const {
    // Only handle operations with sparse input/output with non-identity dim2lvl
    // maps.
    if (!hasAnyNonIdentityOperandsOrResults(op))
      return failure();

    // TODO: demap constant as well.
    if (auto constOp = op.getTensor().getDefiningOp<arith::ConstantOp>())
      if (auto attr = dyn_cast<SparseElementsAttr>(constOp.getValue()))
        return failure();

    Location loc = op.getLoc();
    // Cache the type information since we update the foreach op in-place.
    auto srcStt = getSparseTensorType(op.getTensor());
    SmallVector<Type> prevRetTps(op.getResultTypes());

    rewriter.startOpModification(op);
    op.getTensorMutable().assign(adaptor.getTensor());
    op.getInitArgsMutable().assign(adaptor.getInitArgs());
    // Update results' types.
    for (auto r : op.getResults())
      if (auto stt = tryGetSparseTensorType(r); stt && !stt->isIdentity())
        r.setType(stt->getDemappedType());

    Level lvlRank = getSparseTensorType(adaptor.getTensor()).getLvlRank();
    // Update the foreach body.
    SmallVector<Type> blockArgTps(lvlRank, rewriter.getIndexType());
    blockArgTps.push_back(srcStt.getElementType());
    blockArgTps.append(adaptor.getInitArgs().getTypes().begin(),
                       adaptor.getInitArgs().getTypes().end());
    Block *body = op.getBody();
    // Block Args: [dimCrd, val, initArgs]
    unsigned preArgNum = body->getNumArguments();
    for (Type t : blockArgTps)
      body->addArgument(t, loc);

    // Block Args: [dimCrd, val, initArgs, lvlCrds, val, DemappedArgs]
    rewriter.setInsertionPointToStart(body);
    ValueRange lvlCrds = body->getArguments().slice(preArgNum, lvlRank);

    ValueRange dimCrds = srcStt.translateCrds(rewriter, loc, lvlCrds,
                                              CrdTransDirectionKind::lvl2dim);
    rewriter.replaceAllUsesWith(
        body->getArguments().take_front(srcStt.getDimRank()), dimCrds);
    body->eraseArguments(0, srcStt.getDimRank());
    // Block Args: [val, initArgs, lvlCrds, val, DemappedArgs]
    unsigned numInitArgs = op.getInitArgs().size();
    rewriter.replaceAllUsesWith(body->getArgument(0),
                                body->getArgument(lvlRank + numInitArgs + 1));
    body->eraseArgument(0);
    // Block Args: [initArgs, lvlCrds, val, DemappedArgs]
    ValueRange srcArgs = body->getArguments().take_front(numInitArgs);
    ValueRange dstArgs = body->getArguments().take_back(numInitArgs);
    // Remap back before replacement.
    SmallVector<Value> reMappedArgs =
        remapValueRange(rewriter, srcArgs.getTypes(), dstArgs);
    rewriter.replaceAllUsesWith(srcArgs, reMappedArgs);
    body->eraseArguments(0, numInitArgs);
    // Block Args: [lvlCrds, DemappedArgs] and we are done.

    // Update yield operations.
    if (numInitArgs != 0) {
      rewriter.setInsertionPointToEnd(body);
      auto yield = llvm::cast<YieldOp>(body->getTerminator());
      if (auto stt = tryGetSparseTensorType(yield.getResult());
          stt && !stt->isIdentity()) {
        Value y = genDemap(rewriter, stt->getEncoding(), yield.getResult());
        rewriter.create<YieldOp>(loc, y);
        rewriter.eraseOp(yield);
      }
    }
    rewriter.finalizeOpModification(op);

    rewriter.setInsertionPointAfter(op);
    SmallVector<Value> outs =
        remapValueRange(rewriter, prevRetTps, op.getResults());

    // Replace all the uses of the foreach results, expect the use in
    // reinterpret_map used to remap the output.
    for (auto [from, to] : llvm::zip(op.getResults(), outs))
      rewriter.replaceAllUsesExcept(from, to, to.getDefiningOp());

    return success();
  }
};

} // namespace

void mlir::populateSparseReinterpretMap(RewritePatternSet &patterns,
                                        ReinterpretMapScope scope) {
  if (scope == ReinterpretMapScope::kAll ||
      scope == ReinterpretMapScope::kGenericOnly) {
    patterns.add<GenericOpReinterpretMap, GenericOpScheduler>(
        patterns.getContext());
  }
  if (scope == ReinterpretMapScope::kAll ||
      scope == ReinterpretMapScope::kExceptGeneric) {
    patterns.add<TensorAllocDemapper<bufferization::AllocTensorOp>,
                 TensorAllocDemapper<tensor::EmptyOp>, SparseAssembleDemapper,
                 SparseDisassembleDemapper, TensorInsertDemapper,
                 ForeachOpDemapper>(patterns.getContext());
  }
}
