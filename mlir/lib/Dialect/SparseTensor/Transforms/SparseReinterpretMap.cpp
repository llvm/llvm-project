//===- SparseReinterpretMap.cpp - reinterpret sparse tensor maps ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
// File Local Helper methods.
//===----------------------------------------------------------------------===//

// Translates a "simple" map according to an identity lvl-map.
static AffineMap translateMap(OpBuilder &builder, SparseTensorType stt,
                              AffineMap map) {
  unsigned lvlRank = stt.getLvlRank();
  AffineMap lvl2dim = stt.getLvlToDim();
  assert(lvl2dim.getNumInputs() == lvlRank);
  SmallVector<AffineExpr> exps;
  for (unsigned i = 0, n = map.getNumResults(); i < n; i++) {
    unsigned pos = map.getResult(i).cast<AffineDimExpr>().getPosition();
    exps.push_back(lvl2dim.getResult(pos));
  }
  return AffineMap::get(lvlRank, 0, exps, builder.getContext());
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

/// Whether the operation has any sparse tensor with non-identity dim2lvl maps.
static bool hasNonIdentityOperandsOrResults(Operation *op) {
  auto hasNonIdentityMap = [](Value v) {
    auto stt = tryGetSparseTensorType(v);
    return stt && !stt->isIdentity();
  };

  return llvm::any_of(op->getOperands(), hasNonIdentityMap) ||
         llvm::any_of(op->getResults(), hasNonIdentityMap);
}

// Generates a clone of the given linalg generic operation, but with
// remapped arguments, index maps, and iteration types.
//
// TODO: As decribed below, this is proof-of-concept code which makes a lot
//       of simplifying assumptions for now.
//
static linalg::GenericOp genGenericLinalg(PatternRewriter &rewriter,
                                          linalg::GenericOp linalgOp,
                                          SparseTensorType stt, Value out) {
  unsigned dimRank = stt.getDimRank();
  unsigned lvlRank = stt.getLvlRank();
  SmallVector<Value> inputOps = linalgOp.getInputs();
  SmallVector<Value> outputOps = {out};
  SmallVector<AffineMap> indexMaps;
  SmallVector<utils::IteratorType> iterTypes;
  // Translate the index maps, except output map, which is lvl-identity.
  auto maps = linalgOp.getIndexingMapsArray();
  for (unsigned i = 0, n = maps.size() - 1; i < n; i++)
    indexMaps.push_back(translateMap(rewriter, stt, maps[i]));
  indexMaps.push_back(
      AffineMap::getMultiDimIdentityMap(lvlRank, rewriter.getContext()));
  // Add additional "parallel" iteration types at the top.
  for (unsigned i = 0, diff = lvlRank = dimRank; i < diff; i++)
    iterTypes.push_back(utils::IteratorType::parallel);
  for (auto &i : linalgOp.getIteratorTypesArray())
    iterTypes.push_back(i);
  // Generate the new linalg generic operation and clone body.
  auto newOp = rewriter.create<linalg::GenericOp>(
      linalgOp.getLoc(), out.getType(), inputOps, outputOps, indexMaps,
      iterTypes);
  rewriter.cloneRegionBefore(linalgOp.getRegion(), newOp.getRegion(),
                             newOp.getRegion().begin());
  return newOp;
}

namespace {

//===----------------------------------------------------------------------===//
// Rewriting rules for linalg generic ops.
//===----------------------------------------------------------------------===//

/// Sparse rewriting rule for the generic `linalg` operation.
struct GenericOpReinterpretMap : public OpRewritePattern<linalg::GenericOp> {
public:
  GenericOpReinterpretMap(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context) {}

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // Only rewrite single output operations with pure tensor semantics.
    if (linalgOp.getNumDpsInits() != 1 || !linalgOp.hasTensorSemantics())
      return failure();
    // Scan all operands, inspect sparse tensors.
    //
    // TODO: generalize this proof-of-concept algorithm, since the current
    //       implementation accepts only simple indexing maps, and one
    //       non-permutation sparse tensor, which must have an identity
    //       indexing map and be the output.
    //
    OpOperand *tx = nullptr;
    for (OpOperand &t : linalgOp->getOpOperands()) {
      // Ensure every index map is "simple".
      const auto map = linalgOp.getMatchingIndexingMap(&t);
      for (unsigned i = 0, n = map.getNumResults(); i < n; i++)
        if (map.getResult(i).getKind() != AffineExprKind::DimId)
          return failure();
      // Inspect sparse operands.
      auto stt = tryGetSparseTensorType(t.get());
      if (stt && stt->hasEncoding()) {
        if (stt->isPermutation())
          continue;
        assert(stt->getDimRank() < stt->getLvlRank()); // only allowed non-perm
        if (tx)
          return failure(); // more than one non-perm
        if (!map.isIdentity())
          return failure(); // no ID indexing map on the non-perm
        tx = &t;
      }
    }
    // Found a non-permutation, rewrite when this is the output.
    if (tx && tx == linalgOp.getDpsInitOperand(0)) {
      auto stt = getSparseTensorType(tx->get());
      auto demap = genDemap(rewriter, stt.getEncoding(), tx->get());
      auto newOp = genGenericLinalg(rewriter, linalgOp, stt, demap);
      auto remap = genRemap(rewriter, stt.getEncoding(), newOp.getResult(0));
      rewriter.replaceOp(linalgOp, remap);
      return success();
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Reinterpret Map Rewriters for operations other than linalg.generics
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
    SmallVector<Value> deMappedIns(op->getOperands());
    for (Value &in : deMappedIns)
      if (auto stt = tryGetSparseTensorType(in); stt && !stt->isIdentity())
        in = rewriter.create<ReinterpretMapOp>(loc, stt->getDemappedType(), in);

    // CRTP call.
    OpAdaptor adaptor(deMappedIns);
    return static_cast<const SubClass *>(this)->rewriteOp(op, adaptor,
                                                          rewriter);
  }
};

struct TensorAllocDemapper
    : public OpRewritePattern<bufferization::AllocTensorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(bufferization::AllocTensorOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasNonIdentityOperandsOrResults(op))
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
    rewriter.startRootUpdate(op);
    op->setOperands(dynLvlSzs);
    op.getResult().setType(stt.getDemappedType());
    rewriter.finalizeRootUpdate(op);
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

struct ForeachOpDemapper
    : public DemapInsRewriter<ForeachOpDemapper, ForeachOp> {
  using DemapInsRewriter::DemapInsRewriter;
  LogicalResult rewriteOp(ForeachOp op, OpAdaptor adaptor,
                          PatternRewriter &rewriter) const {
    // Only handle operations with sparse input/output with non-identity dim2lvl
    // maps.
    if (!hasNonIdentityOperandsOrResults(op))
      return failure();

    // TODO: demap constant as well.
    if (auto constOp = op.getTensor().getDefiningOp<arith::ConstantOp>())
      if (auto attr = dyn_cast<SparseElementsAttr>(constOp.getValue()))
        return failure();

    Location loc = op.getLoc();
    // Cache the type information since we update the foreach op in-place.
    auto srcStt = getSparseTensorType(op.getTensor());
    SmallVector<Type> prevRetTps(op.getResultTypes());

    rewriter.startRootUpdate(op);
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
    rewriter.finalizeRootUpdate(op);

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
    patterns.add<GenericOpReinterpretMap>(patterns.getContext());
  }
  if (scope == ReinterpretMapScope::kAll ||
      scope == ReinterpretMapScope::kExceptGeneric) {
    patterns.add<TensorAllocDemapper, TensorInsertDemapper, ForeachOpDemapper>(
        patterns.getContext());
  }
}
