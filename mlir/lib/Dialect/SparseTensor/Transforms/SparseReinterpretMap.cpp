//===- SparseReinterpretMap.cpp - reinterpret sparse tensor maps ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

//===----------------------------------------------------------------------===//
// Helper methods.
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
      auto stt = getSparseTensorType(t.get());
      if (stt.hasEncoding()) {
        if (stt.isPermutation())
          continue;
        assert(stt.getDimRank() < stt.getLvlRank()); // only allowed non-perm
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
// Rewriting rules for operations other than linalg generic ops.
//===----------------------------------------------------------------------===//

// CRTP to help implementing a rewriter that demaps all its inputs and remaps
// all its outputs.
template <typename SubClass, typename SourceOp>
struct DemapInsRemapOutsRewriter : public OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    if (!static_cast<const SubClass *>(this)->matchOp(op))
      return failure();

    Location loc = op.getLoc();
    // Demaps non-trivial inputs.
    SmallVector<Value> deMappedIns(op->getOperands());
    for (Value &in : deMappedIns)
      if (auto stt = tryGetSparseTensorType(in); stt && !stt->isIdentity())
        in = rewriter.create<ReinterpretMapOp>(loc, stt->getDemappedType(), in);

    // CRTP call.
    OpAdaptor adaptor(deMappedIns);
    ValueRange outs =
        static_cast<const SubClass *>(this)->rewriteOp(op, adaptor, rewriter);
    assert(outs.size() == op->getResults().size());

    // Remap  outputs.
    SmallVector<Value> reMappedOuts(outs);
    for (auto [r, a] : llvm::zip(reMappedOuts, op->getResults()))
      if (r.getType() != a.getType())
        r = rewriter.create<ReinterpretMapOp>(loc, a.getType(), r);

    rewriter.replaceOp(op, reMappedOuts);
    return success();
  }
};

struct CrdTranslateRewriter : public OpRewritePattern<CrdTranslateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CrdTranslateOp op,
                                PatternRewriter &rewriter) const override {
    AffineMap map = op.getDirection() == CrdTransDirectionKind::dim2lvl
                        ? op.getEncoder().getDimToLvl()
                        : op.getEncoder().getLvlToDim();

    SmallVector<Value> outCrds;
    for (AffineExpr result : map.getResults()) {
      // TODO: we should probably expand the affine map to IR using our own
      // rules, since affine.apply assume signed value, while the cooridinates
      // we provided must always be signless.
      Value trans = rewriter.create<affine::AffineApplyOp>(
          op.getLoc(), AffineMap::get(map.getNumDims(), 0, result),
          op.getInCrds());
      outCrds.push_back(trans);
    }
    rewriter.replaceOp(op, outCrds);
    return success();
  }
};

struct TensorInsertRewriter
    : public DemapInsRemapOutsRewriter<TensorInsertRewriter, tensor::InsertOp> {
  using DemapInsRemapOutsRewriter::DemapInsRemapOutsRewriter;

  bool matchOp(tensor::InsertOp op) const {
    return op.getResult().getType().getEncoding() != nullptr;
  }

  ValueRange rewriteOp(tensor::InsertOp op, OpAdaptor adaptor,
                       PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto stt = getSparseTensorType(op.getResult());
    ValueRange lvlCrd = stt.translateCrds(rewriter, loc, op.getIndices(),
                                          CrdTransDirectionKind::dim2lvl);
    Operation *insertOp = rewriter.create<sparse_tensor::InsertOp>(
        loc, op.getScalar(), adaptor.getDest(), lvlCrd);
    return insertOp->getResults();
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
    patterns.add<CrdTranslateRewriter, TensorInsertRewriter>(
        patterns.getContext());
  }
}
