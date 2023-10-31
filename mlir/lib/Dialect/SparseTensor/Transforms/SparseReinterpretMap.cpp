//===- SparseReinterpretMap.cpp - reinterpret sparse tensor maps ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

// TODO:
//   (1) insert the zero-cost sparse_tensor.reinterpret_map ops
//   (2) rewrite linalg.generic ops traits on level crds
//   (3) compute topsort, and resolve cyles with sparse_tensor.convert ops

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

//===----------------------------------------------------------------------===//
// Reinterpret Map Rewriters for operations other than linalg.generics
//===----------------------------------------------------------------------===//

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
      scope == ReinterpretMapScope::kExceptGeneric) {
    patterns.add<CrdTranslateRewriter, TensorInsertRewriter>(
        patterns.getContext());
  }
}
