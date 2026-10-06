//===- CopyToContiguous.cpp - Split non-contiguous memref.copy -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass rewrites a `memref.copy` whose source or target is not contiguous
// into a loop nest over the leading dimensions around a `memref.copy` of the
// longest contiguous trailing block.
//
// The MemRef-to-LLVM lowering emits `llvm.intr.memcpy` for a copy only when
// both operands satisfy `memref::isStaticShapeAndContiguousRowMajor`; any other
// copy is lowered to a call to the `memrefCopy` runtime function, which walks
// the iteration space one element at a time. A copy into a padded buffer, or
// an `insert_slice` into a larger buffer, typically has a contiguous inner
// block but a mismatched leading stride, and so takes the slow path for the
// whole copy even though most of the bytes could be moved with `memcpy`.
//
// The pass determines the contiguous suffix with the same rule the lowering
// applies, so the copies it produces are guaranteed to take the `memcpy` path.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_COPYTOCONTIGUOUSPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

namespace {

/// Returns the first dimension of the trailing run that
/// `memref::isStaticShapeAndContiguousRowMajor` accepts as contiguous, i.e.
/// dimensions are consumed from the back while `stride == product of the
/// trailing extents`. This mirrors the first phase of that predicate exactly;
/// the second phase (leading unit dimensions) is handled by the loop nest,
/// since a loop over a unit extent is harmless.
static int64_t contiguousSuffixStart(ArrayRef<int64_t> shape,
                                     ArrayRef<int64_t> strides) {
  int64_t running = 1;
  int64_t cur = static_cast<int64_t>(shape.size()) - 1;
  while (cur >= 0 && strides[cur] == running) {
    running *= shape[cur];
    --cur;
  }
  return cur + 1;
}

/// Static strides of `type`, or failure if any stride or extent is dynamic.
static LogicalResult getStaticStrides(MemRefType type,
                                      SmallVectorImpl<int64_t> &strides) {
  if (!type.hasStaticShape())
    return failure();
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return failure();
  if (llvm::any_of(strides, ShapedType::isDynamic))
    return failure();
  return success();
}

struct CopyToContiguousPattern : public OpRewritePattern<memref::CopyOp> {
  CopyToContiguousPattern(MLIRContext *ctx, int64_t minSuffixElements)
      : OpRewritePattern(ctx), minSuffixElements(minSuffixElements) {}

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<MemRefType>(op.getSource().getType());
    auto dstType = dyn_cast<MemRefType>(op.getTarget().getType());
    if (!srcType || !dstType)
      return rewriter.notifyMatchFailure(op, "unranked operand");
    if (srcType.getShape() != dstType.getShape())
      return rewriter.notifyMatchFailure(op, "shape mismatch");

    SmallVector<int64_t> srcStrides, dstStrides;
    if (failed(getStaticStrides(srcType, srcStrides)) ||
        failed(getStaticStrides(dstType, dstStrides)))
      return rewriter.notifyMatchFailure(op, "dynamic shape or strides");

    ArrayRef<int64_t> shape = srcType.getShape();
    int64_t rank = shape.size();
    if (rank == 0 || srcType.getNumElements() == 0)
      return rewriter.notifyMatchFailure(op, "rank-0 or empty");

    // Both operands must be contiguous over the suffix, so take the shorter.
    int64_t k = std::max(contiguousSuffixStart(shape, srcStrides),
                         contiguousSuffixStart(shape, dstStrides));
    if (k == 0)
      return rewriter.notifyMatchFailure(op, "already contiguous");

    // If every leading dimension is a unit extent the lowering already takes
    // the memcpy path (second phase of its predicate); nothing to do.
    if (llvm::all_of(shape.take_front(k), [](int64_t d) { return d == 1; }))
      return rewriter.notifyMatchFailure(op, "leading dims are all unit");

    int64_t suffixElements = 1;
    for (int64_t d : shape.drop_front(k))
      suffixElements *= d;
    if (suffixElements < minSuffixElements)
      return rewriter.notifyMatchFailure(op, "suffix below threshold");

    // Infer the subview types up front and verify, with the lowering's own
    // predicate, that the inner copy will take the memcpy path.
    SmallVector<int64_t> staticOffsets(rank, 0), staticSizes(shape),
        staticStrides(rank, 1);
    for (int64_t i = 0; i < k; ++i) {
      staticOffsets[i] = ShapedType::kDynamic;
      staticSizes[i] = 1;
    }
    auto srcSubType = memref::SubViewOp::inferResultType(
        srcType, staticOffsets, staticSizes, staticStrides);
    auto dstSubType = memref::SubViewOp::inferResultType(
        dstType, staticOffsets, staticSizes, staticStrides);
    if (!memref::isStaticShapeAndContiguousRowMajor(srcSubType) ||
        !memref::isStaticShapeAndContiguousRowMajor(dstSubType))
      return rewriter.notifyMatchFailure(op, "suffix would not be contiguous");

    Location loc = op.getLoc();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<Value> lbs(k, zero), steps(k, one), ubs;
    ubs.reserve(k);
    for (int64_t i = 0; i < k; ++i)
      ubs.push_back(arith::ConstantIndexOp::create(rewriter, loc, shape[i]));

    Value src = op.getSource(), dst = op.getTarget();
    scf::buildLoopNest(rewriter, loc, lbs, ubs, steps,
                       [&](OpBuilder &b, Location nloc, ValueRange ivs) {
                         SmallVector<OpFoldResult> offsets, sizes, strides;
                         offsets.reserve(rank);
                         sizes.reserve(rank);
                         strides.reserve(rank);
                         for (int64_t i = 0; i < rank; ++i) {
                           if (i < k) {
                             offsets.push_back(ivs[i]);
                             sizes.push_back(b.getIndexAttr(1));
                           } else {
                             offsets.push_back(b.getIndexAttr(0));
                             sizes.push_back(b.getIndexAttr(shape[i]));
                           }
                           strides.push_back(b.getIndexAttr(1));
                         }
                         // Rank-preserving on purpose: the leading unit
                         // dimensions keep the original strides, and the
                         // contiguity predicate ignores them.
                         Value s = memref::SubViewOp::create(
                             b, nloc, srcSubType, src, offsets, sizes, strides);
                         Value d = memref::SubViewOp::create(
                             b, nloc, dstSubType, dst, offsets, sizes, strides);
                         memref::CopyOp::create(b, nloc, s, d);
                       });

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t minSuffixElements;
};

struct CopyToContiguousPass final
    : public memref::impl::CopyToContiguousPassBase<CopyToContiguousPass> {
  using CopyToContiguousPassBase::CopyToContiguousPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    memref::populateCopyToContiguousPatterns(patterns, minSuffixElements);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace

void memref::populateCopyToContiguousPatterns(RewritePatternSet &patterns,
                                              int64_t minSuffixElements) {
  patterns.add<CopyToContiguousPattern>(patterns.getContext(),
                                        minSuffixElements);
}
