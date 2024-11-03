//===- StdExpandDivs.cpp - Code to prepare Std for lowering Divs to LLVM  -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file Std transformations to expand Divs operation to help for the
// lowering to LLVM. Currently implemented transformations are Ceil and Floor
// for Signed Integers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_EXPANDOPS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

namespace {

/// Converts `atomic_rmw` that cannot be lowered to a simple atomic op with
/// AtomicRMWOpLowering pattern, e.g. with "minf" or "maxf" attributes, to
/// `memref.generic_atomic_rmw` with the expanded code.
///
/// %x = atomic_rmw "maxf" %fval, %F[%i] : (f32, memref<10xf32>) -> f32
///
/// will be lowered to
///
/// %x = memref.generic_atomic_rmw %F[%i] : memref<10xf32> {
/// ^bb0(%current: f32):
///   %cmp = arith.cmpf "ogt", %current, %fval : f32
///   %new_value = select %cmp, %current, %fval : f32
///   memref.atomic_yield %new_value : f32
/// }
struct AtomicRMWOpConverter : public OpRewritePattern<memref::AtomicRMWOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AtomicRMWOp op,
                                PatternRewriter &rewriter) const final {
    arith::CmpFPredicate predicate;
    switch (op.getKind()) {
    case arith::AtomicRMWKind::maxf:
      predicate = arith::CmpFPredicate::OGT;
      break;
    case arith::AtomicRMWKind::minf:
      predicate = arith::CmpFPredicate::OLT;
      break;
    default:
      return failure();
    }

    auto loc = op.getLoc();
    auto genericOp = rewriter.create<memref::GenericAtomicRMWOp>(
        loc, op.getMemref(), op.getIndices());
    OpBuilder bodyBuilder =
        OpBuilder::atBlockEnd(genericOp.getBody(), rewriter.getListener());

    Value lhs = genericOp.getCurrentValue();
    Value rhs = op.getValue();
    Value cmp = bodyBuilder.create<arith::CmpFOp>(loc, predicate, lhs, rhs);
    Value select = bodyBuilder.create<arith::SelectOp>(loc, cmp, lhs, rhs);
    bodyBuilder.create<memref::AtomicYieldOp>(loc, select);

    rewriter.replaceOp(op, genericOp.getResult());
    return success();
  }
};

/// Converts `memref.reshape` that has a target shape of a statically-known
/// size to `memref.reinterpret_cast`.
struct MemRefReshapeOpConverter : public OpRewritePattern<memref::ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ReshapeOp op,
                                PatternRewriter &rewriter) const final {
    auto shapeType = cast<MemRefType>(op.getShape().getType());
    if (!shapeType.hasStaticShape())
      return failure();

    int64_t rank = cast<MemRefType>(shapeType).getDimSize(0);
    SmallVector<OpFoldResult, 4> sizes, strides;
    sizes.resize(rank);
    strides.resize(rank);

    Location loc = op.getLoc();
    Value stride = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (int i = rank - 1; i >= 0; --i) {
      Value size;
      // Load dynamic sizes from the shape input, use constants for static dims.
      if (op.getType().isDynamicDim(i)) {
        Value index = rewriter.create<arith::ConstantIndexOp>(loc, i);
        size = rewriter.create<memref::LoadOp>(loc, op.getShape(), index);
        if (!isa<IndexType>(size.getType()))
          size = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getIndexType(), size);
        sizes[i] = size;
      } else {
        auto sizeAttr = rewriter.getIndexAttr(op.getType().getDimSize(i));
        size = rewriter.create<arith::ConstantOp>(loc, sizeAttr);
        sizes[i] = sizeAttr;
      }
      strides[i] = stride;
      if (i > 0)
        stride = rewriter.create<arith::MulIOp>(loc, stride, size);
    }
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, op.getType(), op.getSource(), /*offset=*/rewriter.getIndexAttr(0),
        sizes, strides);
    return success();
  }
};

struct ExpandOpsPass : public memref::impl::ExpandOpsBase<ExpandOpsPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();

    RewritePatternSet patterns(&ctx);
    memref::populateExpandOpsPatterns(patterns);
    ConversionTarget target(ctx);

    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect>();
    target.addDynamicallyLegalOp<memref::AtomicRMWOp>(
        [](memref::AtomicRMWOp op) {
          return op.getKind() != arith::AtomicRMWKind::maxf &&
                 op.getKind() != arith::AtomicRMWKind::minf;
        });
    target.addDynamicallyLegalOp<memref::ReshapeOp>([](memref::ReshapeOp op) {
      return !cast<MemRefType>(op.getShape().getType()).hasStaticShape();
    });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::memref::populateExpandOpsPatterns(RewritePatternSet &patterns) {
  patterns.add<AtomicRMWOpConverter, MemRefReshapeOpConverter>(
      patterns.getContext());
}

std::unique_ptr<Pass> mlir::memref::createExpandOpsPass() {
  return std::make_unique<ExpandOpsPass>();
}
