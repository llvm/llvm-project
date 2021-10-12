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

#include "PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace {

/// Converts `atomic_rmw` that cannot be lowered to a simple atomic op with
/// AtomicRMWOpLowering pattern, e.g. with "minf" or "maxf" attributes, to
/// `generic_atomic_rmw` with the expanded code.
///
/// %x = atomic_rmw "maxf" %fval, %F[%i] : (f32, memref<10xf32>) -> f32
///
/// will be lowered to
///
/// %x = std.generic_atomic_rmw %F[%i] : memref<10xf32> {
/// ^bb0(%current: f32):
///   %cmp = arith.cmpf "ogt", %current, %fval : f32
///   %new_value = select %cmp, %current, %fval : f32
///   atomic_yield %new_value : f32
/// }
struct AtomicRMWOpConverter : public OpRewritePattern<AtomicRMWOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AtomicRMWOp op,
                                PatternRewriter &rewriter) const final {
    arith::CmpFPredicate predicate;
    switch (op.kind()) {
    case AtomicRMWKind::maxf:
      predicate = arith::CmpFPredicate::OGT;
      break;
    case AtomicRMWKind::minf:
      predicate = arith::CmpFPredicate::OLT;
      break;
    default:
      return failure();
    }

    auto loc = op.getLoc();
    auto genericOp =
        rewriter.create<GenericAtomicRMWOp>(loc, op.memref(), op.indices());
    OpBuilder bodyBuilder =
        OpBuilder::atBlockEnd(genericOp.getBody(), rewriter.getListener());

    Value lhs = genericOp.getCurrentValue();
    Value rhs = op.value();
    Value cmp = bodyBuilder.create<arith::CmpFOp>(loc, predicate, lhs, rhs);
    Value select = bodyBuilder.create<SelectOp>(loc, cmp, lhs, rhs);
    bodyBuilder.create<AtomicYieldOp>(loc, select);

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
    auto shapeType = op.shape().getType().cast<MemRefType>();
    if (!shapeType.hasStaticShape())
      return failure();

    int64_t rank = shapeType.cast<MemRefType>().getDimSize(0);
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
        size = rewriter.create<memref::LoadOp>(loc, op.shape(), index);
        if (!size.getType().isa<IndexType>())
          size = rewriter.create<arith::IndexCastOp>(loc, size,
                                                     rewriter.getIndexType());
        sizes[i] = size;
      } else {
        sizes[i] = rewriter.getIndexAttr(op.getType().getDimSize(i));
        size =
            rewriter.create<arith::ConstantOp>(loc, sizes[i].get<Attribute>());
      }
      strides[i] = stride;
      if (i > 0)
        stride = rewriter.create<arith::MulIOp>(loc, stride, size);
    }
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, op.getType(), op.source(), /*offset=*/rewriter.getIndexAttr(0),
        sizes, strides);
    return success();
  }
};

template <typename OpTy, arith::CmpFPredicate pred>
struct MaxMinFOpConverter : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    Value lhs = op.lhs();
    Value rhs = op.rhs();

    Location loc = op.getLoc();
    Value cmp = rewriter.create<arith::CmpFOp>(loc, pred, lhs, rhs);
    Value select = rewriter.create<SelectOp>(loc, cmp, lhs, rhs);

    auto floatType = getElementTypeOrSelf(lhs.getType()).cast<FloatType>();
    Value isNaN = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO,
                                                 lhs, rhs);

    Value nan = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat::getQNaN(floatType.getFloatSemantics()), floatType);
    if (VectorType vectorType = lhs.getType().dyn_cast<VectorType>())
      nan = rewriter.create<SplatOp>(loc, vectorType, nan);

    rewriter.replaceOpWithNewOp<SelectOp>(op, isNaN, nan, select);
    return success();
  }
};

template <typename OpTy, arith::CmpIPredicate pred>
struct MaxMinIOpConverter : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    Value lhs = op.lhs();
    Value rhs = op.rhs();

    Location loc = op.getLoc();
    Value cmp = rewriter.create<arith::CmpIOp>(loc, pred, lhs, rhs);
    rewriter.replaceOpWithNewOp<SelectOp>(op, cmp, lhs, rhs);
    return success();
  }
};

struct StdExpandOpsPass : public StdExpandOpsBase<StdExpandOpsPass> {
  void runOnFunction() override {
    MLIRContext &ctx = getContext();

    RewritePatternSet patterns(&ctx);
    populateStdExpandOpsPatterns(patterns);
    arith::populateArithmeticExpandOpsPatterns(patterns);

    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithmeticDialect, memref::MemRefDialect,
                           StandardOpsDialect>();
    target.addIllegalOp<arith::CeilDivSIOp, arith::FloorDivSIOp>();
    target.addDynamicallyLegalOp<AtomicRMWOp>([](AtomicRMWOp op) {
      return op.kind() != AtomicRMWKind::maxf &&
             op.kind() != AtomicRMWKind::minf;
    });
    target.addDynamicallyLegalOp<memref::ReshapeOp>([](memref::ReshapeOp op) {
      return !op.shape().getType().cast<MemRefType>().hasStaticShape();
    });
    // clang-format off
    target.addIllegalOp<
      MaxFOp,
      MaxSIOp,
      MaxUIOp,
      MinFOp,
      MinSIOp,
      MinUIOp
    >();
    // clang-format on
    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateStdExpandOpsPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    AtomicRMWOpConverter,
    MaxMinFOpConverter<MaxFOp, arith::CmpFPredicate::OGT>,
    MaxMinFOpConverter<MinFOp, arith::CmpFPredicate::OLT>,
    MaxMinIOpConverter<MaxSIOp, arith::CmpIPredicate::sgt>,
    MaxMinIOpConverter<MaxUIOp, arith::CmpIPredicate::ugt>,
    MaxMinIOpConverter<MinSIOp, arith::CmpIPredicate::slt>,
    MaxMinIOpConverter<MinUIOp, arith::CmpIPredicate::ult>,
    MemRefReshapeOpConverter
  >(patterns.getContext());
  // clang-format on
}

std::unique_ptr<Pass> mlir::createStdExpandOpsPass() {
  return std::make_unique<StdExpandOpsPass>();
}
