//===- ExpandAtomic.cpp - Code to perform expanding atomic ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements expansion of AtomicRMWOp into GenericAtomicRMWOp.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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
///   %cmp = cmpf "ogt", %current, %fval : f32
///   %new_value = select %cmp, %current, %fval : f32
///   atomic_yield %new_value : f32
/// }
struct AtomicRMWOpConverter : public OpRewritePattern<AtomicRMWOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AtomicRMWOp op,
                                PatternRewriter &rewriter) const final {
    CmpFPredicate predicate;
    switch (op.kind()) {
    case AtomicRMWKind::maxf:
      predicate = CmpFPredicate::OGT;
      break;
    case AtomicRMWKind::minf:
      predicate = CmpFPredicate::OLT;
      break;
    default:
      return failure();
    }

    auto loc = op.getLoc();
    auto genericOp =
        rewriter.create<GenericAtomicRMWOp>(loc, op.memref(), op.indices());
    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(genericOp.getBody());

    Value lhs = genericOp.getCurrentValue();
    Value rhs = op.value();
    Value cmp = bodyBuilder.create<CmpFOp>(loc, predicate, lhs, rhs);
    Value select = bodyBuilder.create<SelectOp>(loc, cmp, lhs, rhs);
    bodyBuilder.create<AtomicYieldOp>(loc, select);

    rewriter.replaceOp(op, genericOp.getResult());
    return success();
  }
};

struct ExpandAtomic : public ExpandAtomicBase<ExpandAtomic> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    patterns.insert<AtomicRMWOpConverter>(&getContext());

    ConversionTarget target(getContext());
    target.addLegalOp<GenericAtomicRMWOp>();
    target.addDynamicallyLegalOp<AtomicRMWOp>([](AtomicRMWOp op) {
      return op.kind() != AtomicRMWKind::maxf &&
             op.kind() != AtomicRMWKind::minf;
    });
    if (failed(mlir::applyPartialConversion(getFunction(), target, patterns)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createExpandAtomicPass() {
  return std::make_unique<ExpandAtomic>();
}
