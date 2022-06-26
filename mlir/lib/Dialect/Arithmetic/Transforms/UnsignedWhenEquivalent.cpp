//===- UnsignedWhenEquivalent.cpp - Pass to replace signed operations with
// unsigned
// ones when all their arguments and results are statically non-negative --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/IntRangeAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::arith;

/// Succeeds when a value is statically non-negative in that it has a lower
/// bound on its value (if it is treated as signed) and that bound is
/// non-negative.
static LogicalResult staticallyNonNegative(IntRangeAnalysis &analysis,
                                           Value v) {
  Optional<ConstantIntRanges> result = analysis.getResult(v);
  if (!result.has_value())
    return failure();
  const ConstantIntRanges &range = result.value();
  return success(range.smin().isNonNegative());
}

/// Succeeds if an op can be converted to its unsigned equivalent without
/// changing its semantics. This is the case when none of its openands or
/// results can be below 0 when analyzed from a signed perspective.
static LogicalResult staticallyNonNegative(IntRangeAnalysis &analysis,
                                           Operation *op) {
  auto nonNegativePred = [&analysis](Value v) -> bool {
    return succeeded(staticallyNonNegative(analysis, v));
  };
  return success(llvm::all_of(op->getOperands(), nonNegativePred) &&
                 llvm::all_of(op->getResults(), nonNegativePred));
}

/// Succeeds when the comparison predicate is a signed operation and all the
/// operands are non-negative, indicating that the cmpi operation `op` can have
/// its predicate changed to an unsigned equivalent.
static LogicalResult isCmpIConvertable(IntRangeAnalysis &analysis, CmpIOp op) {
  CmpIPredicate pred = op.getPredicate();
  switch (pred) {
  case CmpIPredicate::sle:
  case CmpIPredicate::slt:
  case CmpIPredicate::sge:
  case CmpIPredicate::sgt:
    return success(llvm::all_of(op.getOperands(), [&analysis](Value v) -> bool {
      return succeeded(staticallyNonNegative(analysis, v));
    }));
  default:
    return failure();
  }
}

/// Return the unsigned equivalent of a signed comparison predicate,
/// or the predicate itself if there is none.
static CmpIPredicate toUnsignedPred(CmpIPredicate pred) {
  switch (pred) {
  case CmpIPredicate::sle:
    return CmpIPredicate::ule;
  case CmpIPredicate::slt:
    return CmpIPredicate::ult;
  case CmpIPredicate::sge:
    return CmpIPredicate::uge;
  case CmpIPredicate::sgt:
    return CmpIPredicate::ugt;
  default:
    return pred;
  }
}

namespace {
template <typename Signed, typename Unsigned>
struct ConvertOpToUnsigned : OpConversionPattern<Signed> {
  using OpConversionPattern<Signed>::OpConversionPattern;

  LogicalResult matchAndRewrite(Signed op, typename Signed::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const override {
    rw.replaceOpWithNewOp<Unsigned>(op, op->getResultTypes(),
                                    adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

struct ConvertCmpIToUnsigned : OpConversionPattern<CmpIOp> {
  using OpConversionPattern<CmpIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(CmpIOp op, CmpIOpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const override {
    rw.replaceOpWithNewOp<CmpIOp>(op, toUnsignedPred(op.getPredicate()),
                                  op.getLhs(), op.getRhs());
    return success();
  }
};

struct ArithmeticUnsignedWhenEquivalentPass
    : public ArithmeticUnsignedWhenEquivalentBase<
          ArithmeticUnsignedWhenEquivalentPass> {
  /// Implementation structure: first find all equivalent ops and collect them,
  /// then perform all the rewrites in a second pass over the target op. This
  /// ensures that analysis results are not invalidated during rewriting.
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    IntRangeAnalysis analysis(op);

    ConversionTarget target(*ctx);
    target.addLegalDialect<ArithmeticDialect>();
    target
        .addDynamicallyLegalOp<DivSIOp, CeilDivSIOp, CeilDivUIOp, FloorDivSIOp,
                               RemSIOp, MinSIOp, MaxSIOp, ExtSIOp>(
            [&analysis](Operation *op) -> Optional<bool> {
              return failed(staticallyNonNegative(analysis, op));
            });
    target.addDynamicallyLegalOp<CmpIOp>(
        [&analysis](CmpIOp op) -> Optional<bool> {
          return failed(isCmpIConvertable(analysis, op));
        });

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertOpToUnsigned<DivSIOp, DivUIOp>,
                 ConvertOpToUnsigned<CeilDivSIOp, CeilDivUIOp>,
                 ConvertOpToUnsigned<FloorDivSIOp, DivUIOp>,
                 ConvertOpToUnsigned<RemSIOp, RemUIOp>,
                 ConvertOpToUnsigned<MinSIOp, MinUIOp>,
                 ConvertOpToUnsigned<MaxSIOp, MaxUIOp>,
                 ConvertOpToUnsigned<ExtSIOp, ExtUIOp>, ConvertCmpIToUnsigned>(
        ctx);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

std::unique_ptr<Pass>
mlir::arith::createArithmeticUnsignedWhenEquivalentPass() {
  return std::make_unique<ArithmeticUnsignedWhenEquivalentPass>();
}
