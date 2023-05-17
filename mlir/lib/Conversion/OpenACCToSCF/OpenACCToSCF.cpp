//===- OpenACCToSCF.cpp - OpenACC condition to SCF if conversion ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToSCF/ConvertOpenACCToSCF.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTOPENACCTOSCF
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Pattern to transform the `getIfCond` on operation without region into a
/// scf.if and move the operation into the `then` region.
template <typename OpTy>
class ExpandIfCondition : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Early exit if there is no condition.
    if (!op.getIfCond())
      return failure();

    IntegerAttr constAttr;
    if (!matchPattern(op.getIfCond(), m_Constant(&constAttr))) {
      auto ifOp = rewriter.create<scf::IfOp>(op.getLoc(), TypeRange(),
                                             op.getIfCond(), false);
      rewriter.updateRootInPlace(op, [&]() { op.getIfCondMutable().erase(0); });
      auto thenBodyBuilder = ifOp.getThenBodyBuilder();
      thenBodyBuilder.setListener(rewriter.getListener());
      thenBodyBuilder.clone(*op.getOperation());
      rewriter.eraseOp(op);
    } else {
      if (constAttr.getInt())
        rewriter.updateRootInPlace(op,
                                   [&]() { op.getIfCondMutable().erase(0); });
      else
        rewriter.eraseOp(op);
    }
    return success();
  }
};
} // namespace

void mlir::populateOpenACCToSCFConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ExpandIfCondition<acc::EnterDataOp>>(patterns.getContext());
  patterns.add<ExpandIfCondition<acc::ExitDataOp>>(patterns.getContext());
  patterns.add<ExpandIfCondition<acc::UpdateOp>>(patterns.getContext());
}

namespace {
struct ConvertOpenACCToSCFPass
    : public impl::ConvertOpenACCToSCFBase<ConvertOpenACCToSCFPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertOpenACCToSCFPass::runOnOperation() {
  auto op = getOperation();
  auto *context = op.getContext();

  RewritePatternSet patterns(context);
  ConversionTarget target(*context);
  populateOpenACCToSCFConversionPatterns(patterns);

  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<acc::OpenACCDialect>();

  target.addDynamicallyLegalOp<acc::EnterDataOp>(
      [](acc::EnterDataOp op) { return !op.getIfCond(); });

  target.addDynamicallyLegalOp<acc::ExitDataOp>(
      [](acc::ExitDataOp op) { return !op.getIfCond(); });

  target.addDynamicallyLegalOp<acc::UpdateOp>(
      [](acc::UpdateOp op) { return !op.getIfCond(); });

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertOpenACCToSCFPass() {
  return std::make_unique<ConvertOpenACCToSCFPass>();
}
