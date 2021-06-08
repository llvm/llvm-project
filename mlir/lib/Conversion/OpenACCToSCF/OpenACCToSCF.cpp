//===- OpenACCToSCF.cpp - OpenACC condition to SCF if conversion ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/OpenACCToSCF/ConvertOpenACCToSCF.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Pattern to transform the `ifCond` on operation without region into a scf.if
/// and move the operation into the `then` region.
template <typename OpTy>
class ExpandIfCondition : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Early exit if there is no condition.
    if (!op.ifCond())
      return success();

    // Condition is not a constant.
    if (!op.ifCond().template getDefiningOp<ConstantOp>()) {
      auto ifOp = rewriter.create<scf::IfOp>(op.getLoc(), TypeRange(),
                                             op.ifCond(), false);
      rewriter.updateRootInPlace(op, [&]() { op.ifCondMutable().erase(0); });
      auto thenBodyBuilder = ifOp.getThenBodyBuilder();
      thenBodyBuilder.setListener(rewriter.getListener());
      thenBodyBuilder.clone(*op.getOperation());
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
    : public ConvertOpenACCToSCFBase<ConvertOpenACCToSCFPass> {
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
      [](acc::EnterDataOp op) { return !op.ifCond(); });

  target.addDynamicallyLegalOp<acc::ExitDataOp>(
      [](acc::ExitDataOp op) { return !op.ifCond(); });

  target.addDynamicallyLegalOp<acc::UpdateOp>(
      [](acc::UpdateOp op) { return !op.ifCond(); });

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertOpenACCToSCFPass() {
  return std::make_unique<ConvertOpenACCToSCFPass>();
}
