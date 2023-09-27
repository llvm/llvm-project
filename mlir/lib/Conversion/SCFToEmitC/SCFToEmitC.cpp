//===- SCFToEmitC.cpp - SCF to EmitC conversion ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert scf.if ops into emitc ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_SCFTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace {

struct SCFToEmitCPass : public impl::SCFToEmitCBase<SCFToEmitCPass> {
  void runOnOperation() override;
};

// Lower scf::if to emitc::if, implementing return values as emitc::variable's
// updated within the then and else regions.
struct IfLowering : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace

LogicalResult IfLowering::matchAndRewrite(IfOp ifOp,
                                          PatternRewriter &rewriter) const {
  Location loc = ifOp.getLoc();

  SmallVector<Value> resultVariables;

  // Create an emitc::variable op for each result. These variables will be
  // assigned to by emitc::assign ops within the then & else regions.
  if (ifOp.getNumResults()) {
    MLIRContext *context = ifOp.getContext();
    rewriter.setInsertionPoint(ifOp);
    for (OpResult result : ifOp.getResults()) {
      Type resultType = result.getType();
      auto noInit = emitc::OpaqueAttr::get(context, "");
      auto var = rewriter.create<emitc::VariableOp>(loc, resultType, noInit);
      resultVariables.push_back(var);
    }
  }

  // Utility function to lower the contents of an scf::if region to an emitc::if
  // region. The contents of the scf::if regions is moved into the respective
  // emitc::if regions, but the scf::yield is replaced not only with an
  // emitc::yield, but also with a sequence of emitc::assign ops that set the
  // yielded values into the result variables.
  auto lowerRegion = [&resultVariables, &rewriter](Region &region,
                                                   Region &loweredRegion) {
    rewriter.inlineRegionBefore(region, loweredRegion, loweredRegion.end());
    Operation *terminator = loweredRegion.back().getTerminator();
    Location terminatorLoc = terminator->getLoc();
    ValueRange terminatorOperands = terminator->getOperands();
    rewriter.setInsertionPointToEnd(&loweredRegion.back());
    for (auto value2Var : llvm::zip(terminatorOperands, resultVariables)) {
      Value resultValue = std::get<0>(value2Var);
      Value resultVar = std::get<1>(value2Var);
      rewriter.create<emitc::AssignOp>(terminatorLoc, resultVar, resultValue);
    }
    rewriter.create<emitc::YieldOp>(terminatorLoc);
    rewriter.eraseOp(terminator);
  };

  Region &thenRegion = ifOp.getThenRegion();
  Region &elseRegion = ifOp.getElseRegion();

  bool hasElseBlock = !elseRegion.empty();

  auto loweredIf =
      rewriter.create<emitc::IfOp>(loc, ifOp.getCondition(), false, false);

  Region &loweredThenRegion = loweredIf.getThenRegion();
  lowerRegion(thenRegion, loweredThenRegion);

  if (hasElseBlock) {
    Region &loweredElseRegion = loweredIf.getElseRegion();
    lowerRegion(elseRegion, loweredElseRegion);
  }

  rewriter.replaceOp(ifOp, resultVariables);
  return success();
}

void mlir::populateSCFToEmitCConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<IfLowering>(patterns.getContext());
}

void SCFToEmitCPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateSCFToEmitCConversionPatterns(patterns);

  // Configure conversion to lower out SCF operations.
  ConversionTarget target(getContext());
  target.addIllegalOp<scf::IfOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createConvertSCFToEmitCPass() {
  return std::make_unique<SCFToEmitCPass>();
}
