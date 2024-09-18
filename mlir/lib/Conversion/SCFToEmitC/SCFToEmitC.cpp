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

// Lower scf::for to emitc::for, implementing result values using
// emitc::variable's updated within the loop body.
struct ForLowering : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override;
};

// Create an uninitialized emitc::variable op for each result of the given op.
template <typename T>
static SmallVector<Value> createVariablesForResults(T op,
                                                    PatternRewriter &rewriter) {
  SmallVector<Value> resultVariables;

  if (!op.getNumResults())
    return resultVariables;

  Location loc = op->getLoc();
  MLIRContext *context = op.getContext();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  for (OpResult result : op.getResults()) {
    Type resultType = result.getType();
    Type varType = emitc::LValueType::get(resultType);
    emitc::OpaqueAttr noInit = emitc::OpaqueAttr::get(context, "");
    emitc::VariableOp var =
        rewriter.create<emitc::VariableOp>(loc, varType, noInit);
    resultVariables.push_back(var);
  }

  return resultVariables;
}

// Create a series of assign ops assigning given values to given variables at
// the current insertion point of given rewriter.
static void assignValues(ValueRange values, SmallVector<Value> &variables,
                         PatternRewriter &rewriter, Location loc) {
  for (auto [value, var] : llvm::zip(values, variables))
    rewriter.create<emitc::AssignOp>(loc, var, value);
}

SmallVector<Value> loadValues(const SmallVector<Value> &variables,
                              PatternRewriter &rewriter, Location loc) {
  return llvm::map_to_vector<>(variables, [&](Value var) {
    Type type = cast<emitc::LValueType>(var.getType()).getValueType();
    return rewriter.create<emitc::LoadOp>(loc, type, var).getResult();
  });
}

static void lowerYield(SmallVector<Value> &resultVariables,
                       PatternRewriter &rewriter, scf::YieldOp yield) {
  Location loc = yield.getLoc();
  ValueRange operands = yield.getOperands();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(yield);

  assignValues(operands, resultVariables, rewriter, loc);

  rewriter.create<emitc::YieldOp>(loc);
  rewriter.eraseOp(yield);
}

// Lower the contents of an scf::if/scf::index_switch regions to an
// emitc::if/emitc::switch region. The contents of the lowering region is
// moved into the respective lowered region, but the scf::yield is replaced not
// only with an emitc::yield, but also with a sequence of emitc::assign ops that
// set the yielded values into the result variables.
static void lowerRegion(SmallVector<Value> &resultVariables,
                        PatternRewriter &rewriter, Region &region,
                        Region &loweredRegion) {
  rewriter.inlineRegionBefore(region, loweredRegion, loweredRegion.end());
  Operation *terminator = loweredRegion.back().getTerminator();
  lowerYield(resultVariables, rewriter, cast<scf::YieldOp>(terminator));
}

LogicalResult ForLowering::matchAndRewrite(ForOp forOp,
                                           PatternRewriter &rewriter) const {
  Location loc = forOp.getLoc();

  // Create an emitc::variable op for each result. These variables will be
  // assigned to by emitc::assign ops within the loop body.
  SmallVector<Value> resultVariables =
      createVariablesForResults(forOp, rewriter);

  assignValues(forOp.getInits(), resultVariables, rewriter, loc);

  emitc::ForOp loweredFor = rewriter.create<emitc::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep());

  Block *loweredBody = loweredFor.getBody();

  // Erase the auto-generated terminator for the lowered for op.
  rewriter.eraseOp(loweredBody->getTerminator());

  IRRewriter::InsertPoint ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(loweredBody);

  SmallVector<Value> iterArgsValues =
      loadValues(resultVariables, rewriter, loc);

  rewriter.restoreInsertionPoint(ip);

  SmallVector<Value> replacingValues;
  replacingValues.push_back(loweredFor.getInductionVar());
  replacingValues.append(iterArgsValues.begin(), iterArgsValues.end());

  rewriter.mergeBlocks(forOp.getBody(), loweredBody, replacingValues);
  lowerYield(resultVariables, rewriter,
             cast<scf::YieldOp>(loweredBody->getTerminator()));

  // Load variables into SSA values after the for loop.
  SmallVector<Value> resultValues = loadValues(resultVariables, rewriter, loc);

  rewriter.replaceOp(forOp, resultValues);
  return success();
}

// Lower scf::if to emitc::if, implementing result values as emitc::variable's
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

  // Create an emitc::variable op for each result. These variables will be
  // assigned to by emitc::assign ops within the then & else regions.
  SmallVector<Value> resultVariables =
      createVariablesForResults(ifOp, rewriter);

  Region &thenRegion = ifOp.getThenRegion();
  Region &elseRegion = ifOp.getElseRegion();

  bool hasElseBlock = !elseRegion.empty();

  auto loweredIf =
      rewriter.create<emitc::IfOp>(loc, ifOp.getCondition(), false, false);

  Region &loweredThenRegion = loweredIf.getThenRegion();
  lowerRegion(resultVariables, rewriter, thenRegion, loweredThenRegion);

  if (hasElseBlock) {
    Region &loweredElseRegion = loweredIf.getElseRegion();
    lowerRegion(resultVariables, rewriter, elseRegion, loweredElseRegion);
  }

  rewriter.setInsertionPointAfter(ifOp);
  SmallVector<Value> results = loadValues(resultVariables, rewriter, loc);

  rewriter.replaceOp(ifOp, results);
  return success();
}

// Lower scf::index_switch to emitc::switch, implementing result values as
// emitc::variable's updated within the case and default regions.
struct IndexSwitchOpLowering : public OpRewritePattern<IndexSwitchOp> {
  using OpRewritePattern<IndexSwitchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexSwitchOp indexSwitchOp,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
IndexSwitchOpLowering::matchAndRewrite(IndexSwitchOp indexSwitchOp,
                                       PatternRewriter &rewriter) const {
  Location loc = indexSwitchOp.getLoc();

  // Create an emitc::variable op for each result. These variables will be
  // assigned to by emitc::assign ops within the case and default regions.
  SmallVector<Value> resultVariables =
      createVariablesForResults(indexSwitchOp, rewriter);

  auto loweredSwitch = rewriter.create<emitc::SwitchOp>(
      loc, indexSwitchOp.getArg(), indexSwitchOp.getCases(),
      indexSwitchOp.getNumCases());

  // Lowering all case regions.
  for (auto pair : llvm::zip(indexSwitchOp.getCaseRegions(),
                             loweredSwitch.getCaseRegions())) {
    lowerRegion(resultVariables, rewriter, std::get<0>(pair),
                std::get<1>(pair));
  }

  // Lowering default region.
  lowerRegion(resultVariables, rewriter, indexSwitchOp.getDefaultRegion(),
              loweredSwitch.getDefaultRegion());

  rewriter.setInsertionPointAfter(indexSwitchOp);
  SmallVector<Value> results = loadValues(resultVariables, rewriter, loc);

  rewriter.replaceOp(indexSwitchOp, results);
  return success();
}

void mlir::populateSCFToEmitCConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ForLowering>(patterns.getContext());
  patterns.add<IfLowering>(patterns.getContext());
  patterns.add<IndexSwitchOpLowering>(patterns.getContext());
}

void SCFToEmitCPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateSCFToEmitCConversionPatterns(patterns);

  // Configure conversion to lower out SCF operations.
  ConversionTarget target(getContext());
  target.addIllegalOp<scf::ForOp, scf::IfOp, scf::IndexSwitchOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
