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

#include "mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
#define GEN_PASS_DEF_SCFTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace {

/// Implement the interface to convert SCF to EmitC.
struct SCFToEmitCDialectInterface : public ConvertToEmitCPatternInterface {
  using ConvertToEmitCPatternInterface::ConvertToEmitCPatternInterface;

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToEmitCConversionPatterns(
      ConversionTarget &target, TypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateEmitCSizeTTypeConversions(typeConverter);
    populateSCFToEmitCConversionPatterns(patterns, typeConverter);
  }
};
} // namespace

void mlir::registerConvertSCFToEmitCInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
    dialect->addInterfaces<SCFToEmitCDialectInterface>();
  });
}

namespace {

struct SCFToEmitCPass : public impl::SCFToEmitCBase<SCFToEmitCPass> {
  void runOnOperation() override;
};

// Lower scf::for to emitc::for, implementing result values using
// emitc::variable's updated within the loop body.
struct ForLowering : public OpConversionPattern<ForOp> {
  using OpConversionPattern<ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ForOp forOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// Create an uninitialized emitc::variable op for each result of the given op.
template <typename T>
static LogicalResult
createVariablesForResults(T op, const TypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter,
                          SmallVector<Value> &resultVariables) {
  if (!op.getNumResults())
    return success();

  Location loc = op->getLoc();
  MLIRContext *context = op.getContext();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  for (OpResult result : op.getResults()) {
    Type resultType = typeConverter->convertType(result.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    Type varType = emitc::LValueType::get(resultType);
    emitc::OpaqueAttr noInit = emitc::OpaqueAttr::get(context, "");
    emitc::VariableOp var =
        emitc::VariableOp::create(rewriter, loc, varType, noInit);
    resultVariables.push_back(var);
  }

  return success();
}

// Create a series of assign ops assigning given values to given variables at
// the current insertion point of given rewriter.
static void assignValues(ValueRange values, ValueRange variables,
                         ConversionPatternRewriter &rewriter, Location loc) {
  for (auto [value, var] : llvm::zip(values, variables))
    emitc::AssignOp::create(rewriter, loc, var, value);
}

SmallVector<Value> loadValues(ArrayRef<Value> variables,
                              PatternRewriter &rewriter, Location loc) {
  return llvm::map_to_vector<>(variables, [&](Value var) {
    Type type = cast<emitc::LValueType>(var.getType()).getValueType();
    return emitc::LoadOp::create(rewriter, loc, type, var).getResult();
  });
}

static LogicalResult lowerYield(Operation *op, ValueRange resultVariables,
                                ConversionPatternRewriter &rewriter,
                                scf::YieldOp yield, bool createYield = true) {
  Location loc = yield.getLoc();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(yield);

  SmallVector<Value> yieldOperands;
  if (failed(rewriter.getRemappedValues(yield.getOperands(), yieldOperands)))
    return rewriter.notifyMatchFailure(op, "failed to lower yield operands");

  assignValues(yieldOperands, resultVariables, rewriter, loc);

  emitc::YieldOp::create(rewriter, loc);
  rewriter.eraseOp(yield);

  return success();
}

// Lower the contents of an scf::if/scf::index_switch regions to an
// emitc::if/emitc::switch region. The contents of the lowering region is
// moved into the respective lowered region, but the scf::yield is replaced not
// only with an emitc::yield, but also with a sequence of emitc::assign ops that
// set the yielded values into the result variables.
static LogicalResult lowerRegion(Operation *op, ValueRange resultVariables,
                                 ConversionPatternRewriter &rewriter,
                                 Region &region, Region &loweredRegion) {
  rewriter.inlineRegionBefore(region, loweredRegion, loweredRegion.end());
  Operation *terminator = loweredRegion.back().getTerminator();
  return lowerYield(op, resultVariables, rewriter,
                    cast<scf::YieldOp>(terminator));
}

LogicalResult
ForLowering::matchAndRewrite(ForOp forOp, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
  Location loc = forOp.getLoc();

  if (forOp.getUnsignedCmp())
    return rewriter.notifyMatchFailure(forOp,
                                       "unsigned loops are not supported");

  // Create an emitc::variable op for each result. These variables will be
  // assigned to by emitc::assign ops within the loop body.
  SmallVector<Value> resultVariables;
  if (failed(createVariablesForResults(forOp, getTypeConverter(), rewriter,
                                       resultVariables)))
    return rewriter.notifyMatchFailure(forOp,
                                       "create variables for results failed");

  assignValues(adaptor.getInitArgs(), resultVariables, rewriter, loc);

  emitc::ForOp loweredFor =
      emitc::ForOp::create(rewriter, loc, adaptor.getLowerBound(),
                           adaptor.getUpperBound(), adaptor.getStep());

  Block *loweredBody = loweredFor.getBody();

  // Erase the auto-generated terminator for the lowered for op.
  rewriter.eraseOp(loweredBody->getTerminator());

  IRRewriter::InsertPoint ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(loweredBody);

  SmallVector<Value> iterArgsValues =
      loadValues(resultVariables, rewriter, loc);

  rewriter.restoreInsertionPoint(ip);

  // Convert the original region types into the new types by adding unrealized
  // casts in the beginning of the loop. This performs the conversion in place.
  if (failed(rewriter.convertRegionTypes(&forOp.getRegion(),
                                         *getTypeConverter(), nullptr))) {
    return rewriter.notifyMatchFailure(forOp, "region types conversion failed");
  }

  // Register the replacements for the block arguments and inline the body of
  // the scf.for loop into the body of the emitc::for loop.
  Block *scfBody = &(forOp.getRegion().front());
  SmallVector<Value> replacingValues;
  replacingValues.push_back(loweredFor.getInductionVar());
  replacingValues.append(iterArgsValues.begin(), iterArgsValues.end());
  rewriter.mergeBlocks(scfBody, loweredBody, replacingValues);

  auto result = lowerYield(forOp, resultVariables, rewriter,
                           cast<scf::YieldOp>(loweredBody->getTerminator()));

  if (failed(result)) {
    return result;
  }

  // Load variables into SSA values after the for loop.
  SmallVector<Value> resultValues = loadValues(resultVariables, rewriter, loc);

  rewriter.replaceOp(forOp, resultValues);
  return success();
}

// Lower scf::if to emitc::if, implementing result values as emitc::variable's
// updated within the then and else regions.
struct IfLowering : public OpConversionPattern<IfOp> {
  using OpConversionPattern<IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfOp ifOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

LogicalResult
IfLowering::matchAndRewrite(IfOp ifOp, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {
  Location loc = ifOp.getLoc();

  // Create an emitc::variable op for each result. These variables will be
  // assigned to by emitc::assign ops within the then & else regions.
  SmallVector<Value> resultVariables;
  if (failed(createVariablesForResults(ifOp, getTypeConverter(), rewriter,
                                       resultVariables)))
    return rewriter.notifyMatchFailure(ifOp,
                                       "create variables for results failed");

  // Utility function to lower the contents of an scf::if region to an emitc::if
  // region. The contents of the scf::if regions is moved into the respective
  // emitc::if regions, but the scf::yield is replaced not only with an
  // emitc::yield, but also with a sequence of emitc::assign ops that set the
  // yielded values into the result variables.
  auto lowerRegion = [&resultVariables, &rewriter,
                      &ifOp](Region &region, Region &loweredRegion) {
    rewriter.inlineRegionBefore(region, loweredRegion, loweredRegion.end());
    Operation *terminator = loweredRegion.back().getTerminator();
    auto result = lowerYield(ifOp, resultVariables, rewriter,
                             cast<scf::YieldOp>(terminator));
    if (failed(result)) {
      return result;
    }
    return success();
  };

  Region &thenRegion = adaptor.getThenRegion();
  Region &elseRegion = adaptor.getElseRegion();

  bool hasElseBlock = !elseRegion.empty();

  auto loweredIf =
      emitc::IfOp::create(rewriter, loc, adaptor.getCondition(), false, false);

  Region &loweredThenRegion = loweredIf.getThenRegion();
  auto result = lowerRegion(thenRegion, loweredThenRegion);
  if (failed(result)) {
    return result;
  }

  if (hasElseBlock) {
    Region &loweredElseRegion = loweredIf.getElseRegion();
    auto result = lowerRegion(elseRegion, loweredElseRegion);
    if (failed(result)) {
      return result;
    }
  }

  rewriter.setInsertionPointAfter(ifOp);
  SmallVector<Value> results = loadValues(resultVariables, rewriter, loc);

  rewriter.replaceOp(ifOp, results);
  return success();
}

// Lower scf::index_switch to emitc::switch, implementing result values as
// emitc::variable's updated within the case and default regions.
struct IndexSwitchOpLowering : public OpConversionPattern<IndexSwitchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IndexSwitchOp indexSwitchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

LogicalResult IndexSwitchOpLowering::matchAndRewrite(
    IndexSwitchOp indexSwitchOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = indexSwitchOp.getLoc();

  // Create an emitc::variable op for each result. These variables will be
  // assigned to by emitc::assign ops within the case and default regions.
  SmallVector<Value> resultVariables;
  if (failed(createVariablesForResults(indexSwitchOp, getTypeConverter(),
                                       rewriter, resultVariables))) {
    return rewriter.notifyMatchFailure(indexSwitchOp,
                                       "create variables for results failed");
  }

  auto loweredSwitch =
      emitc::SwitchOp::create(rewriter, loc, adaptor.getArg(),
                              adaptor.getCases(), indexSwitchOp.getNumCases());

  // Lowering all case regions.
  for (auto pair :
       llvm::zip(adaptor.getCaseRegions(), loweredSwitch.getCaseRegions())) {
    if (failed(lowerRegion(indexSwitchOp, resultVariables, rewriter,
                           *std::get<0>(pair), std::get<1>(pair)))) {
      return failure();
    }
  }

  // Lowering default region.
  if (failed(lowerRegion(indexSwitchOp, resultVariables, rewriter,
                         adaptor.getDefaultRegion(),
                         loweredSwitch.getDefaultRegion()))) {
    return failure();
  }

  rewriter.setInsertionPointAfter(indexSwitchOp);
  SmallVector<Value> results = loadValues(resultVariables, rewriter, loc);

  rewriter.replaceOp(indexSwitchOp, results);
  return success();
}

// Lower scf::while to emitc::do using mutable variables to maintain loop state
// across iterations. The do-while structure ensures the condition is evaluated
// after each iteration, matching SCF while semantics.
struct WhileLowering : public OpConversionPattern<WhileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WhileOp whileOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = whileOp.getLoc();
    MLIRContext *context = loc.getContext();

    // Create an emitc::variable op for each result. These variables will be
    // assigned to by emitc::assign ops within the loop body.
    SmallVector<Value> resultVariables;
    if (failed(createVariablesForResults(whileOp, getTypeConverter(), rewriter,
                                         resultVariables)))
      return rewriter.notifyMatchFailure(whileOp,
                                         "Failed to create result variables");

    // Create variable storage for loop-carried values to enable imperative
    // updates while maintaining SSA semantics at conversion boundaries.
    SmallVector<Value> loopVariables;
    if (failed(createVariablesForLoopCarriedValues(
            whileOp, rewriter, loopVariables, loc, context)))
      return failure();

    if (failed(lowerDoWhile(whileOp, loopVariables, resultVariables, context,
                            rewriter, loc)))
      return failure();

    rewriter.setInsertionPointAfter(whileOp);

    // Load the final result values from result variables.
    SmallVector<Value> finalResults =
        loadValues(resultVariables, rewriter, loc);
    rewriter.replaceOp(whileOp, finalResults);

    return success();
  }

private:
  // Initialize variables for loop-carried values to enable state updates
  // across iterations without SSA argument passing.
  LogicalResult createVariablesForLoopCarriedValues(
      WhileOp whileOp, ConversionPatternRewriter &rewriter,
      SmallVectorImpl<Value> &loopVars, Location loc,
      MLIRContext *context) const {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(whileOp);

    emitc::OpaqueAttr noInit = emitc::OpaqueAttr::get(context, "");

    for (Value init : whileOp.getInits()) {
      Type convertedType = getTypeConverter()->convertType(init.getType());
      if (!convertedType)
        return rewriter.notifyMatchFailure(whileOp, "type conversion failed");

      emitc::VariableOp var = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(convertedType), noInit);
      rewriter.create<emitc::AssignOp>(loc, var.getResult(), init);
      loopVars.push_back(var);
    }

    return success();
  }

  // Lower scf.while to emitc.do.
  LogicalResult lowerDoWhile(WhileOp whileOp, ArrayRef<Value> loopVars,
                             ArrayRef<Value> resultVars, MLIRContext *context,
                             ConversionPatternRewriter &rewriter,
                             Location loc) const {
    // Create a global boolean variable to store the loop condition state.
    Type i1Type = IntegerType::get(context, 1);
    auto globalCondition =
        rewriter.create<emitc::VariableOp>(loc, emitc::LValueType::get(i1Type),
                                           emitc::OpaqueAttr::get(context, ""));
    Value conditionVal = globalCondition.getResult();

    auto loweredDo = rewriter.create<emitc::DoOp>(loc);

    // Convert region types to match the target dialect type system.
    if (failed(rewriter.convertRegionTypes(&whileOp.getBefore(),
                                           *getTypeConverter(), nullptr)) ||
        failed(rewriter.convertRegionTypes(&whileOp.getAfter(),
                                           *getTypeConverter(), nullptr))) {
      return rewriter.notifyMatchFailure(whileOp,
                                         "region types conversion failed");
    }

    // Prepare the before region (condition evaluation) for merging.
    Block *beforeBlock = &whileOp.getBefore().front();
    Block *bodyBlock = rewriter.createBlock(&loweredDo.getBodyRegion());
    rewriter.setInsertionPointToStart(bodyBlock);

    // Load current variable values to use as initial arguments for the
    // condition block.
    SmallVector<Value> replacingValues = loadValues(loopVars, rewriter, loc);
    rewriter.mergeBlocks(beforeBlock, bodyBlock, replacingValues);

    Operation *condTerminator =
        loweredDo.getBodyRegion().back().getTerminator();
    scf::ConditionOp condOp = cast<scf::ConditionOp>(condTerminator);
    rewriter.setInsertionPoint(condOp);

    // Update result variables with values from scf::condition.
    SmallVector<Value> conditionArgs;
    for (Value arg : condOp.getArgs()) {
      conditionArgs.push_back(rewriter.getRemappedValue(arg));
    }
    assignValues(conditionArgs, resultVars, rewriter, loc);

    // Convert scf.condition to condition variable assignment.
    Value condition = rewriter.getRemappedValue(condOp.getCondition());
    rewriter.create<emitc::AssignOp>(loc, conditionVal, condition);

    // Wrap body region in conditional to preserve scf semantics. Only create
    // ifOp if after-region is non-empty.
    if (whileOp.getAfterBody()->getOperations().size() > 1) {
      auto ifOp = rewriter.create<emitc::IfOp>(loc, condition, false, false);

      // Prepare the after region (loop body) for merging.
      Block *afterBlock = &whileOp.getAfter().front();
      Block *ifBodyBlock = rewriter.createBlock(&ifOp.getBodyRegion());

      // Replacement values for after block using condition op arguments.
      SmallVector<Value> afterReplacingValues;
      for (Value arg : condOp.getArgs())
        afterReplacingValues.push_back(rewriter.getRemappedValue(arg));

      rewriter.mergeBlocks(afterBlock, ifBodyBlock, afterReplacingValues);

      if (failed(lowerYield(whileOp, loopVars, rewriter,
                            cast<scf::YieldOp>(ifBodyBlock->getTerminator()))))
        return failure();
    }

    rewriter.eraseOp(condOp);

    // Create condition region that loads from the flag variable.
    Region &condRegion = loweredDo.getConditionRegion();
    Block *condBlock = rewriter.createBlock(&condRegion);
    rewriter.setInsertionPointToStart(condBlock);

    auto exprOp = rewriter.create<emitc::ExpressionOp>(
        loc, i1Type, conditionVal, /*do_not_inline=*/false);
    Block *exprBlock = rewriter.createBlock(&exprOp.getBodyRegion());

    // Set up the expression block to load the condition variable.
    exprBlock->addArgument(conditionVal.getType(), loc);
    rewriter.setInsertionPointToStart(exprBlock);

    // Load the condition value and yield it as the expression result.
    Value cond =
        rewriter.create<emitc::LoadOp>(loc, i1Type, exprBlock->getArgument(0));
    rewriter.create<emitc::YieldOp>(loc, cond);

    // Yield the expression as the condition region result.
    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<emitc::YieldOp>(loc, exprOp);

    return success();
  }
};

void mlir::populateSCFToEmitCConversionPatterns(RewritePatternSet &patterns,
                                                TypeConverter &typeConverter) {
  patterns.add<ForLowering>(typeConverter, patterns.getContext());
  patterns.add<IfLowering>(typeConverter, patterns.getContext());
  patterns.add<IndexSwitchOpLowering>(typeConverter, patterns.getContext());
  patterns.add<WhileLowering>(typeConverter, patterns.getContext());
}

void SCFToEmitCPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  TypeConverter typeConverter;
  // Fallback for other types.
  typeConverter.addConversion([](Type type) -> std::optional<Type> {
    if (!emitc::isSupportedEmitCType(type))
      return {};
    return type;
  });
  populateEmitCSizeTTypeConversions(typeConverter);
  populateSCFToEmitCConversionPatterns(patterns, typeConverter);

  // Configure conversion to lower out SCF operations.
  ConversionTarget target(getContext());
  target
      .addIllegalOp<scf::ForOp, scf::IfOp, scf::IndexSwitchOp, scf::WhileOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
