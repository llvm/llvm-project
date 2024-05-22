//===- ControlFlowToSCF.h - ControlFlow to SCF -------------*- C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the ControlFlow dialect to the SCF dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CFGToSCF.h"

namespace mlir {
#define GEN_PASS_DEF_LIFTCONTROLFLOWTOSCFPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

FailureOr<Operation *>
ControlFlowToSCFTransformation::createStructuredBranchRegionOp(
    OpBuilder &builder, Operation *controlFlowCondOp, TypeRange resultTypes,
    MutableArrayRef<Region> regions) {
  if (auto condBrOp = dyn_cast<cf::CondBranchOp>(controlFlowCondOp)) {
    assert(regions.size() == 2);
    auto ifOp = builder.create<scf::IfOp>(controlFlowCondOp->getLoc(),
                                          resultTypes, condBrOp.getCondition());
    ifOp.getThenRegion().takeBody(regions[0]);
    ifOp.getElseRegion().takeBody(regions[1]);
    return ifOp.getOperation();
  }

  if (auto switchOp = dyn_cast<cf::SwitchOp>(controlFlowCondOp)) {
    // `getCFGSwitchValue` returns an i32 that we need to convert to index
    // fist.
    auto cast = builder.create<arith::IndexCastUIOp>(
        controlFlowCondOp->getLoc(), builder.getIndexType(),
        switchOp.getFlag());
    SmallVector<int64_t> cases;
    if (auto caseValues = switchOp.getCaseValues())
      llvm::append_range(
          cases, llvm::map_range(*caseValues, [](const llvm::APInt &apInt) {
            return apInt.getZExtValue();
          }));

    assert(regions.size() == cases.size() + 1);

    auto indexSwitchOp = builder.create<scf::IndexSwitchOp>(
        controlFlowCondOp->getLoc(), resultTypes, cast, cases, cases.size());

    indexSwitchOp.getDefaultRegion().takeBody(regions[0]);
    for (auto &&[targetRegion, sourceRegion] :
         llvm::zip(indexSwitchOp.getCaseRegions(), llvm::drop_begin(regions)))
      targetRegion.takeBody(sourceRegion);

    return indexSwitchOp.getOperation();
  }

  controlFlowCondOp->emitOpError(
      "Cannot convert unknown control flow op to structured control flow");
  return failure();
}

LogicalResult
ControlFlowToSCFTransformation::createStructuredBranchRegionTerminatorOp(
    Location loc, OpBuilder &builder, Operation *branchRegionOp,
    Operation *replacedControlFlowOp, ValueRange results) {
  builder.create<scf::YieldOp>(loc, results);
  return success();
}

FailureOr<Operation *>
ControlFlowToSCFTransformation::createStructuredDoWhileLoopOp(
    OpBuilder &builder, Operation *replacedOp, ValueRange loopVariablesInit,
    Value condition, ValueRange loopVariablesNextIter, Region &&loopBody) {
  Location loc = replacedOp->getLoc();
  auto whileOp = builder.create<scf::WhileOp>(loc, loopVariablesInit.getTypes(),
                                              loopVariablesInit);

  whileOp.getBefore().takeBody(loopBody);

  builder.setInsertionPointToEnd(&whileOp.getBefore().back());
  // `getCFGSwitchValue` returns a i32. We therefore need to truncate the
  // condition to i1 first. It is guaranteed to be either 0 or 1 already.
  builder.create<scf::ConditionOp>(
      loc, builder.create<arith::TruncIOp>(loc, builder.getI1Type(), condition),
      loopVariablesNextIter);

  auto *afterBlock = new Block;
  whileOp.getAfter().push_back(afterBlock);
  afterBlock->addArguments(
      loopVariablesInit.getTypes(),
      SmallVector<Location>(loopVariablesInit.size(), loc));
  builder.setInsertionPointToEnd(afterBlock);
  builder.create<scf::YieldOp>(loc, afterBlock->getArguments());

  return whileOp.getOperation();
}

Value ControlFlowToSCFTransformation::getCFGSwitchValue(Location loc,
                                                        OpBuilder &builder,
                                                        unsigned int value) {
  return builder.create<arith::ConstantOp>(loc,
                                           builder.getI32IntegerAttr(value));
}

void ControlFlowToSCFTransformation::createCFGSwitchOp(
    Location loc, OpBuilder &builder, Value flag,
    ArrayRef<unsigned int> caseValues, BlockRange caseDestinations,
    ArrayRef<ValueRange> caseArguments, Block *defaultDest,
    ValueRange defaultArgs) {
  builder.create<cf::SwitchOp>(loc, flag, defaultDest, defaultArgs,
                               llvm::to_vector_of<int32_t>(caseValues),
                               caseDestinations, caseArguments);
}

Value ControlFlowToSCFTransformation::getUndefValue(Location loc,
                                                    OpBuilder &builder,
                                                    Type type) {
  return builder.create<ub::PoisonOp>(loc, type, nullptr);
}

FailureOr<Operation *>
ControlFlowToSCFTransformation::createUnreachableTerminator(Location loc,
                                                            OpBuilder &builder,
                                                            Region &region) {

  // TODO: This should create a `ub.unreachable` op. Once such an operation
  //       exists to make the pass independent of the func dialect. For now just
  //       return poison values.
  Operation *parentOp = region.getParentOp();
  auto funcOp = dyn_cast<func::FuncOp>(parentOp);
  if (!funcOp)
    return emitError(loc, "Cannot create unreachable terminator for '")
           << parentOp->getName() << "'";

  return builder
      .create<func::ReturnOp>(
          loc, llvm::map_to_vector(funcOp.getResultTypes(),
                                   [&](Type type) {
                                     return getUndefValue(loc, builder, type);
                                   }))
      .getOperation();
}

namespace {

struct LiftControlFlowToSCF
    : public impl::LiftControlFlowToSCFPassBase<LiftControlFlowToSCF> {

  using Base::Base;

  void runOnOperation() override {
    ControlFlowToSCFTransformation transformation;

    bool changed = false;
    Operation *op = getOperation();
    WalkResult result = op->walk([&](func::FuncOp funcOp) {
      if (funcOp.getBody().empty())
        return WalkResult::advance();

      auto &domInfo = funcOp != op ? getChildAnalysis<DominanceInfo>(funcOp)
                                   : getAnalysis<DominanceInfo>();

      auto visitor = [&](Operation *innerOp) -> WalkResult {
        for (Region &reg : innerOp->getRegions()) {
          FailureOr<bool> changedFunc =
              transformCFGToSCF(reg, transformation, domInfo);
          if (failed(changedFunc))
            return WalkResult::interrupt();

          changed |= *changedFunc;
        }
        return WalkResult::advance();
      };

      if (funcOp->walk<WalkOrder::PostOrder>(visitor).wasInterrupted())
        return WalkResult::interrupt();

      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();

    if (!changed)
      markAllAnalysesPreserved();
  }
};
} // namespace
