//===- ACCIfClauseLowering.cpp - Lower ACC compute construct if clauses --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers OpenACC compute constructs (parallel, kernels, serial) with
// `if` clauses using region specialization. It creates two execution paths:
// device execution when the condition is true, host execution when false.
//
// Overview:
// ---------
// When an ACC compute construct has an `if` clause, the construct should only
// execute on the device when the condition is true. If the condition is false,
// the code should execute on the host instead. This pass transforms:
//
//   acc.parallel if(%cond) { ... }
//
// Into:
//
//   scf.if %cond {
//     // Device path: clone data ops, compute construct without if, exit ops
//     acc.parallel { ... }
//   } else {
//     // Host path: original region body with ACC ops converted to host
//   }
//
// Transformations:
// ----------------
// For each compute construct with an `if` clause:
//
// 1. Device Path (true branch):
//    - Clone data entry operations (acc.copyin, acc.create, etc.)
//    - Clone the compute construct without the `if` clause
//    - Clone data exit operations (acc.copyout, acc.delete, etc.)
//
// 2. Host Path (false branch):
//    - Move the original region body to the else branch
//    - Apply host fallback patterns to convert ACC ops to host equivalents
//
// 3. Cleanup:
//    - Erase the original compute construct and data operations
//    - Replace uses of ACC variables with host variables in the else branch
//
// Requirements:
// -------------
// To use this pass in a pipeline, the following requirements exist:
//
// 1. Analysis Registration (Optional): If custom behavior is needed for
//    emitting not-yet-implemented messages for unsupported cases, the pipeline
//    should pre-register the `acc::OpenACCSupport` analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsLoop.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCIFCLAUSELOWERING
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-if-clause-lowering"

using namespace mlir;
using namespace mlir::acc;

namespace {

class ACCIfClauseLowering
    : public acc::impl::ACCIfClauseLoweringBase<ACCIfClauseLowering> {
  using ACCIfClauseLoweringBase<ACCIfClauseLowering>::ACCIfClauseLoweringBase;

private:
  OpenACCSupport *accSupport = nullptr;

  void convertHostRegion(Operation *computeOp, Region &region);

  template <typename OpTy>
  void lowerIfClauseForComputeConstruct(
      OpTy computeConstructOp, SmallVector<Operation *> &eraseOps,
      llvm::SetVector<Operation *> &condEraseOps);

public:
  void runOnOperation() override;
};

void ACCIfClauseLowering::convertHostRegion(Operation *computeOp,
                                            Region &region) {
  // Only collect ACC dialect operations - other ops don't need conversion
  SmallVector<Operation *> hostOps;
  region.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<acc::OpenACCDialect>(op->getDialect()))
      hostOps.push_back(op);
  });

  RewritePatternSet patterns(computeOp->getContext());
  populateACCHostFallbackPatterns(patterns, *accSupport);

  GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  config.setStrictness(GreedyRewriteStrictness::ExistingOps);
  if (failed(applyOpPatternsGreedily(hostOps, std::move(patterns), config)))
    accSupport->emitNYI(computeOp->getLoc(), "failed to convert host region");
}

// Template function to handle if condition conversion for ACC compute
// constructs
template <typename OpTy>
void ACCIfClauseLowering::lowerIfClauseForComputeConstruct(
    OpTy computeConstructOp, SmallVector<Operation *> &eraseOps,
    llvm::SetVector<Operation *> &condEraseOps) {
  Value ifCond = computeConstructOp.getIfCond();
  if (!ifCond)
    return;

  IRRewriter rewriter(computeConstructOp);

  LLVM_DEBUG(llvm::dbgs() << "Converting " << computeConstructOp->getName()
                          << " with if condition: " << computeConstructOp
                          << "\n");

  // Collect data clause operations that need to be recreated in the if
  // condition
  SmallVector<Operation *> dataEntryOps;
  SmallVector<Operation *> dataExitOps;
  SmallVector<Operation *> firstprivateOps;
  SmallVector<Operation *> privateOps;
  SmallVector<Operation *> reductionOps;

  // A data entry op is "externally owned" when it is also used by a construct
  // that encloses this one (e.g. an acc.data wrapping this compute construct
  // that shares the same data entry op). Such ops - and their data exit ops -
  // belong to the enclosing construct and must not be cloned or erased here.
  auto isExternallyOwned = [&](Operation *dataOp) {
    for (Operation *user : dataOp->getUsers())
      for (Operation *anc = computeConstructOp->getParentOp(); anc;
           anc = anc->getParentOp())
        if (anc == user)
          return true;
    return false;
  };

  // Collect data entry operations
  for (Value operand : computeConstructOp.getDataClauseOperands())
    if (Operation *defOp = operand.getDefiningOp())
      if (isa<ACC_DATA_ENTRY_OPS>(defOp))
        dataEntryOps.push_back(defOp);

  // Find corresponding exit operations for each local entry operation. Exit ops
  // of externally owned entry ops belong to the enclosing construct.
  // Iterate backwards through entry ops since exit ops appear in reverse order.
  for (Operation *dataEntryOp : llvm::reverse(dataEntryOps))
    if (!isExternallyOwned(dataEntryOp))
      for (Operation *user : dataEntryOp->getUsers())
        if (isa<ACC_DATA_EXIT_OPS>(user))
          dataExitOps.push_back(user);

  // Collect firstprivate, private, and reduction operations
  auto collectOps = [&](SmallVector<Operation *> &ops, OperandRange operands) {
    for (Value operand : operands)
      if (Operation *defOp = operand.getDefiningOp())
        ops.push_back(defOp);
  };
  collectOps(firstprivateOps, computeConstructOp.getFirstprivateOperands());
  collectOps(privateOps, computeConstructOp.getPrivateOperands());
  collectOps(reductionOps, computeConstructOp.getReductionOperands());

  // Create scf.if with device and host execution paths
  auto ifOp = scf::IfOp::create(rewriter, computeConstructOp.getLoc(),
                                TypeRange{}, ifCond, /*withElseRegion=*/true);

  LLVM_DEBUG(llvm::dbgs() << "Cloning " << dataEntryOps.size()
                          << " data entry operations for device path\n");

  // Device execution path (true branch)
  Block &thenBlock = ifOp.getThenRegion().front();
  rewriter.setInsertionPointToStart(&thenBlock);

  // Clone data entry operations
  SmallVector<Value> deviceDataOperands;
  SmallVector<Value> firstprivateOperands;
  SmallVector<Value> privateOperands;
  SmallVector<Value> reductionOperands;

  // Map the data entry and firstprivate ops for the cloned region
  IRMapping deviceMapping;
  auto cloneAndMapOps = [&](SmallVector<Operation *> &ops,
                            SmallVector<Value> &operands) {
    for (Operation *op : ops) {
      Operation *clonedOp = rewriter.clone(*op, deviceMapping);
      operands.push_back(clonedOp->getResult(0));
      deviceMapping.map(op->getResult(0), clonedOp->getResult(0));
    }
  };
  // Local data entry ops are cloned into the device path; externally owned ones
  // (mapped by an enclosing construct such as acc.data) are referenced directly.
  for (Operation *op : dataEntryOps) {
    if (isExternallyOwned(op)) {
      deviceDataOperands.push_back(op->getResult(0));
      continue;
    }
    Operation *clonedOp = rewriter.clone(*op, deviceMapping);
    deviceDataOperands.push_back(clonedOp->getResult(0));
    deviceMapping.map(op->getResult(0), clonedOp->getResult(0));
  }
  cloneAndMapOps(firstprivateOps, firstprivateOperands);
  cloneAndMapOps(privateOps, privateOperands);
  cloneAndMapOps(reductionOps, reductionOperands);

  // Create new compute op without if condition for device execution by
  // cloning
  OpTy newComputeOp = cast<OpTy>(
      rewriter.clone(*computeConstructOp.getOperation(), deviceMapping));
  newComputeOp.getIfCondMutable().clear();
  newComputeOp.getDataClauseOperandsMutable().assign(deviceDataOperands);
  newComputeOp.getFirstprivateOperandsMutable().assign(firstprivateOperands);
  newComputeOp.getPrivateOperandsMutable().assign(privateOperands);
  newComputeOp.getReductionOperandsMutable().assign(reductionOperands);

  // Clone data exit operations
  rewriter.setInsertionPointAfter(newComputeOp);
  for (Operation *dataOp : dataExitOps)
    rewriter.clone(*dataOp, deviceMapping);

  rewriter.setInsertionPointToEnd(&thenBlock);
  if (!thenBlock.getTerminator())
    scf::YieldOp::create(rewriter, computeConstructOp.getLoc());

  // Host execution path (false branch)
  Region &hostRegion = computeConstructOp.getRegion();
  if (hostRegion.hasOneBlock()) {
    // Don't need to clone original ops, just take them and legalize for host.
    ifOp.getElseRegion().takeBody(hostRegion);

    // Swap acc yield for scf yield.
    Block &elseBlock = ifOp.getElseRegion().front();
    elseBlock.getTerminator()->erase();
    rewriter.setInsertionPointToEnd(&elseBlock);
    scf::YieldOp::create(rewriter, computeConstructOp.getLoc());

    convertHostRegion(computeConstructOp, ifOp.getElseRegion());
  } else {
    // scf.if regions must stay single-block. Wrap the original multi-block ACC
    // body in scf.execute_region so it can be hosted in the else branch.
    Block &elseBlock = ifOp.getElseRegion().front();
    rewriter.setInsertionPoint(elseBlock.getTerminator());
    IRMapping hostMapping;
    auto hostExecuteRegion = wrapMultiBlockRegionWithSCFExecuteRegion(
        hostRegion, hostMapping, computeConstructOp.getLoc(), rewriter);
    convertHostRegion(computeConstructOp, hostExecuteRegion.getRegion());
  }

  // The original op is now empty and can be erased
  eraseOps.push_back(computeConstructOp);

  // TODO: Can probably 'move' the data ops instead of cloning them
  // which would eliminate need to explicitly erase
  for (Operation *dataOp : dataExitOps)
    eraseOps.push_back(dataOp);

  // The host (else) region may contain uses of the acc variables. Redirect
  // only those host-side uses to the host values. Other uses (e.g. an enclosing
  // acc.data that shares the same data entry op) must be preserved, so these ops
  // are only erased once they have no remaining users.
  Region &elseRegion = ifOp.getElseRegion();
  auto replaceHostUsesAndScheduleErase = [&](SmallVector<Operation *> &ops) {
    for (Operation *op : ops) {
      getAccVar(op).replaceUsesWithIf(getVar(op), [&](OpOperand &use) {
        return elseRegion.isAncestor(use.getOwner()->getParentRegion());
      });
      condEraseOps.insert(op);
    }
  };
  replaceHostUsesAndScheduleErase(dataEntryOps);
  replaceHostUsesAndScheduleErase(firstprivateOps);
  replaceHostUsesAndScheduleErase(privateOps);
  replaceHostUsesAndScheduleErase(reductionOps);
}

void ACCIfClauseLowering::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  accSupport = &getAnalysis<OpenACCSupport>();

  SmallVector<Operation *> eraseOps;
  llvm::SetVector<Operation *> condEraseOps;
  funcOp.walk([&](Operation *op) {
    if (auto parallelOp = dyn_cast<acc::ParallelOp>(op))
      lowerIfClauseForComputeConstruct(parallelOp, eraseOps, condEraseOps);
    else if (auto kernelsOp = dyn_cast<acc::KernelsOp>(op))
      lowerIfClauseForComputeConstruct(kernelsOp, eraseOps, condEraseOps);
    else if (auto serialOp = dyn_cast<acc::SerialOp>(op))
      lowerIfClauseForComputeConstruct(serialOp, eraseOps, condEraseOps);
  });

  for (Operation *op : eraseOps)
    op->erase();
  // Data entry/private/reduction ops may be shared with other constructs (e.g.
  // an enclosing acc.data). Erase them only after their users are gone, in
  // reverse insertion order so consumers are removed before producers.
  for (Operation *op : llvm::reverse(condEraseOps))
    if (op->use_empty())
      op->erase();
}

} // namespace
