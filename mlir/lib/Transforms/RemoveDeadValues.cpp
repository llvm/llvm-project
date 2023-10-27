//===- RemoveDeadValues.cpp - Remove Dead Values --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The goal of this pass is optimization (reducing runtime) by removing
// unnecessary instructions. Unlike other passes that rely on local information
// gathered from patterns to accomplish optimization, this pass uses a full
// analysis of the IR, specifically, liveness analysis, and is thus more
// powerful.
//
// Currently, this pass performs the following optimizations:
// (A) Removes function arguments that are not live,
// (B) Removes function return values that are not live across all callers of
// the function,
// (C) Removes unneccesary operands, results, region arguments, and region
// terminator operands of region branch ops, and,
// (D) Removes simple and region branch ops that have all non-live results and
// don't affect memory in any way,
//
// iff
//
// the IR doesn't have any non-function symbol ops, non-call symbol user ops and
// branch ops.
//
// Here, a "simple op" refers to an op that isn't a symbol op, symbol-user op,
// region branch op, branch op, region branch terminator op, or return-like.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/LivenessAnalysis.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_REMOVEDEADVALUES
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// RemoveDeadValues Pass
//===----------------------------------------------------------------------===//

namespace {

// Some helper functions...

/// Return true iff at least one value in `values` is live, given the liveness
/// information in `la`.
static bool hasLive(ValueRange values, RunLivenessAnalysis &la) {
  for (Value value : values) {
    // If there is a null value, it implies that it was dropped during the
    // execution of this pass, implying that it was non-live.
    if (!value)
      continue;

    const Liveness *liveness = la.getLiveness(value);
    if (!liveness || liveness->isLive)
      return true;
  }
  return false;
}

/// Return a BitVector of size `values.size()` where its i-th bit is 1 iff the
/// i-th value in `values` is live, given the liveness information in `la`.
static BitVector markLives(ValueRange values, RunLivenessAnalysis &la) {
  BitVector lives(values.size(), true);

  for (auto [index, value] : llvm::enumerate(values)) {
    if (!value) {
      lives.reset(index);
      continue;
    }

    const Liveness *liveness = la.getLiveness(value);
    // It is important to note that when `liveness` is null, we can't tell if
    // `value` is live or not. So, the safe option is to consider it live. Also,
    // the execution of this pass might create new SSA values when erasing some
    // of the results of an op and we know that these new values are live
    // (because they weren't erased) and also their liveness is null because
    // liveness analysis ran before their creation.
    if (liveness && !liveness->isLive)
      lives.reset(index);
  }

  return lives;
}

/// Drop the uses of the i-th result of `op` and then erase it iff toErase[i]
/// is 1.
static void dropUsesAndEraseResults(Operation *op, BitVector toErase) {
  assert(op->getNumResults() == toErase.size() &&
         "expected the number of results in `op` and the size of `toErase` to "
         "be the same");

  std::vector<Type> newResultTypes;
  for (OpResult result : op->getResults())
    if (!toErase[result.getResultNumber()])
      newResultTypes.push_back(result.getType());
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  OperationState state(op->getLoc(), op->getName().getStringRef(),
                       op->getOperands(), newResultTypes, op->getAttrs());
  for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
    state.addRegion();
  Operation *newOp = builder.create(state);
  for (const auto &[index, region] : llvm::enumerate(op->getRegions())) {
    Region &newRegion = newOp->getRegion(index);
    // Move all blocks of `region` into `newRegion`.
    Block *temp = new Block();
    newRegion.push_back(temp);
    while (!region.empty())
      region.front().moveBefore(temp);
    temp->erase();
  }

  unsigned indexOfNextNewCallOpResultToReplace = 0;
  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    assert(result && "expected result to be non-null");
    if (toErase[index]) {
      result.dropAllUses();
    } else {
      result.replaceAllUsesWith(
          newOp->getResult(indexOfNextNewCallOpResultToReplace++));
    }
  }
  op->erase();
}

/// Convert a list of `Operand`s to a list of `OpOperand`s.
static SmallVector<OpOperand *> operandsToOpOperands(OperandRange operands) {
  OpOperand *values = operands.getBase();
  SmallVector<OpOperand *> opOperands;
  for (unsigned i = 0, e = operands.size(); i < e; i++)
    opOperands.push_back(&values[i]);
  return opOperands;
}

/// Clean a simple op `op`, given the liveness analysis information in `la`.
/// Here, cleaning means:
///   (1) Dropping all its uses, AND
///   (2) Erasing it
/// iff it has no memory effects and none of its results are live.
///
/// It is assumed that `op` is simple. Here, a simple op is one which isn't a
/// symbol op, a symbol-user op, a region branch op, a branch op, a region
/// branch terminator op, or return-like.
static void cleanSimpleOp(Operation *op, RunLivenessAnalysis &la) {
  if (!isMemoryEffectFree(op) || hasLive(op->getResults(), la))
    return;

  op->dropAllUses();
  op->erase();
}

/// Clean a function-like op `funcOp`, given the liveness information in `la`
/// and the IR in `module`. Here, cleaning means:
///   (1) Dropping the uses of its unnecessary (non-live) arguments,
///   (2) Erasing these arguments,
///   (3) Erasing their corresponding operands from its callers,
///   (4) Erasing its unnecessary terminator operands (return values that are
///   non-live across all callers),
///   (5) Dropping the uses of these return values from its callers, AND
///   (6) Erasing these return values
/// iff it is not public.
static void cleanFuncOp(FunctionOpInterface funcOp, Operation *module,
                        RunLivenessAnalysis &la) {
  if (funcOp.isPublic())
    return;

  // Get the list of unnecessary (non-live) arguments in `nonLiveArgs`.
  SmallVector<Value> arguments(funcOp.getArguments());
  BitVector nonLiveArgs = markLives(arguments, la);
  nonLiveArgs = nonLiveArgs.flip();

  // Do (1).
  for (auto [index, arg] : llvm::enumerate(arguments))
    if (arg && nonLiveArgs[index])
      arg.dropAllUses();

  // Do (2).
  funcOp.eraseArguments(nonLiveArgs);

  // Do (3).
  SymbolTable::UseRange uses = *funcOp.getSymbolUses(module);
  for (SymbolTable::SymbolUse use : uses) {
    Operation *callOp = use.getUser();
    assert(isa<CallOpInterface>(callOp) && "expected a call-like user");
    // The number of operands in the call op may not match the number of
    // arguments in the func op.
    BitVector nonLiveCallOperands(callOp->getNumOperands(), false);
    SmallVector<OpOperand *> callOpOperands =
        operandsToOpOperands(cast<CallOpInterface>(callOp).getArgOperands());
    for (int index : nonLiveArgs.set_bits())
      nonLiveCallOperands.set(callOpOperands[index]->getOperandNumber());
    callOp->eraseOperands(nonLiveCallOperands);
  }

  // Get the list of unnecessary terminator operands (return values that are
  // non-live across all callers) in `nonLiveRets`. There is a very important
  // subtlety here. Unnecessary terminator operands are NOT the operands of the
  // terminator that are non-live. Instead, these are the return values of the
  // callers such that a given return value is non-live across all callers. Such
  // corresponding operands in the terminator could be live. An example to
  // demonstrate this:
  //  func.func private @f(%arg0: memref<i32>) -> (i32, i32) {
  //    %c0_i32 = arith.constant 0 : i32
  //    %0 = arith.addi %c0_i32, %c0_i32 : i32
  //    memref.store %0, %arg0[] : memref<i32>
  //    return %c0_i32, %0 : i32, i32
  //  }
  //  func.func @main(%arg0: i32, %arg1: memref<i32>) -> (i32) {
  //    %1:2 = call @f(%arg1) : (memref<i32>) -> i32
  //    return %1#0 : i32
  //  }
  // Here, we can see that %1#1 is never used. It is non-live. Thus, @f doesn't
  // need to return %0. But, %0 is live. And, still, we want to stop it from
  // being returned, in order to optimize our IR. So, this demonstrates how we
  // can make our optimization strong by even removing a live return value (%0),
  // since it forwards only to non-live value(s) (%1#1).
  Operation *lastReturnOp = funcOp.back().getTerminator();
  size_t numReturns = lastReturnOp->getNumOperands();
  BitVector nonLiveRets(numReturns, true);
  for (SymbolTable::SymbolUse use : uses) {
    Operation *callOp = use.getUser();
    assert(isa<CallOpInterface>(callOp) && "expected a call-like user");
    BitVector liveCallRets = markLives(callOp->getResults(), la);
    nonLiveRets &= liveCallRets.flip();
  }

  // Do (4).
  // Note that in the absence of control flow ops forcing the control to go from
  // the entry (first) block to the other blocks, the control never reaches any
  // block other than the entry block, because every block has a terminator.
  for (Block &block : funcOp.getBlocks()) {
    Operation *returnOp = block.getTerminator();
    if (returnOp && returnOp->getNumOperands() == numReturns)
      returnOp->eraseOperands(nonLiveRets);
  }
  funcOp.eraseResults(nonLiveRets);

  // Do (5) and (6).
  for (SymbolTable::SymbolUse use : uses) {
    Operation *callOp = use.getUser();
    assert(isa<CallOpInterface>(callOp) && "expected a call-like user");
    dropUsesAndEraseResults(callOp, nonLiveRets);
  }
}

/// Clean a region branch op `regionBranchOp`, given the liveness information in
/// `la`. Here, cleaning means:
///   (1') Dropping all its uses, AND
///   (2') Erasing it
/// if it has no memory effects and none of its results are live, AND
///   (1) Erasing its unnecessary operands (operands that are forwarded to
///   unneccesary results and arguments),
///   (2) Cleaning each of its regions,
///   (3) Dropping the uses of its unnecessary results (results that are
///   forwarded from unnecessary operands and terminator operands), AND
///   (4) Erasing these results
/// otherwise.
/// Note that here, cleaning a region means:
///   (2.a) Dropping the uses of its unnecessary arguments (arguments that are
///   forwarded from unneccesary operands and terminator operands),
///   (2.b) Erasing these arguments, AND
///   (2.c) Erasing its unnecessary terminator operands (terminator operands
///   that are forwarded to unneccesary results and arguments).
/// It is important to note that values in this op flow from operands and
/// terminator operands (successor operands) to arguments and results (successor
/// inputs).
static void cleanRegionBranchOp(RegionBranchOpInterface regionBranchOp,
                                RunLivenessAnalysis &la) {
  // Mark live results of `regionBranchOp` in `liveResults`.
  auto markLiveResults = [&](BitVector &liveResults) {
    liveResults = markLives(regionBranchOp->getResults(), la);
  };

  // Mark live arguments in the regions of `regionBranchOp` in `liveArgs`.
  auto markLiveArgs = [&](DenseMap<Region *, BitVector> &liveArgs) {
    for (Region &region : regionBranchOp->getRegions()) {
      SmallVector<Value> arguments(region.front().getArguments());
      BitVector regionLiveArgs = markLives(arguments, la);
      liveArgs[&region] = regionLiveArgs;
    }
  };

  // Return the successors of `region` if the latter is not null. Else return
  // the successors of `regionBranchOp`.
  auto getSuccessors = [&](Region *region = nullptr) {
    auto point = region ? region : RegionBranchPoint::parent();
    SmallVector<Attribute> operandAttributes(regionBranchOp->getNumOperands(),
                                             nullptr);
    SmallVector<RegionSuccessor> successors;
    regionBranchOp.getSuccessorRegions(point, successors);
    return successors;
  };

  // Return the operands of `terminator` that are forwarded to `successor` if
  // the former is not null. Else return the operands of `regionBranchOp`
  // forwarded to `successor`.
  auto getForwardedOpOperands = [&](const RegionSuccessor &successor,
                                    Operation *terminator = nullptr) {
    OperandRange operands =
        terminator ? cast<RegionBranchTerminatorOpInterface>(terminator)
                         .getSuccessorOperands(successor)
                   : regionBranchOp.getEntrySuccessorOperands(successor);
    SmallVector<OpOperand *> opOperands = operandsToOpOperands(operands);
    return opOperands;
  };

  // Mark the non-forwarded operands of `regionBranchOp` in
  // `nonForwardedOperands`.
  auto markNonForwardedOperands = [&](BitVector &nonForwardedOperands) {
    nonForwardedOperands.resize(regionBranchOp->getNumOperands(), true);
    for (const RegionSuccessor &successor : getSuccessors()) {
      for (OpOperand *opOperand : getForwardedOpOperands(successor))
        nonForwardedOperands.reset(opOperand->getOperandNumber());
    }
  };

  // Mark the non-forwarded terminator operands of the various regions of
  // `regionBranchOp` in `nonForwardedRets`.
  auto markNonForwardedReturnValues =
      [&](DenseMap<Operation *, BitVector> &nonForwardedRets) {
        for (Region &region : regionBranchOp->getRegions()) {
          Operation *terminator = region.front().getTerminator();
          nonForwardedRets[terminator] =
              BitVector(terminator->getNumOperands(), true);
          for (const RegionSuccessor &successor : getSuccessors(&region)) {
            for (OpOperand *opOperand :
                 getForwardedOpOperands(successor, terminator))
              nonForwardedRets[terminator].reset(opOperand->getOperandNumber());
          }
        }
      };

  // Update `valuesToKeep` (which is expected to correspond to operands or
  // terminator operands) based on `resultsToKeep` and `argsToKeep`, given
  // `region`. When `valuesToKeep` correspond to operands, `region` is null.
  // Else, `region` is the parent region of the terminator.
  auto updateOperandsOrTerminatorOperandsToKeep =
      [&](BitVector &valuesToKeep, BitVector &resultsToKeep,
          DenseMap<Region *, BitVector> &argsToKeep, Region *region = nullptr) {
        Operation *terminator =
            region ? region->front().getTerminator() : nullptr;

        for (const RegionSuccessor &successor : getSuccessors(region)) {
          Region *successorRegion = successor.getSuccessor();
          for (auto [opOperand, input] :
               llvm::zip(getForwardedOpOperands(successor, terminator),
                         successor.getSuccessorInputs())) {
            size_t operandNum = opOperand->getOperandNumber();
            bool updateBasedOn =
                successorRegion
                    ? argsToKeep[successorRegion]
                                [cast<BlockArgument>(input).getArgNumber()]
                    : resultsToKeep[cast<OpResult>(input).getResultNumber()];
            valuesToKeep[operandNum] = valuesToKeep[operandNum] | updateBasedOn;
          }
        }
      };

  // Recompute `resultsToKeep` and `argsToKeep` based on `operandsToKeep` and
  // `terminatorOperandsToKeep`. Store true in `resultsOrArgsToKeepChanged` if a
  // value is modified, else, false.
  auto recomputeResultsAndArgsToKeep =
      [&](BitVector &resultsToKeep, DenseMap<Region *, BitVector> &argsToKeep,
          BitVector &operandsToKeep,
          DenseMap<Operation *, BitVector> &terminatorOperandsToKeep,
          bool &resultsOrArgsToKeepChanged) {
        resultsOrArgsToKeepChanged = false;

        // Recompute `resultsToKeep` and `argsToKeep` based on `operandsToKeep`.
        for (const RegionSuccessor &successor : getSuccessors()) {
          Region *successorRegion = successor.getSuccessor();
          for (auto [opOperand, input] :
               llvm::zip(getForwardedOpOperands(successor),
                         successor.getSuccessorInputs())) {
            bool recomputeBasedOn =
                operandsToKeep[opOperand->getOperandNumber()];
            bool toRecompute =
                successorRegion
                    ? argsToKeep[successorRegion]
                                [cast<BlockArgument>(input).getArgNumber()]
                    : resultsToKeep[cast<OpResult>(input).getResultNumber()];
            if (!toRecompute && recomputeBasedOn)
              resultsOrArgsToKeepChanged = true;
            if (successorRegion) {
              argsToKeep[successorRegion][cast<BlockArgument>(input)
                                              .getArgNumber()] =
                  argsToKeep[successorRegion]
                            [cast<BlockArgument>(input).getArgNumber()] |
                  recomputeBasedOn;
            } else {
              resultsToKeep[cast<OpResult>(input).getResultNumber()] =
                  resultsToKeep[cast<OpResult>(input).getResultNumber()] |
                  recomputeBasedOn;
            }
          }
        }

        // Recompute `resultsToKeep` and `argsToKeep` based on
        // `terminatorOperandsToKeep`.
        for (Region &region : regionBranchOp->getRegions()) {
          Operation *terminator = region.front().getTerminator();
          for (const RegionSuccessor &successor : getSuccessors(&region)) {
            Region *successorRegion = successor.getSuccessor();
            for (auto [opOperand, input] :
                 llvm::zip(getForwardedOpOperands(successor, terminator),
                           successor.getSuccessorInputs())) {
              bool recomputeBasedOn =
                  terminatorOperandsToKeep[region.back().getTerminator()]
                                          [opOperand->getOperandNumber()];
              bool toRecompute =
                  successorRegion
                      ? argsToKeep[successorRegion]
                                  [cast<BlockArgument>(input).getArgNumber()]
                      : resultsToKeep[cast<OpResult>(input).getResultNumber()];
              if (!toRecompute && recomputeBasedOn)
                resultsOrArgsToKeepChanged = true;
              if (successorRegion) {
                argsToKeep[successorRegion][cast<BlockArgument>(input)
                                                .getArgNumber()] =
                    argsToKeep[successorRegion]
                              [cast<BlockArgument>(input).getArgNumber()] |
                    recomputeBasedOn;
              } else {
                resultsToKeep[cast<OpResult>(input).getResultNumber()] =
                    resultsToKeep[cast<OpResult>(input).getResultNumber()] |
                    recomputeBasedOn;
              }
            }
          }
        }
      };

  // Mark the values that we want to keep in `resultsToKeep`, `argsToKeep`,
  // `operandsToKeep`, and `terminatorOperandsToKeep`.
  auto markValuesToKeep =
      [&](BitVector &resultsToKeep, DenseMap<Region *, BitVector> &argsToKeep,
          BitVector &operandsToKeep,
          DenseMap<Operation *, BitVector> &terminatorOperandsToKeep) {
        bool resultsOrArgsToKeepChanged = true;
        // We keep updating and recomputing the values until we reach a point
        // where they stop changing.
        while (resultsOrArgsToKeepChanged) {
          // Update the operands that need to be kept.
          updateOperandsOrTerminatorOperandsToKeep(operandsToKeep,
                                                   resultsToKeep, argsToKeep);

          // Update the terminator operands that need to be kept.
          for (Region &region : regionBranchOp->getRegions()) {
            updateOperandsOrTerminatorOperandsToKeep(
                terminatorOperandsToKeep[region.back().getTerminator()],
                resultsToKeep, argsToKeep, &region);
          }

          // Recompute the results and arguments that need to be kept.
          recomputeResultsAndArgsToKeep(
              resultsToKeep, argsToKeep, operandsToKeep,
              terminatorOperandsToKeep, resultsOrArgsToKeepChanged);
        }
      };

  // Do (1') and (2'). This is the only case where the entire `regionBranchOp`
  // is removed. It will not happen in any other scenario. Note that in this
  // case, a non-forwarded operand of `regionBranchOp` could be live/non-live.
  // It could never be live because of this op but its liveness could have been
  // attributed to something else.
  if (isMemoryEffectFree(regionBranchOp.getOperation()) &&
      !hasLive(regionBranchOp->getResults(), la)) {
    regionBranchOp->dropAllUses();
    regionBranchOp->erase();
    return;
  }

  // At this point, we know that every non-forwarded operand of `regionBranchOp`
  // is live.

  // Stores the results of `regionBranchOp` that we want to keep.
  BitVector resultsToKeep;
  // Stores the mapping from regions of `regionBranchOp` to their arguments that
  // we want to keep.
  DenseMap<Region *, BitVector> argsToKeep;
  // Stores the operands of `regionBranchOp` that we want to keep.
  BitVector operandsToKeep;
  // Stores the mapping from region terminators in `regionBranchOp` to their
  // operands that we want to keep.
  DenseMap<Operation *, BitVector> terminatorOperandsToKeep;

  // Initializing the above variables...

  // The live results of `regionBranchOp` definitely need to be kept.
  markLiveResults(resultsToKeep);
  // Similarly, the live arguments of the regions in `regionBranchOp` definitely
  // need to be kept.
  markLiveArgs(argsToKeep);
  // The non-forwarded operands of `regionBranchOp` definitely need to be kept.
  // A live forwarded operand can be removed but no non-forwarded operand can be
  // removed since it "controls" the flow of data in this control flow op.
  markNonForwardedOperands(operandsToKeep);
  // Similarly, the non-forwarded terminator operands of the regions in
  // `regionBranchOp` definitely need to be kept.
  markNonForwardedReturnValues(terminatorOperandsToKeep);

  // Mark the values (results, arguments, operands, and terminator operands)
  // that we want to keep.
  markValuesToKeep(resultsToKeep, argsToKeep, operandsToKeep,
                   terminatorOperandsToKeep);

  // Do (1).
  regionBranchOp->eraseOperands(operandsToKeep.flip());

  // Do (2.a) and (2.b).
  for (Region &region : regionBranchOp->getRegions()) {
    assert(!region.empty() && "expected a non-empty region in an op "
                              "implementing `RegionBranchOpInterface`");
    for (auto [index, arg] : llvm::enumerate(region.front().getArguments())) {
      if (argsToKeep[&region][index])
        continue;
      if (arg)
        arg.dropAllUses();
    }
    region.front().eraseArguments(argsToKeep[&region].flip());
  }

  // Do (2.c).
  for (Region &region : regionBranchOp->getRegions()) {
    Operation *terminator = region.front().getTerminator();
    terminator->eraseOperands(terminatorOperandsToKeep[terminator].flip());
  }

  // Do (3) and (4).
  dropUsesAndEraseResults(regionBranchOp.getOperation(), resultsToKeep.flip());
}

struct RemoveDeadValues : public impl::RemoveDeadValuesBase<RemoveDeadValues> {
  void runOnOperation() override;
};
} // namespace

void RemoveDeadValues::runOnOperation() {
  auto &la = getAnalysis<RunLivenessAnalysis>();
  Operation *module = getOperation();

  // The removal of non-live values is performed iff there are no branch ops,
  // all symbol ops present in the IR are function-like, and all symbol user ops
  // present in the IR are call-like.
  WalkResult acceptableIR = module->walk([&](Operation *op) {
    if (isa<BranchOpInterface>(op) ||
        (isa<SymbolOpInterface>(op) && !isa<FunctionOpInterface>(op)) ||
        (isa<SymbolUserOpInterface>(op) && !isa<CallOpInterface>(op))) {
      op->emitError() << "cannot optimize an IR with non-function symbol ops, "
                         "non-call symbol user ops or branch ops\n";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (acceptableIR.wasInterrupted())
    return;

  module->walk([&](Operation *op) {
    if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
      cleanFuncOp(funcOp, module, la);
    } else if (auto regionBranchOp = dyn_cast<RegionBranchOpInterface>(op)) {
      cleanRegionBranchOp(regionBranchOp, la);
    } else if (op->hasTrait<OpTrait::ReturnLike>()) {
      // Nothing to do because this terminator is associated with either a
      // function op or a region branch op and gets cleaned when these ops are
      // cleaned.
    } else if (isa<RegionBranchTerminatorOpInterface>(op)) {
      // Nothing to do because this terminator is associated with a region
      // branch op and gets cleaned when the latter is cleaned.
    } else if (isa<CallOpInterface>(op)) {
      // Nothing to do because this op is associated with a function op and gets
      // cleaned when the latter is cleaned.
    } else {
      cleanSimpleOp(op, la);
    }
  });
}

std::unique_ptr<Pass> mlir::createRemoveDeadValuesPass() {
  return std::make_unique<RemoveDeadValues>();
}
