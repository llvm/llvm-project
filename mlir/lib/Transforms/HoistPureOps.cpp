//===- HoistPureOps.cpp - Hoist Pure ops ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the function of hoist the pure op based on SSA
// dominance.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/DebugLog.h"

namespace mlir {
#define GEN_PASS_DEF_HOISTPUREOPS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hoist-pure-ops"

using namespace mlir;

namespace {

/// Return the dominated Value.
static Value getDomaincedValue(DominanceInfo &dominanceInfo, Value a, Value b) {
  Block *aB = a.getParentBlock();
  Block *bB = b.getParentBlock();
  if (isa<BlockArgument>(a) && isa<BlockArgument>(b)) {
    return dominanceInfo.dominates(aB, bB) ? b : a;
  } else if (isa<BlockArgument>(a) || isa<BlockArgument>(b)) {
    if (aB != bB)
      return dominanceInfo.dominates(aB, bB) ? b : a;
    if (auto aArg = dyn_cast<BlockArgument>(a)) {
      Operation *aFrontOp = &aArg.getOwner()->front();
      if (aFrontOp == b.getDefiningOp())
        return b;
      return dominanceInfo.dominates(aFrontOp, b.getDefiningOp()) ? b : a;
    }
    auto bArg = cast<BlockArgument>(b);
    Operation *bFrontOp = &bArg.getOwner()->front();
    if (bFrontOp == a.getDefiningOp())
      return a;
    return dominanceInfo.dominates(a.getDefiningOp(), bFrontOp) ? b : a;
  } else {
    Operation *aDefineOp = a.getDefiningOp();
    Operation *bDefineOp = b.getDefiningOp();
    return dominanceInfo.dominates(aDefineOp, bDefineOp) ? b : a;
  }
}

static bool isOpContainBlock(Operation *op, Block *block) {
  Operation *parentOp = block->getParentOp();
  while (parentOp && parentOp != op) {
    parentOp = parentOp->getParentOp();
  }
  return parentOp == op ? true : false;
}

/// Find the hoisting position for the pure op.
static Value getDestPos(Operation *op) {
  DominanceInfo dominanceInfo(op);
  SmallVector<Value> operands(op->getOperands());
  if (op->getNumRegions()) {
    op->walk([&](Operation *operation) {
      for (auto operand : operation->getOperands()) {
        Operation *defineOp = operand.getDefiningOp();
        if (!defineOp) {
          BlockArgument argument = cast<BlockArgument>(operand);
          if (!isOpContainBlock(op, argument.getOwner()))
            operands.push_back(operand);
          continue;
        }
        if (!isOpContainBlock(op, defineOp->getBlock())) {
          operands.push_back(operand);
        }
      }
    });
  }
  if (operands.empty())
    return {};
  Value ret = operands[0];
  for (int i = 1, e = operands.size(); i < e; ++i) {
    ret = getDomaincedValue(dominanceInfo, ret, operands[i]);
  }
  return ret;
}

/// Hoist single pure op.
static void hoistPureOp(RewriterBase &rewriter, Operation *op) {
  LDBG() << "hoistPureOp: " << OpWithFlags(op, OpPrintingFlags().skipRegions());
  Value pos = getDestPos(op);
  if (!pos)
    return;

  if (Operation *defineOp = pos.getDefiningOp()) {
    if (op == defineOp)
      return;

    LDBG() << "move " << OpWithFlags(op, OpPrintingFlags().skipRegions())
           << " after "
           << OpWithFlags(defineOp, OpPrintingFlags().skipRegions());
    rewriter.moveOpAfter(op, defineOp);
    return;
  }
  auto argument = cast<BlockArgument>(pos);
  LDBG() << "move " << OpWithFlags(op, OpPrintingFlags().skipRegions())
         << " before "
         << OpWithFlags(&argument.getOwner()->front(),
                        OpPrintingFlags().skipRegions());
  rewriter.moveOpBefore(op, &argument.getOwner()->front());
}

struct HoistPureOps : public impl::HoistPureOpsBase<HoistPureOps> {
  void runOnOperation() override;
};
} // namespace

void HoistPureOps::runOnOperation() {
  Operation *module = getOperation();
  IRRewriter rewriter(module->getContext());
  module->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->hasTrait<mlir::OpTrait::IsTerminator>())
      return;
    if (isPure(op)) {
      hoistPureOp(rewriter, op);
    }
  });
}
