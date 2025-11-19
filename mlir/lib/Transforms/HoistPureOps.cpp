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

namespace mlir {
#define GEN_PASS_DEF_HOISTPUREOPS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Return the dominated Value.
static Value getDomaincedValue(DominanceInfo &dominanceInfo, Value a, Value b) {
  Block *aB = a.getParentBlock();
  Block *bB = b.getParentBlock();
  if (isa_and_present<BlockArgument>(a) && isa_and_present<BlockArgument>(b)) {
    return dominanceInfo.dominates(aB, bB) ? b : a;
  } else if (isa_and_present<BlockArgument>(a) ||
             isa_and_present<BlockArgument>(b)) {
    if (aB == bB)
      return b;
    return dominanceInfo.dominates(aB, bB) ? b : a;
  } else {
    Operation *aDefineOp = a.getDefiningOp();
    Operation *bDefineOp = b.getDefiningOp();
    return dominanceInfo.dominates(aDefineOp, bDefineOp) ? b : a;
  }
}

/// Find the hoisting position for the pure op.
static Value getDestPos(Operation *op) {
  DominanceInfo dominanceInfo(op);
  SmallVector<Value> operands(op->getOperands());
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
  Value pos = getDestPos(op);
  if (!pos)
    return;

  if (Operation *defineOp = pos.getDefiningOp()) {
    rewriter.moveOpAfter(op, defineOp);
    return;
  }
  auto argument = cast<BlockArgument>(pos);
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
