//===- SincosFusion.cpp - Fuse sin/cos into sincos -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::math;

namespace mlir::math {
#define GEN_PASS_DEF_MATHSINCOSFUSIONPASS
#include "mlir/Dialect/Math/Transforms/Passes.h.inc"
} // namespace mlir::math

namespace {

/// A math.sin and a math.cos in the same block, on the same operand and with
/// identical fastmath flags, that can be replaced by a single math.sincos.
struct SincosPair {
  math::SinOp sinOp;
  math::CosOp cosOp;
  /// Whichever of the two comes first in the block; the math.sincos is
  /// inserted there so that its results dominate both uses.
  Operation *firstOp;
};

/// Find the math.cos that should be fused with `sinOp`: the earliest one in the
/// same block that uses the same operand with the same fastmath flags and has
/// not already been paired with another math.sin.
static math::CosOp
findFusionCandidate(math::SinOp sinOp,
                    const llvm::DenseSet<Operation *> &pairedCosOps) {
  Value operand = sinOp.getOperand();
  arith::FastMathFlags sinFastMathFlags = sinOp.getFastmath();
  Block *block = sinOp->getBlock();

  math::CosOp candidate = nullptr;
  for (Operation *user : operand.getUsers()) {
    auto cosOp = dyn_cast<math::CosOp>(user);
    if (!cosOp || cosOp->getBlock() != block)
      continue;
    if (cosOp.getFastmath() != sinFastMathFlags)
      continue;
    if (pairedCosOps.contains(cosOp))
      continue;
    // The operand use list is not in program order, so keep the earliest
    // candidate to make the choice independent of use list order.
    if (!candidate || cosOp->isBeforeInBlock(candidate))
      candidate = cosOp;
  }
  return candidate;
}

struct MathSincosFusionPass final
    : math::impl::MathSincosFusionPassBase<MathSincosFusionPass> {
  using MathSincosFusionPassBase::MathSincosFusionPassBase;

  void runOnOperation() override {
    // Collect the pairs before touching the IR: fusing erases the math.cos,
    // which may be the operation the walk is about to visit next.
    llvm::SmallVector<SincosPair> pairs;
    llvm::DenseSet<Operation *> pairedCosOps;
    getOperation()->walk([&](math::SinOp sinOp) {
      math::CosOp cosOp = findFusionCandidate(sinOp, pairedCosOps);
      if (!cosOp)
        return;
      pairedCosOps.insert(cosOp);
      Operation *firstOp = sinOp->isBeforeInBlock(cosOp) ? sinOp.getOperation()
                                                         : cosOp.getOperation();
      pairs.push_back({sinOp, cosOp, firstOp});
    });

    IRRewriter rewriter(&getContext());
    for (SincosPair &pair : pairs) {
      rewriter.setInsertionPoint(pair.firstOp);
      Type elemType = pair.sinOp.getType();
      auto sincos = math::SincosOp::create(
          rewriter, pair.firstOp->getLoc(), TypeRange{elemType, elemType},
          pair.sinOp.getOperand(), pair.sinOp.getFastmathAttr());
      rewriter.replaceOp(pair.sinOp, sincos.getSin());
      rewriter.replaceOp(pair.cosOp, sincos.getCos());
    }
  }
};

} // namespace
