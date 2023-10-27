//===- DecomposeAffineOps.cpp - Decompose affine ops into finer-grained ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functionality to progressively decompose coarse-grained
// affine ops into finer-grained ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::affine;

#define DEBUG_TYPE "decompose-affine-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

/// Count the number of loops surrounding `operand` such that operand could be
/// hoisted above.
/// Stop counting at the first loop over which the operand cannot be hoisted.
static int64_t numEnclosingInvariantLoops(OpOperand &operand) {
  int64_t count = 0;
  Operation *currentOp = operand.getOwner();
  while (auto loopOp = currentOp->getParentOfType<LoopLikeOpInterface>()) {
    if (!loopOp.isDefinedOutsideOfLoop(operand.get()))
      break;
    currentOp = loopOp;
    count++;
  }
  return count;
}

void mlir::affine::reorderOperandsByHoistability(RewriterBase &rewriter,
                                                 AffineApplyOp op) {
  SmallVector<int64_t> numInvariant = llvm::to_vector(
      llvm::map_range(op->getOpOperands(), [&](OpOperand &operand) {
        return numEnclosingInvariantLoops(operand);
      }));

  int64_t numOperands = op.getNumOperands();
  SmallVector<int64_t> operandPositions =
      llvm::to_vector(llvm::seq<int64_t>(0, numOperands));
  llvm::stable_sort(operandPositions, [&numInvariant](size_t i1, size_t i2) {
    return numInvariant[i1] > numInvariant[i2];
  });

  SmallVector<AffineExpr> replacements(numOperands);
  SmallVector<Value> operands(numOperands);
  for (int64_t i = 0; i < numOperands; ++i) {
    operands[i] = op.getOperand(operandPositions[i]);
    replacements[operandPositions[i]] = getAffineSymbolExpr(i, op.getContext());
  }

  AffineMap map = op.getAffineMap();
  ArrayRef<AffineExpr> repls{replacements};
  map = map.replaceDimsAndSymbols(repls.take_front(map.getNumDims()),
                                  repls.drop_front(map.getNumDims()),
                                  /*numResultDims=*/0,
                                  /*numResultSyms=*/numOperands);
  map = AffineMap::get(0, numOperands,
                       simplifyAffineExpr(map.getResult(0), 0, numOperands),
                       op->getContext());
  canonicalizeMapAndOperands(&map, &operands);

  rewriter.startRootUpdate(op);
  op.setMap(map);
  op->setOperands(operands);
  rewriter.finalizeRootUpdate(op);
}

/// Build an affine.apply that is a subexpression `expr` of `originalOp`s affine
/// map and with the same operands.
/// Canonicalize the map and operands to deduplicate and drop dead operands
/// before returning but do not perform maximal composition of AffineApplyOp
/// which would defeat the purpose.
static AffineApplyOp createSubApply(RewriterBase &rewriter,
                                    AffineApplyOp originalOp, AffineExpr expr) {
  MLIRContext *ctx = originalOp->getContext();
  AffineMap m = originalOp.getAffineMap();
  auto rhsMap = AffineMap::get(m.getNumDims(), m.getNumSymbols(), expr, ctx);
  SmallVector<Value> rhsOperands = originalOp->getOperands();
  canonicalizeMapAndOperands(&rhsMap, &rhsOperands);
  return rewriter.create<AffineApplyOp>(originalOp.getLoc(), rhsMap,
                                        rhsOperands);
}

FailureOr<AffineApplyOp> mlir::affine::decompose(RewriterBase &rewriter,
                                                 AffineApplyOp op) {
  // 1. Preconditions: only handle dimensionless AffineApplyOp maps with a
  // top-level binary expression that we can reassociate (i.e. add or mul).
  AffineMap m = op.getAffineMap();
  if (m.getNumDims() > 0)
    return rewriter.notifyMatchFailure(op, "expected no dims");

  AffineExpr remainingExp = m.getResult(0);
  auto binExpr = remainingExp.dyn_cast<AffineBinaryOpExpr>();
  if (!binExpr)
    return rewriter.notifyMatchFailure(op, "terminal affine.apply");

  if (!binExpr.getLHS().isa<AffineBinaryOpExpr>() &&
      !binExpr.getRHS().isa<AffineBinaryOpExpr>())
    return rewriter.notifyMatchFailure(op, "terminal affine.apply");

  bool supportedKind = ((binExpr.getKind() == AffineExprKind::Add) ||
                        (binExpr.getKind() == AffineExprKind::Mul));
  if (!supportedKind)
    return rewriter.notifyMatchFailure(
        op, "only add or mul binary expr can be reassociated");

  LLVM_DEBUG(DBGS() << "Start decomposeIntoFinerGrainedOps: " << op << "\n");

  // 2. Iteratively extract the RHS subexpressions while the top-level binary
  // expr kind remains the same.
  MLIRContext *ctx = op->getContext();
  SmallVector<AffineExpr> subExpressions;
  while (true) {
    auto currentBinExpr = remainingExp.dyn_cast<AffineBinaryOpExpr>();
    if (!currentBinExpr || currentBinExpr.getKind() != binExpr.getKind()) {
      subExpressions.push_back(remainingExp);
      LLVM_DEBUG(DBGS() << "--terminal: " << subExpressions.back() << "\n");
      break;
    }
    subExpressions.push_back(currentBinExpr.getRHS());
    LLVM_DEBUG(DBGS() << "--subExpr: " << subExpressions.back() << "\n");
    remainingExp = currentBinExpr.getLHS();
  }

  // 3. Reorder subExpressions by the min symbol they are a function of.
  // This also takes care of properly reordering local variables.
  // This however won't be able to split expression that cannot be reassociated
  // such as ones that involve divs and multiple symbols.
  auto getMaxSymbol = [&](AffineExpr e) -> int64_t {
    for (int64_t i = m.getNumSymbols(); i >= 0; --i)
      if (e.isFunctionOfSymbol(i))
        return i;
    return -1;
  };
  llvm::stable_sort(subExpressions, [&](AffineExpr e1, AffineExpr e2) {
    return getMaxSymbol(e1) < getMaxSymbol(e2);
  });
  LLVM_DEBUG(
      llvm::interleaveComma(subExpressions, DBGS() << "--sorted subexprs: ");
      llvm::dbgs() << "\n");

  // 4. Merge sorted subExpressions iteratively, thus achieving reassociation.
  auto s0 = getAffineSymbolExpr(0, ctx);
  auto s1 = getAffineSymbolExpr(1, ctx);
  AffineMap binMap = AffineMap::get(
      /*dimCount=*/0, /*symbolCount=*/2,
      getAffineBinaryOpExpr(binExpr.getKind(), s0, s1), ctx);

  auto current = createSubApply(rewriter, op, subExpressions[0]);
  for (int64_t i = 1, e = subExpressions.size(); i < e; ++i) {
    Value tmp = createSubApply(rewriter, op, subExpressions[i]);
    current = rewriter.create<AffineApplyOp>(op.getLoc(), binMap,
                                             ValueRange{current, tmp});
    LLVM_DEBUG(DBGS() << "--reassociate into: " << current << "\n");
  }

  // 5. Replace original op.
  rewriter.replaceOp(op, current.getResult());
  return current;
}
