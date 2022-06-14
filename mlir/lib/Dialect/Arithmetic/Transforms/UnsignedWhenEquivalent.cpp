//===- UnsignedWhenEquivalent.cpp - Pass to replace signed operations with
// unsigned
// ones when all their arguments and results are statically non-negative --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/IntRangeAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::arith;

using OpList = llvm::SmallVector<Operation *>;

/// Returns true when a value is statically non-negative in that it has a lower
/// bound on its value (if it is treated as signed) and that bound is
/// non-negative.
static bool staticallyNonNegative(IntRangeAnalysis &analysis, Value v) {
  Optional<ConstantIntRanges> result = analysis.getResult(v);
  if (!result.hasValue())
    return false;
  const ConstantIntRanges &range = result.getValue();
  return (range.smin().isNonNegative());
}

/// Identify all operations in a block that have signed equivalents and have
/// operands and results that are statically non-negative.
template <typename... Ts>
static void getConvertableOps(Operation *root, OpList &toRewrite,
                              IntRangeAnalysis &analysis) {
  auto nonNegativePred = [&analysis](Value v) -> bool {
    return staticallyNonNegative(analysis, v);
  };
  root->walk([&nonNegativePred, &toRewrite](Operation *orig) {
    if (isa<Ts...>(orig) &&
        llvm::all_of(orig->getOperands(), nonNegativePred) &&
        llvm::all_of(orig->getResults(), nonNegativePred)) {
      toRewrite.push_back(orig);
    }
  });
}

static CmpIPredicate toUnsignedPred(CmpIPredicate pred) {
  switch (pred) {
  case CmpIPredicate::sle:
    return CmpIPredicate::ule;
  case CmpIPredicate::slt:
    return CmpIPredicate::ult;
  case CmpIPredicate::sge:
    return CmpIPredicate::uge;
  case CmpIPredicate::sgt:
    return CmpIPredicate::ugt;
  default:
    return pred;
  }
}

/// Find all cmpi ops that can be replaced by their unsigned equivalents.
static void getConvertableCmpi(Operation *root, OpList &toRewrite,
                               IntRangeAnalysis &analysis) {
  auto nonNegativePred = [&analysis](Value v) -> bool {
    return staticallyNonNegative(analysis, v);
  };
  root->walk([&nonNegativePred, &toRewrite](arith::CmpIOp orig) {
    CmpIPredicate pred = orig.getPredicate();
    if (toUnsignedPred(pred) != pred &&
        // i1 will spuriously and trivially show up as pontentially negative,
        // so don't check the results
        llvm::all_of(orig->getOperands(), nonNegativePred)) {
      toRewrite.push_back(orig.getOperation());
    }
  });
}

/// Return ops to be replaced in the order they should be rewritten.
static OpList getMatching(Operation *root, IntRangeAnalysis &analysis) {
  OpList ret;
  getConvertableOps<DivSIOp, CeilDivSIOp, FloorDivSIOp, RemSIOp, MinSIOp,
                    MaxSIOp, ExtSIOp>(root, ret, analysis);
  // Since these are in-place changes, they don't need to be topological order
  // like the others.
  getConvertableCmpi(root, ret, analysis);
  return ret;
}

template <typename T, typename U>
static void rewriteOp(Operation *op, OpBuilder &b) {
  if (isa<T>(op)) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(op);
    Operation *newOp = b.create<U>(op->getLoc(), op->getResultTypes(),
                                   op->getOperands(), op->getAttrs());
    op->replaceAllUsesWith(newOp->getResults());
    op->erase();
  }
}

static void rewriteCmpI(Operation *op, OpBuilder &b) {
  if (auto cmpOp = dyn_cast<CmpIOp>(op)) {
    cmpOp.setPredicateAttr(CmpIPredicateAttr::get(
        b.getContext(), toUnsignedPred(cmpOp.getPredicate())));
  }
}

static void rewrite(Operation *root, const OpList &toReplace) {
  OpBuilder b(root->getContext());
  b.setInsertionPoint(root);
  for (Operation *op : toReplace) {
    rewriteOp<DivSIOp, DivUIOp>(op, b);
    rewriteOp<CeilDivSIOp, CeilDivUIOp>(op, b);
    rewriteOp<FloorDivSIOp, DivUIOp>(op, b);
    rewriteOp<RemSIOp, RemUIOp>(op, b);
    rewriteOp<MinSIOp, MinUIOp>(op, b);
    rewriteOp<MaxSIOp, MaxUIOp>(op, b);
    rewriteOp<ExtSIOp, ExtUIOp>(op, b);
    rewriteCmpI(op, b);
  }
}

namespace {
struct ArithmeticUnsignedWhenEquivalentPass
    : public ArithmeticUnsignedWhenEquivalentBase<
          ArithmeticUnsignedWhenEquivalentPass> {
  /// Implementation structure: first find all equivalent ops and collect them,
  /// then perform all the rewrites in a second pass over the target op. This
  /// ensures that analysis results are not invalidated during rewriting.
  void runOnOperation() override {
    Operation *op = getOperation();
    IntRangeAnalysis analysis(op);
    rewrite(op, getMatching(op, analysis));
  }
};
} // end anonymous namespace

std::unique_ptr<Pass>
mlir::arith::createArithmeticUnsignedWhenEquivalentPass() {
  return std::make_unique<ArithmeticUnsignedWhenEquivalentPass>();
}
