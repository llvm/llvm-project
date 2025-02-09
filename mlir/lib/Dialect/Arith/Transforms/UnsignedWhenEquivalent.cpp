//===- UnsignedWhenEquivalent.cpp - Pass to replace signed operations with
// unsigned
// ones when all their arguments and results are statically non-negative --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir {
namespace arith {
#define GEN_PASS_DEF_ARITHUNSIGNEDWHENEQUIVALENT
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace arith
} // namespace mlir

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::dataflow;

/// Succeeds when a value is statically non-negative in that it has a lower
/// bound on its value (if it is treated as signed) and that bound is
/// non-negative.
// TODO: IntegerRangeAnalysis internally assumes index is 64bit and this pattern
// relies on this. These transformations may not be valid for 32bit index,
// need more investigation.
static LogicalResult staticallyNonNegative(DataFlowSolver &solver, Value v) {
  auto *result = solver.lookupState<IntegerValueRangeLattice>(v);
  if (!result || result->getValue().isUninitialized())
    return failure();
  const ConstantIntRanges &range = result->getValue().getValue();
  return success(range.smin().isNonNegative());
}

/// Succeeds if an op can be converted to its unsigned equivalent without
/// changing its semantics. This is the case when none of its openands or
/// results can be below 0 when analyzed from a signed perspective.
static LogicalResult staticallyNonNegative(DataFlowSolver &solver,
                                           Operation *op) {
  auto nonNegativePred = [&solver](Value v) -> bool {
    return succeeded(staticallyNonNegative(solver, v));
  };
  return success(llvm::all_of(op->getOperands(), nonNegativePred) &&
                 llvm::all_of(op->getResults(), nonNegativePred));
}

/// Succeeds when the comparison predicate is a signed operation and all the
/// operands are non-negative, indicating that the cmpi operation `op` can have
/// its predicate changed to an unsigned equivalent.
static LogicalResult isCmpIConvertable(DataFlowSolver &solver, CmpIOp op) {
  CmpIPredicate pred = op.getPredicate();
  switch (pred) {
  case CmpIPredicate::sle:
  case CmpIPredicate::slt:
  case CmpIPredicate::sge:
  case CmpIPredicate::sgt:
    return success(llvm::all_of(op.getOperands(), [&solver](Value v) -> bool {
      return succeeded(staticallyNonNegative(solver, v));
    }));
  default:
    return failure();
  }
}

/// Return the unsigned equivalent of a signed comparison predicate,
/// or the predicate itself if there is none.
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

namespace {
class DataFlowListener : public RewriterBase::Listener {
public:
  DataFlowListener(DataFlowSolver &s) : s(s) {}

protected:
  void notifyOperationErased(Operation *op) override {
    s.eraseState(s.getProgramPointAfter(op));
    for (Value res : op->getResults())
      s.eraseState(res);
  }

  DataFlowSolver &s;
};

template <typename Signed, typename Unsigned>
struct ConvertOpToUnsigned final : OpRewritePattern<Signed> {
  ConvertOpToUnsigned(MLIRContext *context, DataFlowSolver &s)
      : OpRewritePattern<Signed>(context), solver(s) {}

  LogicalResult matchAndRewrite(Signed op, PatternRewriter &rw) const override {
    if (failed(
            staticallyNonNegative(this->solver, static_cast<Operation *>(op))))
      return failure();

    rw.replaceOpWithNewOp<Unsigned>(op, op->getResultTypes(), op->getOperands(),
                                    op->getAttrs());
    return success();
  }

private:
  DataFlowSolver &solver;
};

struct ConvertCmpIToUnsigned final : OpRewritePattern<CmpIOp> {
  ConvertCmpIToUnsigned(MLIRContext *context, DataFlowSolver &s)
      : OpRewritePattern<CmpIOp>(context), solver(s) {}

  LogicalResult matchAndRewrite(CmpIOp op, PatternRewriter &rw) const override {
    if (failed(isCmpIConvertable(this->solver, op)))
      return failure();

    rw.replaceOpWithNewOp<CmpIOp>(op, toUnsignedPred(op.getPredicate()),
                                  op.getLhs(), op.getRhs());
    return success();
  }

private:
  DataFlowSolver &solver;
};

struct ArithUnsignedWhenEquivalentPass
    : public arith::impl::ArithUnsignedWhenEquivalentBase<
          ArithUnsignedWhenEquivalentPass> {

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    DataFlowListener listener(solver);

    RewritePatternSet patterns(ctx);
    populateUnsignedWhenEquivalentPatterns(patterns, solver);

    walkAndApplyPatterns(op, std::move(patterns), &listener);
  }
};
} // end anonymous namespace

void mlir::arith::populateUnsignedWhenEquivalentPatterns(
    RewritePatternSet &patterns, DataFlowSolver &solver) {
  patterns.add<ConvertOpToUnsigned<DivSIOp, DivUIOp>,
               ConvertOpToUnsigned<CeilDivSIOp, CeilDivUIOp>,
               ConvertOpToUnsigned<FloorDivSIOp, DivUIOp>,
               ConvertOpToUnsigned<RemSIOp, RemUIOp>,
               ConvertOpToUnsigned<MinSIOp, MinUIOp>,
               ConvertOpToUnsigned<MaxSIOp, MaxUIOp>,
               ConvertOpToUnsigned<ExtSIOp, ExtUIOp>, ConvertCmpIToUnsigned>(
      patterns.getContext(), solver);
}

std::unique_ptr<Pass> mlir::arith::createArithUnsignedWhenEquivalentPass() {
  return std::make_unique<ArithUnsignedWhenEquivalentPass>();
}
