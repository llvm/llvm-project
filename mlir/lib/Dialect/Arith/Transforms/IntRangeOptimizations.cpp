//===- IntRangeOptimizations.cpp - Optimizations based on integer ranges --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::arith {
#define GEN_PASS_DEF_ARITHINTRANGEOPTS
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace mlir::arith

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::dataflow;

/// Returns true if 2 integer ranges have intersection.
static bool intersects(const ConstantIntRanges &lhs,
                       const ConstantIntRanges &rhs) {
  return !((lhs.smax().slt(rhs.smin()) || lhs.smin().sgt(rhs.smax())) &&
           (lhs.umax().ult(rhs.umin()) || lhs.umin().ugt(rhs.umax())));
}

static FailureOr<bool> handleEq(ConstantIntRanges lhs, ConstantIntRanges rhs) {
  if (!intersects(std::move(lhs), std::move(rhs)))
    return false;

  return failure();
}

static FailureOr<bool> handleNe(ConstantIntRanges lhs, ConstantIntRanges rhs) {
  if (!intersects(std::move(lhs), std::move(rhs)))
    return true;

  return failure();
}

static FailureOr<bool> handleSlt(ConstantIntRanges lhs, ConstantIntRanges rhs) {
  if (lhs.smax().slt(rhs.smin()))
    return true;

  if (lhs.smin().sge(rhs.smax()))
    return false;

  return failure();
}

static FailureOr<bool> handleSle(ConstantIntRanges lhs, ConstantIntRanges rhs) {
  if (lhs.smax().sle(rhs.smin()))
    return true;

  if (lhs.smin().sgt(rhs.smax()))
    return false;

  return failure();
}

static FailureOr<bool> handleSgt(ConstantIntRanges lhs, ConstantIntRanges rhs) {
  return handleSlt(std::move(rhs), std::move(lhs));
}

static FailureOr<bool> handleSge(ConstantIntRanges lhs, ConstantIntRanges rhs) {
  return handleSle(std::move(rhs), std::move(lhs));
}

static FailureOr<bool> handleUlt(ConstantIntRanges lhs, ConstantIntRanges rhs) {
  if (lhs.umax().ult(rhs.umin()))
    return true;

  if (lhs.umin().uge(rhs.umax()))
    return false;

  return failure();
}

static FailureOr<bool> handleUle(ConstantIntRanges lhs, ConstantIntRanges rhs) {
  if (lhs.umax().ule(rhs.umin()))
    return true;

  if (lhs.umin().ugt(rhs.umax()))
    return false;

  return failure();
}

static FailureOr<bool> handleUgt(ConstantIntRanges lhs, ConstantIntRanges rhs) {
  return handleUlt(std::move(rhs), std::move(lhs));
}

static FailureOr<bool> handleUge(ConstantIntRanges lhs, ConstantIntRanges rhs) {
  return handleUle(std::move(rhs), std::move(lhs));
}

namespace {
struct ConvertCmpOp : public OpRewritePattern<arith::CmpIOp> {

  ConvertCmpOp(MLIRContext *context, DataFlowSolver &s)
      : OpRewritePattern<arith::CmpIOp>(context), solver(s) {}

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter &rewriter) const override {
    auto *lhsResult =
        solver.lookupState<dataflow::IntegerValueRangeLattice>(op.getLhs());
    if (!lhsResult || lhsResult->getValue().isUninitialized())
      return failure();

    auto *rhsResult =
        solver.lookupState<dataflow::IntegerValueRangeLattice>(op.getRhs());
    if (!rhsResult || rhsResult->getValue().isUninitialized())
      return failure();

    using HandlerFunc =
        FailureOr<bool> (*)(ConstantIntRanges, ConstantIntRanges);
    std::array<HandlerFunc, arith::getMaxEnumValForCmpIPredicate() + 1>
        handlers{};
    using Pred = arith::CmpIPredicate;
    handlers[static_cast<size_t>(Pred::eq)] = &handleEq;
    handlers[static_cast<size_t>(Pred::ne)] = &handleNe;
    handlers[static_cast<size_t>(Pred::slt)] = &handleSlt;
    handlers[static_cast<size_t>(Pred::sle)] = &handleSle;
    handlers[static_cast<size_t>(Pred::sgt)] = &handleSgt;
    handlers[static_cast<size_t>(Pred::sge)] = &handleSge;
    handlers[static_cast<size_t>(Pred::ult)] = &handleUlt;
    handlers[static_cast<size_t>(Pred::ule)] = &handleUle;
    handlers[static_cast<size_t>(Pred::ugt)] = &handleUgt;
    handlers[static_cast<size_t>(Pred::uge)] = &handleUge;

    HandlerFunc handler = handlers[static_cast<size_t>(op.getPredicate())];
    if (!handler)
      return failure();

    ConstantIntRanges lhsValue = lhsResult->getValue().getValue();
    ConstantIntRanges rhsValue = rhsResult->getValue().getValue();
    FailureOr<bool> result = handler(lhsValue, rhsValue);

    if (failed(result))
      return failure();

    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(
        op, static_cast<int64_t>(*result), /*width*/ 1);
    return success();
  }

private:
  DataFlowSolver &solver;
};

struct IntRangeOptimizationsPass
    : public arith::impl::ArithIntRangeOptsBase<IntRangeOptimizationsPass> {

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    RewritePatternSet patterns(ctx);
    populateIntRangeOptimizationsPatterns(patterns, solver);

    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::arith::populateIntRangeOptimizationsPatterns(
    RewritePatternSet &patterns, DataFlowSolver &solver) {
  patterns.add<ConvertCmpOp>(patterns.getContext(), solver);
}

std::unique_ptr<Pass> mlir::arith::createIntRangeOptimizationsPass() {
  return std::make_unique<IntRangeOptimizationsPass>();
}
