//===- TestReifyValueBounds.cpp - Test value bounds reification -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/ScalableValueBoundsConstraintSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "test-affine-reify-value-bounds"

using namespace mlir;
using namespace mlir::affine;
using mlir::presburger::BoundType;

namespace {

/// This pass applies the permutation on the first maximal perfect nest.
struct TestReifyValueBounds
    : public PassWrapper<TestReifyValueBounds,
                         InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestReifyValueBounds)

  StringRef getArgument() const final { return PASS_NAME; }
  StringRef getDescription() const final {
    return "Tests ValueBoundsOpInterface with affine dialect reification";
  }
  TestReifyValueBounds() = default;
  TestReifyValueBounds(const TestReifyValueBounds &pass) : PassWrapper(pass){};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, tensor::TensorDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override;

private:
  Option<bool> reifyToFuncArgs{
      *this, "reify-to-func-args",
      llvm::cl::desc("Reify in terms of function args"), llvm::cl::init(false)};

  Option<bool> useArithOps{*this, "use-arith-ops",
                           llvm::cl::desc("Reify with arith dialect ops"),
                           llvm::cl::init(false)};
};

} // namespace

static ValueBoundsConstraintSet::ComparisonOperator
invertComparisonOperator(ValueBoundsConstraintSet::ComparisonOperator cmp) {
  if (cmp == ValueBoundsConstraintSet::ComparisonOperator::LT)
    return ValueBoundsConstraintSet::ComparisonOperator::GE;
  if (cmp == ValueBoundsConstraintSet::ComparisonOperator::LE)
    return ValueBoundsConstraintSet::ComparisonOperator::GT;
  if (cmp == ValueBoundsConstraintSet::ComparisonOperator::GT)
    return ValueBoundsConstraintSet::ComparisonOperator::LE;
  if (cmp == ValueBoundsConstraintSet::ComparisonOperator::GE)
    return ValueBoundsConstraintSet::ComparisonOperator::LT;
  llvm_unreachable("unsupported comparison operator");
}

/// Look for "test.reify_bound" ops in the input and replace their results with
/// the reified values.
static LogicalResult testReifyValueBounds(FunctionOpInterface funcOp,
                                          bool reifyToFuncArgs,
                                          bool useArithOps) {
  IRRewriter rewriter(funcOp.getContext());
  WalkResult result = funcOp.walk([&](test::ReifyBoundOp op) {
    auto boundType = op.getBoundType();
    Value value = op.getVar();
    std::optional<int64_t> dim = op.getDim();
    bool constant = op.getConstant();
    bool scalable = op.getScalable();

    // Prepare stop condition. By default, reify in terms of the op's
    // operands. No stop condition is used when a constant was requested.
    std::function<bool(Value, std::optional<int64_t>,
                       ValueBoundsConstraintSet & cstr)>
        stopCondition = [&](Value v, std::optional<int64_t> d,
                            ValueBoundsConstraintSet &cstr) {
          // Reify in terms of SSA values that are different from `value`.
          return v != value;
        };
    if (reifyToFuncArgs) {
      // Reify in terms of function block arguments.
      stopCondition = [](Value v, std::optional<int64_t> d,
                         ValueBoundsConstraintSet &cstr) {
        auto bbArg = dyn_cast<BlockArgument>(v);
        if (!bbArg)
          return false;
        return isa<FunctionOpInterface>(bbArg.getParentBlock()->getParentOp());
      };
    }

    // Reify value bound
    rewriter.setInsertionPointAfter(op);
    FailureOr<OpFoldResult> reified = failure();
    if (constant) {
      auto reifiedConst = ValueBoundsConstraintSet::computeConstantBound(
          boundType, {value, dim}, /*stopCondition=*/nullptr);
      if (succeeded(reifiedConst))
        reified = FailureOr<OpFoldResult>(rewriter.getIndexAttr(*reifiedConst));
    } else if (scalable) {
      auto loc = op->getLoc();
      auto reifiedScalable =
          vector::ScalableValueBoundsConstraintSet::computeScalableBound(
              value, dim, *op.getVscaleMin(), *op.getVscaleMax(), boundType);
      if (succeeded(reifiedScalable)) {
        SmallVector<std::pair<Value, std::optional<int64_t>>, 1> vscaleOperand;
        if (reifiedScalable->map.getNumInputs() == 1) {
          // The only possible input to the bound is vscale.
          vscaleOperand.push_back(std::make_pair(
              rewriter.create<vector::VectorScaleOp>(loc), std::nullopt));
        }
        reified = affine::materializeComputedBound(
            rewriter, loc, reifiedScalable->map, vscaleOperand);
      }
    } else {
      if (useArithOps) {
        reified = arith::reifyValueBound(rewriter, op->getLoc(), boundType,
                                         op.getVariable(), stopCondition);
      } else {
        reified = reifyValueBound(rewriter, op->getLoc(), boundType,
                                  op.getVariable(), stopCondition);
      }
    }
    if (failed(reified)) {
      op->emitOpError("could not reify bound");
      return WalkResult::interrupt();
    }

    // Replace the op with the reified bound.
    if (auto val = llvm::dyn_cast_if_present<Value>(*reified)) {
      rewriter.replaceOp(op, val);
      return WalkResult::skip();
    }
    Value constOp = rewriter.create<arith::ConstantIndexOp>(
        op->getLoc(), cast<IntegerAttr>(cast<Attribute>(*reified)).getInt());
    rewriter.replaceOp(op, constOp);
    return WalkResult::skip();
  });
  return failure(result.wasInterrupted());
}

/// Look for "test.compare" ops and emit errors/remarks.
static LogicalResult testEquality(FunctionOpInterface funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  WalkResult result = funcOp.walk([&](test::CompareOp op) {
    auto cmpType = op.getComparisonOperator();
    if (op.getCompose()) {
      if (cmpType != ValueBoundsConstraintSet::EQ) {
        op->emitOpError(
            "comparison operator must be EQ when 'composed' is specified");
        return WalkResult::interrupt();
      }
      FailureOr<int64_t> delta = affine::fullyComposeAndComputeConstantDelta(
          op->getOperand(0), op->getOperand(1));
      if (failed(delta)) {
        op->emitError("could not determine equality");
      } else if (*delta == 0) {
        op->emitRemark("equal");
      } else {
        op->emitRemark("different");
      }
      return WalkResult::advance();
    }

    auto compare = [&](ValueBoundsConstraintSet::ComparisonOperator cmp) {
      return ValueBoundsConstraintSet::compare(op.getLhs(), cmp, op.getRhs());
    };
    if (compare(cmpType)) {
      op->emitRemark("true");
    } else if (cmpType != ValueBoundsConstraintSet::EQ &&
               compare(invertComparisonOperator(cmpType))) {
      op->emitRemark("false");
    } else if (cmpType == ValueBoundsConstraintSet::EQ &&
               (compare(ValueBoundsConstraintSet::ComparisonOperator::LT) ||
                compare(ValueBoundsConstraintSet::ComparisonOperator::GT))) {
      op->emitRemark("false");
    } else {
      op->emitError("unknown");
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void TestReifyValueBounds::runOnOperation() {
  if (failed(
          testReifyValueBounds(getOperation(), reifyToFuncArgs, useArithOps)))
    signalPassFailure();
  if (failed(testEquality(getOperation())))
    signalPassFailure();
}

namespace mlir {
void registerTestAffineReifyValueBoundsPass() {
  PassRegistration<TestReifyValueBounds>();
}
} // namespace mlir
