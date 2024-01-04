//===- TestReifyValueBounds.cpp - Test value bounds reification -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "test-affine-reify-value-bounds"

using namespace mlir;
using namespace mlir::affine;
using mlir::presburger::BoundType;

namespace {

/// This pass applies the permutation on the first maximal perfect nest.
struct TestReifyValueBounds
    : public PassWrapper<TestReifyValueBounds, OperationPass<func::FuncOp>> {
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

FailureOr<BoundType> parseBoundType(std::string type) {
  if (type == "EQ")
    return BoundType::EQ;
  if (type == "LB")
    return BoundType::LB;
  if (type == "UB")
    return BoundType::UB;
  return failure();
}

/// Look for "test.reify_bound" ops in the input and replace their results with
/// the reified values.
static LogicalResult testReifyValueBounds(func::FuncOp funcOp,
                                          bool reifyToFuncArgs,
                                          bool useArithOps) {
  IRRewriter rewriter(funcOp.getContext());
  WalkResult result = funcOp.walk([&](Operation *op) {
    // Look for test.reify_bound ops.
    if (op->getName().getStringRef() == "test.reify_bound" ||
        op->getName().getStringRef() == "test.reify_constant_bound") {
      if (op->getNumOperands() != 1 || op->getNumResults() != 1 ||
          !op->getResultTypes()[0].isIndex()) {
        op->emitOpError("invalid op");
        return WalkResult::skip();
      }
      Value value = op->getOperand(0);
      if (isa<IndexType>(value.getType()) !=
          !op->hasAttrOfType<IntegerAttr>("dim")) {
        // Op should have "dim" attribute if and only if the operand is an
        // index-typed value.
        op->emitOpError("invalid op");
        return WalkResult::skip();
      }

      // Get bound type.
      std::string boundTypeStr = "EQ";
      if (auto boundTypeAttr = op->getAttrOfType<StringAttr>("type"))
        boundTypeStr = boundTypeAttr.str();
      auto boundType = parseBoundType(boundTypeStr);
      if (failed(boundType)) {
        op->emitOpError("invalid op");
        return WalkResult::interrupt();
      }

      // Get shape dimension (if any).
      auto dim = value.getType().isIndex()
                     ? std::nullopt
                     : std::make_optional<int64_t>(
                           op->getAttrOfType<IntegerAttr>("dim").getInt());

      // Check if a constant was requested.
      bool constant =
          op->getName().getStringRef() == "test.reify_constant_bound";

      // Prepare stop condition. By default, reify in terms of the op's
      // operands. No stop condition is used when a constant was requested.
      std::function<bool(Value, std::optional<int64_t>)> stopCondition =
          [&](Value v, std::optional<int64_t> d) {
            // Reify in terms of SSA values that are different from `value`.
            return v != value;
          };
      if (reifyToFuncArgs) {
        // Reify in terms of function block arguments.
        stopCondition = stopCondition = [](Value v, std::optional<int64_t> d) {
          auto bbArg = dyn_cast<BlockArgument>(v);
          if (!bbArg)
            return false;
          return isa<FunctionOpInterface>(
              bbArg.getParentBlock()->getParentOp());
        };
      }

      // Reify value bound
      rewriter.setInsertionPointAfter(op);
      FailureOr<OpFoldResult> reified = failure();
      if (constant) {
        auto reifiedConst = ValueBoundsConstraintSet::computeConstantBound(
            *boundType, value, dim, /*stopCondition=*/nullptr);
        if (succeeded(reifiedConst))
          reified =
              FailureOr<OpFoldResult>(rewriter.getIndexAttr(*reifiedConst));
      } else {
        if (dim) {
          if (useArithOps) {
            reified = arith::reifyShapedValueDimBound(
                rewriter, op->getLoc(), *boundType, value, *dim, stopCondition);
          } else {
            reified = reifyShapedValueDimBound(
                rewriter, op->getLoc(), *boundType, value, *dim, stopCondition);
          }
        } else {
          if (useArithOps) {
            reified = arith::reifyIndexValueBound(
                rewriter, op->getLoc(), *boundType, value, stopCondition);
          } else {
            reified = reifyIndexValueBound(rewriter, op->getLoc(), *boundType,
                                           value, stopCondition);
          }
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
          op->getLoc(), cast<IntegerAttr>(reified->get<Attribute>()).getInt());
      rewriter.replaceOp(op, constOp);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

/// Look for "test.are_equal" ops and emit errors/remarks.
static LogicalResult testEquality(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  WalkResult result = funcOp.walk([&](Operation *op) {
    // Look for test.are_equal ops.
    if (op->getName().getStringRef() == "test.are_equal") {
      if (op->getNumOperands() != 2 || !op->getOperand(0).getType().isIndex() ||
          !op->getOperand(1).getType().isIndex()) {
        op->emitOpError("invalid op");
        return WalkResult::skip();
      }
      if (op->hasAttr("compose")) {
        FailureOr<int64_t> delta = affine::fullyComposeAndComputeConstantDelta(
            op->getOperand(0), op->getOperand(1));
        if (failed(delta)) {
          op->emitError("could not determine equality");
        } else if (*delta == 0) {
          op->emitRemark("equal");
        } else {
          op->emitRemark("different");
        }
      } else {
        FailureOr<bool> equal = ValueBoundsConstraintSet::areEqual(
            op->getOperand(0), op->getOperand(1));
        if (failed(equal)) {
          op->emitError("could not determine equality");
        } else if (*equal) {
          op->emitRemark("equal");
        } else {
          op->emitRemark("different");
        }
      }
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
