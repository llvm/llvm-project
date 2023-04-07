//===- TestReifyValueBounds.cpp - Test value bounds reification -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "test-affine-reify-value-bounds"

using namespace mlir;
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
    registry.insert<AffineDialect>();
  }

  void runOnOperation() override;

private:
  Option<bool> reifyToFuncArgs{
      *this, "reify-to-func-args",
      llvm::cl::desc("Reify in terms of function args"), llvm::cl::init(false)};
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
                                          bool reifyToFuncArgs) {
  IRRewriter rewriter(funcOp.getContext());
  WalkResult result = funcOp.walk([&](Operation *op) {
    // Look for test.reify_bound ops.
    if (op->getName().getStringRef() == "test.reify_bound") {
      if (op->getNumOperands() != 1 || op->getNumResults() != 1 ||
          !op->getResultTypes()[0].isIndex()) {
        op->emitOpError("invalid op");
        return WalkResult::skip();
      }
      Value value = op->getOperand(0);
      if (value.getType().isa<IndexType>() !=
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

      // Reify value bound.
      rewriter.setInsertionPointAfter(op);
      FailureOr<OpFoldResult> reified;
      if (!reifyToFuncArgs) {
        // Reify in terms of the op's operands.
        reified =
            reifyValueBound(rewriter, op->getLoc(), *boundType, value, dim);
      } else {
        // Reify in terms of function block arguments.
        auto stopCondition = [](Value v) {
          auto bbArg = v.dyn_cast<BlockArgument>();
          if (!bbArg)
            return false;
          return isa<FunctionOpInterface>(
              bbArg.getParentBlock()->getParentOp());
        };
        reified = reifyValueBound(rewriter, op->getLoc(), *boundType, value,
                                  dim, stopCondition);
      }
      if (failed(reified)) {
        op->emitOpError("could not reify bound");
        return WalkResult::interrupt();
      }

      // Replace the op with the reified bound.
      if (auto val = reified->dyn_cast<Value>()) {
        rewriter.replaceOp(op, val);
        return WalkResult::skip();
      }
      Value constOp = rewriter.create<arith::ConstantIndexOp>(
          op->getLoc(), reified->get<Attribute>().cast<IntegerAttr>().getInt());
      rewriter.replaceOp(op, constOp);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void TestReifyValueBounds::runOnOperation() {
  if (failed(testReifyValueBounds(getOperation(), reifyToFuncArgs)))
    signalPassFailure();
}

namespace mlir {
void registerTestAffineReifyValueBoundsPass() {
  PassRegistration<TestReifyValueBounds>();
}
} // namespace mlir
