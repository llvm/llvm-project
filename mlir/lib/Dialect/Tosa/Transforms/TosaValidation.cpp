//===- TosaValidation.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Validate if TOSA dialect input matchs with the specification for given
// requirements.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/PassesEnums.cpp.inc"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAVALIDATION
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

static LogicalResult checkConstantOperandPad(Operation *op) {
  if (auto pad_op = dyn_cast<tosa::PadOp>(op)) {
    DenseElementsAttr paddings;
    if (!matchPattern(pad_op.getPadding(), m_Constant(&paddings)))
      return op->emitOpError("padding of pad is not constant");

    DenseElementsAttr pad_const;
    // Assume this op is zero-padding if pad_const is not presented.
    if (pad_op.getPadConst() &&
        !matchPattern(pad_op.getPadConst(), m_Constant(&pad_const)))
      return op->emitOpError("pad_const of pad is not constant");
  }
  return success();
}

static LogicalResult checkConstantOperandTranspose(Operation *op) {
  if (auto transpose_op = dyn_cast<tosa::TransposeOp>(op)) {
    DenseElementsAttr perms;
    if (!matchPattern(transpose_op.getPerms(), m_Constant(&perms)))
      return op->emitOpError("perms of transpose is not constant");
  }
  return success();
}

static LogicalResult checkConstantOperandFullyConnected(Operation *op) {
  if (auto fc_op = dyn_cast<tosa::FullyConnectedOp>(op)) {
    DenseElementsAttr weight;
    if (!matchPattern(fc_op.getWeight(), m_Constant(&weight)))
      return op->emitOpError("weight of fully_connected is not constant");

    DenseElementsAttr bias;
    if (!matchPattern(fc_op.getBias(), m_Constant(&bias)))
      return op->emitOpError("bias of fully_connected is not constant");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TOSA Validation Pass.
//===----------------------------------------------------------------------===//

struct TosaValidation : public tosa::impl::TosaValidationBase<TosaValidation> {
public:
  explicit TosaValidation() { populateConstantOperandChecks(); }
  void runOnOperation() override;

  LogicalResult applyConstantOperandCheck(Operation *op) {
    for (auto &checker : const_checkers) {
      if (failed(checker(op)))
        return failure();
    }
    return success();
  }

private:
  void populateConstantOperandChecks() {
    const_checkers.emplace_back(checkConstantOperandPad);
    const_checkers.emplace_back(checkConstantOperandTranspose);
    const_checkers.emplace_back(checkConstantOperandFullyConnected);
  }

  SmallVector<std::function<LogicalResult(Operation *)>> const_checkers;
  std::optional<TosaProfileEnum> profileType;
};

void TosaValidation::runOnOperation() {
  profileType = symbolizeEnum<TosaProfileEnum>(profileName);

  getOperation().walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if ((profileType == TosaProfileEnum::BaseInference) &&
          isa<FloatType>(getElementTypeOrSelf(operand))) {
        return signalPassFailure();
      }
      if (getElementTypeOrSelf(operand).isF64()) {
        return signalPassFailure();
      }
    }

    // Some uses of TOSA rely on the constant operands of particular operations.
    if (StrictOperationSpecAlignment && failed(applyConstantOperandCheck(op)))
      signalPassFailure();
  });
}
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaValidationPass() {
  return std::make_unique<TosaValidation>();
}
