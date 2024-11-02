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

//===----------------------------------------------------------------------===//
// TOSA Validation Pass.
//===----------------------------------------------------------------------===//

struct TosaValidation : public tosa::impl::TosaValidationBase<TosaValidation> {
public:
  explicit TosaValidation() = default;

private:
  void runOnOperation() override;

  llvm::Optional<TosaProfileEnum> profileType;
};

void TosaValidation::runOnOperation() {
  profileType = symbolizeEnum<TosaProfileEnum>(profileName);

  getOperation().walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if ((profileType == TosaProfileEnum::BaseInference) &&
          getElementTypeOrSelf(operand).isa<FloatType>()) {
        return signalPassFailure();
      }
    }
  });
}
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaValidationPass() {
  return std::make_unique<TosaValidation>();
}
