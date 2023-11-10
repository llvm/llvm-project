//===- VectorToArmSMEPass.cpp - Conversion from Vector to the ArmSME dialect =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToArmSME/VectorToArmSME.h"

#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTVECTORTOARMSME
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::vector;

namespace {
struct ConvertVectorToArmSMEPass
    : public impl::ConvertVectorToArmSMEBase<ConvertVectorToArmSMEPass> {

  void runOnOperation() override;
};
} // namespace

void ConvertVectorToArmSMEPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateVectorToArmSMEPatterns(patterns, getContext());

  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> mlir::createConvertVectorToArmSMEPass() {
  return std::make_unique<ConvertVectorToArmSMEPass>();
}
