//===- RuntimeOpVerification.cpp - Op Verification ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/RuntimeVerifiableOpInterface.h"

namespace mlir {
#define GEN_PASS_DEF_GENERATERUNTIMEVERIFICATION
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct GenerateRuntimeVerificationPass
    : public impl::GenerateRuntimeVerificationBase<
          GenerateRuntimeVerificationPass> {
  void runOnOperation() override;
};
} // namespace

void GenerateRuntimeVerificationPass::runOnOperation() {
  getOperation()->walk([&](RuntimeVerifiableOpInterface verifiableOp) {
    OpBuilder builder(getOperation()->getContext());
    builder.setInsertionPoint(verifiableOp);
    verifiableOp.generateRuntimeVerification(builder, verifiableOp.getLoc());
  });
}

std::unique_ptr<Pass> mlir::createGenerateRuntimeVerificationPass() {
  return std::make_unique<GenerateRuntimeVerificationPass>();
}
