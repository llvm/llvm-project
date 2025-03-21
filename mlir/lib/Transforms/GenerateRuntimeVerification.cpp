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
  // The implementation of the RuntimeVerifiableOpInterface may create ops that
  // can be verified. We don't want to generate verification for IR that
  // performs verification, so gather all runtime-verifiable ops first.
  SmallVector<RuntimeVerifiableOpInterface> ops;
  getOperation()->walk([&](RuntimeVerifiableOpInterface verifiableOp) {
    ops.push_back(verifiableOp);
  });

  OpBuilder builder(getOperation()->getContext());
  for (RuntimeVerifiableOpInterface verifiableOp : ops) {
    builder.setInsertionPoint(verifiableOp);
    verifiableOp.generateRuntimeVerification(builder, verifiableOp.getLoc());
  };
}

std::unique_ptr<Pass> mlir::createGenerateRuntimeVerificationPass() {
  return std::make_unique<GenerateRuntimeVerificationPass>();
}
