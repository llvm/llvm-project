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

#define DEBUG_TYPE "generate-runtime-verification"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

static bool defaultShouldHandleVerifiableOpFn(RuntimeVerifiableOpInterface op) {
  // By default, all verifiable ops are considered
  return true;
}

namespace {
struct GenerateRuntimeVerificationPass
    : public impl::GenerateRuntimeVerificationBase<
          GenerateRuntimeVerificationPass> {

  GenerateRuntimeVerificationPass();
  GenerateRuntimeVerificationPass(const GenerateRuntimeVerificationPass &) =
      default;
  GenerateRuntimeVerificationPass(
      std::function<bool(RuntimeVerifiableOpInterface)>
          shouldHandleVerifiableOpFn);

  void runOnOperation() override;

private:
  // A filter function to select verifiable ops to generate verification for.
  // If empty, all verifiable ops are considered.
  std::function<bool(RuntimeVerifiableOpInterface)> shouldHandleVerifiableOpFn;
};
} // namespace

GenerateRuntimeVerificationPass::GenerateRuntimeVerificationPass()
    : shouldHandleVerifiableOpFn(defaultShouldHandleVerifiableOpFn) {}

GenerateRuntimeVerificationPass::GenerateRuntimeVerificationPass(
    std::function<bool(RuntimeVerifiableOpInterface)>
        shouldHandleVerifiableOpFn)
    : shouldHandleVerifiableOpFn(std::move(shouldHandleVerifiableOpFn)) {}

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
    if (shouldHandleVerifiableOpFn(verifiableOp)) {
      builder.setInsertionPoint(verifiableOp);
      verifiableOp.generateRuntimeVerification(builder, verifiableOp.getLoc());
    } else {
      LDBG("Skipping operation: " << verifiableOp.getOperation());
    }
  }
}

std::unique_ptr<Pass> mlir::createGenerateRuntimeVerificationPass() {
  return std::make_unique<GenerateRuntimeVerificationPass>();
}

std::unique_ptr<Pass> mlir::createGenerateRuntimeVerificationPass(
    std::function<bool(RuntimeVerifiableOpInterface)>
        shouldHandleVerifiableOpFn) {
  return std::make_unique<GenerateRuntimeVerificationPass>(
      std::move(shouldHandleVerifiableOpFn));
}
