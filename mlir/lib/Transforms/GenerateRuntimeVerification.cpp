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
  // Check verboseLevel is in range [0, 2].
  if (verboseLevel > 2) {
    getOperation()->emitError(
        "generate-runtime-verification pass: set verboseLevel to 0, 1 or 2");
    signalPassFailure();
    return;
  }

  // The implementation of the RuntimeVerifiableOpInterface may create ops that
  // can be verified. We don't want to generate verification for IR that
  // performs verification, so gather all runtime-verifiable ops first.
  SmallVector<RuntimeVerifiableOpInterface> ops;
  getOperation()->walk([&](RuntimeVerifiableOpInterface verifiableOp) {
    ops.push_back(verifiableOp);
  });

  // Create error message generator based on verboseLevel
  auto errorMsgGenerator = [vLevel = verboseLevel.getValue()](
                               Operation *op, StringRef msg) -> std::string {
    std::string buffer;
    llvm::raw_string_ostream stream(buffer);
    OpPrintingFlags flags;
    // We may generate a lot of error messages and so we need to ensure the
    // printing is fast.
    flags.elideLargeElementsAttrs();
    flags.printGenericOpForm();
    flags.skipRegions();
    flags.useLocalScope();
    stream << "ERROR: Runtime op verification failed\n";
    if (vLevel == 2) {
      // print full op including operand names, very expensive
      op->print(stream, flags);
      stream << "\n " << msg;
    } else if (vLevel == 1) {
      // print op name and operand types
      stream << "Op: " << op->getName().getStringRef() << "\n";
      stream << "Operand Types:";
      for (const auto &operand : op->getOpOperands()) {
        stream << " " << operand.get().getType();
      }
      stream << "\n" << msg;
      stream << "Result Types:";
      for (const auto &result : op->getResults()) {
        stream << " " << result.getType();
      }
    }
    // all verbose levels include location
    stream << "^\nLocation: ";
    op->getLoc().print(stream);
    return buffer;
  };

  OpBuilder builder(getOperation()->getContext());
  for (RuntimeVerifiableOpInterface verifiableOp : ops) {
    builder.setInsertionPoint(verifiableOp);
    verifiableOp.generateRuntimeVerification(builder, verifiableOp.getLoc(),
                                             errorMsgGenerator);
  };
}

std::unique_ptr<Pass> mlir::createGenerateRuntimeVerificationPass() {
  return std::make_unique<GenerateRuntimeVerificationPass>();
}
