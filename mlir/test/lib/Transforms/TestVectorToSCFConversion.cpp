//===- TestVectorToSCFConversion.cpp - Test VectorTransfers lowering ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

struct TestVectorToSCFPass
    : public PassWrapper<TestVectorToSCFPass, FunctionPass> {
  TestVectorToSCFPass() = default;
  TestVectorToSCFPass(const TestVectorToSCFPass &pass) {}

  Option<bool> fullUnroll{
      *this, "full-unroll",
      llvm::cl::desc(
          "Perform full unrolling when converting vector transfers to SCF"),
      llvm::cl::init(false)};

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    populateVectorToSCFConversionPatterns(
        patterns, context, VectorTransferToSCFOptions().setUnroll(fullUnroll));
    applyPatternsAndFoldGreedily(getFunction(), patterns);
  }
};

} // end anonymous namespace

namespace mlir {
void registerTestVectorToSCFPass() {
  PassRegistration<TestVectorToSCFPass> pass(
      "test-convert-vector-to-scf",
      "Converts vector transfer ops to loops over scalars and vector casts");
}
} // namespace mlir
