//===- TestLoopPermutation.cpp - Test affine loop permutation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test the affine for op permutation utility.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#define PASS_NAME "test-loop-permutation"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory(PASS_NAME " options");

namespace {

/// This pass applies the permutation on the first maximal perfect nest.
struct TestLoopPermutation
    : public PassWrapper<TestLoopPermutation, FunctionPass> {
  TestLoopPermutation() = default;
  TestLoopPermutation(const TestLoopPermutation &pass){};

  void runOnFunction() override;

private:
  /// Permutation specifying loop i is mapped to permList[i] in
  /// transformed nest (with i going from outermost to innermost).
  ListOption<unsigned> permList{*this, "permutation-map",
                                llvm::cl::desc("Specify the loop permutation"),
                                llvm::cl::OneOrMore, llvm::cl::CommaSeparated};
};

} // end anonymous namespace

void TestLoopPermutation::runOnFunction() {
  // Get the first maximal perfect nest.
  SmallVector<AffineForOp, 6> nest;
  for (auto &op : getFunction().front()) {
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      getPerfectlyNestedLoops(nest, forOp);
      break;
    }
  }

  // Nothing to do.
  if (nest.size() < 2)
    return;

  SmallVector<unsigned, 4> permMap(permList.begin(), permList.end());
  permuteLoops(nest, permMap);
}

namespace mlir {
void registerTestLoopPermutationPass() {
  PassRegistration<TestLoopPermutation>(
      PASS_NAME, "Tests affine loop permutation utility");
}
} // namespace mlir
