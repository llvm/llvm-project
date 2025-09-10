//===- TestSingleFold.cpp - Pass to test single-pass folding --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

using namespace mlir;

namespace {
/// Test pass for single-pass constant folding.
///
/// This pass tests the behavior of operations when folded exactly once. Unlike
/// canonicalization passes that may apply multiple rounds of folding, this pass
/// ensures that each operation is folded at most once, which is useful for
/// testing scenarios where the fold implementation should handle complex cases
/// without requiring multiple iterations.
///
/// The pass also removes dead constants after folding to clean up unused
/// intermediate results.
struct TestSingleFold : public PassWrapper<TestSingleFold, OperationPass<>>,
                        public RewriterBase::Listener {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSingleFold)

  StringRef getArgument() const final { return "test-single-fold"; }
  StringRef getDescription() const final {
    return "Test single-pass operation folding and dead constant elimination";
  }
  // All constants in the operation post folding.
  SmallVector<Operation *> existingConstants;

  void foldOperation(Operation *op, OperationFolder &helper);
  void runOnOperation() override;

  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override {
    existingConstants.push_back(op);
  }
  void notifyOperationErased(Operation *op) override {
    auto *it = llvm::find(existingConstants, op);
    if (it != existingConstants.end())
      existingConstants.erase(it);
  }
};
} // namespace

void TestSingleFold::foldOperation(Operation *op, OperationFolder &helper) {
  // Attempt to fold the specified operation, including handling unused or
  // duplicated constants.
  (void)helper.tryToFold(op);
}

void TestSingleFold::runOnOperation() {
  existingConstants.clear();

  // Collect and fold the operations within the operation.
  SmallVector<Operation *, 8> ops;
  getOperation()->walk<mlir::WalkOrder::PreOrder>(
      [&](Operation *op) { ops.push_back(op); });

  // Fold the constants in reverse so that the last generated constants from
  // folding are at the beginning. This creates somewhat of a linear ordering to
  // the newly generated constants that matches the operation order and improves
  // the readability of test cases.
  OperationFolder helper(&getContext(), /*listener=*/this);
  for (Operation *op : llvm::reverse(ops))
    foldOperation(op, helper);

  // By the time we are done, we may have simplified a bunch of code, leaving
  // around dead constants. Check for them now and remove them.
  for (auto *cst : existingConstants) {
    if (cst->use_empty())
      cst->erase();
  }
}

namespace mlir {
namespace test {
void registerTestSingleFold() { PassRegistration<TestSingleFold>(); }
} // namespace test
} // namespace mlir
