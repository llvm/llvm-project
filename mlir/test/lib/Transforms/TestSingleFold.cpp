//===- TestSingleFold.cpp - Pass to test single-pass folding --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../IR/DumpNotifications.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/SetVector.h"

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

  TestSingleFold() = default;
  TestSingleFold(const TestSingleFold &pass) : PassWrapper(pass) {}

  StringRef getArgument() const final { return "test-single-fold"; }
  StringRef getDescription() const final {
    return "Test single-pass operation folding and dead constant elimination";
  }
  // Moves can report the same constant more than once. Keep each constant
  // once so that cleanup cannot visit an operation after erasing it.
  llvm::SmallSetVector<Operation *, 8> existingConstants;

  void foldOperation(Operation *op, OperationFolder &helper);
  void runOnOperation() override;

  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override {
    if (dumpNotifications)
      notificationPrinter.notifyOperationInserted(op, previous);
    existingConstants.insert(op);
  }
  void notifyOperationErased(Operation *op) override {
    if (dumpNotifications)
      notificationPrinter.notifyOperationErased(op);
    existingConstants.remove(op);
  }

  DumpNotifications notificationPrinter;
  Option<bool> dumpNotifications{
      *this, "dump-notifications",
      llvm::cl::desc("Print insertion and erasure notifications"),
      llvm::cl::init(false)};
  Option<int> maxIterations{*this, "max-iterations",
                            llvm::cl::desc("Max iterations in the tryToFold"),
                            llvm::cl::init(1)};
};
} // namespace

void TestSingleFold::foldOperation(Operation *op, OperationFolder &helper) {
  // Attempt to fold the specified operation, including handling unused or
  // duplicated constants.
  bool inPlaceUpdate = false;
  (void)helper.tryToFold(op, &inPlaceUpdate, maxIterations);
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
