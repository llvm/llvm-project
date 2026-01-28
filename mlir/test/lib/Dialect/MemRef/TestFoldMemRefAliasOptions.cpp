//===- TestFoldMemRefAliasOptions.cpp - Test FoldMemRefAlias options ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass to exercise the optional arguments of
// FoldMemRefAliasOps (excluded patterns and control callback).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/TypeName.h"

using namespace mlir;

namespace {
struct TestFoldMemRefAliasOptionsPass
    : public PassWrapper<TestFoldMemRefAliasOptionsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFoldMemRefAliasOptionsPass)

  TestFoldMemRefAliasOptionsPass() = default;
  TestFoldMemRefAliasOptionsPass(const TestFoldMemRefAliasOptionsPass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final {
    return "test-fold-memref-alias-options";
  }
  StringRef getDescription() const final {
    return "Test FoldMemRefAliasOps optional arguments";
  }

  ListOption<std::string> excludedPatternTokens{
      *this, "exclude-pattern",
      llvm::cl::desc("Comma-separated tokens to exclude certain patterns "
                     "(e.g., load-subview)")};
  Option<std::string> controlAttr{
      *this, "control-attr",
      llvm::cl::desc(
          "Attribute name that disables rewrites when present on the "
          "matched operation"),
      llvm::cl::init("")};

  void runOnOperation() override;
};

void TestFoldMemRefAliasOptionsPass::runOnOperation() {
  // Custom version of "FoldMemRefAliasOps" to test its options, by:
  // 1) Excluding patterns that fold memref.subview into load ops
  // 2) Ignoring user ops that have a specific attribute.

  // Map friendly tokens to concrete pattern names expected by the exclusion
  // mechanism.
  SmallVector<std::string> disabledPatternNames;
  if (llvm::is_contained(excludedPatternTokens, "load-subview")) {
    // Resolve pattern debug names from a populated set.
    RewritePatternSet patternsSet(&getContext());
    memref::populateFoldMemRefAliasOpPatterns(patternsSet);
    for (auto &pattern : patternsSet.getNativePatterns()) {
      std::optional<OperationName> rootKind = pattern->getRootKind();
      if (rootKind &&
          rootKind->getStringRef() == memref::LoadOp::getOperationName()) {
        disabledPatternNames.push_back(pattern->getDebugName().str());
        break;
      }
    }
  }

  std::function<bool(Operation *)> controlFnStorage;
  function_ref<bool(Operation *)> controlFnRef;
  if (!controlAttr.empty()) {
    StringAttr attrName = StringAttr::get(&getContext(), controlAttr);
    controlFnStorage = [attrName](Operation *op) {
      return !op->hasAttr(attrName);
    };
    controlFnRef = controlFnStorage;
  }

  RewritePatternSet owningPatterns(&getContext());
  memref::populateFoldMemRefAliasOpPatterns(owningPatterns, controlFnRef);
  FrozenRewritePatternSet patterns(std::move(owningPatterns),
                                   disabledPatternNames);
  (void)applyPatternsGreedily(getOperation(), patterns);
}
} // namespace

namespace mlir {
namespace test {
void registerTestFoldMemRefAliasOptionsPass() {
  PassRegistration<TestFoldMemRefAliasOptionsPass>();
}
} // namespace test
} // namespace mlir
