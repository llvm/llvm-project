//===- TestInlining.cpp - Pass to inline calls in the test dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO: This pass is only necessary because the main inlining pass
// has no abstracted away the call+callee relationship. When the inlining
// interface has this support, this pass should be removed.
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace test;

namespace {
struct Inliner : public PassWrapper<Inliner, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "test-inline"; }
  StringRef getDescription() const final {
    return "Test inlining region calls";
  }

  void runOnOperation() override {
    auto function = getOperation();

    // Collect each of the direct function calls within the module.
    SmallVector<CallIndirectOp, 16> callers;
    function.walk([&](CallIndirectOp caller) { callers.push_back(caller); });

    // Build the inliner interface.
    InlinerInterface interface(&getContext());

    // Try to inline each of the call operations.
    for (auto caller : callers) {
      auto callee = dyn_cast_or_null<FunctionalRegionOp>(
          caller.getCallee().getDefiningOp());
      if (!callee)
        continue;

      // Inline the functional region operation, but only clone the internal
      // region if there is more than one use.
      if (failed(inlineRegion(
              interface, &callee.getBody(), caller, caller.getArgOperands(),
              caller.getResults(), caller.getLoc(),
              /*shouldCloneInlinedRegion=*/!callee.getResult().hasOneUse())))
        continue;

      // If the inlining was successful then erase the call and callee if
      // possible.
      caller.erase();
      if (callee.use_empty())
        callee.erase();
    }
  }
};
} // namespace

namespace mlir {
namespace test {
void registerInliner() { PassRegistration<Inliner>(); }
} // namespace test
} // namespace mlir
