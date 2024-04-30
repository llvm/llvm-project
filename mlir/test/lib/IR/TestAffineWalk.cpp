//===- TestAffineWalk.cpp - Pass to test affine walks
//----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"

#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {
/// A test pass for verifying walk interrupts.
struct TestAffineWalk
    : public PassWrapper<TestAffineWalk, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAffineWalk)

  void runOnOperation() override;
  StringRef getArgument() const final { return "test-affine-walk"; }
  StringRef getDescription() const final { return "Test affine walk method."; }
};
} // namespace

/// Emits a remark for the first `map`'s result expression that contains a
/// mod expression.
static void checkMod(AffineMap map, Location loc) {
  for (AffineExpr e : map.getResults()) {
    e.walk([&](AffineExpr s) {
      if (s.getKind() == mlir::AffineExprKind::Mod) {
        emitRemark(loc, "mod expression: ");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
}

void TestAffineWalk::runOnOperation() {
  auto m = getOperation();
  // Test whether the walk is being correctly interrupted.
  m.walk([](Operation *op) {
    for (NamedAttribute attr : op->getAttrs()) {
      auto mapAttr = dyn_cast<AffineMapAttr>(attr.getValue());
      if (!mapAttr)
        return;
      checkMod(mapAttr.getAffineMap(), op->getLoc());
    }
  });
}

namespace mlir {
void registerTestAffineWalk() { PassRegistration<TestAffineWalk>(); }
} // namespace mlir
