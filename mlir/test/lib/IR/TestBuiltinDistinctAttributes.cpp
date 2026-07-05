//===- TestBuiltinDistinctAttributes.cpp - Test DistinctAttributes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This is a distinct attribute test pass that tests if distinct attributes can
/// be created in parallel in a deterministic way.
struct DistinctAttributesPass
    : public PassWrapper<DistinctAttributesPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistinctAttributesPass)

  StringRef getArgument() const final { return "test-distinct-attrs"; }
  StringRef getDescription() const final {
    return "Test parallel creation of distinct attributes";
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

    /// Walk all operations and create a distinct output attribute given a
    /// distinct input attribute.
    funcOp->walk([](Operation *op) {
      auto distinctAttr = op->getAttrOfType<DistinctAttr>("distinct.input");
      if (!distinctAttr)
        return;
      op->setAttr("distinct.output",
                  DistinctAttr::create(distinctAttr.getReferencedAttr()));
    });
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestBuiltinDistinctAttributes() {
  PassRegistration<DistinctAttributesPass>();
}
} // namespace test
} // namespace mlir
