//===- TestBuiltinAttributeInterfaces.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace test;

namespace {
struct TestElementsAttrInterface
    : public PassWrapper<TestElementsAttrInterface, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "test-elements-attr-interface"; }
  StringRef getDescription() const final {
    return "Test ElementsAttr interface support.";
  }
  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      for (NamedAttribute attr : op->getAttrs()) {
        auto elementsAttr = attr.getValue().dyn_cast<ElementsAttr>();
        if (!elementsAttr)
          continue;
        testElementsAttrIteration<uint64_t>(op, elementsAttr, "uint64_t");
        testElementsAttrIteration<APInt>(op, elementsAttr, "APInt");
        testElementsAttrIteration<IntegerAttr>(op, elementsAttr, "IntegerAttr");
      }
    });
  }

  template <typename T>
  void testElementsAttrIteration(Operation *op, ElementsAttr attr,
                                 StringRef type) {
    InFlightDiagnostic diag = op->emitError()
                              << "Test iterating `" << type << "`: ";

    auto values = attr.tryGetValues<T>();
    if (!values) {
      diag << "unable to iterate type";
      return;
    }

    llvm::interleaveComma(*values, diag, [&](T value) {
      diag << llvm::formatv("{0}", value).str();
    });
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestBuiltinAttributeInterfaces() {
  PassRegistration<TestElementsAttrInterface>();
}
} // namespace test
} // namespace mlir
