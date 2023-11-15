//===- TestBuiltinAttributeInterfaces.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace test;

// Helper to print one scalar value, force int8_t to print as integer instead of
// char.
template <typename T>
static void printOneElement(InFlightDiagnostic &os, T value) {
  os << llvm::formatv("{0}", value).str();
}

namespace {
struct TestElementsAttrInterface
    : public PassWrapper<TestElementsAttrInterface, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestElementsAttrInterface)

  StringRef getArgument() const final { return "test-elements-attr-interface"; }
  StringRef getDescription() const final {
    return "Test ElementsAttr interface support.";
  }
  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      for (NamedAttribute attr : op->getAttrs()) {
        auto elementsAttr = dyn_cast<ElementsAttr>(attr.getValue());
        if (!elementsAttr)
          continue;
        testElementsAttrIteration<int64_t>(op, elementsAttr, "int64_t");
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

    if (!attr.getElementType().isa<mlir::IntegerType>()) {
      diag << "expected element type to be an integer type";
      return;
    }

    auto values = attr.tryGetValues<T>();
    if (!values) {
      diag << "unable to iterate type";
      return;
    }

    llvm::interleaveComma(*values, diag,
                          [&](T value) { printOneElement(diag, value); });
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
