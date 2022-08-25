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
template <>
void printOneElement<int8_t>(InFlightDiagnostic &os, int8_t value) {
  os << llvm::formatv("{0}", static_cast<int64_t>(value)).str();
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
        auto elementsAttr = attr.getValue().dyn_cast<ElementsAttr>();
        if (!elementsAttr)
          continue;
        if (auto concreteAttr =
                attr.getValue().dyn_cast<DenseArrayBaseAttr>()) {
          llvm::TypeSwitch<DenseArrayBaseAttr>(concreteAttr)
              .Case([&](DenseBoolArrayAttr attr) {
                testElementsAttrIteration<bool>(op, attr, "bool");
              })
              .Case([&](DenseI8ArrayAttr attr) {
                testElementsAttrIteration<int8_t>(op, attr, "int8_t");
              })
              .Case([&](DenseI16ArrayAttr attr) {
                testElementsAttrIteration<int16_t>(op, attr, "int16_t");
              })
              .Case([&](DenseI32ArrayAttr attr) {
                testElementsAttrIteration<int32_t>(op, attr, "int32_t");
              })
              .Case([&](DenseI64ArrayAttr attr) {
                testElementsAttrIteration<int64_t>(op, attr, "int64_t");
              })
              .Case([&](DenseF32ArrayAttr attr) {
                testElementsAttrIteration<float>(op, attr, "float");
              })
              .Case([&](DenseF64ArrayAttr attr) {
                testElementsAttrIteration<double>(op, attr, "double");
              });
          continue;
        }
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
