//===- TestShapeFunctions.cpp - Passes to test shape function  ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <queue>

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This is a pass that reports shape functions associated with ops.
struct ReportShapeFnPass
    : public PassWrapper<ReportShapeFnPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};
} // end anonymous namespace

void ReportShapeFnPass::runOnOperation() {
  auto module = getOperation();

  // Report the shape function available to refine the op.
  auto shapeFnId = Identifier::get("shape.function", &getContext());
  auto remarkShapeFn = [&](shape::FunctionLibraryOp shapeFnLib, Operation *op) {
    if (op->isKnownTerminator())
      return true;
    if (auto typeInterface = dyn_cast<InferTypeOpInterface>(op)) {
      op->emitRemark() << "implements InferType op interface";
      return true;
    }
    if (auto fn = shapeFnLib.getShapeFunction(op)) {
      op->emitRemark() << "associated shape function: " << fn.getName();
      return true;
    }
    if (auto symbol = op->getAttrOfType<SymbolRefAttr>(shapeFnId)) {
      auto fn = cast<FuncOp>(SymbolTable::lookupSymbolIn(module, symbol));
      op->emitRemark() << "associated shape function: " << fn.getName();
      return true;
    }
    return false;
  };

  // Lookup shape function library.
  SmallVector<shape::FunctionLibraryOp, 4> libraries;
  auto attr = module->getAttr("shape.lib");
  if (attr) {
    auto lookup = [&](Attribute attr) {
      return cast<shape::FunctionLibraryOp>(
          SymbolTable::lookupSymbolIn(module, attr.cast<SymbolRefAttr>()));
    };
    if (auto arrayAttr = attr.dyn_cast<ArrayAttr>()) {
      libraries.reserve(arrayAttr.size());
      for (auto attr : arrayAttr)
        libraries.push_back(lookup(attr));
    } else {
      libraries.reserve(1);
      libraries.push_back(lookup(attr));
    }
  }

  module.getBodyRegion().walk([&](FuncOp func) {
    // Skip ops in the shape function library.
    if (isa<shape::FunctionLibraryOp>(func->getParentOp()))
      return;

    func.walk([&](Operation *op) {
      bool found = llvm::any_of(libraries, [&](shape::FunctionLibraryOp lib) {
        return remarkShapeFn(lib, op);
      });
      if (!found)
        op->emitRemark() << "no associated way to refine shape";
    });
  });
}

namespace mlir {
void registerShapeFunctionTestPasses() {
  PassRegistration<ReportShapeFnPass>(
      "test-shape-function-report",
      "Test pass to report associated shape functions");
}
} // namespace mlir
