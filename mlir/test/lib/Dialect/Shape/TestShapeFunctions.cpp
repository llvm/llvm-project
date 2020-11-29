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

  // Lookup shape function library.
  shape::FunctionLibraryOp shapeFnLib = nullptr;
  for (auto lib : module.getOps<shape::FunctionLibraryOp>()) {
    if (shapeFnLib) {
      lib.emitError("duplicate shape library op")
              .attachNote(shapeFnLib.getLoc())
          << "previous mapping";
      return signalPassFailure();
    }
    shapeFnLib = lib;
  };

  // Report the shape function available to refine the op.
  auto shapeFnId = Identifier::get("shape.function", &getContext());
  auto remarkShapeFn = [&](Operation *op) {
    if (op->isKnownTerminator())
      return;
    if (auto typeInterface = dyn_cast<InferTypeOpInterface>(op)) {
      op->emitRemark() << "implements InferType op interface";
    } else if (auto fn = shapeFnLib.getShapeFunction(op)) {
      op->emitRemark() << "associated shape function: " << fn.getName();
    } else if (auto symbol = op->getAttrOfType<SymbolRefAttr>(shapeFnId)) {
      auto fn = cast<FuncOp>(SymbolTable::lookupSymbolIn(module, symbol));
      op->emitRemark() << "associated shape function: " << fn.getName();
    } else {
      op->emitRemark() << "no associated way to refine shape";
    }
  };

  module.getBodyRegion().walk([&](FuncOp func) {
    // Skip ops in the shape function library.
    if (isa<shape::FunctionLibraryOp>(func.getParentOp()))
      return;

    func.walk([&](Operation *op) { remarkShapeFn(op); });
  });
}

namespace mlir {
void registerShapeFunctionTestPasses() {
  PassRegistration<ReportShapeFnPass>(
      "test-shape-function-report",
      "Test pass to report associated shape functions");
}
} // namespace mlir
