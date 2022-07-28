//====----- InsertDimensionSymbols.cpp --------------------------*- C++-*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include <queue>
#include <vector>

using namespace mlir;

namespace {

struct InsertDimensionSymbolsPass
    : public InsertDimensionSymbolsBase<InsertDimensionSymbolsPass> {

  InsertDimensionSymbolsPass(const std::string &entryFunc)
      : InsertDimensionSymbolsBase() {
    this->entryFunc = entryFunc;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    std::string dynamicSourceNamePrefix = "s";
    int dynamicSourceNameIdx = 0;

    moduleOp.walk([&](func::FuncOp funcOp) {
      if (funcOp.getName() != entryFunc)
        return;

      bool changed = false;
      SmallVector<Type> argTypes;
      for (Value arg : funcOp.getArguments()) {
        RankedTensorType rankedType =
            arg.getType().dyn_cast<RankedTensorType>();
        if (rankedType != nullptr && !rankedType.hasStaticShape()) {
          changed = true;
          SmallVector<FlatSymbolRefAttr> symbols;
          for (unsigned i = 0; i < rankedType.getRank(); ++i) {
            if (rankedType.isDynamicDim(i)) {
              std::string name = dynamicSourceNamePrefix +
                                 std::to_string(dynamicSourceNameIdx++);
              auto symbol = FlatSymbolRefAttr::get(context, name);
              symbols.push_back(symbol);
            }
          }
          auto shapeInfo = shape::ExtShapeInfoAttr::get(context, symbols);
          arg.setType(RankedTensorType::get(
              rankedType.getShape(), rankedType.getElementType(), shapeInfo));
        }
        argTypes.push_back(arg.getType());
      }

      if (changed)
        funcOp.setType(FunctionType::get(
            funcOp.getContext(), argTypes,
            funcOp.front().getTerminator()->getOperandTypes()));
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createInsertDimensionSymbolsPass(const std::string &entryFunc) {
  return std::make_unique<InsertDimensionSymbolsPass>(entryFunc);
}
