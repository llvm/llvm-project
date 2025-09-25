//===- EliminateFunctionParameter.cpp.cpp - Eliminate function Parameter --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"

namespace mlir {
namespace func {
#define GEN_PASS_DEF_ELIMINATEFUNCTIONPARAMETERPASS
#include "mlir/Dialect/Func/Transforms/Passes.h.inc"
} // namespace func

/// This function eliminates unnecessary parameters within the function.
static LogicalResult updateFunc(func::FuncOp funcOp, BitVector &arguemntNoUse) {
  Block &entryBlock = funcOp.front();
  bool change = false;
  FunctionType origType = funcOp.getFunctionType();
  llvm::ArrayRef<Type> origInputTypes = origType.getInputs();
  SmallVector<Type, 4> newInputTypes;
  for (auto iter : llvm::enumerate(funcOp.getArguments())) {
    size_t position = iter.index();
    if (!iter.value().use_empty()) {
      newInputTypes.push_back(origInputTypes[position]);
      continue;
    }
    arguemntNoUse.set(position);
    entryBlock.eraseArgument(position);
    change = true;
  }

  if (change) {
    auto newFunctionType = FunctionType::get(funcOp.getContext(), newInputTypes,
                                             origType.getResults());
    funcOp.setFunctionType(newFunctionType);
  }
  return success(change);
}

/// After eliminating redundant parameters from the function, update the
/// function calls.
static LogicalResult updateCall(func::CallOp callOp,
                                BitVector &argumentsNoUse) {
  ValueRange origOperands = callOp.getOperands();
  SmallVector<Value, 4> newOperands;
  for (auto iter : llvm::enumerate(origOperands)) {
    if (!argumentsNoUse[iter.index()])
      newOperands.push_back(iter.value());
  }
  callOp->setOperands(newOperands);
  return success();
}

namespace {
struct EliminateFunctionParameterPass
    : public func::impl::EliminateFunctionParameterPassBase<
          EliminateFunctionParameterPass> {
  using EliminateFunctionParameterPassBase<
      EliminateFunctionParameterPass>::EliminateFunctionParameterPassBase;
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      size_t argumentSize = funcOp.getArguments().size();
      if (!argumentSize)
        continue;
      BitVector argumentNoUse(argumentSize);
      if (failed(updateFunc(funcOp, argumentNoUse)))
        continue;

      auto symbolOp = mlir::cast<SymbolOpInterface>(funcOp.getOperation());
      auto users = symbolOp.getSymbolUses(moduleOp);

      if (!users.has_value())
        continue;
      for (SymbolTable::SymbolUse user : *users) {
        Operation *call = user.getUser();
        (void)updateCall(mlir::cast<func::CallOp>(call), argumentNoUse);
      }
    }
  }
};

} // namespace
} // namespace mlir
