//===- DataLayoutAnalysis.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include <memory>

using namespace mlir;

DataLayoutAnalysis::DataLayoutAnalysis(Operation *root)
    : defaultLayout(std::make_unique<DataLayout>(DataLayoutOpInterface())) {
  // Construct a DataLayout if possible from the op.
  auto computeLayout = [this](Operation *op) {
    if (auto iface = dyn_cast<DataLayoutOpInterface>(op))
      layouts[op] = std::make_unique<DataLayout>(iface);
    if (auto module = dyn_cast<ModuleOp>(op))
      layouts[op] = std::make_unique<DataLayout>(module);
  };

  // Compute layouts for both ancestors and descendants.
  root->walk(computeLayout);
  for (Operation *ancestor = root->getParentOp(); ancestor != nullptr;
       ancestor = ancestor->getParentOp()) {
    computeLayout(ancestor);
  }
}

const DataLayout &DataLayoutAnalysis::getAbove(Operation *operation) const {
  for (Operation *ancestor = operation->getParentOp(); ancestor != nullptr;
       ancestor = ancestor->getParentOp()) {
    auto it = layouts.find(ancestor);
    if (it != layouts.end())
      return *it->getSecond();
  }

  // Fallback to the default layout.
  return *defaultLayout;
}

const DataLayout &DataLayoutAnalysis::getAtOrAbove(Operation *operation) const {
  auto it = layouts.find(operation);
  if (it != layouts.end())
    return *it->getSecond();
  return getAbove(operation);
}
