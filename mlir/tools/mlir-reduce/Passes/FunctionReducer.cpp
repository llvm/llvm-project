//===- FunctionReducer.cpp - MLIR Reduce Function Reducer -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FunctionReducer class. It defines a variant generator
// class to be used in a Reduction Tree Pass instantiation with the aim of
// reducing the number of function operations in an MLIR Module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Reducer/Passes/FunctionReducer.h"
#include "mlir/IR/Function.h"

using namespace mlir;

/// Return the number of function operations in the module's body.
int countFunctions(ModuleOp module) {
  auto ops = module.getOps<FuncOp>();
  return std::distance(ops.begin(), ops.end());
}

/// Generate variants by removing function operations from the module in the
/// parent and link the variants as children in the Reduction Tree Pass.
void FunctionReducer::generateVariants(ReductionNode *parent,
                                       const Tester *test) {
  ModuleOp module = parent->getModule();
  int opCount = countFunctions(module);
  int sectionSize = opCount / 2;
  std::vector<Operation *> opsToRemove;

  if (opCount == 0)
    return;

  // Create a variant by deleting all ops.
  if (opCount == 1) {
    opsToRemove.clear();
    ModuleOp moduleVariant = module.clone();

    for (FuncOp op : moduleVariant.getOps<FuncOp>())
      opsToRemove.push_back(op);

    for (Operation *o : opsToRemove)
      o->erase();

    new ReductionNode(moduleVariant, parent);

    return;
  }

  // Create two variants by bisecting the module.
  for (int i = 0; i < 2; ++i) {
    opsToRemove.clear();
    ModuleOp moduleVariant = module.clone();

    for (auto op : enumerate(moduleVariant.getOps<FuncOp>())) {
      int index = op.index();
      if (index >= sectionSize * i && index < sectionSize * (i + 1))
        opsToRemove.push_back(op.value());
    }

    for (Operation *o : opsToRemove)
      o->erase();

    new ReductionNode(moduleVariant, parent);
  }

  return;
}
