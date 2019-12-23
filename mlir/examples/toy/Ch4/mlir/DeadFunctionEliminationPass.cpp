//===- DeadFunctionEliminationPass.cpp - Eliminate inlined functions ------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a Module level pass performing dead function
// elimination. This is required as a post-processing step after function
// inlining.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace {
/// This is a simple function DCE pass that deletes all non-main functions after
/// inlining.
/// TODO(riverriddle) This is only necessary because MLIR currently does not
/// have generic DCE support for functions.
class DeadFunctionEliminationPass
    : public mlir::ModulePass<DeadFunctionEliminationPass> {
public:
  void runOnModule() override {
    mlir::ModuleOp module = getModule();
    mlir::SymbolTable moduleSymTable(module);

    // Eliminate non-main functions.
    auto mainFn = moduleSymTable.lookup<mlir::FuncOp>("main");
    for (mlir::FuncOp func :
         llvm::make_early_inc_range(module.getOps<mlir::FuncOp>())) {
      if (func != mainFn)
        func.erase();
    }
  }
};
} // end anonymous namespace

/// Create a pass that eliminates inlined functions in toy.
std::unique_ptr<mlir::Pass> mlir::toy::createDeadFunctionEliminationPass() {
  return std::make_unique<DeadFunctionEliminationPass>();
}
