//===-- StringConstantPropagation.cpp - dataflow tutorial -------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is contains the implementations of the visit and initialize
// methods for StringConstantPropagation.
//
//===----------------------------------------------------------------------===//

#include "StringConstantPropagation.h"
#include "StringDialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "string-constant-propagation"

using namespace mlir;
using namespace dataflow;

LogicalResult StringConstantPropagation::visit(ProgramPoint *point) {
  LDBG() << "Visiting program point: " << point << " " << *point;
  // This function expects only to receive operations.
  auto *op = point->getPrevOp();

  // Get or create the constant string values of the operands.
  SmallVector<StringConstant *> operandValues;
  for (Value operand : op->getOperands()) {
    auto *value = getOrCreate<StringConstant>(operand);
    // Create a dependency from the state to this analysis. When the string
    // value of one of the operation's operands are updated, invoke the
    // transfer function again.
    addDependency(value, point);
    // If the state is uninitialized, bail out and come back later when it is
    // initialized.
    if (value->isUninitialized())
      return success();
    operandValues.push_back(value);
  }

  // Try to compute a constant value of the result.
  auto *result = getOrCreate<StringConstant>(op->getResult(0));
  if (auto constant = dyn_cast<string::ConstantOp>(op)) {
    // Just grab and set the constant value of the result of the operation.
    // Propagate an update to the state if it changed.
    propagateIfChanged(result, result->join(constant.getValue()));
  } else if (auto concat = dyn_cast<string::ConcatOp>(op)) {
    StringRef lhs = operandValues[0]->getStringValue();
    StringRef rhs = operandValues[1]->getStringValue();
    // If either operand is overdefined, the results are overdefined.
    if (lhs.empty() || rhs.empty()) {
      propagateIfChanged(result, result->defaultInitialize());

      // Otherwise, compute the constant value and join it with the result.
    } else {
      propagateIfChanged(result, result->join(lhs + rhs));
    }
  } else {
    // We don't know how to implement the transfer function for this
    // operation. Mark its results as overdefined.
    propagateIfChanged(result, result->defaultInitialize());
  }
  return success();
}

LogicalResult StringConstantPropagation::initialize(Operation *top) {
  LDBG() << "Initializing DeadCodeAnalysis for top-level op: "
         << top->getName();
  // Visit every nested string operation and set up its dependencies.
  top->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      auto *state = getOrCreate<StringConstant>(operand);
      addDependency(state, getProgramPointAfter(op));
    }
  });
  // Now that the dependency graph has been set up, "seed" the evolution of the
  // analysis by marking the constant values of all block arguments as
  // overdefined and the results of (non-constant) operations with no operands.
  auto defaultInitializeAll = [&](ValueRange values) {
    for (Value value : values) {
      auto *state = getOrCreate<StringConstant>(value);
      propagateIfChanged(state, state->defaultInitialize());
    }
  };
  top->walk([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        defaultInitializeAll(block.getArguments());
    if (auto constant = dyn_cast<string::ConstantOp>(op)) {
      auto *result = getOrCreate<StringConstant>(constant.getResult());
      propagateIfChanged(result, result->join(constant.getValue()));
    } else if (op->getNumOperands() == 0) {
      defaultInitializeAll(op->getResults());
    }
  });
  // The dependency graph has been set up and the analysis has been seeded.
  // Finish initialization and let the solver run.
  return success();
}
