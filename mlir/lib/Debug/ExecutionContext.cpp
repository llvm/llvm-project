//===- ExecutionContext.cpp - Debug Execution Context Support -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Debug/ExecutionContext.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FormatVariadic.h"

#include <cstddef>

using namespace mlir;
using namespace mlir::tracing;

//===----------------------------------------------------------------------===//
// ActionActiveStack
//===----------------------------------------------------------------------===//

void ActionActiveStack::print(raw_ostream &os, bool withContext) const {
  os << "ActionActiveStack depth " << getDepth() << "\n";
  const ActionActiveStack *current = this;
  int count = 0;
  while (current) {
    llvm::errs() << llvm::formatv("#{0,3}: ", count++);
    current->action.print(llvm::errs());
    llvm::errs() << "\n";
    ArrayRef<IRUnit> context = current->action.getContextIRUnits();
    if (withContext && !context.empty()) {
      llvm::errs() << "Context:\n";
      llvm::interleave(
          current->action.getContextIRUnits(),
          [&](const IRUnit &unit) {
            llvm::errs() << "  - ";
            unit.print(llvm::errs());
          },
          [&]() { llvm::errs() << "\n"; });
      llvm::errs() << "\n";
    }
    current = current->parent;
  }
}

//===----------------------------------------------------------------------===//
// ExecutionContext
//===----------------------------------------------------------------------===//

static const LLVM_THREAD_LOCAL ActionActiveStack *actionStack = nullptr;

void ExecutionContext::registerObserver(Observer *observer) {
  observers.push_back(observer);
}

void ExecutionContext::operator()(llvm::function_ref<void()> transform,
                                  const Action &action) {
  // Update the top of the stack with the current action.
  int depth = 0;
  if (actionStack)
    depth = actionStack->getDepth() + 1;
  ActionActiveStack info{actionStack, action, depth};
  actionStack = &info;
  auto raii = llvm::make_scope_exit([&]() { actionStack = info.getParent(); });
  Breakpoint *breakpoint = nullptr;

  // Invoke the callback here and handles control requests here.
  auto handleUserInput = [&]() -> bool {
    if (!onBreakpointControlExecutionCallback)
      return true;
    auto todoNext = onBreakpointControlExecutionCallback(actionStack);
    switch (todoNext) {
    case ExecutionContext::Apply:
      depthToBreak = std::nullopt;
      return true;
    case ExecutionContext::Skip:
      depthToBreak = std::nullopt;
      return false;
    case ExecutionContext::Step:
      depthToBreak = depth + 1;
      return true;
    case ExecutionContext::Next:
      depthToBreak = depth;
      return true;
    case ExecutionContext::Finish:
      depthToBreak = depth - 1;
      return true;
    }
    llvm::report_fatal_error("Unknown control request");
  };

  // Try to find a breakpoint that would hit on this action.
  // Right now there is no way to collect them all, we stop at the first one.
  for (auto *breakpointManager : breakpoints) {
    breakpoint = breakpointManager->match(action);
    if (breakpoint)
      break;
  }
  info.setBreakpoint(breakpoint);

  bool shouldExecuteAction = true;
  // If we have a breakpoint, or if `depthToBreak` was previously set and the
  // current depth matches, we invoke the user-provided callback.
  if (breakpoint || (depthToBreak && depth <= depthToBreak))
    shouldExecuteAction = handleUserInput();

  // Notify the observers about the current action.
  for (auto *observer : observers)
    observer->beforeExecute(actionStack, breakpoint, shouldExecuteAction);

  if (shouldExecuteAction) {
    // Execute the action here.
    transform();

    // Notify the observers about completion of the action.
    for (auto *observer : observers)
      observer->afterExecute(actionStack);
  }

  if (depthToBreak && depth <= depthToBreak)
    handleUserInput();
}
