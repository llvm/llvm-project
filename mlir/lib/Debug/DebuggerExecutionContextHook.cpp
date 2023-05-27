//===- DebuggerExecutionContextHook.cpp - Debugger Support ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Debug/DebuggerExecutionContextHook.h"

#include "mlir/Debug/BreakpointManagers/FileLineColLocBreakpointManager.h"
#include "mlir/Debug/BreakpointManagers/TagBreakpointManager.h"

using namespace mlir;
using namespace mlir::tracing;

namespace {
/// This structure tracks the state of the interactive debugger.
struct DebuggerState {
  /// This variable keeps track of the current control option. This is set by
  /// the debugger when control is handed over to it.
  ExecutionContext::Control debuggerControl = ExecutionContext::Apply;

  /// The breakpoint manager that allows the debugger to set breakpoints on
  /// action tags.
  TagBreakpointManager tagBreakpointManager;

  /// The breakpoint manager that allows the debugger to set breakpoints on
  /// FileLineColLoc locations.
  FileLineColLocBreakpointManager fileLineColLocBreakpointManager;

  /// Map of breakpoint IDs to breakpoint objects.
  DenseMap<unsigned, Breakpoint *> breakpointIdsMap;

  /// The current stack of actiive actions.
  const tracing::ActionActiveStack *actionActiveStack;

  /// This is a "cursor" in the IR, it is used for the debugger to navigate the
  /// IR associated to the actions.
  IRUnit cursor;
};
} // namespace

static DebuggerState &getGlobalDebuggerState() {
  static LLVM_THREAD_LOCAL DebuggerState debuggerState;
  return debuggerState;
}

extern "C" {
void mlirDebuggerSetControl(int controlOption) {
  getGlobalDebuggerState().debuggerControl =
      static_cast<ExecutionContext::Control>(controlOption);
}

void mlirDebuggerPrintContext() {
  DebuggerState &state = getGlobalDebuggerState();
  if (!state.actionActiveStack) {
    llvm::outs() << "No active action.\n";
    return;
  }
  const ArrayRef<IRUnit> &units =
      state.actionActiveStack->getAction().getContextIRUnits();
  llvm::outs() << units.size() << " available IRUnits:\n";
  for (const IRUnit &unit : units) {
    llvm::outs() << "  - ";
    unit.print(
        llvm::outs(),
        OpPrintingFlags().useLocalScope().skipRegions().enableDebugInfo());
    llvm::outs() << "\n";
  }
}

void mlirDebuggerPrintActionBacktrace(bool withContext) {
  DebuggerState &state = getGlobalDebuggerState();
  if (!state.actionActiveStack) {
    llvm::outs() << "No active action.\n";
    return;
  }
  state.actionActiveStack->print(llvm::outs(), withContext);
}

//===----------------------------------------------------------------------===//
// Cursor Management
//===----------------------------------------------------------------------===//

void mlirDebuggerCursorPrint(bool withRegion) {
  auto &state = getGlobalDebuggerState();
  if (!state.cursor) {
    llvm::outs() << "No active MLIR cursor, select from the context first\n";
    return;
  }
  state.cursor.print(llvm::outs(), OpPrintingFlags()
                                       .skipRegions(!withRegion)
                                       .useLocalScope()
                                       .enableDebugInfo());
  llvm::outs() << "\n";
}

void mlirDebuggerCursorSelectIRUnitFromContext(int index) {
  auto &state = getGlobalDebuggerState();
  if (!state.actionActiveStack) {
    llvm::outs() << "No active MLIR Action stack\n";
    return;
  }
  ArrayRef<IRUnit> units =
      state.actionActiveStack->getAction().getContextIRUnits();
  if (index < 0 || index >= static_cast<int>(units.size())) {
    llvm::outs() << "Index invalid, bounds: [0, " << units.size()
                 << "] but got " << index << "\n";
    return;
  }
  state.cursor = units[index];
  state.cursor.print(llvm::outs());
  llvm::outs() << "\n";
}

void mlirDebuggerCursorSelectParentIRUnit() {
  auto &state = getGlobalDebuggerState();
  if (!state.cursor) {
    llvm::outs() << "No active MLIR cursor, select from the context first\n";
    return;
  }
  IRUnit *unit = &state.cursor;
  if (auto *op = llvm::dyn_cast_if_present<Operation *>(*unit)) {
    state.cursor = op->getBlock();
  } else if (auto *region = llvm::dyn_cast_if_present<Region *>(*unit)) {
    state.cursor = region->getParentOp();
  } else if (auto *block = llvm::dyn_cast_if_present<Block *>(*unit)) {
    state.cursor = block->getParent();
  } else {
    llvm::outs() << "Current cursor is not a valid IRUnit";
    return;
  }
  state.cursor.print(llvm::outs());
  llvm::outs() << "\n";
}

void mlirDebuggerCursorSelectChildIRUnit(int index) {
  auto &state = getGlobalDebuggerState();
  if (!state.cursor) {
    llvm::outs() << "No active MLIR cursor, select from the context first\n";
    return;
  }
  IRUnit *unit = &state.cursor;
  if (auto *op = llvm::dyn_cast_if_present<Operation *>(*unit)) {
    if (index < 0 || index >= static_cast<int>(op->getNumRegions())) {
      llvm::outs() << "Index invalid, op has " << op->getNumRegions()
                   << " but got " << index << "\n";
      return;
    }
    state.cursor = &op->getRegion(index);
  } else if (auto *region = llvm::dyn_cast_if_present<Region *>(*unit)) {
    auto block = region->begin();
    int count = 0;
    while (block != region->end() && count != index) {
      ++block;
      ++count;
    }

    if (block == region->end()) {
      llvm::outs() << "Index invalid, region has " << count << " block but got "
                   << index << "\n";
      return;
    }
    state.cursor = &*block;
  } else if (auto *block = llvm::dyn_cast_if_present<Block *>(*unit)) {
    auto op = block->begin();
    int count = 0;
    while (op != block->end() && count != index) {
      ++op;
      ++count;
    }

    if (op == block->end()) {
      llvm::outs() << "Index invalid, block has " << count
                   << "operations but got " << index << "\n";
      return;
    }
    state.cursor = &*op;
  } else {
    llvm::outs() << "Current cursor is not a valid IRUnit";
    return;
  }
  state.cursor.print(llvm::outs());
  llvm::outs() << "\n";
}

void mlirDebuggerCursorSelectPreviousIRUnit() {
  auto &state = getGlobalDebuggerState();
  if (!state.cursor) {
    llvm::outs() << "No active MLIR cursor, select from the context first\n";
    return;
  }
  IRUnit *unit = &state.cursor;
  if (auto *op = llvm::dyn_cast_if_present<Operation *>(*unit)) {
    Operation *previous = op->getPrevNode();
    if (!previous) {
      llvm::outs() << "No previous operation in the current block\n";
      return;
    }
    state.cursor = previous;
  } else if (auto *region = llvm::dyn_cast_if_present<Region *>(*unit)) {
    llvm::outs() << "Has region\n";
    Operation *parent = region->getParentOp();
    if (!parent) {
      llvm::outs() << "No parent operation for the current region\n";
      return;
    }
    if (region->getRegionNumber() == 0) {
      llvm::outs() << "No previous region in the current operation\n";
      return;
    }
    state.cursor =
        &region->getParentOp()->getRegion(region->getRegionNumber() - 1);
  } else if (auto *block = llvm::dyn_cast_if_present<Block *>(*unit)) {
    Block *previous = block->getPrevNode();
    if (!previous) {
      llvm::outs() << "No previous block in the current region\n";
      return;
    }
    state.cursor = previous;
  } else {
    llvm::outs() << "Current cursor is not a valid IRUnit";
    return;
  }
  state.cursor.print(llvm::outs());
  llvm::outs() << "\n";
}

void mlirDebuggerCursorSelectNextIRUnit() {
  auto &state = getGlobalDebuggerState();
  if (!state.cursor) {
    llvm::outs() << "No active MLIR cursor, select from the context first\n";
    return;
  }
  IRUnit *unit = &state.cursor;
  if (auto *op = llvm::dyn_cast_if_present<Operation *>(*unit)) {
    Operation *next = op->getNextNode();
    if (!next) {
      llvm::outs() << "No next operation in the current block\n";
      return;
    }
    state.cursor = next;
  } else if (auto *region = llvm::dyn_cast_if_present<Region *>(*unit)) {
    Operation *parent = region->getParentOp();
    if (!parent) {
      llvm::outs() << "No parent operation for the current region\n";
      return;
    }
    if (region->getRegionNumber() == parent->getNumRegions() - 1) {
      llvm::outs() << "No next region in the current operation\n";
      return;
    }
    state.cursor =
        &region->getParentOp()->getRegion(region->getRegionNumber() + 1);
  } else if (auto *block = llvm::dyn_cast_if_present<Block *>(*unit)) {
    Block *next = block->getNextNode();
    if (!next) {
      llvm::outs() << "No next block in the current region\n";
      return;
    }
    state.cursor = next;
  } else {
    llvm::outs() << "Current cursor is not a valid IRUnit";
    return;
  }
  state.cursor.print(llvm::outs());
  llvm::outs() << "\n";
}

//===----------------------------------------------------------------------===//
// Breakpoint Management
//===----------------------------------------------------------------------===//

void mlirDebuggerEnableBreakpoint(BreakpointHandle breakpoint) {
  reinterpret_cast<Breakpoint *>(breakpoint)->enable();
}

void mlirDebuggerDisableBreakpoint(BreakpointHandle breakpoint) {
  reinterpret_cast<Breakpoint *>(breakpoint)->disable();
}

BreakpointHandle mlirDebuggerAddTagBreakpoint(const char *tag) {
  DebuggerState &state = getGlobalDebuggerState();
  Breakpoint *breakpoint =
      state.tagBreakpointManager.addBreakpoint(StringRef(tag, strlen(tag)));
  int breakpointId = state.breakpointIdsMap.size() + 1;
  state.breakpointIdsMap[breakpointId] = breakpoint;
  return reinterpret_cast<BreakpointHandle>(breakpoint);
}

void mlirDebuggerAddRewritePatternBreakpoint(const char *patternNameInfo) {}

void mlirDebuggerAddFileLineColLocBreakpoint(const char *file, int line,
                                             int col) {
  getGlobalDebuggerState().fileLineColLocBreakpointManager.addBreakpoint(
      StringRef(file, strlen(file)), line, col);
}

} // extern "C"

LLVM_ATTRIBUTE_NOINLINE void mlirDebuggerBreakpointHook() {
  static LLVM_THREAD_LOCAL void *volatile sink;
  sink = (void *)&sink;
}

static void preventLinkerDeadCodeElim() {
  static void *volatile sink;
  static bool initialized = [&]() {
    sink = (void *)mlirDebuggerSetControl;
    sink = (void *)mlirDebuggerEnableBreakpoint;
    sink = (void *)mlirDebuggerDisableBreakpoint;
    sink = (void *)mlirDebuggerPrintContext;
    sink = (void *)mlirDebuggerPrintActionBacktrace;
    sink = (void *)mlirDebuggerCursorPrint;
    sink = (void *)mlirDebuggerCursorSelectIRUnitFromContext;
    sink = (void *)mlirDebuggerCursorSelectParentIRUnit;
    sink = (void *)mlirDebuggerCursorSelectChildIRUnit;
    sink = (void *)mlirDebuggerCursorSelectPreviousIRUnit;
    sink = (void *)mlirDebuggerCursorSelectNextIRUnit;
    sink = (void *)mlirDebuggerAddTagBreakpoint;
    sink = (void *)mlirDebuggerAddRewritePatternBreakpoint;
    sink = (void *)mlirDebuggerAddFileLineColLocBreakpoint;
    sink = (void *)&sink;
    return true;
  }();
  (void)initialized;
}

static tracing::ExecutionContext::Control
debuggerCallBackFunction(const tracing::ActionActiveStack *actionStack) {
  preventLinkerDeadCodeElim();
  // Invoke the breakpoint hook, the debugger is supposed to trap this.
  // The debugger controls the execution from there by invoking
  // `mlirDebuggerSetControl()`.
  auto &state = getGlobalDebuggerState();
  state.actionActiveStack = actionStack;
  getGlobalDebuggerState().debuggerControl = ExecutionContext::Apply;
  actionStack->getAction().print(llvm::outs());
  llvm::outs() << "\n";
  mlirDebuggerBreakpointHook();
  return getGlobalDebuggerState().debuggerControl;
}

namespace {
/// Manage the stack of actions that are currently active.
class DebuggerObserver : public ExecutionContext::Observer {
  void beforeExecute(const ActionActiveStack *action, Breakpoint *breakpoint,
                     bool willExecute) override {
    auto &state = getGlobalDebuggerState();
    state.actionActiveStack = action;
  }
  void afterExecute(const ActionActiveStack *action) override {
    auto &state = getGlobalDebuggerState();
    state.actionActiveStack = action->getParent();
    state.cursor = nullptr;
  }
};
} // namespace

void mlir::setupDebuggerExecutionContextHook(
    tracing::ExecutionContext &executionContext) {
  executionContext.setCallback(debuggerCallBackFunction);
  DebuggerState &state = getGlobalDebuggerState();
  static DebuggerObserver observer;
  executionContext.registerObserver(&observer);
  executionContext.addBreakpointManager(&state.fileLineColLocBreakpointManager);
  executionContext.addBreakpointManager(&state.tagBreakpointManager);
}
