//===- ExecutionContext.h -  Execution Context Support *- C++ -*-=============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRACING_EXECUTIONCONTEXT_H
#define MLIR_TRACING_EXECUTIONCONTEXT_H

#include "mlir/Debug/BreakpointManager.h"
#include "mlir/IR/Action.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace tracing {

/// This class is used to keep track of the active actions in the stack.
/// It provides the current action but also access to the parent entry in the
/// stack. This allows to keep track of the nested nature in which actions may
/// be executed.
struct ActionActiveStack {
public:
  ActionActiveStack(const ActionActiveStack *parent, const Action &action,
                    int depth)
      : parent(parent), action(action), depth(depth) {}
  const ActionActiveStack *getParent() const { return parent; }
  const Action &getAction() const { return action; }
  int getDepth() const { return depth; }
  void print(raw_ostream &os, bool withContext) const;
  void dump() const {
    print(llvm::errs(), /*withContext=*/true);
    llvm::errs() << "\n";
  }
  Breakpoint *getBreakpoint() const { return breakpoint; }
  void setBreakpoint(Breakpoint *breakpoint) { this->breakpoint = breakpoint; }

private:
  Breakpoint *breakpoint = nullptr;
  const ActionActiveStack *parent;
  const Action &action;
  int depth;
};

/// The ExecutionContext is the main orchestration of the infrastructure, it
/// acts as a handler in the MLIRContext for executing an Action. When an action
/// is dispatched, it'll query its set of Breakpoints managers for a breakpoint
/// matching this action. If a breakpoint is hit, it passes the action and the
/// breakpoint information to a callback. The callback is responsible for
/// controlling the execution of the action through an enum value it returns.
/// Optionally, observers can be registered to be notified before and after the
/// callback is executed.
class ExecutionContext {
public:
  /// Enum that allows the client of the context to control the execution of the
  /// action.
  /// - Apply: The action is executed.
  /// - Skip: The action is skipped.
  /// - Step: The action is executed and the execution is paused before the next
  ///         action, including for nested actions encountered before the
  ///         current action finishes.
  /// - Next: The action is executed and the execution is paused after the
  ///         current action finishes before the next action.
  /// - Finish: The action is executed and the execution is paused only when we
  ///           reach the parent/enclosing operation. If there are no enclosing
  ///           operation, the execution continues without stopping.
  enum Control { Apply = 1, Skip = 2, Step = 3, Next = 4, Finish = 5 };

  /// The type of the callback that is used to control the execution.
  /// The callback is passed the current action.
  using CallbackTy = function_ref<Control(const ActionActiveStack *)>;

  /// Create an ExecutionContext with a callback that is used to control the
  /// execution.
  ExecutionContext(CallbackTy callback) { setCallback(callback); }
  ExecutionContext() = default;

  /// Set the callback that is used to control the execution.
  void setCallback(CallbackTy callback) {
    onBreakpointControlExecutionCallback = callback;
  }

  /// This abstract class defines the interface used to observe an Action
  /// execution. It allows to be notified before and after the callback is
  /// processed, but can't affect the execution.
  struct Observer {
    virtual ~Observer() = default;
    /// This method is called before the Action is executed
    /// If a breakpoint was hit, it is passed as an argument to the callback.
    /// The `willExecute` argument indicates whether the action will be executed
    /// or not.
    /// Note that this method will be called from multiple threads concurrently
    /// when MLIR multi-threading is enabled.
    virtual void beforeExecute(const ActionActiveStack *action,
                               Breakpoint *breakpoint, bool willExecute) {}

    /// This method is called after the Action is executed, if it was executed.
    /// It is not called if the action is skipped.
    /// Note that this method will be called from multiple threads concurrently
    /// when MLIR multi-threading is enabled.
    virtual void afterExecute(const ActionActiveStack *action) {}
  };

  /// Register a new `Observer` on this context. It'll be notified before and
  /// after executing an action. Note that this method is not thread-safe: it
  /// isn't supported to add a new observer while actions may be executed.
  void registerObserver(Observer *observer);

  /// Register a new `BreakpointManager` on this context. It'll have a chance to
  /// match an action before it gets executed. Note that this method is not
  /// thread-safe: it isn't supported to add a new manager while actions may be
  /// executed.
  void addBreakpointManager(BreakpointManager *manager) {
    breakpoints.push_back(manager);
  }

  /// Process the given action. This is the operator called by MLIRContext on
  /// `executeAction()`.
  void operator()(function_ref<void()> transform, const Action &action);

private:
  /// Callback that is executed when a breakpoint is hit and allows the client
  /// to control the execution.
  CallbackTy onBreakpointControlExecutionCallback;

  /// Next point to stop execution as describe by `Control` enum.
  /// This is handle by indicating at which levels of depth the next
  /// break should happen.
  std::optional<int> depthToBreak;

  /// Observers that are notified before and after the callback is executed.
  SmallVector<Observer *> observers;

  /// The list of managers that are queried for breakpoints.
  SmallVector<BreakpointManager *> breakpoints;
};

} // namespace tracing
} // namespace mlir

#endif // MLIR_TRACING_EXECUTIONCONTEXT_H
