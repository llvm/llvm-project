//===----------- TaskDispatcher.h - Task dispatch utils ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Task and TaskDispatcher classes.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_TASKDISPATCHER_H
#define ORC_RT_TASKDISPATCHER_H

#include "orc-rt/RTTI.h"

#include <memory>
#include <utility>

namespace orc_rt {

/// Represents an abstract task to be run.
class Task : public RTTIExtends<Task, RTTIRoot> {
public:
  virtual ~Task();
  virtual void run() = 0;
};

/// Base class for generic tasks.
class GenericTask : public RTTIExtends<GenericTask, Task> {};

/// Generic task implementation.
template <typename FnT> class GenericTaskImpl : public GenericTask {
public:
  GenericTaskImpl(FnT &&Fn) : Fn(std::forward<FnT>(Fn)) {}
  void run() override { Fn(); }

private:
  FnT Fn;
};

/// Create a generic task from a function object.
template <typename FnT> std::unique_ptr<GenericTask> makeGenericTask(FnT &&Fn) {
  return std::make_unique<GenericTaskImpl<std::decay_t<FnT>>>(
      std::forward<FnT>(Fn));
}

/// Abstract base for classes that dispatch Tasks.
class TaskDispatcher {
public:
  virtual ~TaskDispatcher();

  /// Run the given task.
  virtual void dispatch(std::unique_ptr<Task> T) = 0;

  /// Called by Session. Should cause further dispatches to be rejected, and
  /// wait until all previously dispatched tasks have completed.
  virtual void shutdown() = 0;
};

} // End namespace orc_rt

#endif // ORC_RT_TASKDISPATCHER_H
