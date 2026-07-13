//===--- TaskGroup.h - Tracks completion of a group of tasks ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TaskGroup and related APIs.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_TASKGROUP_H
#define ORC_RT_TASKGROUP_H

#include "move_only_function.h"

#include <cassert>
#include <memory>
#include <mutex>
#include <vector>

namespace orc_rt {

/// TaskGroup tracks execution of a set of tasks, providing notification when
/// all tasks have completed.
class TaskGroup {
public:
  /// Token represents the right to proceed with a task as part of a
  /// TaskGroup.
  ///
  /// Construction (from a TaskGroup or by copy) may fail if the group is
  /// closed. Always check validity with operator bool() before proceeding:
  ///
  ///   Token T(TG);
  ///   if (!T) return;  // Group was closed
  ///
  /// WARNING: Avoid storing Tokens in long-lived data structures. The TaskGroup
  /// cannot complete while any Token exists, so stashing copies may
  /// unintentionally defer completion.
  class Token {
  public:
    /// Construct an empty Token not associated with any TaskGroup.
    Token() = default;

    /// Attempt to create a copy of the given Token.
    /// Note that this may fail if the TaskGroup has been closed. Clients must
    /// check whether the Token is valid (using operator bool()) before
    /// continuing with their task.
    Token(const Token &Other) {
      if (Other.G && Other.G->acquireToken())
        G = Other.G;
    }

    /// Attempt to overwrite this Token.
    /// Note that this will:
    ///   1. Trigger task group completion if this Token represented the last
    ///      running task in the TaskGroup and Other is an empty Token.
    ///   2. Fail if the TaskGroup referenced by Other has been closed. Clients
    ///      must check whether the Token is valid (using operator bool())
    ///      before continuing with their task.
    Token &operator=(const Token &Other) {
      if (&Other == this)
        return *this;

      if (G)
        G->releaseToken();
      if (Other.G && Other.G->acquireToken())
        G = Other.G;
      else
        G = nullptr;

      return *this;
    }

    /// Move-construct from Other.
    Token(Token &&Other) { std::swap(G, Other.G); }

    /// Move-assign from Other.
    ///
    /// Note that this will trigger task group completion if this Token
    /// represented the last running task in the TaskGroup and Other is an
    /// empty Token.
    Token &operator=(Token &&Other) {
      if (this == &Other)
        return *this;
      if (G) {
        G->releaseToken();
        G = nullptr;
      }
      std::swap(G, Other.G);
      return *this;
    }

    /// Construct a Token from the given TaskGroup.
    /// Note that this may fail if the TaskGroup has been closed. Clients must
    /// check whether the resulting Token is valid (using operator bool())
    /// before continuing with their task.
    Token(std::shared_ptr<TaskGroup> G) {
      if (G && G->acquireToken())
        this->G = std::move(G);
    }

    /// Destroys this Token, potentially triggering task group completion if
    /// this Token represented the last running task in the TaskGroup.
    ~Token() {
      if (G)
        G->releaseToken();
    }

    /// Returns true if this Token is valid and attached to a task group.
    explicit operator bool() const noexcept { return !!G; }

  private:
    std::shared_ptr<TaskGroup> G;
  };

  using OnCompleteFn = move_only_function<void()>;

  TaskGroup(const TaskGroup &) = delete;
  TaskGroup &operator=(const TaskGroup &) = delete;
  TaskGroup(TaskGroup &&) = delete;
  TaskGroup &operator=(TaskGroup &&) = delete;

  static std::shared_ptr<TaskGroup> Create() noexcept {
    return std::shared_ptr<TaskGroup>(new TaskGroup());
  }

  /// Increment the number of tasks in this group if it is still open.
  /// Returns true on success, false on failure.
  bool acquireToken() noexcept {
    std::scoped_lock<std::mutex> Lock(M);
    if (Closed)
      return false;
    ++NumTasks;
    return true;
  }

  /// Decrement the number of tasks in this group. This will trigger any
  /// OnComplete callbacks if the TaskGroup has been closed and the count
  /// reaches zero.
  void releaseToken() noexcept {
    std::vector<OnCompleteFn> ToRun;
    {
      std::scoped_lock<std::mutex> Lock(M);
      assert(NumTasks > 0 && "TaskCount is invalid");
      --NumTasks;
      if (NumTasks == 0 && Closed)
        ToRun = std::move(OnCompletes);
    }
    if (ToRun.empty())
      return;
    runOnCompletes(std::move(ToRun));
  }

  /// Close the TaskGroup. No new Tokens will be issued. OnComplete callbacks
  /// will be run once the task count reaches zero.
  void close() {
    std::vector<OnCompleteFn> ToRun;
    {
      std::scoped_lock<std::mutex> Lock(M);
      Closed = true;
      if (NumTasks == 0)
        ToRun = std::move(OnCompletes);
    }
    if (ToRun.empty())
      return;
    runOnCompletes(std::move(ToRun));
  }

  /// Register an OnComplete callback. The given callback will be run once the
  /// group is closed and all tasks in it have completed.
  void addOnComplete(OnCompleteFn OnComplete) {
    assert(OnComplete && "OnComplete cannot be null");
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (!Closed || NumTasks > 0) {
        OnCompletes.push_back(std::move(OnComplete));
        return;
      }
    }
    assert(OnComplete && "OnComplete should still be present here");
    OnComplete();
  }

private:
  TaskGroup() noexcept = default;

  static void runOnCompletes(std::vector<OnCompleteFn> ToRun) {
    // TODO: Exception handling
    for (auto &OnComplete : ToRun)
      OnComplete();
  }

  std::mutex M;
  bool Closed = false;
  size_t NumTasks = 0;
  std::vector<OnCompleteFn> OnCompletes;
};

} // namespace orc_rt

#endif // ORC_RT_TASKGROUP_H
