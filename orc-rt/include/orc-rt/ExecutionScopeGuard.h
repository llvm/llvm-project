//===-- ExecutionScopeGuard.h - RAII guard for execution scope --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_EXECUTIONSCOPEGUARD_H
#define ORC_RT_EXECUTIONSCOPEGUARD_H

#include "Session.h"

namespace orc_rt {

/// RAII guard for retaining the Session's execution scope.
///
/// Each Session has a single conceptual execution scope. While the execution
/// scope is retained, Session shutdown will not proceed. ExecutionScopeGuard
/// instances retain the execution scope on construction and release it on
/// destruction.
class ExecutionScopeGuard {
public:
  ExecutionScopeGuard() = default;
  explicit ExecutionScopeGuard(Session &S) : S(&S) { retain(); }
  ExecutionScopeGuard(const ExecutionScopeGuard &Other) : S(Other.S) {
    retain();
  }
  ExecutionScopeGuard &operator=(const ExecutionScopeGuard &Other) {
    // Retain before release to avoid triggering shutdown if this held the
    // last reference.
    Other.retain();
    release();
    S = Other.S;
    return *this;
  }
  ExecutionScopeGuard(ExecutionScopeGuard &&Other) : S(Other.S) {
    Other.S = nullptr;
  }
  ExecutionScopeGuard &operator=(ExecutionScopeGuard &&Other) {
    release();
    S = Other.S;
    Other.S = nullptr;
    return *this;
  }
  ~ExecutionScopeGuard() { release(); }
  Session *getSession() const { return S; }
  explicit operator bool() const { return !!S; }
  void reset() {
    release();
    S = nullptr;
  }

private:
  void retain() const {
    if (S)
      S->retainExecutionScope();
  }
  void release() const {
    if (S)
      S->releaseExecutionScope();
  }
  Session *S = nullptr;
};

} // namespace orc_rt

#endif // ORC_RT_EXECUTIONSCOPEGUARD_H
