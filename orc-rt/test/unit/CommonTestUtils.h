//===- CommonTestUtils.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_UNITTEST_COMMONTESTUTILS_H
#define ORC_RT_UNITTEST_COMMONTESTUTILS_H

#include "orc-rt/Error.h"
#include "orc-rt/ExecutorProcessInfo.h"
#include "orc-rt/WrapperFunction.h"
#include "orc-rt/move_only_function.h"

#include "orc-rt-c/CoreTypes.h"
#include "orc-rt-c/WrapperFunction.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include "gtest/gtest.h"

inline void noErrors(orc_rt::Error Err) { orc_rt::cantFail(std::move(Err)); }

/// ReportError callback for tests that records the message of every reported
/// error, in the order reported.
class AccumulateErrors {
public:
  AccumulateErrors(std::vector<std::string> &ErrMsgs) : ErrMsgs(ErrMsgs) {}

  void operator()(orc_rt::Error Err) {
    ErrMsgs.push_back(orc_rt::toString(std::move(Err)));
  }

private:
  std::vector<std::string> &ErrMsgs;
};

inline orc_rt::ExecutorProcessInfo mockExecutorProcessInfo() noexcept {
  return orc_rt::ExecutorProcessInfo("arm64-apple-darwin", 16384);
}

/// DispatchFn for tests that should never dispatch a task. Records a test
/// failure on invocation, then runs the task inline so that any caller
/// awaiting a result unblocks (rather than hanging) and the managed-code token
/// is released, even in -Asserts builds or when the dispatch arrives on a
/// non-test thread.
inline void noDispatch(orc_rt::move_only_function<void()> Task) {
  ADD_FAILURE() << "unexpected dispatch in a no-dispatch session";
  Task();
}

template <size_t Idx = 0> class OpCounter {
public:
  OpCounter() { ++DefaultConstructions; }
  OpCounter(const OpCounter &Other) { ++CopyConstructions; }
  OpCounter &operator=(const OpCounter &Other) {
    ++CopyAssignments;
    return *this;
  }
  OpCounter(OpCounter &&Other) { ++MoveConstructions; }
  OpCounter &operator=(OpCounter &&Other) {
    ++MoveAssignments;
    return *this;
  }
  ~OpCounter() { ++Destructions; }

  static size_t defaultConstructions() { return DefaultConstructions; }
  static size_t copyConstructions() { return CopyConstructions; }
  static size_t copyAssignments() { return CopyAssignments; }
  static size_t copies() { return copyConstructions() + copyAssignments(); }
  static size_t moveConstructions() { return MoveConstructions; }
  static size_t moveAssignments() { return MoveAssignments; }
  static size_t moves() { return moveConstructions() + moveAssignments(); }
  static size_t destructions() { return Destructions; }

  static bool destructionsMatch() {
    return destructions() == defaultConstructions() + copies() + moves();
  }

  static void reset() {
    DefaultConstructions = 0;
    CopyConstructions = 0;
    CopyAssignments = 0;
    MoveConstructions = 0;
    MoveAssignments = 0;
    Destructions = 0;
  }

private:
  static size_t DefaultConstructions;
  static size_t CopyConstructions;
  static size_t CopyAssignments;
  static size_t MoveConstructions;
  static size_t MoveAssignments;
  static size_t Destructions;
};

template <size_t Idx> size_t OpCounter<Idx>::DefaultConstructions = 0;
template <size_t Idx> size_t OpCounter<Idx>::CopyConstructions = 0;
template <size_t Idx> size_t OpCounter<Idx>::CopyAssignments = 0;
template <size_t Idx> size_t OpCounter<Idx>::MoveConstructions = 0;
template <size_t Idx> size_t OpCounter<Idx>::MoveAssignments = 0;
template <size_t Idx> size_t OpCounter<Idx>::Destructions = 0;

template <typename T>
orc_rt::move_only_function<void(T)> waitFor(std::future<T> &F) {
  std::promise<T> P;
  F = P.get_future();
  return [P = std::move(P)](T Val) mutable { P.set_value(std::move(Val)); };
}

inline orc_rt::move_only_function<void()> waitFor(std::future<void> &F) {
  std::promise<void> P;
  F = P.get_future();
  return [P = std::move(P)]() mutable { P.set_value(); };
}

#endif // ORC_RT_UNITTEST_COMMONTESTUTILS_H
