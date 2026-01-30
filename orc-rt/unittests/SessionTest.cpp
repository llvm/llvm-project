//===- SessionTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's Session.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/Session.h"
#include "orc-rt/ExecutionScopeGuard.h"
#include "orc-rt/SPSWrapperFunction.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <chrono>
#include <deque>
#include <future>
#include <optional>

using namespace orc_rt;
using ::testing::Eq;
using ::testing::Optional;

class MockResourceManager : public ResourceManager {
public:
  enum class Op { Detach, Shutdown };

  static Error alwaysSucceed(Op) { return Error::success(); }

  MockResourceManager(std::optional<size_t> &DetachOpIdx,
                      std::optional<size_t> &ShutdownOpIdx, size_t &OpIdx,
                      move_only_function<Error(Op)> GenResult = alwaysSucceed)
      : DetachOpIdx(DetachOpIdx), ShutdownOpIdx(ShutdownOpIdx), OpIdx(OpIdx),
        GenResult(std::move(GenResult)) {}

  void detach(OnCompleteFn OnComplete) override {
    DetachOpIdx = OpIdx++;
    OnComplete(GenResult(Op::Detach));
  }

  void shutdown(OnCompleteFn OnComplete) override {
    ShutdownOpIdx = OpIdx++;
    OnComplete(GenResult(Op::Shutdown));
  }

private:
  std::optional<size_t> &DetachOpIdx;
  std::optional<size_t> &ShutdownOpIdx;
  size_t &OpIdx;
  move_only_function<Error(Op)> GenResult;
};

class MockControllerAccess : public Session::ControllerAccess {
public:
  using Task = move_only_function<void()>;
  using EnqueueTaskFn = move_only_function<void(Task)>;

  MockControllerAccess(Session &S, EnqueueTaskFn EnqueueTask)
      : Session::ControllerAccess(S), EnqueueTask(std::move(EnqueueTask)) {}

  void disconnect() override {
    std::unique_lock<std::mutex> Lock(M);
    Shutdown = true;
    ShutdownCV.wait(Lock, [this]() { return Shutdown && Outstanding == 0; });
  }

  void callController(OnCallHandlerCompleteFn OnComplete, HandlerTag T,
                      WrapperFunctionBuffer ArgBytes) override {
    // Simulate a call to the controller by dispatching a task to run the
    // requested function.
    size_t CId;
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (Shutdown)
        return;
      CId = CallId++;
      Pending[CId] = std::move(OnComplete);
      ++Outstanding;
    }

    EnqueueTask([this, CId, OnComplete = std::move(OnComplete), T,
                 ArgBytes = std::move(ArgBytes)]() mutable {
      auto Fn = reinterpret_cast<orc_rt_WrapperFunction>(T);
      Fn(reinterpret_cast<orc_rt_SessionRef>(this), CId, wfReturn,
         ArgBytes.release());
    });

    bool Notify = false;
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (--Outstanding == 0 && Shutdown)
        Notify = true;
    }
    if (Notify)
      ShutdownCV.notify_all();
  }

  void sendWrapperResult(uint64_t CallId,
                         WrapperFunctionBuffer ResultBytes) override {
    // Respond to a simulated call by the controller.
    OnCallHandlerCompleteFn OnComplete;
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (Shutdown) {
        assert(Pending.empty() && "Shut down but results still pending?");
        return;
      }
      auto I = Pending.find(CallId);
      assert(I != Pending.end());
      OnComplete = std::move(I->second);
      Pending.erase(I);
      ++Outstanding;
    }

    EnqueueTask([OnComplete = std::move(OnComplete),
                 ResultBytes = std::move(ResultBytes)]() mutable {
      OnComplete(std::move(ResultBytes));
    });

    bool Notify = false;
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (--Outstanding == 0 && Shutdown)
        Notify = true;
    }
    if (Notify)
      ShutdownCV.notify_all();
  }

  void callFromController(OnCallHandlerCompleteFn OnComplete,
                          orc_rt_WrapperFunction Fn,
                          WrapperFunctionBuffer ArgBytes) {
    size_t CId = 0;
    bool BailOut = false;
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (!Shutdown) {
        CId = CallId++;
        Pending[CId] = std::move(OnComplete);
        ++Outstanding;
      } else
        BailOut = true;
    }
    if (BailOut)
      return OnComplete(WrapperFunctionBuffer::createOutOfBandError(
          "Controller disconnected"));

    EnqueueTask([this, CId, Fn, ArgBytes = std::move(ArgBytes)]() mutable {
      handleWrapperCall(CId, Fn, std::move(ArgBytes));
    });

    bool Notify = false;
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (--Outstanding == 0 && Shutdown)
        Notify = true;
    }

    if (Notify)
      ShutdownCV.notify_all();
  }

  /// Simulate start of outstanding operation.
  void incOutstanding() {
    std::scoped_lock<std::mutex> Lock(M);
    ++Outstanding;
  }

  /// Simulate end of outstanding operation.
  void decOutstanding() {
    bool Notify = false;
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (--Outstanding == 0 && Shutdown)
        Notify = true;
    }
    if (Notify)
      ShutdownCV.notify_all();
  }

private:
  static void wfReturn(orc_rt_SessionRef S, uint64_t CallId,
                       orc_rt_WrapperFunctionBuffer ResultBytes) {
    // Abuse "session" to refer to the ControllerAccess object.
    // We can just re-use sendFunctionResult for this.
    reinterpret_cast<MockControllerAccess *>(S)->sendWrapperResult(
        CallId, WrapperFunctionBuffer(ResultBytes));
  }

  EnqueueTaskFn EnqueueTask;

  std::mutex M;
  bool Shutdown = false;
  size_t Outstanding = 0;
  size_t CallId = 0;
  std::unordered_map<size_t, OnCallHandlerCompleteFn> Pending;
  std::condition_variable ShutdownCV;
};

class TaskQueue {
public:
  TaskQueue(ExecutionScopeGuard ER) : ER(std::move(ER)) {
    assert(this->ER && "Invalid ExecutionScopeGuard");
  }

  MockControllerAccess::EnqueueTaskFn getEnqueuer() {
    assert(ER && "getEnqueuer can only be called once");
    return [this, ER = std::move(ER)](MockControllerAccess::Task T) {
      append(std::move(T));
    };
  }

  void runFromFront() {
    while (true) {
      MockControllerAccess::Task T;
      {
        std::scoped_lock<std::mutex> Lock(M);
        if (!Q.empty()) {
          T = std::move(Q.front());
          Q.pop_front();
        }
      }
      if (T)
        T();
      else
        break;
    }
  }

private:
  void append(MockControllerAccess::Task T) {
    std::scoped_lock<std::mutex> Lock(M);
    Q.push_back(std::move(T));
  }

  ExecutionScopeGuard ER;
  std::mutex M;
  std::deque<MockControllerAccess::Task> Q;
};

class CallViaMockControllerAccess {
public:
  CallViaMockControllerAccess(MockControllerAccess &CA,
                              orc_rt_WrapperFunction Fn)
      : CA(CA), Fn(Fn) {}
  void operator()(Session::OnCallHandlerCompleteFn OnComplete,
                  WrapperFunctionBuffer ArgBytes) {
    CA.callFromController(std::move(OnComplete), Fn, std::move(ArgBytes));
  }

private:
  MockControllerAccess &CA;
  orc_rt_WrapperFunction Fn;
};

// Non-overloaded version of cantFail: allows easy construction of
// move_only_functions<void(Error)>s.
static void noErrors(Error Err) { cantFail(std::move(Err)); }

TEST(SessionTest, TrivialConstructionAndDestruction) { Session S(noErrors); }

TEST(SessionTest, ReportError) {
  Error E = Error::success();
  cantFail(std::move(E)); // Force error into checked state.

  Session S([&](Error Err) { E = std::move(Err); });
  S.reportError(make_error<StringError>("foo"));

  if (E)
    EXPECT_EQ(toString(std::move(E)), "foo");
  else
    ADD_FAILURE() << "Missing error value";
}

TEST(SessionTest, SingleResourceManager) {
  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;

  {
    Session S(noErrors);
    S.addResourceManager(std::make_unique<MockResourceManager>(
        DetachOpIdx, ShutdownOpIdx, OpIdx));
  }

  EXPECT_EQ(OpIdx, 1U);
  EXPECT_EQ(DetachOpIdx, std::nullopt);
  EXPECT_THAT(ShutdownOpIdx, Optional(Eq(0)));
}

TEST(SessionTest, MultipleResourceManagers) {
  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx[3];
  std::optional<size_t> ShutdownOpIdx[3];

  {
    Session S(noErrors);
    for (size_t I = 0; I != 3; ++I)
      S.addResourceManager(std::make_unique<MockResourceManager>(
          DetachOpIdx[I], ShutdownOpIdx[I], OpIdx));
  }

  EXPECT_EQ(OpIdx, 3U);
  // Expect shutdown in reverse order.
  for (size_t I = 0; I != 3; ++I) {
    EXPECT_EQ(DetachOpIdx[I], std::nullopt);
    EXPECT_THAT(ShutdownOpIdx[I], Optional(Eq(2 - I)));
  }
}

TEST(SessionTest, ExpectedShutdownSequence) {
  // Check that Session shutdown results in...
  // 1. ResourceManagers being shut down.
  // 2. A call to OnShutdownComplete.

  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;

  bool SessionShutdownComplete = false;
  Session S(noErrors);
  S.addResourceManager(
      std::make_unique<MockResourceManager>(DetachOpIdx, ShutdownOpIdx, OpIdx));

  S.shutdown([&]() {
    EXPECT_EQ(DetachOpIdx, std::nullopt);
    EXPECT_EQ(ShutdownOpIdx, 0U);
    SessionShutdownComplete = true;
  });
  S.waitForShutdown();

  EXPECT_TRUE(SessionShutdownComplete);
}

TEST(ControllerAccessTest, Basics) {
  // Test that we can set the ControllerAccess implementation and still shut
  // down as expected.
  Session S(noErrors);
  TaskQueue Tasks{ExecutionScopeGuard(S)};
  S.setController(
      std::make_shared<MockControllerAccess>(S, Tasks.getEnqueuer()));

  Tasks.runFromFront();

  S.waitForShutdown();
}

static void add_sps_wrapper(orc_rt_SessionRef S, uint64_t CallId,
                            orc_rt_WrapperFunctionReturn Return,
                            orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::handle(
      S, CallId, Return, ArgBytes,
      [](move_only_function<void(int32_t)> Return, int32_t X, int32_t Y) {
        Return(X + Y);
      });
}

TEST(ControllerAccessTest, ValidCallToController) {
  // Simulate a call to a controller handler.
  Session S(noErrors);
  TaskQueue Tasks{ExecutionScopeGuard(S)};
  S.setController(
      std::make_shared<MockControllerAccess>(S, Tasks.getEnqueuer()));

  int32_t Result = 0;
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      CallViaSession(S, reinterpret_cast<Session::HandlerTag>(add_sps_wrapper)),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); }, 41, 1);

  Tasks.runFromFront();

  EXPECT_EQ(Result, 42);

  S.waitForShutdown();
}

TEST(ControllerAccessTest, CallToControllerBeforeAttach) {
  // Expect calls to the controller prior to attaching to fail.
  Session S(noErrors);

  Error Err = Error::success();
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      CallViaSession(S, reinterpret_cast<Session::HandlerTag>(add_sps_wrapper)),
      [&](Expected<int32_t> R) {
        ErrorAsOutParameter _(Err);
        Err = R.takeError();
      },
      41, 1);

  EXPECT_EQ(toString(std::move(Err)), "no controller attached");

  S.waitForShutdown();
}

TEST(ControllerAccessTest, CallToControllerAfterDetach) {
  // Expect calls to the controller prior to attaching to fail.
  Session S(noErrors);
  TaskQueue Tasks{ExecutionScopeGuard(S)};
  S.setController(
      std::make_shared<MockControllerAccess>(S, Tasks.getEnqueuer()));

  S.detachFromController();

  Error Err = Error::success();
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      CallViaSession(S, reinterpret_cast<Session::HandlerTag>(add_sps_wrapper)),
      [&](Expected<int32_t> R) {
        ErrorAsOutParameter _(Err);
        Err = R.takeError();
      },
      41, 1);

  EXPECT_EQ(toString(std::move(Err)), "no controller attached");

  S.waitForShutdown();
}

TEST(ControllerAccessTest, CallFromController) {
  // Simulate a call from the controller.
  Session S(noErrors);
  TaskQueue Tasks{ExecutionScopeGuard(S)};
  auto CA = std::make_shared<MockControllerAccess>(S, Tasks.getEnqueuer());
  S.setController(CA);

  int32_t Result = 0;
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      CallViaMockControllerAccess(*CA, add_sps_wrapper),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); }, 41, 1);

  Tasks.runFromFront();

  EXPECT_EQ(Result, 42);

  // We need to explicitly reset CA here, since it holds an ExecutionScopeGuard
  // that would prevent shutdown from proceeding.
  CA.reset();

  S.waitForShutdown();
}

TEST(ControllerAccessTest, RedundantAsyncShutdown) {
  // Check that redundant calls to shutdown have their callbacks run.
  Session S(noErrors);
  S.waitForShutdown();

  bool RedundantCallbackRan = false;
  S.shutdown([&]() { RedundantCallbackRan = true; });
  EXPECT_TRUE(RedundantCallbackRan);
}
