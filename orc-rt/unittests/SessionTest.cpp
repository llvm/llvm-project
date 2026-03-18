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
#include "orc-rt/SPSWrapperFunction.h"
#include "orc-rt/ThreadPoolTaskDispatcher.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <chrono>
#include <deque>
#include <future>
#include <optional>

using namespace orc_rt;
using ::testing::Eq;
using ::testing::Optional;

class MockService : public Service {
public:
  enum class Op { Detach, Shutdown };

  static void noop(Op) {}

  MockService(std::optional<size_t> &DetachOpIdx,
              std::optional<size_t> &ShutdownOpIdx, size_t &OpIdx,
              move_only_function<void(Op)> GenResult = noop)
      : DetachOpIdx(DetachOpIdx), ShutdownOpIdx(ShutdownOpIdx), OpIdx(OpIdx),
        GenResult(std::move(GenResult)) {}

  void onDetach(OnCompleteFn OnComplete) override {
    DetachOpIdx = OpIdx++;
    GenResult(Op::Detach);
    OnComplete();
  }

  void onShutdown(OnCompleteFn OnComplete) override {
    ShutdownOpIdx = OpIdx++;
    GenResult(Op::Shutdown);
    OnComplete();
  }

private:
  std::optional<size_t> &DetachOpIdx;
  std::optional<size_t> &ShutdownOpIdx;
  size_t &OpIdx;
  move_only_function<void(Op)> GenResult;
};

class ConfigurableService : public Service {
public:
  ConfigurableService(int ConstructorOption) {}

  void onDetach(OnCompleteFn OnComplete) override { OnComplete(); }

  void onShutdown(OnCompleteFn OnComplete) override { OnComplete(); }

  void doMoreConfig(int) noexcept {}
};

static ExecutorProcessInfo mockExecutorProcessInfo() noexcept {
  return ExecutorProcessInfo("arm64-apple-darwin", 16384);
}

class NoDispatcher : public TaskDispatcher {
public:
  void dispatch(std::unique_ptr<Task> T) override {
    assert(false && "strictly no dispatching!");
  }
  void shutdown() override {}
};

class EnqueueingDispatcher : public TaskDispatcher {
public:
  using OnShutdownRunFn = move_only_function<void()>;
  EnqueueingDispatcher(std::deque<std::unique_ptr<Task>> &Tasks,
                       OnShutdownRunFn OnShutdownRun = {})
      : Tasks(Tasks), OnShutdownRun(std::move(OnShutdownRun)) {}
  void dispatch(std::unique_ptr<Task> T) override {
    Tasks.push_back(std::move(T));
  }
  void shutdown() override {
    if (OnShutdownRun)
      OnShutdownRun();
  }

  /// Run up to NumTasks (arbitrarily many if NumTasks == std::nullopt) tasks
  /// from the front of the queue, returning the number actually run.
  static size_t
  runTasksFromFront(std::deque<std::unique_ptr<Task>> &Tasks,
                    std::optional<size_t> NumTasks = std::nullopt) {
    size_t NumRun = 0;

    while (!Tasks.empty() && (!NumTasks || NumRun != *NumTasks)) {
      auto T = std::move(Tasks.front());
      Tasks.pop_front();
      T->run();
      ++NumRun;
    }

    return NumRun;
  }

private:
  std::deque<std::unique_ptr<Task>> &Tasks;
  OnShutdownRunFn OnShutdownRun;
};

class MockControllerAccess : public Session::ControllerAccess {
public:
  MockControllerAccess(Session &SS) : Session::ControllerAccess(SS), SS(SS) {}

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

    SS.dispatch(makeGenericTask([this, CId, OnComplete = std::move(OnComplete),
                                 T, ArgBytes = std::move(ArgBytes)]() mutable {
      auto Fn = reinterpret_cast<orc_rt_WrapperFunction>(T);
      Fn(reinterpret_cast<orc_rt_SessionRef>(this), CId, wfReturn,
         ArgBytes.release());
    }));

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

    SS.dispatch(
        makeGenericTask([OnComplete = std::move(OnComplete),
                         ResultBytes = std::move(ResultBytes)]() mutable {
          OnComplete(std::move(ResultBytes));
        }));

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

    handleWrapperCall(CId, Fn, std::move(ArgBytes));

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

  Session &SS;

  std::mutex M;
  bool Shutdown = false;
  size_t Outstanding = 0;
  size_t CallId = 0;
  std::unordered_map<size_t, OnCallHandlerCompleteFn> Pending;
  std::condition_variable ShutdownCV;
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

TEST(SessionTest, TrivialConstructionAndDestruction) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
}

TEST(SessionTest, ReportError) {
  Error E = Error::success();
  cantFail(std::move(E)); // Force error into checked state.

  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            [&](Error Err) { E = std::move(Err); });
  S.reportError(make_error<StringError>("foo"));

  if (E)
    EXPECT_EQ(toString(std::move(E)), "foo");
  else
    ADD_FAILURE() << "Missing error value";
}

TEST(SessionTest, DispatchTask) {
  int X = 0;
  std::deque<std::unique_ptr<Task>> Tasks;
  Session S(mockExecutorProcessInfo(),
            std::make_unique<EnqueueingDispatcher>(Tasks), noErrors);

  EXPECT_EQ(Tasks.size(), 0U);
  S.dispatch(makeGenericTask([&]() { ++X; }));
  EXPECT_EQ(Tasks.size(), 1U);
  auto T = std::move(Tasks.front());
  Tasks.pop_front();
  T->run();
  EXPECT_EQ(X, 1);
}

TEST(SessionTest, SingleService) {
  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;

  {
    Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
              noErrors);
    S.addService(
        std::make_unique<MockService>(DetachOpIdx, ShutdownOpIdx, OpIdx));
  }

  EXPECT_EQ(OpIdx, 1U);
  EXPECT_EQ(DetachOpIdx, std::nullopt);
  EXPECT_THAT(ShutdownOpIdx, Optional(Eq(0)));
}

TEST(SessionTest, MultipleServices) {
  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx[3];
  std::optional<size_t> ShutdownOpIdx[3];

  {
    Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
              noErrors);
    for (size_t I = 0; I != 3; ++I)
      S.addService(std::make_unique<MockService>(DetachOpIdx[I],
                                                 ShutdownOpIdx[I], OpIdx));
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
  // 1. Services being shut down.
  // 2. The TaskDispatcher being shut down.
  // 3. A call to OnShutdownComplete.

  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;

  bool DispatcherShutDown = false;
  bool SessionShutdownComplete = false;
  std::deque<std::unique_ptr<Task>> Tasks;
  Session S(mockExecutorProcessInfo(),
            std::make_unique<EnqueueingDispatcher>(
                Tasks,
                [&]() {
                  EXPECT_TRUE(ShutdownOpIdx);
                  EXPECT_EQ(*ShutdownOpIdx, 0);
                  EXPECT_FALSE(SessionShutdownComplete);
                  DispatcherShutDown = true;
                }),
            noErrors);
  S.addService(
      std::make_unique<MockService>(DetachOpIdx, ShutdownOpIdx, OpIdx));

  S.shutdown([&]() {
    EXPECT_TRUE(DispatcherShutDown);
    SessionShutdownComplete = true;
  });
  S.waitForShutdown();

  EXPECT_TRUE(SessionShutdownComplete);
}

TEST(SessionTest, AddServiceAndUseRef) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  auto &CS = S.addService(std::make_unique<ConfigurableService>(42));
  CS.doMoreConfig(1);
}

TEST(SessionTest, CreateServiceAndUseRef) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  auto &CS = S.createService<ConfigurableService>(42);
  CS.doMoreConfig(1);
}

TEST(SessionTest, ControllerInterfaceContainsSessionByDefault) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  ASSERT_TRUE(S.controllerInterface()->count("orc_rt_SessionInstance"));
  EXPECT_EQ(S.controllerInterface()->at("orc_rt_SessionInstance"),
            static_cast<void *>(&S));
}

TEST(SessionTest, ControllerInterfaceWithRef) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  int X = 0, Y = 0;
  S.controllerInterface().with_ref([&](SimpleSymbolTable &ST) {
    std::pair<const char *, void *> Syms[] = {
        {"orc_rt_A", static_cast<void *>(&X)},
        {"orc_rt_B", static_cast<void *>(&Y)}};
    cantFail(ST.addUnique(Syms));
  });

  EXPECT_EQ(S.controllerInterface()->at("orc_rt_A"), &X);
  EXPECT_EQ(S.controllerInterface()->at("orc_rt_B"), &Y);
}

TEST(SessionTest, ControllerInterfaceConstAccess) {
  Session S(mockExecutorProcessInfo(), std::make_unique<NoDispatcher>(),
            noErrors);
  int X = 0;
  std::pair<const char *, void *> Syms[] = {{"orc_rt_X", &X}};
  cantFail(S.controllerInterface()->addUnique(Syms));

  const Session &CS = S;
  ASSERT_TRUE(CS.controllerInterface()->count("orc_rt_X"));
  EXPECT_EQ(CS.controllerInterface()->at("orc_rt_X"), &X);
}

TEST(ControllerAccessTest, Basics) {
  // Test that we can set the ControllerAccess implementation and still shut
  // down as expected.
  std::deque<std::unique_ptr<Task>> Tasks;
  Session S(mockExecutorProcessInfo(),
            std::make_unique<EnqueueingDispatcher>(Tasks), noErrors);
  auto CA = std::make_shared<MockControllerAccess>(S);
  S.setController(CA);

  EnqueueingDispatcher::runTasksFromFront(Tasks);

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
  std::deque<std::unique_ptr<Task>> Tasks;
  Session S(mockExecutorProcessInfo(),
            std::make_unique<EnqueueingDispatcher>(Tasks), noErrors);
  auto CA = std::make_shared<MockControllerAccess>(S);
  S.setController(CA);

  int32_t Result = 0;
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      CallViaSession(S, reinterpret_cast<Session::HandlerTag>(add_sps_wrapper)),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); }, 41, 1);

  EnqueueingDispatcher::runTasksFromFront(Tasks);

  EXPECT_EQ(Result, 42);

  S.waitForShutdown();
}

TEST(ControllerAccessTest, CallToControllerBeforeAttach) {
  // Expect calls to the controller prior to attaching to fail.
  std::deque<std::unique_ptr<Task>> Tasks;
  Session S(mockExecutorProcessInfo(),
            std::make_unique<EnqueueingDispatcher>(Tasks), noErrors);

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
  std::deque<std::unique_ptr<Task>> Tasks;
  Session S(mockExecutorProcessInfo(),
            std::make_unique<EnqueueingDispatcher>(Tasks), noErrors);
  auto CA = std::make_shared<MockControllerAccess>(S);
  S.setController(CA);

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
  std::deque<std::unique_ptr<Task>> Tasks;
  Session S(mockExecutorProcessInfo(),
            std::make_unique<EnqueueingDispatcher>(Tasks), noErrors);
  auto CA = std::make_shared<MockControllerAccess>(S);
  S.setController(CA);

  int32_t Result = 0;
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      CallViaMockControllerAccess(*CA, add_sps_wrapper),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); }, 41, 1);

  EnqueueingDispatcher::runTasksFromFront(Tasks);

  EXPECT_EQ(Result, 42);

  S.waitForShutdown();
}

TEST(ControllerAccessTest, RedundantAsyncShutdown) {
  // Check that redundant calls to shutdown have their callbacks run.
  std::deque<std::unique_ptr<Task>> Tasks;
  Session S(mockExecutorProcessInfo(),
            std::make_unique<EnqueueingDispatcher>(Tasks), noErrors);
  S.waitForShutdown();

  bool RedundantCallbackRan = false;
  S.shutdown([&]() { RedundantCallbackRan = true; });
  EXPECT_TRUE(RedundantCallbackRan);
}
