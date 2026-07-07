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
#include "orc-rt/QueueingRunner.h"
#include "orc-rt/SPSWrapperFunction.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "CommonTestUtils.h"

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

  void onDetach(OnCompleteFn OnComplete, bool ShutdownRequested) override {
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

  /// Fallible named constructor for testing tryCreateService.
  static Expected<std::unique_ptr<ConfigurableService>> Create(bool Fail) {
    if (Fail)
      return make_error<StringError>("failed to create service");
    return std::make_unique<ConfigurableService>(42);
  }

  void onDetach(OnCompleteFn OnComplete, bool ShutdownRequested) override {
    OnComplete();
  }

  void onShutdown(OnCompleteFn OnComplete) override { OnComplete(); }

  void doMoreConfig(int) noexcept {}
};

class MockControllerAccess : public Session::ControllerAccess {
public:
  using OnConnectFn = move_only_function<Error(BootstrapInfo &BI)>;

  /// Hook used to defer controller-side work (the simulated controller
  /// handler invocation, and the OnComplete callback for completed calls).
  /// Tests typically wire this to push onto the same WorkQueue that the
  /// Session's QueueingRunner uses, so that a single drain advances both
  /// sides.
  using PostFn = move_only_function<void(move_only_function<void()>)>;

  MockControllerAccess(Session &SS, PostFn Post = {},
                       OnConnectFn OnConnect = {},
                       MockControllerAccess **Self = nullptr)
      : Session::ControllerAccess(SS), Post(std::move(Post)),
        OnConnect(std::move(OnConnect)) {
    // Optionally publish this instance so tests that need to drive the
    // controller side directly can reach it after attach constructs it.
    // (attach constructs the ControllerAccess internally and does not hand
    // back a reference, since the object may not outlive the attach call.)
    if (Self)
      *Self = this;
  }

  /// Fallible named constructor for testing tryAttach. Returns an error if
  /// Fail is true, otherwise a MockControllerAccess forwarding the remaining
  /// arguments to the constructor.
  static Expected<std::shared_ptr<MockControllerAccess>>
  Create(Session &S, bool Fail, PostFn Post = {}, OnConnectFn OnConnect = {}) {
    if (Fail)
      return make_error<StringError>("failed to create controller access");
    return std::make_shared<MockControllerAccess>(S, std::move(Post),
                                                  std::move(OnConnect));
  }

  void connect(BootstrapInfo BI) override {
    if (OnConnect) {
      if (auto Err = OnConnect(BI)) {
        reportError(std::move(Err));
        notifyDisconnected();
      }
    }
  }

  void disconnect() override {
    std::unique_lock<std::mutex> Lock(M);
    Shutdown = true;
    ShutdownCV.wait(Lock, [this]() { return Shutdown && Outstanding == 0; });
    notifyDisconnected();
  }

  void callController(OnCallHandlerCompleteFn OnComplete, HandlerTag T,
                      WrapperFunctionBuffer ArgBytes) override {
    // Simulate a call to the controller by running the requested function via
    // the test-supplied Post hook (or inline, if no hook was provided).
    size_t CId;
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (Shutdown)
        return;
      CId = CallId++;
      Pending[CId] = std::move(OnComplete);
      ++Outstanding;
    }

    runOrPost([this, CId, T, ArgBytes = std::move(ArgBytes)]() mutable {
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

    runOrPost([OnComplete = std::move(OnComplete),
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
  void runOrPost(move_only_function<void()> Work) {
    if (Post)
      Post(std::move(Work));
    else
      Work();
  }

  static void wfReturn(orc_rt_SessionRef S, uint64_t CallId,
                       orc_rt_WrapperFunctionBuffer ResultBytes) {
    // Abuse "session" to refer to the ControllerAccess object.
    // We can just re-use sendFunctionResult for this.
    reinterpret_cast<MockControllerAccess *>(S)->sendWrapperResult(
        CallId, WrapperFunctionBuffer(ResultBytes));
  }

  PostFn Post;

  std::mutex M;
  bool Shutdown = false;
  size_t Outstanding = 0;
  size_t CallId = 0;
  std::unordered_map<size_t, OnCallHandlerCompleteFn> Pending;
  std::condition_variable ShutdownCV;
  OnConnectFn OnConnect;
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

/// Build a PostFn for MockControllerAccess that pushes its work onto the
/// supplied queue. With this, a single QueueingRunner::runFIFOUntilEmpty(Q)
/// call advances both Session-side and controller-side work.
inline MockControllerAccess::PostFn postOnto(QueueingRunner<>::WorkQueue &Q) {
  return
      [&Q](move_only_function<void()> Work) { Q.push_back(std::move(Work)); };
}

void waitForShutdown(Session &S) {
  std::promise<void> P;
  auto F = P.get_future();
  S.shutdown([P = std::move(P)]() mutable { P.set_value(); });
  F.get();
}

TEST(SessionTest, TrivialConstructionAndDestruction) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
}

TEST(SessionTest, ReportError) {
  Error E = Error::success();
  cantFail(std::move(E)); // Force error into checked state.

  Session S(mockExecutorProcessInfo(), noDispatch,
            [&](Error Err) { E = std::move(Err); });
  S.reportError(make_error<StringError>("foo"));

  if (E)
    EXPECT_EQ(toString(std::move(E)), "foo");
  else
    ADD_FAILURE() << "Missing error value";
}

TEST(SessionTest, SingleService) {
  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;

  {
    Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
    S.addService(
        std::make_unique<MockService>(DetachOpIdx, ShutdownOpIdx, OpIdx));
  }

  EXPECT_EQ(OpIdx, 2U);
  EXPECT_EQ(DetachOpIdx, 0U);
  EXPECT_EQ(ShutdownOpIdx, 1U);
}

TEST(SessionTest, MultipleServices) {
  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx[3];
  std::optional<size_t> ShutdownOpIdx[3];

  {
    Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
    for (size_t I = 0; I != 3; ++I)
      S.addService(std::make_unique<MockService>(DetachOpIdx[I],
                                                 ShutdownOpIdx[I], OpIdx));
  }

  EXPECT_EQ(OpIdx, 6U);
  // Expect shutdown in reverse order.
  for (size_t I = 0; I != 3; ++I) {
    EXPECT_EQ(DetachOpIdx[I], 2 - I);
    EXPECT_EQ(ShutdownOpIdx[I], 5 - I);
  }
}

TEST(SessionTest, ScheduleShutdownFromOnDetachHandler) {
  // Check that when we schedule a shutdown from an onDetach handler:
  // 1. The shutdown is scheduled.
  // 2. All onDetach handlers run before any onShutdown handlers.

  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  int OnDetachHandlersRun = 0;
  bool OnShutdownHandlerRun = false;

  S.addOnDetach([&]() { ++OnDetachHandlersRun; });
  S.addOnDetach([&]() { S.shutdown(); });
  S.addOnDetach([&]() { ++OnDetachHandlersRun; });
  S.addOnShutdown([&]() {
    EXPECT_EQ(OnDetachHandlersRun, 2);
    OnShutdownHandlerRun = true;
  });

  S.detach();

  EXPECT_TRUE(OnShutdownHandlerRun);
}

TEST(SessionTest, RedundantAsyncShutdown) {
  // Check that redundant calls to shutdown have their callbacks run.
  QueueingRunner<>::WorkQueue Tasks;
  Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);

  // Initiate shutdown here, and wait for the on-shutdown callbacks to start
  // running.
  waitForShutdown(S);

  // Now try to add a new on-shutdown callback and verify that it runs.
  bool RedundantCallbackRan = false;
  S.shutdown([&]() { RedundantCallbackRan = true; });
  EXPECT_TRUE(RedundantCallbackRan);
}

TEST(SessionTest, ExpectedShutdownSequenceWithNoActiveManagedCodeCalls) {
  // Check that Session shutdown results in...
  // 1. Services being shut down.
  // 2. A call to OnShutdownComplete.

  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;
  bool SessionShutdownComplete = false;

  {
    QueueingRunner<>::WorkQueue Tasks;
    Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);
    S.addService(
        std::make_unique<MockService>(DetachOpIdx, ShutdownOpIdx, OpIdx));

    S.shutdown([&]() {
      EXPECT_TRUE(ShutdownOpIdx);
      EXPECT_EQ(*ShutdownOpIdx, 1);
      SessionShutdownComplete = true;
    });
  }

  EXPECT_TRUE(SessionShutdownComplete);
}

TEST(SessionTest, ActiveManagedCallsDelayShutdown) {
  QueueingRunner<>::WorkQueue Tasks;
  Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);

  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;
  S.createService<MockService>(DetachOpIdx, ShutdownOpIdx, OpIdx);

  ASSERT_FALSE(DetachOpIdx);
  ASSERT_FALSE(ShutdownOpIdx);

  // Take a managed code call token. This should succeed.
  auto Tok = TaskGroup::Token(S.managedCodeTaskGroup());
  ASSERT_TRUE(Tok);

  // We expect shutdown to wait for any active managed calls to complete.
  bool ShutdownComplete = false;
  S.shutdown([&]() { ShutdownComplete = true; });

  // Detach should have happened, but shutdown should be waiting on token.
  EXPECT_EQ(DetachOpIdx, 0U);
  EXPECT_FALSE(ShutdownOpIdx);
  EXPECT_FALSE(ShutdownComplete);

  // The managed calls code group should have been closed. Assert that we
  // can't get a new token.
  ASSERT_FALSE(TaskGroup::Token(S.managedCodeTaskGroup()));

  Tok = TaskGroup::Token(); // Reset token.

  EXPECT_EQ(ShutdownOpIdx, 1U);

  EXPECT_TRUE(ShutdownComplete);
}

static void managedSyncVoidFunction(int *P) { *P = 42; }

TEST(SessionTest, SyncCallManagedCodeVoidFn) {
  // Test synchronous calls to a void function while holding a
  // ManagedCodeTaskGroup token.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  {
    // Pre-shutdown we expect token acquisition to succeed and the function to
    // run.
    int X = 0;
    bool CallSucceeded = S.callManagedCodeSync(managedSyncVoidFunction, &X);

    EXPECT_TRUE(CallSucceeded);
    EXPECT_EQ(X, 42U);
  }

  waitForShutdown(S);

  {
    // Post-shutdown we expect token acquisition to fail, and
    // callManagedCodeSync to return false.
    int X = 0;
    bool CallSucceeded = S.callManagedCodeSync(managedSyncVoidFunction, &X);

    EXPECT_FALSE(CallSucceeded);
  }
}

static int managedSyncNonVoidFunction(int N) { return N + 1; }

TEST(SessionTest, SyncCallManagedCodeNonVoidFn) {
  // Test synchronous calls to a non-void function while holding a
  // ManagedCodeTaskGroup token.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  {
    // Pre-shutdown we expect token acquisition to succeed, the function to be
    // run, and the result to be returned.
    auto Result = S.callManagedCodeSync(managedSyncNonVoidFunction, 41);

    EXPECT_TRUE(Result);
    EXPECT_EQ(*Result, 42U);
  }

  waitForShutdown(S);

  {
    // Post-shutdown we expect token acquisition to fail, and
    // callManagedCodeSync to return std::nullopt.
    auto Result = S.callManagedCodeSync(managedSyncNonVoidFunction, 41);

    EXPECT_EQ(Result, std::nullopt);
  }
}

static void managedAsyncVoidFunction(move_only_function<void()> Return,
                                     int *P) {
  *P = 42;
  Return();
}

TEST(SessionTest, AsyncCallManagedCodeVoidFn) {
  // Test asynchronous calls to a void function while holding a
  // ManagedCodeTaskGroup token.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  {
    // Pre-shutdown we expect token acquisition to succeed, and the function
    // and Return callback to be run.
    int X = 0;
    bool ReturnSucceeded = false;
    S.callManagedCodeAsync([&](bool B) { ReturnSucceeded = B; },
                           managedAsyncVoidFunction, &X);
    EXPECT_TRUE(ReturnSucceeded);
    EXPECT_EQ(X, 42U);
  }

  waitForShutdown(S);

  {
    // Post-shutdown we expect token acquisition to fail. Return should be
    // with `false` and the function should not be called.
    int X = 0;
    bool ReturnSucceeded = false;
    S.callManagedCodeAsync([&](bool B) { ReturnSucceeded = B; },
                           managedAsyncVoidFunction, &X);
    EXPECT_FALSE(ReturnSucceeded);
    EXPECT_EQ(X, 0U);
  }
}

static void managedAsyncNonVoidFunction(move_only_function<void(int)> Return,
                                        int *P) {
  Return(++*P);
}

TEST(SessionTest, AsyncCallManagedCodeNonVoidFn) {
  // Test asynchronous calls to a non-void function while holding a
  // ManagedCodeTaskGroup token.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  {
    // Pre-shutdown we expect token acquisition to succeed, and the function
    // and Return callback to be run.
    int N = 41;
    std::optional<int> Result;
    S.callManagedCodeAsync([&](std::optional<int> N) { Result = N; },
                           managedAsyncNonVoidFunction, &N);
    EXPECT_TRUE(Result);
    EXPECT_EQ(*Result, 42U);
    EXPECT_EQ(N, 42U);
  }

  waitForShutdown(S);

  {
    // Post-shutdown we expect token acquisition to fail. Return should be
    // with `std::nullopt` and the function should not be called.
    int N = 41;
    std::optional<int> Result;
    S.callManagedCodeAsync([&](std::optional<int> N) { Result = N; },
                           managedAsyncNonVoidFunction, &N);
    EXPECT_EQ(Result, std::nullopt);
    EXPECT_EQ(N, 41U);
  }
}

TEST(SessionTest, AsyncCallManagedCodeHoldsTokenAcrossAsyncGap) {
  // Verify that the ManagedCodeTaskGroup token is held until the async
  // continuation runs, not just until callManagedCodeAsync returns. This
  // ensures shutdown blocks for the duration of the actual async work.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;
  S.createService<MockService>(DetachOpIdx, ShutdownOpIdx, OpIdx);

  // The managed code function stashes its continuation instead of calling it.
  std::optional<int> Result;
  move_only_function<void(int)> StashedContinuation;
  S.callManagedCodeAsync([&](std::optional<int> N) { Result = std::move(N); },
                         [&](move_only_function<void(int)> Return, int N) {
                           // Stash the continuation and return without calling
                           // it.
                           StashedContinuation = std::move(Return);
                         },
                         41);

  // callManagedCodeAsync has returned, but the continuation hasn't been
  // called yet. The token should still be held inside StashedContinuation.
  ASSERT_TRUE(StashedContinuation);

  // Request shutdown. It should detach but block on the outstanding token.
  bool ShutdownComplete = false;
  S.shutdown([&]() { ShutdownComplete = true; });

  EXPECT_EQ(DetachOpIdx, 0U);
  EXPECT_FALSE(ShutdownOpIdx);
  EXPECT_FALSE(ShutdownComplete);

  // Now invoke the stashed continuation and then destroy it, releasing the
  // token.
  StashedContinuation(42);
  StashedContinuation = {};

  // Check result.
  EXPECT_EQ(Result, 42);

  // Shutdown should now have completed.
  EXPECT_EQ(ShutdownOpIdx, 1U);
  EXPECT_TRUE(ShutdownComplete);
}

TEST(SessionTest, AddServiceAndUseRef) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  auto &CS = S.addService(std::make_unique<ConfigurableService>(42));
  CS.doMoreConfig(1);
}

TEST(SessionTest, CreateServiceAndUseRef) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  auto &CS = S.createService<ConfigurableService>(42);
  CS.doMoreConfig(1);
}

TEST(SessionTest, TryCreateServiceSuccess) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  auto CS = S.tryCreateService<ConfigurableService>(false);
  if (auto Err = CS.takeError()) {
    ADD_FAILURE() << "expected service creation to succeed";
    consumeError(std::move(Err));
  }
}

TEST(SessionTest, TryCreateServiceFailure) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  auto CS = S.tryCreateService<ConfigurableService>(true);
  if (auto Err = CS.takeError())
    consumeError(std::move(Err));
  else
    ADD_FAILURE() << "expected service creation to fail";
}

TEST(ControllerAccessTest, Basics) {
  // Test that we can set the ControllerAccess implementation and still shut
  // down as expected.
  QueueingRunner<>::WorkQueue Tasks;
  Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);
  S.attach<MockControllerAccess>(BootstrapInfo(S), postOnto(Tasks));

  QueueingRunner<>::runFIFOUntilEmpty(Tasks);
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
  QueueingRunner<>::WorkQueue Tasks;
  Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);
  S.attach<MockControllerAccess>(BootstrapInfo(S), postOnto(Tasks));

  int32_t Result = 0;
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      S.callViaSession(reinterpret_cast<Session::HandlerTag>(add_sps_wrapper)),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); }, 41, 1);

  QueueingRunner<>::runFIFOUntilEmpty(Tasks);

  EXPECT_EQ(Result, 42);
}

TEST(ControllerAccessTest, CallToControllerBeforeAttach) {
  // Expect calls to the controller prior to attaching to fail.
  QueueingRunner<>::WorkQueue Tasks;
  Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);

  Error Err = Error::success();
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      S.callViaSession(reinterpret_cast<Session::HandlerTag>(add_sps_wrapper)),
      [&](Expected<int32_t> R) {
        ErrorAsOutParameter _(Err);
        Err = R.takeError();
      },
      41, 1);

  EXPECT_EQ(toString(std::move(Err)), "no controller attached");
}

TEST(ControllerAccessTest, CallToControllerAfterDetach) {
  // Expect calls to the controller prior to attaching to fail.
  QueueingRunner<>::WorkQueue Tasks;
  Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);
  S.attach<MockControllerAccess>(BootstrapInfo(S), postOnto(Tasks));

  S.detach();

  Error Err = Error::success();
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      S.callViaSession(reinterpret_cast<Session::HandlerTag>(add_sps_wrapper)),
      [&](Expected<int32_t> R) {
        ErrorAsOutParameter _(Err);
        Err = R.takeError();
      },
      41, 1);

  EXPECT_EQ(toString(std::move(Err)), "no controller attached");
}

TEST(ControllerAccessTest, CallFromController) {
  // Simulate a call from the controller.
  QueueingRunner<>::WorkQueue Tasks;
  Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);
  MockControllerAccess *CA = nullptr;
  S.attach<MockControllerAccess>(BootstrapInfo(S), postOnto(Tasks),
                                 MockControllerAccess::OnConnectFn{}, &CA);

  int32_t Result = 0;
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      CallViaMockControllerAccess(*CA, add_sps_wrapper),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); }, 41, 1);

  QueueingRunner<>::runFIFOUntilEmpty(Tasks);

  EXPECT_EQ(Result, 42);
}

TEST(ControllerAccessTest, FailConnect) {
  // Simulate failure to connect.
  bool GotError = false;
  std::string ErrMsg = "failed to connect";
  Session S(mockExecutorProcessInfo(), noDispatch, [&](Error Err) {
    GotError = true;
    EXPECT_EQ(toString(std::move(Err)), ErrMsg);
  });
  BootstrapInfo BI(S);
  S.attach<MockControllerAccess>(
      std::move(BI), MockControllerAccess::PostFn{},
      [&](BootstrapInfo &BI) { return make_error<StringError>(ErrMsg); });
  ASSERT_TRUE(GotError);
}

TEST(ControllerAccessTest, BootstrapInfoPassedToConnect) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  // Test values.
  constexpr const char *SymName = "test_sym";
  const char Sym = '.';
  constexpr const char *SecretKey = "luggage_combo";
  constexpr const char *SecretValue = "12345";

  // Build a BootstrapInfo with custom symbols and values.
  BootstrapInfo BI(S);
  std::pair<const char *, const void *> TestSyms[] = {
      {SymName, static_cast<const void *>(&Sym)}};
  cantFail(BI.symbols().addUnique(TestSyms));
  BI.values()[SecretKey] = SecretValue;

  bool OnConnectRan = false;
  S.attach<MockControllerAccess>(
      std::move(BI), MockControllerAccess::PostFn{}, [&](BootstrapInfo &BI) {
        EXPECT_EQ(BI.symbols().at(SymName), static_cast<const void *>(&Sym));
        EXPECT_EQ(BI.values().at(SecretKey), SecretValue);
        OnConnectRan = true;
        return Error::success();
      });

  ASSERT_TRUE(OnConnectRan);
}

TEST(ControllerAccessTest, TryAttachSuccess) {
  // A successful Create attaches the controller, which then services calls
  // just like one attached via attach<T>.
  QueueingRunner<>::WorkQueue Tasks;
  Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);
  cantFail(S.tryAttach<MockControllerAccess>(BootstrapInfo(S), /*Fail=*/false,
                                             postOnto(Tasks)));

  int32_t Result = 0;
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      S.callViaSession(reinterpret_cast<Session::HandlerTag>(add_sps_wrapper)),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); }, 41, 1);

  QueueingRunner<>::runFIFOUntilEmpty(Tasks);

  EXPECT_EQ(Result, 42);
}

TEST(ControllerAccessTest, TryAttachFailure) {
  // A failing Create surfaces its Error and leaves the Session unattached.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  auto Err = S.tryAttach<MockControllerAccess>(BootstrapInfo(S), /*Fail=*/true);
  ASSERT_TRUE(static_cast<bool>(Err));
  EXPECT_EQ(toString(std::move(Err)), "failed to create controller access");

  // Since nothing was attached, calls to the controller should fail as they
  // would before any attach.
  Error CallErr = Error::success();
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      S.callViaSession(reinterpret_cast<Session::HandlerTag>(add_sps_wrapper)),
      [&](Expected<int32_t> R) {
        ErrorAsOutParameter _(CallErr);
        CallErr = R.takeError();
      },
      41, 1);

  EXPECT_EQ(toString(std::move(CallErr)), "no controller attached");
}
