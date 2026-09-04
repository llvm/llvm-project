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

#include "orc-rt/bedrock/Session.h"
#include "orc-rt/bedrock/QueueingRunner.h"
#include "orc-rt/support/sps/SPSWrapperFunction.h"

#include "orc-rt-c/bedrock/Session.h"

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

private:
  class ConnectGuard {
  public:
    ConnectGuard() = default;
    ConnectGuard(MockControllerAccess *MCA) : MCA(MCA) {
      // Note: Assumes already locked.
      ++MCA->Outstanding;
    }
    ConnectGuard(const ConnectGuard &) = delete;
    ConnectGuard &operator=(const ConnectGuard &) = delete;
    ConnectGuard(ConnectGuard &&Other) : MCA(Other.MCA) { Other.MCA = nullptr; }
    ConnectGuard &operator=(ConnectGuard &&Other) {
      reset();
      MCA = Other.MCA;
      Other.MCA = nullptr;
      return *this;
    };
    ~ConnectGuard() { reset(); }

  private:
    void reset() {
      if (MCA) {
        bool Notify;
        {
          std::scoped_lock Lock(MCA->M);
          --MCA->Outstanding;
          Notify = MCA->Shutdown && MCA->Outstanding == 0;
        }
        if (Notify)
          MCA->ShutdownCV.notify_all();
      }
    }
    MockControllerAccess *MCA = nullptr;
  };

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
    if (OnConnect)
      if (auto Err = OnConnect(BI))
        notifyDisconnected(std::move(Err));
  }

  void disconnect() override {
    // Drain any still-pending controller calls before notifying, so their
    // handlers are failed rather than silently dropped -- see the
    // ControllerAccess::disconnect contract. Shutdown gates callController
    // (which bails via failControllerCallInline once it is set), so no new call
    // can be registered after the snapshot below.
    std::unordered_map<size_t, OnControllerCallReturn> ToDrain;
    {
      std::unique_lock<std::mutex> Lock(M);
      Shutdown = true;
      ShutdownCV.wait(Lock, [this]() { return Shutdown && Outstanding == 0; });
      ToDrain = std::move(PendingOut);
    }
    for (auto &[_, OnComplete] : ToDrain)
      failPendingControllerCall(std::move(OnComplete));
    // A locally requested disconnect is orderly, so the mode is success.
    notifyDisconnected(Error::success());
  }

  void callController(OnControllerCallReturn OnComplete,
                      orc_rt_ControllerHandlerTag T,
                      WrapperFunctionBuffer ArgBytes) override {
    // Simulate a call to the controller by running the requested function via
    // the test-supplied Post hook (or inline, if no hook was provided).
    ConnectGuard CG;
    size_t CId = 0;
    bool BailOut = false;
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (!Shutdown) {
        CG = ConnectGuard(this);
        CId = CallId++;
        assert(!PendingOut.count(CId));
        PendingOut[CId] = std::move(OnComplete);
      } else
        BailOut = true;
    }

    if (BailOut)
      return failControllerCallInline(std::move(OnComplete));

    runOrPost([this, CId, T, ArgBytes = std::move(ArgBytes)]() mutable {
      auto Fn = reinterpret_cast<orc_rt_WrapperFunction>(T);
      Fn(reinterpret_cast<orc_rt_SessionRef>(this), ArgBytes.release(),
         ccReturn, CId);
    });
  }

  void sendWrapperResult(WrapperFunctionBuffer ResultBytes,
                         uint64_t CallId) override {
    // Respond to a simulated call by the controller.
    ConnectGuard CG;
    Session::OnControllerCallReturnFn OnComplete;
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (Shutdown) {
        assert(PendingIn.empty() && "Shut down but results still pending?");
        return;
      }
      CG = ConnectGuard(this);
      auto I = PendingIn.find(CallId);
      assert(I != PendingIn.end());
      OnComplete = std::move(I->second);
      PendingIn.erase(I);
    }

    runOrPost([OnComplete = std::move(OnComplete),
               ResultBytes = std::move(ResultBytes)]() mutable {
      OnComplete(std::move(ResultBytes));
    });
  }

  void returnFromController(uint64_t CallId,
                            WrapperFunctionBuffer ResultBytes) {
    ConnectGuard CG;
    OnControllerCallReturn OnComplete;
    {
      std::scoped_lock<std::mutex> Lock(M);
      CG = ConnectGuard(this);
      auto I = PendingOut.find(CallId);
      assert(I != PendingOut.end());
      OnComplete = std::move(I->second);
      PendingOut.erase(I);
    }

    handleControllerCallResult(std::move(OnComplete), std::move(ResultBytes));
  }

  void callFromController(Session::OnControllerCallReturnFn OnComplete,
                          orc_rt_WrapperFunction Fn,
                          WrapperFunctionBuffer ArgBytes) {
    ConnectGuard CG;
    size_t CId = 0;
    bool BailOut = false;
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (!Shutdown) {
        CG = ConnectGuard(this);
        CId = CallId++;
        PendingIn[CId] = std::move(OnComplete);
      } else
        BailOut = true;
    }
    if (BailOut)
      return runOrPost([OnComplete = std::move(OnComplete)]() mutable {
        OnComplete(WrapperFunctionBuffer::createOutOfBandError(
            "Controller disconnected"));
      });

    handleWrapperCall(Fn, std::move(ArgBytes), CId);
  }

private:
  void runOrPost(move_only_function<void()> Work) {
    if (Post)
      Post(std::move(Work));
    else
      Work();
  }

  static void ccReturn(orc_rt_SessionRef S,
                       orc_rt_WrapperFunctionBuffer ResultBytes,
                       uint64_t CallId) {
    // Abuse "session" to refer to the ControllerAccess object.
    // We can just re-use sendFunctionResult for this.
    reinterpret_cast<MockControllerAccess *>(S)->returnFromController(
        CallId, WrapperFunctionBuffer(ResultBytes));
  }

  PostFn Post;

  std::mutex M;
  bool Shutdown = false;
  size_t CallId = 0;
  size_t Outstanding = 0;
  std::unordered_map<size_t, OnControllerCallReturn> PendingOut;
  std::unordered_map<size_t, Session::OnControllerCallReturnFn> PendingIn;
  std::condition_variable ShutdownCV;
  OnConnectFn OnConnect;
};

class CallFromController {
public:
  CallFromController(MockControllerAccess &CA, orc_rt_WrapperFunction Fn)
      : CA(CA), Fn(Fn) {}
  void operator()(Session::OnControllerCallReturnFn OnComplete,
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

TEST(SessionTest, ReportErrorsViaSession) {
  Error E = Error::success();
  cantFail(std::move(E)); // Force error into checked state.

  // Check that the ReportErrorsViaSession utility works as advertised.
  Session S(mockExecutorProcessInfo(), noDispatch,
            [&](Error Err) { E = std::move(Err); });
  (ReportErrorsViaSession(S))(make_error<StringError>("foo"));

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

TEST(SessionTest, ExpectedShutdownSequenceWithNoOutstandingKeepalives) {
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

TEST(SessionTest, OutstandingKeepalivesDelayShutdown) {
  QueueingRunner<>::WorkQueue Tasks;
  Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);

  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;
  S.createService<MockService>(DetachOpIdx, ShutdownOpIdx, OpIdx);

  ASSERT_FALSE(DetachOpIdx);
  ASSERT_FALSE(ShutdownOpIdx);

  // Take a keepalive. This should succeed.
  auto Tok = TaskGroup::Token(S.keepaliveTokenSource());
  ASSERT_TRUE(Tok);

  // We expect shutdown to wait for any outstanding keepalives to be released.
  bool ShutdownComplete = false;
  S.shutdown([&]() { ShutdownComplete = true; });

  // Detach should have happened, but shutdown should be waiting on token.
  EXPECT_EQ(DetachOpIdx, 0U);
  EXPECT_FALSE(ShutdownOpIdx);
  EXPECT_FALSE(ShutdownComplete);

  // The keepalive group should have been closed. Assert that we
  // can't get a new token.
  ASSERT_FALSE(TaskGroup::Token(S.keepaliveTokenSource()));

  Tok = TaskGroup::Token(); // Reset token.

  EXPECT_EQ(ShutdownOpIdx, 1U);

  EXPECT_TRUE(ShutdownComplete);
}

static void managedVoidFunction(int *P) { *P = 42; }

TEST(SessionTest, CallWithKeepaliveVoidFn) {
  // Test calls to a void function while holding a KeepaliveTaskGroup token.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  {
    // Pre-shutdown we expect token acquisition to succeed and the function to
    // run.
    int X = 0;
    bool CallSucceeded = S.callWithKeepalive(managedVoidFunction, &X);

    EXPECT_TRUE(CallSucceeded);
    EXPECT_EQ(X, 42U);
  }

  waitForShutdown(S);

  {
    // Post-shutdown we expect token acquisition to fail, and
    // callWithKeepalive to return false.
    int X = 0;
    bool CallSucceeded = S.callWithKeepalive(managedVoidFunction, &X);

    EXPECT_FALSE(CallSucceeded);
  }
}

static int managedNonVoidFunction(int N) { return N + 1; }

TEST(SessionTest, CallWithKeepaliveNonVoidFn) {
  // Test calls to a non-void function while holding a KeepaliveTaskGroup
  // token.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  {
    // Pre-shutdown we expect token acquisition to succeed, the function to be
    // run, and the result to be returned.
    auto Result = S.callWithKeepalive(managedNonVoidFunction, 41);

    EXPECT_TRUE(Result);
    EXPECT_EQ(*Result, 42U);
  }

  waitForShutdown(S);

  {
    // Post-shutdown we expect token acquisition to fail, and
    // callWithKeepalive to return std::nullopt.
    auto Result = S.callWithKeepalive(managedNonVoidFunction, 41);

    EXPECT_EQ(Result, std::nullopt);
  }
}

static void managedAsyncFunction(move_only_function<void(int)> Return, int *P) {
  Return(++*P);
}

TEST(SessionTest, CallWithKeepaliveAsyncFn) {
  // Test that calls to managed async functions via callWithKeepalive work.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  {
    // Pre-shutdown we expect token acquisition to succeed, and the function
    // and Return callback to be run.
    int N = 41;
    std::optional<int> Result;
    S.callWithKeepalive(managedAsyncFunction, [&](int N) { Result = N; }, &N);
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
    S.callWithKeepalive(managedAsyncFunction, [&](int N) { Result = N; }, &N);
    EXPECT_EQ(Result, std::nullopt);
    EXPECT_EQ(N, 41U);
  }
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

// A ControllerAccess that is attached but whose controller connection is
// already gone, so every callController fails synchronously. Used to exercise
// the failControllerCallInline path in isolation.
class DeadControllerAccess : public Session::ControllerAccess {
public:
  DeadControllerAccess(Session &S) : ControllerAccess(S) {}
  void connect(BootstrapInfo) override {}
  void disconnect() override { notifyDisconnected(Error::success()); }
  void callController(OnControllerCallReturn OnComplete,
                      orc_rt_ControllerHandlerTag,
                      WrapperFunctionBuffer) override {
    failControllerCallInline(std::move(OnComplete));
  }
  void sendWrapperResult(WrapperFunctionBuffer, uint64_t) override {}
};

TEST(ControllerAccessTest, SynchronousCallControllerFailureRunsInline) {
  // A controller call that fails synchronously (connection already gone) must
  // complete its handler inline -- on the calling thread, before callController
  // returns, and WITHOUT going through Dispatch -- with the canonical
  // "disconnected" out-of-band error. noDispatch ADD_FAILUREs if anything is
  // dispatched, so it doubles as the "ran inline, not via the task queue"
  // check.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  S.attach<DeadControllerAccess>(BootstrapInfo(S));

  bool HandlerRan = false;
  std::string ErrMsg;
  S.callController(
      [&](WrapperFunctionBuffer Result) {
        HandlerRan = true;
        if (const char *E = Result.getOutOfBandError())
          ErrMsg = E;
      },
      /*T=*/nullptr, WrapperFunctionBuffer());

  // Set synchronously, during the call above -> the handler ran inline.
  EXPECT_TRUE(HandlerRan);
  EXPECT_EQ(ErrMsg, "disconnected");
}

static void add_sps_wrapper(orc_rt_SessionRef S,
                            orc_rt_WrapperFunctionBuffer ArgBytes,
                            orc_rt_WrapperFunctionReturn Return,
                            uint64_t CallId) {
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::handle(
      S, ArgBytes, Return, CallId,
      [](move_only_function<void(int32_t)> Return, int32_t X, int32_t Y) {
        Return(X + Y);
      });
}

// Wrapper handler that echoes its argument bytes straight back as the result.
// Used to exercise the C callController entry point without pulling in SPS
// (de)serialization -- the payload is opaque to the code under test.
static void echo_wrapper(orc_rt_SessionRef S,
                         orc_rt_WrapperFunctionBuffer ArgBytes,
                         orc_rt_WrapperFunctionReturn Return, uint64_t CallId) {
  Return(S, ArgBytes, CallId);
}

// Captures what orc_rt_Session_callController threads back to its C return
// function, so the test can inspect it after the call completes.
struct CAPICallControllerResult {
  bool Ran = false;
  orc_rt_SessionRef S = nullptr;
  std::string Result;
};

static void capiCallControllerReturn(orc_rt_SessionRef S,
                                     orc_rt_WrapperFunctionBuffer ResultBytes,
                                     void *Ctx) {
  auto &R = *static_cast<CAPICallControllerResult *>(Ctx);
  R.Ran = true;
  R.S = S;
  WrapperFunctionBuffer Result(ResultBytes);
  R.Result.assign(Result.data(), Result.size());
}

TEST(ControllerAccessTest, CAPICallControllerRoundTrip) {
  // Drive a controller call through the C API entry point
  // (orc_rt_Session_callController) and confirm that the result bytes, the
  // session handle, and the caller-supplied context are all threaded back to
  // the C return function.
  QueueingRunner<>::WorkQueue Tasks;
  Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);
  S.attach<MockControllerAccess>(BootstrapInfo(S), postOnto(Tasks));

  CAPICallControllerResult R;
  orc_rt_Session_callController(
      wrap(&S), reinterpret_cast<orc_rt_ControllerHandlerTag>(&echo_wrapper),
      WrapperFunctionBuffer::copyFrom("hello", 5).release(),
      &capiCallControllerReturn, &R);

  QueueingRunner<>::runFIFOUntilEmpty(Tasks);

  EXPECT_TRUE(R.Ran) << "C return function never fired";
  EXPECT_EQ(R.S, wrap(&S)) << "Session handle not threaded through";
  EXPECT_EQ(R.Result, "hello") << "Result bytes not round-tripped";
}

TEST(ControllerAccessTest, ValidCallToController) {
  // Simulate a call to a controller handler.
  QueueingRunner<>::WorkQueue Tasks;
  Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);
  S.attach<MockControllerAccess>(BootstrapInfo(S), postOnto(Tasks));

  int32_t Result = 0;
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      S.controllerCaller(
          reinterpret_cast<orc_rt_ControllerHandlerTag>(add_sps_wrapper)),
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
      S.controllerCaller(
          reinterpret_cast<orc_rt_ControllerHandlerTag>(add_sps_wrapper)),
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
      S.controllerCaller(
          reinterpret_cast<orc_rt_ControllerHandlerTag>(add_sps_wrapper)),
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
      CallFromController(*CA, add_sps_wrapper),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); }, 41, 1);

  QueueingRunner<>::runFIFOUntilEmpty(Tasks);

  EXPECT_EQ(Result, 42);
}

// Stashes Return, via the address passed as the wrapper call's argument,
// instead of calling it -- simulating a wrapper function that defers
// completion past its own return.
static void deferred_wrapper(orc_rt_SessionRef S,
                             orc_rt_WrapperFunctionBuffer ArgBytes,
                             orc_rt_WrapperFunctionReturn Return,
                             uint64_t CallId) {
  SPSWrapperFunction<void(SPSExecutorAddr)>::handle(
      S, ArgBytes, Return, CallId,
      [](move_only_function<void()> Return, ExecutorAddr P) {
        *P.toPtr<move_only_function<void()> *>() = std::move(Return);
      });
}

TEST(ControllerAccessTest, WrapperCallTokenReleasedWhenFnReturns) {
  // A keepalive acquired for an incoming wrapper call must bracket
  // only the (synchronous) span of Fn's execution -- not the whole
  // call/response chain -- per the "Keepalives and shutdown"
  // policy in docs/Design.md. Check this by having Fn defer its Return call
  // past its own return, then confirming that Session shutdown's drain phase
  // does not wait on that deferred call.
  QueueingRunner<>::WorkQueue Tasks;
  Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);
  MockControllerAccess *CA = nullptr;
  S.attach<MockControllerAccess>(BootstrapInfo(S), postOnto(Tasks),
                                 MockControllerAccess::OnConnectFn{}, &CA);

  move_only_function<void()> DeferredReturn;

  bool GotResult = false;
  SPSWrapperFunction<void(SPSExecutorAddr)>::call(
      CallFromController(*CA, deferred_wrapper),
      [&](Error Err) {
        cantFail(std::move(Err));
        GotResult = true;
      },
      &DeferredReturn);

  QueueingRunner<>::runFIFOUntilEmpty(Tasks);

  // Fn ran and deferred its Return call, so no result should have been sent
  // back yet.
  ASSERT_TRUE(DeferredReturn);
  EXPECT_FALSE(GotResult);

  // Fn has already returned, so its keepalive should have been
  // released even though the call is still logically outstanding. Shutdown's
  // drain phase should therefore complete without waiting on the deferred
  // Return call.
  bool ShutdownComplete = false;
  S.shutdown([&] { ShutdownComplete = true; });
  EXPECT_TRUE(ShutdownComplete);
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
      S.controllerCaller(
          reinterpret_cast<orc_rt_ControllerHandlerTag>(add_sps_wrapper)),
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
      S.controllerCaller(
          reinterpret_cast<orc_rt_ControllerHandlerTag>(add_sps_wrapper)),
      [&](Expected<int32_t> R) {
        ErrorAsOutParameter _(CallErr);
        CallErr = R.takeError();
      },
      41, 1);

  EXPECT_EQ(toString(std::move(CallErr)), "no controller attached");
}

// A ControllerAccess that reports a caller-supplied disconnection mode, used to
// exercise the Error that notifyDisconnected passes to the Session (see
// Session::setOnDisconnect). An empty ErrMsg reports an orderly disconnection,
// any other value an abnormal one.
class DisconnectingControllerAccess : public Session::ControllerAccess {
public:
  DisconnectingControllerAccess(Session &S, std::string ErrMsg = "",
                                DisconnectingControllerAccess **Self = nullptr)
      : ControllerAccess(S), ErrMsg(std::move(ErrMsg)) {
    // Publish this instance so tests that need to drive a controller-initiated
    // disconnection can reach it after attach constructs it.
    if (Self)
      *Self = this;
  }

  /// Simulate the controller going away without the Session asking it to.
  void disconnectFromController() { doDisconnect(); }

  void connect(BootstrapInfo) override {}
  void disconnect() override { doDisconnect(); }
  void callController(OnControllerCallReturn OnComplete,
                      orc_rt_ControllerHandlerTag,
                      WrapperFunctionBuffer) override {
    failControllerCallInline(std::move(OnComplete));
  }
  void sendWrapperResult(WrapperFunctionBuffer, uint64_t) override {}

private:
  void doDisconnect() {
    // notifyDisconnected is exactly-once, so a redundant disconnect -- e.g. a
    // Session-initiated disconnect racing a controller-initiated one -- must be
    // a no-op. These tests are single-threaded, so a plain bool suffices.
    if (Disconnected)
      return;
    Disconnected = true;
    notifyDisconnected(ErrMsg.empty() ? Error::success()
                                      : make_error<StringError>(ErrMsg));
  }

  std::string ErrMsg;
  bool Disconnected = false;
};

/// Consume Err, returning its message, or "" if Err is a success value.
static std::string errMsgOrEmpty(Error Err) {
  if (Err)
    return toString(std::move(Err));
  return "";
}

TEST(ControllerAccessTest, OnDisconnectReportsOrderlyDisconnect) {
  // An orderly disconnection reports a success Error, exactly once.
  size_t HandlerRuns = 0;
  std::string ErrMsg = "<not run>";
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  S.setOnDisconnect([&](Error Err) {
    ++HandlerRuns;
    ErrMsg = errMsgOrEmpty(std::move(Err));
  });
  S.attach<DisconnectingControllerAccess>(BootstrapInfo(S));

  S.detach();

  EXPECT_EQ(HandlerRuns, 1U);
  EXPECT_EQ(ErrMsg, "");
}

TEST(ControllerAccessTest, OnDisconnectReportsAbnormalDisconnect) {
  // An abnormal disconnection reports the ControllerAccess's Error unchanged.
  std::optional<std::string> ErrMsg;
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  S.setOnDisconnect([&](Error Err) { ErrMsg = errMsgOrEmpty(std::move(Err)); });
  S.attach<DisconnectingControllerAccess>(BootstrapInfo(S),
                                          "connection closed without hangup");

  S.detach();

  EXPECT_THAT(ErrMsg,
              Optional(Eq(std::string("connection closed without hangup"))));
}

TEST(ControllerAccessTest, DisconnectErrorRoutedToErrorReporterWithNoHandler) {
  // With no on-disconnect handler installed, an abnormal disconnection is
  // reported via the Session's error reporter -- once, not once per route.
  std::vector<std::string> ErrMsgs;
  Session S(mockExecutorProcessInfo(), noDispatch, AccumulateErrors(ErrMsgs));
  S.attach<DisconnectingControllerAccess>(BootstrapInfo(S),
                                          "connection closed without hangup");

  S.detach();

  ASSERT_EQ(ErrMsgs.size(), 1U);
  EXPECT_EQ(ErrMsgs[0], "connection closed without hangup");
}

TEST(ControllerAccessTest, OrderlyDisconnectNotRoutedToErrorReporter) {
  // An orderly disconnection is not an error, so nothing is reported.
  std::vector<std::string> ErrMsgs;
  Session S(mockExecutorProcessInfo(), noDispatch, AccumulateErrors(ErrMsgs));
  S.attach<DisconnectingControllerAccess>(BootstrapInfo(S));

  S.detach();

  EXPECT_TRUE(ErrMsgs.empty());
}

TEST(ControllerAccessTest, OnDisconnectSuppressesErrorReporterRouting) {
  // Installing a handler takes over routing entirely: the error reporter must
  // not also see the disconnection error.
  std::vector<std::string> ErrMsgs;
  std::optional<std::string> HandlerErrMsg;
  Session S(mockExecutorProcessInfo(), noDispatch, AccumulateErrors(ErrMsgs));
  S.setOnDisconnect(
      [&](Error Err) { HandlerErrMsg = errMsgOrEmpty(std::move(Err)); });
  S.attach<DisconnectingControllerAccess>(BootstrapInfo(S),
                                          "connection closed without hangup");

  S.detach();

  EXPECT_THAT(HandlerErrMsg,
              Optional(Eq(std::string("connection closed without hangup"))));
  EXPECT_TRUE(ErrMsgs.empty());
}

TEST(ControllerAccessTest, OnDisconnectReportsConnectFailure) {
  // A ControllerAccess that fails to connect disconnects during attach, so the
  // handler sees the connect error.
  //
  // This is the boundary between "attempted an attach and failed" and "never
  // attempted one": both reach proceedToDetach with the Session still in the
  // Start state, differing only in the Error the caller supplies, so nothing
  // but this test distinguishes them -- OnDisconnectReportsSuccessWithoutAttach
  // would still pass if a failed connect were reported as success.
  std::optional<std::string> ErrMsg;
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  S.setOnDisconnect([&](Error Err) { ErrMsg = errMsgOrEmpty(std::move(Err)); });

  S.attach<MockControllerAccess>(
      BootstrapInfo(S), MockControllerAccess::PostFn{},
      [](BootstrapInfo &) { return make_error<StringError>("no route"); });

  EXPECT_THAT(ErrMsg, Optional(Eq(std::string("no route"))));
}

TEST(ControllerAccessTest, OnDisconnectReportsSuccessWithoutAttach) {
  // A Session that never attached a controller still reports a disconnection,
  // with success: there was no connection to lose, so it trivially succeeded.
  // This is what makes the handler total -- a client using it to record the
  // Session's result never has to account for it not running.
  //
  // Note that success here is specific to never having attempted an attach:
  // an attach whose connect fails reports the connect error instead, which
  // OnDisconnectReportsConnectFailure covers.
  size_t HandlerRuns = 0;
  std::string ErrMsg = "<not run>";
  {
    Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
    S.setOnDisconnect([&](Error Err) {
      ++HandlerRuns;
      ErrMsg = errMsgOrEmpty(std::move(Err));
    });
  }
  EXPECT_EQ(HandlerRuns, 1U);
  EXPECT_EQ(ErrMsg, "");
}

TEST(ControllerAccessTest, OnDisconnectRunsBeforeDetachAndShutdown) {
  // The on-disconnect handler reports the loss of the controller connection
  // specifically, so it runs before Services are notified via onDetach, and
  // well before shutdown.
  size_t OpIdx = 0;
  std::optional<size_t> DisconnectOpIdx;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;

  {
    Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
    S.setOnDisconnect([&](Error Err) {
      DisconnectOpIdx = OpIdx++;
      cantFail(std::move(Err));
    });
    S.addService(
        std::make_unique<MockService>(DetachOpIdx, ShutdownOpIdx, OpIdx));
    S.attach<DisconnectingControllerAccess>(BootstrapInfo(S));
  }

  EXPECT_EQ(OpIdx, 3U);
  EXPECT_EQ(DisconnectOpIdx, 0U);
  EXPECT_EQ(DetachOpIdx, 1U);
  EXPECT_EQ(ShutdownOpIdx, 2U);
}

TEST(ControllerAccessTest, ShutdownFromOnDisconnectHandler) {
  // Calling shutdown from the on-disconnect handler is supported, and is the
  // expected way to turn the loss of the controller into "finish outstanding
  // work and exit" (see Session::setOnDisconnect). Check that it neither
  // deadlocks nor detaches twice, and that the scheduled shutdown runs.
  //
  // Here the disconnect was requested locally, via detach.
  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;
  bool OnShutdownRan = false;

  {
    Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
    S.addService(
        std::make_unique<MockService>(DetachOpIdx, ShutdownOpIdx, OpIdx));
    S.setOnDisconnect([&](Error Err) {
      cantFail(std::move(Err));
      S.shutdown([&]() { OnShutdownRan = true; });
    });
    S.attach<DisconnectingControllerAccess>(BootstrapInfo(S));

    S.detach();

    EXPECT_TRUE(OnShutdownRan);
  }

  // Exactly one onDetach and one onShutdown: no double detach.
  EXPECT_EQ(OpIdx, 2U);
  EXPECT_EQ(DetachOpIdx, 0U);
  EXPECT_EQ(ShutdownOpIdx, 1U);
}

TEST(ControllerAccessTest, ShutdownFromOnDisconnectHandlerAfterRemoteHangup) {
  // As above, but for a controller-initiated disconnection, where the Session
  // did not ask for the detach. The re-entrant shutdown still has to be
  // absorbed, and Services still detached and shut down exactly once.
  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;
  bool OnShutdownRan = false;

  {
    DisconnectingControllerAccess *CA = nullptr;
    Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
    S.addService(
        std::make_unique<MockService>(DetachOpIdx, ShutdownOpIdx, OpIdx));
    S.setOnDisconnect([&](Error Err) {
      cantFail(std::move(Err));
      S.shutdown([&]() { OnShutdownRan = true; });
    });
    S.attach<DisconnectingControllerAccess>(BootstrapInfo(S), "", &CA);
    ASSERT_TRUE(CA);

    CA->disconnectFromController();

    EXPECT_TRUE(OnShutdownRan);
  }

  EXPECT_EQ(OpIdx, 2U);
  EXPECT_EQ(DetachOpIdx, 0U);
  EXPECT_EQ(ShutdownOpIdx, 1U);
}
