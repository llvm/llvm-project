//===- InProcessControllerAccessTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's InProcessControllerAccess.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/InProcessControllerAccess.h"
#include "orc-rt/QueueingRunner.h"

#include "gtest/gtest.h"

#include "CommonTestUtils.h"

#include <deque>
#include <optional>
#include <string>

using namespace orc_rt;

namespace {

// A minimal stand-in for llvm::orc::InProcessEPC. Registers itself on the
// Connection during OnConnect, exposes hooks for tests to drive cross-calls
// in either direction, and tears the connection down on destruction.
class MockIPEPC {
public:
  using Connection = InProcessControllerAccess::Connection;

  using OnCallJITDispatchFn = move_only_function<void(
      uint64_t CallId, void *HandlerTag, WrapperFunctionBuffer ArgBytes)>;
  using OnReturnWrapperResultFn = move_only_function<void(
      uint64_t CallId, WrapperFunctionBuffer ResultBytes)>;

  MockIPEPC(Connection *C) : C(C) {
    C->Retain(C);
    C->IPEPC = this;
    C->CallJITDispatch = &callJITDispatchEntry;
    C->ReturnWrapperResult = &returnWrapperResultEntry;
  }

  MockIPEPC(const MockIPEPC &) = delete;
  MockIPEPC &operator=(const MockIPEPC &) = delete;

  ~MockIPEPC() {
    C->Disconnect(C);
    C->Release(C);
  }

  void setOnCallJITDispatch(OnCallJITDispatchFn H) {
    OnCallJITDispatch = std::move(H);
  }
  void setOnReturnWrapperResult(OnReturnWrapperResultFn H) {
    OnReturnWrapperResult = std::move(H);
  }

  // Send a result back to a CallJITDispatch invocation.
  void respondToCall(uint64_t CallId, WrapperFunctionBuffer Result) {
    if (C->EnterMessageScope(C)) {
      C->ReturnJITDispatchResult(C->IPCA, CallId, Result.release());
      C->LeaveMessageScope(C);
    }
  }

  // Initiate a wrapper call into the executor, simulating a controller-side
  // wrapper invocation. Returns the CallId that ReturnWrapperResult will
  // refer back to.
  uint64_t callIntoExecutor(orc_rt_WrapperFunction Fn,
                            WrapperFunctionBuffer ArgBytes) {
    uint64_t CallId = NextCallId++;
    if (C->EnterMessageScope(C)) {
      C->CallWrapper(C->IPCA, CallId, reinterpret_cast<void *>(Fn),
                     ArgBytes.release());
      C->LeaveMessageScope(C);
    }
    return CallId;
  }

private:
  static void callJITDispatchEntry(void *IPEPC, uint64_t CallId,
                                   void *HandlerTag,
                                   orc_rt_WrapperFunctionBuffer ArgBytes) {
    auto *Self = static_cast<MockIPEPC *>(IPEPC);
    WrapperFunctionBuffer Buf(ArgBytes);
    if (Self->OnCallJITDispatch)
      Self->OnCallJITDispatch(CallId, HandlerTag, std::move(Buf));
  }

  static void
  returnWrapperResultEntry(void *IPEPC, uint64_t CallId,
                           orc_rt_WrapperFunctionBuffer ResultBytes) {
    auto *Self = static_cast<MockIPEPC *>(IPEPC);
    WrapperFunctionBuffer Buf(ResultBytes);
    if (Self->OnReturnWrapperResult)
      Self->OnReturnWrapperResult(CallId, std::move(Buf));
  }

  Connection *C;
  uint64_t NextCallId = 0;
  OnCallJITDispatchFn OnCallJITDispatch;
  OnReturnWrapperResultFn OnReturnWrapperResult;
};

// Convenience: attach an InProcessControllerAccess to S, constructing a
// MockIPEPC into MockOut from inside OnConnect.
void attachWithMock(Session &S, std::unique_ptr<MockIPEPC> &MockOut) {
  S.attach<InProcessControllerAccess>(
      BootstrapInfo(S),
      [&MockOut](InProcessControllerAccess &, BootstrapInfo &,
                 InProcessControllerAccess::Connection *C,
                 InProcessControllerAccess::BootstrapInfoAccess *) -> Error {
        MockOut = std::make_unique<MockIPEPC>(C);
        return Error::success();
      });
}

} // namespace

TEST(InProcessControllerAccessTest, ConstructAndDestroyWithoutConnect) {
  // An InProcessControllerAccess that is never attached to a Session (so its
  // connect method is never called) must still destroy cleanly. The
  // destructor's `if (C)` guard is what makes this safe.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  InProcessControllerAccess IPCA(
      S,
      [](InProcessControllerAccess &, BootstrapInfo &,
         InProcessControllerAccess::Connection *,
         InProcessControllerAccess::BootstrapInfoAccess *) -> Error {
        ADD_FAILURE() << "OnConnect should not be called";
        return Error::success();
      });

  // IPCA is destroyed at scope exit; test passes if there's no crash.
}

TEST(InProcessControllerAccessTest, AttachAndShutdownViaSession) {
  // Smoke test: attach with a successful OnConnect, verify the mock was
  // constructed, then let scope exit drive shutdown.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  std::unique_ptr<MockIPEPC> Mock;
  attachWithMock(S, Mock);
  EXPECT_TRUE(Mock) << "Expected OnConnect to construct MockIPEPC";
}

TEST(InProcessControllerAccessTest, OnConnectFailureIsReportedAndDetaches) {
  // If OnConnect returns an Error, IPCA::connect should:
  //   (1) Forward the error through Session::reportError.
  //   (2) Trigger disconnect (so the Session ends up detached and no further
  //       calls into the controller succeed).
  Error Reported = Error::success();
  cantFail(std::move(Reported)); // force checked state

  Session S(mockExecutorProcessInfo(), noDispatch,
            [&](Error E) { Reported = std::move(E); });

  S.attach<InProcessControllerAccess>(
      BootstrapInfo(S),
      [](InProcessControllerAccess &, BootstrapInfo &,
         InProcessControllerAccess::Connection *,
         InProcessControllerAccess::BootstrapInfoAccess *) -> Error {
        return make_error<StringError>("fake connect failure");
      });

  if (Reported)
    EXPECT_EQ(toString(std::move(Reported)), "fake connect failure");
  else
    ADD_FAILURE() << "Expected OnConnect error to be reported";

  // A subsequent call to the controller should now fail with "no controller
  // attached" (i.e. the Session detached on the OnConnect error).
  std::optional<std::string> CallErr;
  S.callController(
      [&](WrapperFunctionBuffer R) {
        if (const char *Msg = R.getOutOfBandError())
          CallErr = Msg;
      },
      reinterpret_cast<Session::HandlerTag>(0xdeadbeef),
      WrapperFunctionBuffer::copyFrom("x", 1));

  ASSERT_TRUE(CallErr);
  EXPECT_EQ(*CallErr, "no controller attached");
}

TEST(InProcessControllerAccessTest, CallControllerSuccess) {
  // A callController call routed through MockIPEPC, which echoes the args
  // back as the result. Verify OnComplete fires with the payload.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  std::unique_ptr<MockIPEPC> Mock;
  attachWithMock(S, Mock);
  ASSERT_TRUE(Mock);

  Mock->setOnCallJITDispatch(
      [&](uint64_t CallId, void *, WrapperFunctionBuffer ArgBytes) {
        Mock->respondToCall(CallId, std::move(ArgBytes));
      });

  std::optional<std::string> Result;
  S.callController(
      [&](WrapperFunctionBuffer R) {
        ASSERT_FALSE(R.getOutOfBandError())
            << "Unexpected out-of-band error: " << R.getOutOfBandError();
        Result = std::string(R.data(), R.size());
      },
      reinterpret_cast<Session::HandlerTag>(0xdeadbeef),
      WrapperFunctionBuffer::copyFrom("hello", 5));

  ASSERT_TRUE(Result);
  EXPECT_EQ(*Result, "hello");
}

TEST(InProcessControllerAccessTest, CallControllerOutOfBandError) {
  // A callController call where the mock responds with an out-of-band error.
  // OnComplete should observe the error message intact.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  std::unique_ptr<MockIPEPC> Mock;
  attachWithMock(S, Mock);
  ASSERT_TRUE(Mock);

  Mock->setOnCallJITDispatch(
      [&](uint64_t CallId, void *, WrapperFunctionBuffer) {
        Mock->respondToCall(CallId, WrapperFunctionBuffer::createOutOfBandError(
                                        "simulated failure"));
      });

  std::optional<std::string> ErrMsg;
  S.callController(
      [&](WrapperFunctionBuffer R) {
        if (const char *Msg = R.getOutOfBandError())
          ErrMsg = Msg;
      },
      reinterpret_cast<Session::HandlerTag>(0xdeadbeef),
      WrapperFunctionBuffer::copyFrom("payload", 7));

  ASSERT_TRUE(ErrMsg);
  EXPECT_EQ(*ErrMsg, "simulated failure");
}

TEST(InProcessControllerAccessTest, DisconnectDrainsPendingCalls) {
  // A callController call is in-flight when the connection drops (the mock
  // never responds). Verify that doDisconnect drains the pending handler with
  // a "disconnected" out-of-band error rather than leaving it stranded.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  std::unique_ptr<MockIPEPC> Mock;
  attachWithMock(S, Mock);
  ASSERT_TRUE(Mock);

  // Mock receives the call but never sends a result.
  Mock->setOnCallJITDispatch([](uint64_t, void *, WrapperFunctionBuffer) {});

  std::optional<std::string> ErrMsg;
  S.callController(
      [&](WrapperFunctionBuffer R) {
        if (const char *Msg = R.getOutOfBandError())
          ErrMsg = Msg;
      },
      reinterpret_cast<Session::HandlerTag>(0xdeadbeef),
      WrapperFunctionBuffer::copyFrom("payload", 7));

  ASSERT_FALSE(ErrMsg) << "OnComplete fired prematurely";

  // Tearing down the mock triggers C->Disconnect, which routes through
  // ConnectionImpl::disconnect → IPCA::doDisconnect and drains the pending
  // OnComplete with a "disconnected" error.
  Mock.reset();

  ASSERT_TRUE(ErrMsg);
  EXPECT_EQ(*ErrMsg, "disconnected");
}

// Wrapper function that echoes ArgBytes back as the result. Used to exercise
// the controller-initiated wrapper-call path without pulling in SPS.
static void echoWrapper(orc_rt_SessionRef S, uint64_t CallId,
                        orc_rt_WrapperFunctionReturn Return,
                        orc_rt_WrapperFunctionBuffer ArgBytes) {
  Return(S, CallId, ArgBytes);
}

TEST(InProcessControllerAccessTest, CallFromControllerSuccess) {
  // The mock IPEPC initiates a wrapper call into IPCA. The Session's
  // dispatch hook (a QueueingRunner over `Tasks`) enqueues the
  // invocation; draining the queue runs the wrapper, which echoes its
  // arguments back. Verify the mock receives the echoed bytes via
  // ReturnWrapperResult.
  QueueingRunner<>::WorkQueue Tasks;
  Session S(mockExecutorProcessInfo(), QueueingRunner(Tasks), noErrors);

  std::unique_ptr<MockIPEPC> Mock;
  attachWithMock(S, Mock);
  ASSERT_TRUE(Mock);

  std::optional<std::string> Result;
  Mock->setOnReturnWrapperResult(
      [&](uint64_t, WrapperFunctionBuffer ResultBytes) {
        Result = std::string(ResultBytes.data(), ResultBytes.size());
      });

  Mock->callIntoExecutor(echoWrapper,
                         WrapperFunctionBuffer::copyFrom("world", 5));

  // Nothing has run yet -- the wrapper is sitting in `Tasks` waiting to be
  // dispatched.
  ASSERT_FALSE(Result);

  QueueingRunner<>::runFIFOUntilEmpty(Tasks);

  ASSERT_TRUE(Result);
  EXPECT_EQ(*Result, "world");
}
