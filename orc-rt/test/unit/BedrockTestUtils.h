//===- BedrockTestUtils.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test helpers that need Bedrock. These live apart from CommonTestUtils.h so
// that SupportTests translation units pull in no Bedrock headers at all.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_UNITTEST_BEDROCKTESTUTILS_H
#define ORC_RT_UNITTEST_BEDROCKTESTUTILS_H

#include "orc-rt/bedrock/ExecutorProcessInfo.h"
#include "orc-rt/bedrock/Session.h"

#include "DirectCaller.h"
#include "gtest/gtest.h"

#include <mutex>
#include <unordered_map>

namespace orc_rt::test {

inline ExecutorProcessInfo mockExecutorProcessInfo() noexcept {
  return ExecutorProcessInfo("arm64-apple-darwin", 16384, "+neon, +fullfp16");
}

/// DispatchFn for tests that should never dispatch a task. Records a test
/// failure on invocation, then runs the task inline so that any caller
/// awaiting a result unblocks (rather than hanging) and the keepalive token
/// is released, even in -Asserts builds or when the dispatch arrives on a
/// non-test thread.
inline void noDispatch(Session::Task T) {
  ADD_FAILURE() << "unexpected dispatch in a no-dispatch session";
  T();
}

/// DispatchFn that runs tasks on the current thread.
inline void inlineDispatch(Session::Task T) { T(); }

/// ControllerAccess for testing executor-to-controller calls. Routes each call
/// to the orc_rt_WrapperFunction registered for its tag and runs it directly
/// on the calling thread, so tests can implement controller handlers with
/// SPSWrapperFunction<...>::handle. A call to a tag with no registered handler
/// records a test failure and fails with an out-of-band error.
///
/// Handlers must return their result before callController returns: the
/// result is delivered via handleControllerCallResult, which requires the
/// Session to still be connected. For the same reason, callController must
/// not race with Session shutdown.
class DirectControllerAccess : public Session::ControllerAccess {
public:
  using HandlerMap =
      std::unordered_map<orc_rt_ControllerHandlerTag *, orc_rt_WrapperFunction>;

  DirectControllerAccess(Session &S, HandlerMap Handlers) noexcept
      : ControllerAccess(S), S(S), Handlers(std::move(Handlers)) {}

  void connect(BootstrapInfo BI) override {}

  void disconnect() override {
    {
      std::scoped_lock<std::mutex> Lock(M);
      Disconnected = true;
    }
    // Calls complete synchronously, so there are no pending calls to drain.
    notifyDisconnected(Error::success());
  }

  void callController(OnControllerCallReturn OnComplete,
                      orc_rt_ControllerHandlerTag *T,
                      WrapperFunctionBuffer ArgBytes) override {
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (Disconnected)
        return failControllerCallInline(std::move(OnComplete));
    }

    auto HandlerI = Handlers.find(T);

    // If no such handler exists then fail the call as the controller would:
    // with an out-of-band error result.
    if (HandlerI == Handlers.end()) {
      ADD_FAILURE() << "No handler registered for controller-call tag";
      handleControllerCallResult(
          std::move(OnComplete),
          WrapperFunctionBuffer::createOutOfBandError(
              "DirectControllerAccess: no handler registered for "
              "controller-call tag"));
      return;
    }

    DirectCaller(wrap(&S), HandlerI->second)(
        [this, OnComplete = std::move(OnComplete)](
            WrapperFunctionBuffer ResultBytes) mutable {
          handleControllerCallResult(std::move(OnComplete),
                                     std::move(ResultBytes));
        },
        std::move(ArgBytes));
  }

  void sendWrapperResult(WrapperFunctionBuffer ResultBytes,
                         uint64_t CallId) override {
    ADD_FAILURE() << "DirectControllerAccess does not support calls from the "
                     "controller";
  }

private:
  Session &S;
  HandlerMap Handlers;

  std::mutex M;
  bool Disconnected = false;
};

} // namespace orc_rt::test

#endif // ORC_RT_UNITTEST_BEDROCKTESTUTILS_H
