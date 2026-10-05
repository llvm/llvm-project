//===-- DisconnectTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Handler/RequestHandler.h"
#include "Protocol/ProtocolBase.h"
#include "TestBase.h"
#include "lldb/API/SBDefines.h"
#include "lldb/Host/JSONTransport.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <optional>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap_tests;
using namespace lldb_dap::protocol;
using testing::_;

class DisconnectRequestHandlerTest : public DAPTestBase {
protected:
  void QueueDisconnect() {
    dap->Received(Request{"disconnect", std::nullopt, /*seq=*/1});
  }

  void QueueThreads(Id seq) {
    dap->Received(Request{"threads", std::nullopt, seq});
  }

  /// Handles the queued messages until the session ends, then delivers the
  /// messages sent after the transport thread stopped.
  void RunSession() {
    ASSERT_THAT_ERROR(dap->Loop(), Succeeded());
    Run();
  }
};

TEST_F(DisconnectRequestHandlerTest, DisconnectTriggersTerminated) {
  QueueDisconnect();

  testing::InSequence sequence;
  EXPECT_CALL(client, Received(IsEvent("terminated", _)));
  EXPECT_CALL(client,
              Received(SuccessResponse(HasSeq(1), HasCommand("disconnect"))));
  RunSession();
}

TEST_F(DisconnectRequestHandlerTest, DisconnectCancelsQueuedRequests) {
  QueueDisconnect();
  QueueThreads(2);

  testing::InSequence sequence;
  EXPECT_CALL(client, Received(IsEvent("terminated", _)));
  EXPECT_CALL(client,
              Received(CancelledResponse(HasSeq(2), HasCommand("threads"))));
  EXPECT_CALL(client,
              Received(SuccessResponse(HasSeq(1), HasCommand("disconnect"))));
  RunSession();
}

TEST_F(DisconnectRequestHandlerTest, ClosedConnectionCancelsQueuedRequests) {
  QueueThreads(1);
  dap->OnClosed();

  // The session ends now, without a response.
  testing::InSequence sequence;
  EXPECT_CALL(client, Received(IsEvent("terminated", _)));
  EXPECT_CALL(client,
              Received(CancelledResponse(HasSeq(1), HasCommand("threads"))));
  RunSession();
}

TEST_F(DisconnectRequestHandlerTest, ClosedConnectionCancelsQueuedDisconnect) {
  QueueDisconnect();
  QueueThreads(2);
  dap->OnClosed();

  testing::InSequence sequence;
  EXPECT_CALL(client, Received(IsEvent("terminated", _)));
  EXPECT_CALL(client,
              Received(CancelledResponse(HasSeq(1), HasCommand("disconnect"))));
  EXPECT_CALL(client,
              Received(CancelledResponse(HasSeq(2), HasCommand("threads"))));
  RunSession();
}

TEST_F(DisconnectRequestHandlerTest, TransportErrorEndsSession) {
  QueueThreads(1);
  dap->OnError(llvm::createStringError("read failed"));

  testing::InSequence sequence;
  EXPECT_CALL(client, Received(IsEvent("terminated", _)));
  EXPECT_CALL(client,
              Received(CancelledResponse(HasSeq(1), HasCommand("threads"))));
  EXPECT_THAT_ERROR(dap->Loop(), Failed());
  Run();
}

// Is flaky on Linux, see https://github.com/llvm/llvm-project/issues/154763.
#if LLDB_ENABLE_PYTHON && !defined(__linux__)
TEST_F(DisconnectRequestHandlerTest, DisconnectTriggersTerminateCommands) {
  CreateDebugger();

  if (!GetDebuggerSupportsTarget("X86"))
    GTEST_SKIP() << "Unsupported platform";

  LoadCore();

  dap->configuration.terminateCommands = {"?script print(1)",
                                          "script print(2)"};
  EXPECT_EQ(dap->target.GetProcess().GetState(), lldb::eStateStopped);
  QueueDisconnect();

  EXPECT_CALL(client, Received(Output("1\n"))).Times(testing::AtMost(1));
  // Python print can also writes to the debugger's output file.
  EXPECT_CALL(client, Received(Output("2\n"))).Times(testing::AtLeast(1));

  EXPECT_CALL(client, Received(Output("(lldb) script print(2)\n")));
  EXPECT_CALL(client, Received(Output("Running terminateCommands:\n")));
  EXPECT_CALL(client, Received(IsEvent("terminated", _)));
  EXPECT_CALL(client,
              Received(SuccessResponse(HasSeq(1), HasCommand("disconnect"))));
  RunSession();
}
#endif
