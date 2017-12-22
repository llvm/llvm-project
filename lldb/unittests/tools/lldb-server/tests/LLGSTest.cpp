//===-- LLGSTest.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestBase.h"
#include "lldb/Host/Host.h"
#include "llvm/Testing/Support/Error.h"

using namespace llgs_tests;
using namespace lldb_private;
using namespace llvm;

TEST_F(TestBase, LaunchModePreservesEnvironment) {
  putenv(const_cast<char *>("LLDB_TEST_MAGIC_VARIABLE=LLDB_TEST_MAGIC_VALUE"));

  auto ClientOr = TestClient::launch(getLogFileName(),
                                     {getInferiorPath("environment_check")});
  ASSERT_THAT_EXPECTED(ClientOr, Succeeded());
  auto &Client = **ClientOr;

  ASSERT_THAT_ERROR(Client.ContinueAll(), Succeeded());
  ASSERT_THAT_EXPECTED(
      Client.GetLatestStopReplyAs<StopReplyExit>(),
      HasValue(testing::Property(&StopReply::getKind,
                                 WaitStatus{WaitStatus::Exit, 0})));
}

TEST_F(TestBase, DS_TEST(DebugserverEnv)) {
  // Test that --env takes precedence over inherited environment variables.
  putenv(const_cast<char *>("LLDB_TEST_MAGIC_VARIABLE=foobar"));

  auto ClientOr = TestClient::launchCustom(getLogFileName(),
      { "--env", "LLDB_TEST_MAGIC_VARIABLE=LLDB_TEST_MAGIC_VALUE" },
                                     {getInferiorPath("environment_check")});
  ASSERT_THAT_EXPECTED(ClientOr, Succeeded());
  auto &Client = **ClientOr;

  ASSERT_THAT_ERROR(Client.ContinueAll(), Succeeded());
  ASSERT_THAT_EXPECTED(
      Client.GetLatestStopReplyAs<StopReplyExit>(),
      HasValue(testing::Property(&StopReply::getKind,
                                 WaitStatus{WaitStatus::Exit, 0})));
}
