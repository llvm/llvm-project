//===-- LLGSTest.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestBase.h"
#include "lldb/Host/Host.h"
#include "llvm/Testing/Support/Error.h"

using namespace llgs_tests;
using namespace lldb_private;
using namespace llvm;

#if !defined(__APPLE__)
// FIXME: This test is hanging on Darwin. <rdar://problem/38313510>
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
#endif

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

TEST_F(TestBase, LLGS_TEST(vAttachRichError)) {
  auto ClientOr = TestClient::launch(getLogFileName(),
                                     {getInferiorPath("environment_check")});
  ASSERT_THAT_EXPECTED(ClientOr, Succeeded());
  auto &Client = **ClientOr;

  // Until we enable error strings we should just get the error code.
  ASSERT_THAT_ERROR(Client.SendMessage("vAttach;1"),
                    Failed<ErrorInfoBase>(testing::Property(
                        &ErrorInfoBase::message, "Error 255")));

  ASSERT_THAT_ERROR(Client.SendMessage("QEnableErrorStrings"), Succeeded());

  // Now, we expect the full error message.
  ASSERT_THAT_ERROR(
      Client.SendMessage("vAttach;1"),
      Failed<ErrorInfoBase>(testing::Property(
          &ErrorInfoBase::message,
          testing::StartsWith(
              "cannot attach to process 1 when another process with pid"))));
}
