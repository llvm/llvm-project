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

class DisconnectRequestHandlerTest : public DAPTestBase {};

TEST_F(DisconnectRequestHandlerTest, DisconnectTriggersTerminated) {
  DisconnectRequestHandler handler(*dap);
  EXPECT_FALSE(dap->disconnecting);
  ASSERT_THAT_ERROR(handler.Run(std::nullopt), Succeeded());
  EXPECT_TRUE(dap->disconnecting);
  std::vector<Message> messages = DrainOutput();
  EXPECT_THAT(messages,
              testing::Contains(testing::VariantWith<Event>(testing::FieldsAre(
                  /*event=*/"terminated", /*body=*/testing::_))));
}

TEST_F(DisconnectRequestHandlerTest, DisconnectTriggersTerminateCommands) {
  CreateDebugger();

  if (!GetDebuggerSupportsTarget("X86"))
    GTEST_SKIP() << "Unsupported platform";

  LoadCore();

  DisconnectRequestHandler handler(*dap);

  EXPECT_FALSE(dap->disconnecting);
  dap->configuration.terminateCommands = {"?script print(1)",
                                          "script print(2)"};
  EXPECT_EQ(dap->target.GetProcess().GetState(), lldb::eStateStopped);
  ASSERT_THAT_ERROR(handler.Run(std::nullopt), Succeeded());
  EXPECT_TRUE(dap->disconnecting);
  std::vector<Message> messages = DrainOutput();
  EXPECT_THAT(messages, testing::ElementsAre(
                            OutputMatcher("Running terminateCommands:\n"),
                            OutputMatcher("(lldb) script print(2)\n"),
                            OutputMatcher("2\n"),
                            testing::VariantWith<Event>(testing::FieldsAre(
                                /*event=*/"terminated", /*body=*/testing::_))));
}
