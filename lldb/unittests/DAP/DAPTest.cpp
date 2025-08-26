//===-- DAPTest.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Protocol/ProtocolBase.h"
#include "TestBase.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap_tests;
using namespace lldb_dap::protocol;
using namespace testing;

class DAPTest : public TransportBase {};

TEST_F(DAPTest, SendProtocolMessages) {
  DAP dap{
      /*log=*/nullptr,
      /*default_repl_mode=*/ReplMode::Auto,
      /*pre_init_commands=*/{},
      /*client_name=*/"test_client",
      /*transport=*/*transport,
      /*loop=*/loop,
  };
  dap.Send(Event{/*event=*/"my-event", /*body=*/std::nullopt});
  loop.AddPendingCallback(
      [](lldb_private::MainLoopBase &loop) { loop.RequestTermination(); });
  EXPECT_CALL(client, Received(IsEvent("my-event", std::nullopt)));
  ASSERT_THAT_ERROR(dap.Loop(), llvm::Succeeded());
}
