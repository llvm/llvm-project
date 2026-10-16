//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/MainLoop.h"
#include "lldb/Protocol/MCP/Transport.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

using namespace llvm;
using namespace lldb_private;
using namespace lldb_protocol::mcp;

TEST(MCPTransportTest, LogWithCallback) {
  MainLoop loop;
  std::vector<std::string> messages;
  Transport transport(
      loop, /*in=*/nullptr, /*out=*/nullptr,
      [&](StringRef message) { messages.push_back(message.str()); });

  transport.Log("hello");
  transport.Log("world");

  EXPECT_THAT(messages, testing::ElementsAre("hello", "world"));
}

TEST(MCPTransportTest, LogWithoutCallback) {
  MainLoop loop;
  Transport transport(loop, /*in=*/nullptr, /*out=*/nullptr);

  // Without a log callback, logging is a no-op and must not crash.
  transport.Log("ignored");
}
