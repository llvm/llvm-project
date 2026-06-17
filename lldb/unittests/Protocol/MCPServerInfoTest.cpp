//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport/TestUtilities.h"
#include "lldb/Protocol/MCP/Server.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace lldb_private;
using namespace lldb_protocol::mcp;

// NOTE: ServerInfo::Write/Load and the non-empty ServerInfoHandle paths read
// and write the real `~/.lldb` directory (via HostInfo::GetUserLLDBDir) and so
// can't be exercised hermetically yet. They should be covered once the
// filesystem can be mocked.

TEST(MCPServerInfoTest, JSONRoundtrip) {
  ServerInfo info;
  info.connection_uri = "unix:///tmp/test.sock";

  Expected<ServerInfo> deserialized = roundtripJSON(info);
  ASSERT_THAT_EXPECTED(deserialized, Succeeded());
  EXPECT_EQ(deserialized->connection_uri, info.connection_uri);
}

TEST(MCPServerInfoTest, EmptyHandleRemoveIsNoOp) {
  // A default-constructed handle tracks no file, so Remove is a no-op and does
  // not touch the filesystem.
  ServerInfoHandle handle;
  handle.Remove();
  // Calling Remove again must be safe (idempotent).
  handle.Remove();
}
