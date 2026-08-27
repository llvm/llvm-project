//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Use the umbrella header for -Wdocumentation.
#include "lldb/API/LLDB.h"

#include "TestingSupport/SubsystemRAII.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBProtocolServer.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

class SBProtocolServerTest : public testing::Test {
protected:
  void SetUp() override {
    debugger = SBDebugger::Create(/*source_init_files=*/false);
  }

  void TearDown() override { SBDebugger::Destroy(debugger); }

  SubsystemRAII<lldb::SBDebugger> subsystems;
  SBDebugger debugger;
};

TEST_F(SBProtocolServerTest, DefaultConstructedIsInvalid) {
  SBProtocolServer server;
  EXPECT_FALSE(server.IsValid());
  EXPECT_FALSE(static_cast<bool>(server));
}

TEST_F(SBProtocolServerTest, CreateMCP) {
  SBError error;
  SBProtocolServer server = SBProtocolServer::Create("MCP", error);
  ASSERT_TRUE(error.Success()) << error.GetCString();
  EXPECT_TRUE(server.IsValid());
  EXPECT_TRUE(static_cast<bool>(server));
  // No socket is bound until Start, so there is no connection URI yet.
  EXPECT_EQ(server.GetConnectionURI(), nullptr);
}

TEST_F(SBProtocolServerTest, CreateUnsupportedProtocol) {
  SBError error;
  SBProtocolServer server = SBProtocolServer::Create("bogus", error);
  EXPECT_TRUE(error.Fail());
  EXPECT_FALSE(server.IsValid());
}

TEST_F(SBProtocolServerTest, CreateNullProtocol) {
  SBError error;
  SBProtocolServer server = SBProtocolServer::Create(nullptr, error);
  EXPECT_TRUE(error.Fail());
  EXPECT_FALSE(server.IsValid());
}

TEST_F(SBProtocolServerTest, CreateEmptyProtocol) {
  SBError error;
  SBProtocolServer server = SBProtocolServer::Create("", error);
  EXPECT_TRUE(error.Fail());
  EXPECT_FALSE(server.IsValid());
}

TEST_F(SBProtocolServerTest, StartAndStop) {
  SBError error;
  SBProtocolServer server = SBProtocolServer::Create("MCP", error);
  ASSERT_TRUE(server.IsValid());

  // Bind an ephemeral port on the loopback interface so the test never touches
  // an external network.
  SBError start_error = server.Start("listen://[localhost]:0");
  ASSERT_TRUE(start_error.Success()) << start_error.GetCString();

  // Once listening, the server reports the concrete URI a client can connect
  // to (the ephemeral port resolved above).
  EXPECT_NE(server.GetConnectionURI(), nullptr);

  EXPECT_TRUE(server.Stop().Success());
}

TEST_F(SBProtocolServerTest, StartInvalidServer) {
  // A default-constructed server has no backing protocol server, so Start fails
  // before the connection URI is ever examined.
  SBProtocolServer server;
  EXPECT_TRUE(server.Start("").Fail());
}

TEST_F(SBProtocolServerTest, StopNotRunning) {
  SBError error;
  SBProtocolServer server = SBProtocolServer::Create("MCP", error);
  ASSERT_TRUE(server.IsValid());
  // Stopping a server that was never started is an error.
  EXPECT_TRUE(server.Stop().Fail());
}

TEST_F(SBProtocolServerTest, StartRejectsNonAcceptingURI) {
  SBError error;
  SBProtocolServer server = SBProtocolServer::Create("MCP", error);
  ASSERT_TRUE(server.IsValid());
  // A connecting URI (not an accepting one) is rejected before any socket work.
  EXPECT_TRUE(server.Start("connect://localhost:1234").Fail());
  EXPECT_TRUE(server.Start("not a uri").Fail());
}

TEST_F(SBProtocolServerTest, CopyReferencesSameServer) {
  SBError error;
  SBProtocolServer server = SBProtocolServer::Create("MCP", error);
  ASSERT_TRUE(server.IsValid());

  // The protocol server is owned by the engine, not by SBProtocolServer, so a
  // copy just references the same server and the source stays valid.
  SBProtocolServer copy(server);
  EXPECT_TRUE(copy.IsValid());
  EXPECT_TRUE(server.IsValid());

  // Copy assignment behaves the same way.
  SBProtocolServer assigned;
  EXPECT_FALSE(assigned.IsValid());
  assigned = server;
  EXPECT_TRUE(assigned.IsValid());
  EXPECT_TRUE(server.IsValid());
}
