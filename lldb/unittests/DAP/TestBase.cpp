//===-- TestBase.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestBase.h"
#include "DAPLog.h"
#include "lldb/API/SBDefines.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/Pipe.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <memory>
#include <system_error>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap::protocol;
using namespace lldb_dap_tests;
using lldb_private::File;
using lldb_private::FileSpec;
using lldb_private::FileSystem;
using lldb_private::MainLoop;
using lldb_private::Pipe;

void TransportBase::SetUp() {
  std::tie(to_client, to_server) = TestDAPTransport::createPair();

  std::error_code EC;
  log = std::make_unique<Log>(llvm::outs(), log_mutex);
  dap = std::make_unique<DAP>(
      /*default_repl_mode=*/ReplMode::Auto,
      /*pre_init_commands=*/std::vector<std::string>(),
      /*client_name=*/"test_client",
      /*log=*/*log.get(),
      /*transport=*/*to_client, /*loop=*/loop);

  auto server_handle = to_server->RegisterMessageHandler(loop, *dap);
  EXPECT_THAT_EXPECTED(server_handle, Succeeded());
  handles[0] = std::move(*server_handle);

  auto client_handle = to_client->RegisterMessageHandler(loop, client);
  EXPECT_THAT_EXPECTED(client_handle, Succeeded());
  handles[1] = std::move(*client_handle);
}

void TransportBase::Run() {
  bool addition_succeeded = loop.AddPendingCallback(
      [](lldb_private::MainLoopBase &loop) { loop.RequestTermination(); });
  EXPECT_TRUE(addition_succeeded);
  EXPECT_THAT_ERROR(loop.Run().takeError(), llvm::Succeeded());
}

void DAPTestBase::CreateDebugger() {
  ASSERT_THAT_ERROR(dap->InitializeDebugger(), Succeeded());
  ASSERT_TRUE(dap->debugger);
  fixtures.debugger = dap->debugger;

  if (!fixtures.IsPlatformSupported("X86"))
    GTEST_SKIP() << "Unsupported platform";

  dap->target = dap->debugger.GetDummyTarget();
}

void DAPTestBase::LoadCore() {
  fixtures.LoadTarget();
  fixtures.LoadProcess();
  dap->target = fixtures.target;
}