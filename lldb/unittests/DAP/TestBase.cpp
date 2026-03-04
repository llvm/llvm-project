//===-- TestBase.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestBase.h"
#include "DAP.h"
#include "DAPLog.h"
#include "Handler/RequestHandler.h"
#include "Handler/ResponseHandler.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/Pipe.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <memory>

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
  std::tie(to_client, to_server) = TestDAPTransport::createPair(loop);

  log = std::make_unique<Log>(llvm::outs(), log_mutex);
  dap = std::make_unique<DAP>(
      /*log=*/*log,
      /*default_repl_mode=*/ReplMode::Auto,
      /*pre_init_commands=*/std::vector<String>(),
      /*no_lldbinit=*/false,
      /*client_name=*/"test_client",
      /*transport=*/*to_client, /*loop=*/loop);

  EXPECT_THAT_ERROR(to_server->RegisterMessageHandler(*dap), Succeeded());
  EXPECT_THAT_ERROR(to_client->RegisterMessageHandler(client), Succeeded());
}

void TransportBase::Run() {
  bool addition_succeeded = loop.AddPendingCallback(
      [](lldb_private::MainLoopBase &loop) { loop.RequestTermination(); });
  EXPECT_TRUE(addition_succeeded);
  EXPECT_THAT_ERROR(loop.Run().takeError(), llvm::Succeeded());
}

void DAPTestBase::SetUp() { TransportBase::SetUp(); }

void DAPTestBase::TearDown() {
  if (core)
    ASSERT_THAT_ERROR(core->discard(), Succeeded());
  if (binary)
    ASSERT_THAT_ERROR(binary->discard(), Succeeded());
}

void DAPTestBase::SetUpTestSuite() {
  lldb::SBError error = SBDebugger::InitializeWithErrorHandling();
  EXPECT_TRUE(error.Success());
}
void DAPTestBase::TearDownTestSuite() { SBDebugger::Terminate(); }

void DAPTestBase::CreateDebugger() {
  dap->debugger = lldb::SBDebugger::Create();
  ASSERT_TRUE(dap->debugger);
  dap->target = dap->debugger.GetDummyTarget();

  Expected<lldb::FileUP> dev_null = FileSystem::Instance().Open(
      FileSpec(FileSystem::DEV_NULL), File::eOpenOptionReadWrite);
  ASSERT_THAT_EXPECTED(dev_null, Succeeded());
  lldb::FileSP dev_null_sp = std::move(*dev_null);

  std::FILE *dev_null_stream = dev_null_sp->GetStream();
  ASSERT_THAT_ERROR(dap->ConfigureIO(dev_null_stream, dev_null_stream),
                    Succeeded());

  dap->debugger.SetInputFile(dap->in);
  auto out_fd = dap->out.GetWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(out_fd, Succeeded());
  dap->debugger.SetOutputFile(lldb::SBFile(*out_fd, "w", false));
  auto err_fd = dap->out.GetWriteFileDescriptor();
  ASSERT_THAT_EXPECTED(err_fd, Succeeded());
  dap->debugger.SetErrorFile(lldb::SBFile(*err_fd, "w", false));
}

void DAPTestBase::LoadCore() {
  lldb::SBProcess process;
  std::tie(dap->target, process) =
      lldb_private::LoadCore(dap->debugger, k_linux_binary, k_linux_core);
}
