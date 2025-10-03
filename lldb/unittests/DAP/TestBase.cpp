//===-- TestBase.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestBase.h"
#include "DAPLog.h"
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
  log = std::make_unique<Log>("-", EC);
  dap = std::make_unique<DAP>(
      /*log=*/log.get(),
      /*default_repl_mode=*/ReplMode::Auto,
      /*pre_init_commands=*/std::vector<std::string>(),
      /*no_lldbinit=*/false,
      /*client_name=*/"test_client",
      /*transport=*/*to_client, /*loop=*/loop);

  auto server_handle = to_server->RegisterMessageHandler(loop, *dap.get());
  EXPECT_THAT_EXPECTED(server_handle, Succeeded());
  handles[0] = std::move(*server_handle);

  auto client_handle = to_client->RegisterMessageHandler(loop, client);
  EXPECT_THAT_EXPECTED(client_handle, Succeeded());
  handles[1] = std::move(*client_handle);
}

void TransportBase::Run() {
  loop.AddPendingCallback(
      [](lldb_private::MainLoopBase &loop) { loop.RequestTermination(); });
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
void DAPTestBase::TeatUpTestSuite() { SBDebugger::Terminate(); }

bool DAPTestBase::GetDebuggerSupportsTarget(StringRef platform) {
  EXPECT_TRUE(dap->debugger);

  lldb::SBStructuredData data = dap->debugger.GetBuildConfiguration()
                                    .GetValueForKey("targets")
                                    .GetValueForKey("value");
  for (size_t i = 0; i < data.GetSize(); i++) {
    char buf[100] = {0};
    size_t size = data.GetItemAtIndex(i).GetStringValue(buf, sizeof(buf));
    if (StringRef(buf, size) == platform)
      return true;
  }

  return false;
}

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
  ASSERT_TRUE(dap->debugger);
  llvm::Expected<lldb_private::TestFile> binary_yaml =
      lldb_private::TestFile::fromYamlFile(k_linux_binary);
  ASSERT_THAT_EXPECTED(binary_yaml, Succeeded());
  llvm::Expected<llvm::sys::fs::TempFile> binary_file =
      binary_yaml->writeToTemporaryFile();
  ASSERT_THAT_EXPECTED(binary_file, Succeeded());
  binary = std::move(*binary_file);
  dap->target = dap->debugger.CreateTarget(binary->TmpName.data());
  ASSERT_TRUE(dap->target);
  llvm::Expected<lldb_private::TestFile> core_yaml =
      lldb_private::TestFile::fromYamlFile(k_linux_core);
  ASSERT_THAT_EXPECTED(core_yaml, Succeeded());
  llvm::Expected<llvm::sys::fs::TempFile> core_file =
      core_yaml->writeToTemporaryFile();
  ASSERT_THAT_EXPECTED(core_file, Succeeded());
  this->core = std::move(*core_file);
  SBProcess process = dap->target.LoadCore(this->core->TmpName.data());
  ASSERT_TRUE(process);
}
