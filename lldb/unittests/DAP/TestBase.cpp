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

Expected<MainLoop::ReadHandleUP>
TestTransport::RegisterMessageHandler(MainLoop &loop, MessageHandler &handler) {
  Expected<lldb::FileUP> dummy_file = FileSystem::Instance().Open(
      FileSpec(FileSystem::DEV_NULL), File::eOpenOptionReadWrite);
  if (!dummy_file)
    return dummy_file.takeError();
  m_dummy_file = std::move(*dummy_file);
  lldb_private::Status status;
  auto handle = loop.RegisterReadObject(
      m_dummy_file, [](lldb_private::MainLoopBase &) {}, status);
  if (status.Fail())
    return status.takeError();
  return handle;
}

void DAPTestBase::SetUp() {
  TransportBase::SetUp();
  std::error_code EC;
  log = std::make_unique<Log>("-", EC);
  dap = std::make_unique<DAP>(
      /*log=*/log.get(),
      /*default_repl_mode=*/ReplMode::Auto,
      /*pre_init_commands=*/std::vector<std::string>(),
      /*client_name=*/"test_client",
      /*transport=*/*transport, /*loop=*/loop);
}

void DAPTestBase::TearDown() {
  if (core) {
    ASSERT_THAT_ERROR(core->discard(), Succeeded());
  }
  if (binary) {
    ASSERT_THAT_ERROR(binary->discard(), Succeeded());
  }
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
