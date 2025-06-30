//===-- TestBase.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestBase.h"
#include "Protocol/ProtocolBase.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/Host/File.h"
#include "lldb/Host/Pipe.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap::protocol;
using namespace lldb_dap_tests;
using lldb_private::File;
using lldb_private::NativeFile;
using lldb_private::Pipe;

void TransportBase::SetUp() {
  PipeTest::SetUp();
  to_dap = std::make_unique<Transport>(
      "to_dap", nullptr,
      std::make_shared<NativeFile>(input.GetReadFileDescriptor(),
                                   File::eOpenOptionReadOnly,
                                   NativeFile::Unowned),
      std::make_shared<NativeFile>(output.GetWriteFileDescriptor(),
                                   File::eOpenOptionWriteOnly,
                                   NativeFile::Unowned));
  from_dap = std::make_unique<Transport>(
      "from_dap", nullptr,
      std::make_shared<NativeFile>(output.GetReadFileDescriptor(),
                                   File::eOpenOptionReadOnly,
                                   NativeFile::Unowned),
      std::make_shared<NativeFile>(input.GetWriteFileDescriptor(),
                                   File::eOpenOptionWriteOnly,
                                   NativeFile::Unowned));
}

void DAPTestBase::SetUp() {
  TransportBase::SetUp();
  dap = std::make_unique<DAP>(
      /*log=*/nullptr,
      /*default_repl_mode=*/ReplMode::Auto,
      /*pre_init_commands=*/std::vector<std::string>(),
      /*transport=*/*to_dap);
}

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

bool DAPTestBase::GetDebuggerSupportsTarget(llvm::StringRef platform) {
  EXPECT_TRUE(dap->debugger);

  lldb::SBStructuredData data = dap->debugger.GetBuildConfiguration()
                                    .GetValueForKey("targets")
                                    .GetValueForKey("value");
  for (size_t i = 0; i < data.GetSize(); i++) {
    char buf[100] = {0};
    size_t size = data.GetItemAtIndex(i).GetStringValue(buf, sizeof(buf));
    if (llvm::StringRef(buf, size) == platform)
      return true;
  }

  return false;
}

void DAPTestBase::CreateDebugger() {
  dap->debugger = lldb::SBDebugger::Create();
  ASSERT_TRUE(dap->debugger);
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

std::vector<Message> DAPTestBase::DrainOutput() {
  std::vector<Message> msgs;
  output.CloseWriteFileDescriptor();
  while (true) {
    Expected<Message> next =
        from_dap->Read<protocol::Message>(std::chrono::milliseconds(1));
    if (!next) {
      consumeError(next.takeError());
      break;
    }
    msgs.push_back(*next);
  }
  return msgs;
}
