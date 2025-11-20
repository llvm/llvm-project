//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFixtures.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBStream.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_dap_tests;

TestFixtures::~TestFixtures() {
  if (m_binary)
    EXPECT_THAT_ERROR(m_binary->discard(), Succeeded());
  if (m_core)
    EXPECT_THAT_ERROR(m_core->discard(), Succeeded());
}

bool TestFixtures::IsPlatformSupported(StringRef platform) {
  if (!debugger)
    LoadDebugger();

  SBStructuredData data =
      debugger.GetBuildConfiguration().GetValueForKey("targets").GetValueForKey(
          "value");
  for (size_t i = 0; i < data.GetSize(); i++) {
    char buf[100] = {0};
    size_t size = data.GetItemAtIndex(i).GetStringValue(buf, sizeof(buf));
    if (StringRef(buf, size) == platform)
      return true;
  }

  return false;
}

void TestFixtures::LoadDebugger() {
  ASSERT_FALSE(debugger) << "Debugger already loaded";
  debugger = SBDebugger::Create(false);
  ASSERT_TRUE(debugger);
}

void TestFixtures::LoadTarget(llvm::StringRef path) {
  ASSERT_FALSE(target) << "Target already loaded";
  ASSERT_TRUE(debugger) << "Debugger not loaded";

  Expected<lldb_private::TestFile> binary_yaml =
      lldb_private::TestFile::fromYamlFile(path);
  ASSERT_THAT_EXPECTED(binary_yaml, Succeeded());
  Expected<sys::fs::TempFile> binary_file = binary_yaml->writeToTemporaryFile();
  ASSERT_THAT_EXPECTED(binary_file, Succeeded());
  m_binary = std::move(*binary_file);
  target = debugger.CreateTarget(m_binary->TmpName.data());
  ASSERT_TRUE(target);
}

void TestFixtures::LoadProcess(llvm::StringRef path) {
  llvm::errs() << "LoadProcess(" << path << ")\n";
  ASSERT_FALSE(process) << "Process already loaded";
  ASSERT_TRUE(target) << "Target not loaded";
  ASSERT_TRUE(debugger) << "Debugger not loaded";
  ASSERT_EQ(debugger.GetID(), target.GetDebugger().GetID())
      << "Debugger mismatch";

  Expected<lldb_private::TestFile> core_yaml =
      lldb_private::TestFile::fromYamlFile(path);
  ASSERT_THAT_EXPECTED(core_yaml, Succeeded());
  Expected<sys::fs::TempFile> core_file = core_yaml->writeToTemporaryFile();
  ASSERT_THAT_EXPECTED(core_file, Succeeded());
  m_core = std::move(*core_file);
  lldb::SBError error;
  process = target.LoadCore(m_core->TmpName.data(), error);
  lldb::SBStream str;
  target.GetDescription(str, lldb::eDescriptionLevelFull);
  ASSERT_TRUE(process) << "Failed to load " << m_core->TmpName.data() << " "
                       << error.GetCString() << " and " << str.GetData();
}
