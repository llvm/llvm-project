//===-- TestTypeSystem.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

class TestTypeSystemMap : public testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo, PlatformMacOSX> subsystems;

protected:
  void SetUp() override {
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
  };

  DebuggerSP m_debugger_sp;
  PlatformSP m_platform_sp;
};

TEST_F(TestTypeSystemMap, GetScratchTypeSystemForLanguage) {
  // Set up the debugger, make sure that was done properly.
  TargetSP target_sp;
  ArchSpec arch("x86_64-apple-macosx-");
  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  m_debugger_sp = Debugger::CreateInstance();

  auto &target = m_debugger_sp->GetDummyTarget();
  EXPECT_THAT_EXPECTED(
      target.GetScratchTypeSystemForLanguage(lldb::eLanguageTypeMipsAssembler),
      llvm::FailedWithMessage("No expression support for any languages"));
  EXPECT_THAT_EXPECTED(
      target.GetScratchTypeSystemForLanguage(lldb::eLanguageTypeAssembly),
      llvm::FailedWithMessage("No expression support for any languages"));
  EXPECT_THAT_EXPECTED(
      target.GetScratchTypeSystemForLanguage(lldb::eLanguageTypeUnknown),
      llvm::FailedWithMessage("No expression support for any languages"));
  EXPECT_THAT_EXPECTED(
      target.GetScratchTypeSystemForLanguage(lldb::eLanguageTypeModula2),
      llvm::FailedWithMessage("TypeSystem for language modula2 doesn't exist"));
}
