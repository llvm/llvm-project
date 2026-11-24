//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ScriptInterpreter/None/ScriptInterpreterNone.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "lldb/Breakpoint/BreakpointOptions.h"
#include "lldb/Breakpoint/WatchpointOptions.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Platform.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

namespace {
class ScriptInterpreterNoneTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();

    platform_linux::PlatformLinux::Initialize();
    ArchSpec arch("x86_64-pc-linux");
    Platform::SetHostPlatform(
        platform_linux::PlatformLinux::CreateInstance(true, &arch));

    m_debugger_sp = Debugger::CreateInstance();
    ASSERT_TRUE(m_debugger_sp);
  }

  void TearDown() override {
    PluginManager::SetScriptInterpreterError(eScriptLanguageLua,
                                             llvm::Error::success());
    Debugger::Destroy(m_debugger_sp);
    platform_linux::PlatformLinux::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }

  DebuggerSP m_debugger_sp;
};
} // namespace

TEST_F(ScriptInterpreterNoneTest, ReportsRecordedError) {
  PluginManager::SetScriptInterpreterError(
      eScriptLanguageLua, llvm::createStringError("runtime not found"));
  ScriptInterpreterNone interpreter(*m_debugger_sp, eScriptLanguageLua);

  CommandReturnObject result(/*colors=*/false);
  EXPECT_FALSE(interpreter.ExecuteOneLine("print(1)", &result));
  EXPECT_EQ(result.GetErrorString(),
            "error: Lua scripting is unavailable: runtime not found\n");

  Status error;
  EXPECT_FALSE(interpreter.LoadScriptingModule("foo", {}, error));
  EXPECT_EQ(error.AsCString(),
            std::string("Lua scripting is unavailable: runtime not found"));

  EXPECT_THAT_ERROR(interpreter.ExecuteInterpreterLoop(),
                    llvm::FailedWithMessage(
                        "Lua scripting is unavailable: runtime not found"));
}

TEST_F(ScriptInterpreterNoneTest, ReportsRecordedErrorForCallbacks) {
  PluginManager::SetScriptInterpreterError(
      eScriptLanguageLua, llvm::createStringError("runtime not found"));
  ScriptInterpreterNone interpreter(*m_debugger_sp, eScriptLanguageLua);
  const std::string expected =
      "Lua scripting is unavailable: runtime not found";

  BreakpointOptions bp_options(/*all_flags_set=*/false);
  EXPECT_THAT_ERROR(interpreter.SetBreakpointCommandCallback(
                        bp_options, "print(1)", /*is_callback=*/false),
                    llvm::FailedWithMessage(expected));
  EXPECT_THAT_ERROR(
      interpreter.SetBreakpointCommandCallbackFunction(bp_options, "foo", {}),
      llvm::FailedWithMessage(expected));

  WatchpointOptions wp_options;
  EXPECT_THAT_ERROR(interpreter.SetWatchpointCommandCallback(
                        &wp_options, "print(1)", /*is_callback=*/false),
                    llvm::FailedWithMessage(expected));

  std::vector<std::reference_wrapper<BreakpointOptions>> bp_options_vec = {
      bp_options};
  CommandReturnObject bp_result(/*colors=*/false);
  interpreter.CollectDataForBreakpointCommandCallback(bp_options_vec,
                                                      bp_result);
  EXPECT_FALSE(bp_result.Succeeded());
  EXPECT_EQ(bp_result.GetErrorString(), "error: " + expected + "\n");

  CommandReturnObject wp_result(/*colors=*/false);
  interpreter.CollectDataForWatchpointCommandCallback(&wp_options, wp_result);
  EXPECT_FALSE(wp_result.Succeeded());
  EXPECT_EQ(wp_result.GetErrorString(), "error: " + expected + "\n");
}

TEST_F(ScriptInterpreterNoneTest, ClearRecordedError) {
  PluginManager::SetScriptInterpreterError(
      eScriptLanguageLua, llvm::createStringError("runtime not found"));
  PluginManager::SetScriptInterpreterError(eScriptLanguageLua,
                                           llvm::Error::success());
  EXPECT_THAT_ERROR(
      PluginManager::GetScriptInterpreterError(eScriptLanguageLua),
      llvm::Succeeded());
}

TEST_F(ScriptInterpreterNoneTest, NoRecordedError) {
  ScriptInterpreterNone interpreter(*m_debugger_sp, eScriptLanguageUnknown);

  CommandReturnObject result(/*colors=*/false);
  EXPECT_FALSE(interpreter.ExecuteOneLine("print(1)", &result));
  EXPECT_EQ(result.GetErrorString(),
            "error: Embedded script interpreter unavailable. LLDB was built "
            "without scripting language support.\n");
}
