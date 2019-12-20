//===-- LuaTests.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/ScriptInterpreter/Lua/ScriptInterpreterLua.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Platform.h"
#include "lldb/Utility/Reproducer.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::repro;
using namespace lldb;

namespace {
class ScriptInterpreterTest : public ::testing::Test {
public:
  void SetUp() override {
    llvm::cantFail(Reproducer::Initialize(ReproducerMode::Off, llvm::None));
    FileSystem::Initialize();
    HostInfo::Initialize();

    // Pretend Linux is the host platform.
    platform_linux::PlatformLinux::Initialize();
    ArchSpec arch("powerpc64-pc-linux");
    Platform::SetHostPlatform(
        platform_linux::PlatformLinux::CreateInstance(true, &arch));
  }
  void TearDown() override {
    platform_linux::PlatformLinux::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
    Reproducer::Terminate();
  }
};
} // namespace

TEST_F(ScriptInterpreterTest, Plugin) {
  EXPECT_EQ(ScriptInterpreterLua::GetPluginNameStatic(), "script-lua");
  EXPECT_EQ(ScriptInterpreterLua::GetPluginDescriptionStatic(),
            "Lua script interpreter");
}

TEST_F(ScriptInterpreterTest, ExecuteOneLine) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  ScriptInterpreterLua script_interpreter(*debugger_sp);
  CommandReturnObject result;
  EXPECT_TRUE(script_interpreter.ExecuteOneLine("foo = 1", &result));
  EXPECT_FALSE(script_interpreter.ExecuteOneLine("nil = foo", &result));
  EXPECT_TRUE(result.GetErrorData().startswith(
      "error: lua failed attempting to evaluate 'nil = foo'"));
}
