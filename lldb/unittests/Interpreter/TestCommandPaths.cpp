//===-- ProcessEventDataTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/Status.h"

#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::repro;
using namespace lldb;

namespace {
class VerifyUserMultiwordCmdPathTest : public ::testing::Test {
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
    PlatformMacOSX::Initialize();
  }
  void TearDown() override {
    PlatformMacOSX::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};
} // namespace

class CommandObjectLeaf : public CommandObjectParsed {
public:
  CommandObjectLeaf(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "dummy subcommand leaf",
                            "Does nothing", "dummy subcommand leaf") {
    SetIsUserCommand(true);
  }

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    result.SetStatus(eReturnStatusSuccessFinishResult);
    result.AppendMessage("I did nothing");
    return true;
  }
};

class CommandObjectMultiwordSubDummy : public CommandObjectMultiword {
public:
  CommandObjectMultiwordSubDummy(CommandInterpreter &interpreter)
      : CommandObjectMultiword(interpreter, "dummy subcommand", "Does nothing",
                               "dummy subcommand") {
    SetIsUserCommand(true);
    LoadSubCommand("leaf", CommandObjectSP(new CommandObjectLeaf(interpreter)));
  }

  ~CommandObjectMultiwordSubDummy() override = default;
};

class CommandObjectMultiwordDummy : public CommandObjectMultiword {
public:
  CommandObjectMultiwordDummy(CommandInterpreter &interpreter)
      : CommandObjectMultiword(interpreter, "dummy", "Does nothing", "dummy") {
    SetIsUserCommand(true);
    LoadSubCommand(
        "subcommand",
        CommandObjectSP(new CommandObjectMultiwordSubDummy(interpreter)));
  }

  ~CommandObjectMultiwordDummy() override = default;
};

// Pass in the command path to args.  If success is true, we make sure the MWC
// returned matches the test string.  If success is false, we make sure the
// lookup error matches test_str.
void RunTest(CommandInterpreter &interp, const char *args, bool is_leaf,
             bool success, const char *test_str) {
  CommandObjectMultiword *multi_word_cmd = nullptr;
  Args test_args(args);
  Status error;
  multi_word_cmd =
      interp.VerifyUserMultiwordCmdPath(test_args, is_leaf, error);
  if (success) {
    ASSERT_NE(multi_word_cmd, nullptr);
    ASSERT_TRUE(error.Success());
    ASSERT_STREQ(multi_word_cmd->GetCommandName().str().c_str(), test_str);
  } else {
    ASSERT_EQ(multi_word_cmd, nullptr);
    ASSERT_TRUE(error.Fail());
    ASSERT_STREQ(error.AsCString(), test_str);
  }
}

TEST_F(VerifyUserMultiwordCmdPathTest, TestErrors) {
  ArchSpec arch("x86_64-apple-macosx-");

  Platform::SetHostPlatform(
      PlatformRemoteMacOSX::CreateInstance(true, &arch, /*debugger=*/nullptr,
                                           /*metadata=*/nullptr));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  CommandInterpreter &interp = debugger_sp->GetCommandInterpreter();

  Status error;
  bool success;
  bool is_leaf;

  // Test that we reject non-user path components:
  success = false;
  is_leaf = true;
  RunTest(interp, "process launch", is_leaf, success,
          "Path component: 'process' is not a user command");

  // Test that we reject non-existent commands:
  is_leaf = true;
  success = false;
  RunTest(interp, "wewouldnevernameacommandthis subcommand", is_leaf, success,
          "Path component: 'wewouldnevernameacommandthis' not found");

  // Now we have to add a multiword command, and then probe it.
  error = interp.AddUserCommand(
      "dummy", CommandObjectSP(new CommandObjectMultiwordDummy(interp)), true);
  ASSERT_TRUE(error.Success());

  // Now pass the correct path, and make sure we get back the right MWC.
  is_leaf = false;
  success = true;
  RunTest(interp, "dummy subcommand", is_leaf, success, "dummy subcommand");

  is_leaf = true;
  RunTest(interp, "dummy subcommand", is_leaf, success, "dummy");

  // If you tell us the last node is a leaf, we don't check that.  Make sure
  // that is true:
  is_leaf = true;
  success = true;
  RunTest(interp, "dummy subcommand leaf", is_leaf, success,
          "dummy subcommand");
  // But we should fail if we say the last component is a multiword:

  is_leaf = false;
  success = false;
  RunTest(interp, "dummy subcommand leaf", is_leaf, success,
          "Path component: 'leaf' is not a container command");

  // We should fail if we get the second path component wrong:
  is_leaf = false;
  success = false;
  RunTest(interp, "dummy not-subcommand", is_leaf, success,
          "Path component: 'not-subcommand' not found");
}
