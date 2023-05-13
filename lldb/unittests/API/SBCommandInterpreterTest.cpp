//===-- SBCommandInterpreterTest.cpp ------------------------===----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "gtest/gtest.h"

#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBDebugger.h"

#include <cstring>
#include <string>

using namespace lldb;

class SBCommandInterpreterTest : public testing::Test {
protected:
  void SetUp() override {
    SBDebugger::Initialize();
    m_dbg = SBDebugger::Create(/*source_init_files=*/false);
  }

  SBDebugger m_dbg;
};

class DummyCommand : public SBCommandPluginInterface {
public:
  DummyCommand(const char *message) : m_message(message) {}

  bool DoExecute(SBDebugger dbg, char **command,
                 SBCommandReturnObject &result) override {
    result.PutCString(m_message.c_str());
    result.SetStatus(eReturnStatusSuccessFinishResult);
    return result.Succeeded();
  }

private:
  std::string m_message;
};

TEST_F(SBCommandInterpreterTest, SingleWordCommand) {
  // We first test a command without autorepeat
  DummyCommand dummy("It worked");
  SBCommandInterpreter interp = m_dbg.GetCommandInterpreter();
  interp.AddCommand("dummy", &dummy, /*help=*/nullptr);
  {
    SBCommandReturnObject result;
    interp.HandleCommand("dummy", result, /*add_to_history=*/true);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked\n");
  }
  {
    SBCommandReturnObject result;
    interp.HandleCommand("", result);
    EXPECT_FALSE(result.Succeeded());
    EXPECT_STREQ(result.GetError(), "error: No auto repeat.\n");
  }

  // Now we test a command with autorepeat
  interp.AddCommand("dummy_with_autorepeat", &dummy, /*help=*/nullptr,
                    /*syntax=*/nullptr, /*auto_repeat_command=*/nullptr);
  {
    SBCommandReturnObject result;
    interp.HandleCommand("dummy_with_autorepeat", result,
                         /*add_to_history=*/true);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked\n");
  }
  {
    SBCommandReturnObject result;
    interp.HandleCommand("", result);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked\n");
  }
}

TEST_F(SBCommandInterpreterTest, MultiWordCommand) {
  SBCommandInterpreter interp = m_dbg.GetCommandInterpreter();
  auto command = interp.AddMultiwordCommand("multicommand", /*help=*/nullptr);
  // We first test a subcommand without autorepeat
  DummyCommand subcommand("It worked again");
  command.AddCommand("subcommand", &subcommand, /*help=*/nullptr);
  {
    SBCommandReturnObject result;
    interp.HandleCommand("multicommand subcommand", result,
                         /*add_to_history=*/true);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked again\n");
  }
  {
    SBCommandReturnObject result;
    interp.HandleCommand("", result);
    EXPECT_FALSE(result.Succeeded());
    EXPECT_STREQ(result.GetError(), "error: No auto repeat.\n");
  }

  // We first test a subcommand with autorepeat
  command.AddCommand("subcommand_with_autorepeat", &subcommand,
                     /*help=*/nullptr, /*syntax=*/nullptr,
                     /*auto_repeat_command=*/nullptr);
  {
    SBCommandReturnObject result;
    interp.HandleCommand("multicommand subcommand_with_autorepeat", result,
                         /*add_to_history=*/true);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked again\n");
  }
  {
    SBCommandReturnObject result;
    interp.HandleCommand("", result);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked again\n");
  }

  DummyCommand subcommand2("It worked again 2");
  // We now test a subcommand with autorepeat of the command name
  command.AddCommand(
      "subcommand_with_custom_autorepeat", &subcommand2, /*help=*/nullptr,
      /*syntax=*/nullptr,
      /*auto_repeat_command=*/"multicommand subcommand_with_autorepeat");
  {
    SBCommandReturnObject result;
    interp.HandleCommand("multicommand subcommand_with_custom_autorepeat",
                         result, /*add_to_history=*/true);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked again 2\n");
  }
  {
    SBCommandReturnObject result;
    interp.HandleCommand("", result);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked again\n");
  }
}
