//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/Interfaces/ScriptedCommandInterface.h"
#include "lldb/Interpreter/Interfaces/ScriptedInterface.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {

class DummyScriptedInterface : public ScriptedInterface {
public:
  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return {};
  }
};

class DummyScriptedCommandInterface : public ScriptedCommandInterface {
public:
  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name,
                     lldb::DebuggerSP debugger_sp) override {
    return nullptr;
  }

  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return {};
  }
};

} // namespace

TEST(ScriptedInterfaceTest, ExtensionsCannotBeRunDirectly) {
  DummyScriptedInterface interface;
  EXPECT_FALSE(interface.UserCanRunDirectly());
}

TEST(ScriptedInterfaceTest, CommandsCanBeRunDirectly) {
  DummyScriptedCommandInterface command_interface;
  EXPECT_TRUE(command_interface.UserCanRunDirectly());

  // The scripted-extension policy is pushed through a ScriptedInterface, so the
  // override has to be reachable from the base: a command that looks like any
  // other extension there would silently lose its API mutex.
  ScriptedInterface &as_base = command_interface;
  EXPECT_TRUE(as_base.UserCanRunDirectly());
}
