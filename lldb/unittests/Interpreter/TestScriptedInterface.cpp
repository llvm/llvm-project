//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/Interfaces/ScriptedCommandInterface.h"
#include "lldb/Interpreter/Interfaces/ScriptedInterface.h"
#include "lldb/Interpreter/ScriptedInstanceRegistry.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {

class DummyScriptedInterface : public ScriptedInterface {
public:
  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return {};
  }

  llvm::StringRef GetPluginName() override { return "DummyPlugin"; }
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

  llvm::StringRef GetPluginName() override { return "DummyPlugin"; }
};

class RegisteredScriptedInterface : public DummyScriptedInterface {
public:
  RegisteredScriptedInterface(
      const lldb::ScriptedInstanceRegistrySP &registry_sp,
      llvm::StringRef class_name) {
    RegisterInstance(registry_sp, class_name);
  }

  void Reregister(const lldb::ScriptedInstanceRegistrySP &registry_sp) {
    RegisterInstance(registry_sp, "module.Reregistered");
  }
};

} // namespace

TEST(ScriptedInterfaceTest, DestroyedInstanceIsUnregistered) {
  auto registry_sp = std::make_shared<ScriptedInstanceRegistry>();
  auto first = std::make_unique<RegisteredScriptedInterface>(registry_sp,
                                                             "module.First");
  RegisteredScriptedInterface second(registry_sp, "module.Second");
  ASSERT_EQ(registry_sp->GetInstances().size(), 2u);

  first.reset();
  std::vector<ScriptedInstanceInfo> instances = registry_sp->GetInstances();
  ASSERT_EQ(instances.size(), 1u);
  EXPECT_EQ(instances[0].class_name, "module.Second");
}

TEST(ScriptedInterfaceTest, ReregisteringReplacesEntry) {
  auto registry_sp = std::make_shared<ScriptedInstanceRegistry>();
  RegisteredScriptedInterface interface(registry_sp, "module.First");
  interface.Reregister(registry_sp);

  std::vector<ScriptedInstanceInfo> instances = registry_sp->GetInstances();
  ASSERT_EQ(instances.size(), 1u);
  EXPECT_EQ(instances[0].class_name, "module.Reregistered");
}

TEST(ScriptedInterfaceTest, InterfaceCanOutliveRegistry) {
  auto registry_sp = std::make_shared<ScriptedInstanceRegistry>();
  auto interface =
      std::make_unique<RegisteredScriptedInterface>(registry_sp, "module.A");
  registry_sp.reset();
  interface.reset();
}

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
