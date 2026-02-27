//===-- VariablesTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Variables.h"
#include "DAPLog.h"
#include "Protocol/DAPTypes.h"
#include "Protocol/ProtocolTypes.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBValue.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap::protocol;

class VariablesTest : public ::testing::Test {

public:
  VariablesTest() : log(llvm::nulls(), mutex), vars(log) {}

  static void SetUpTestSuite() {
    lldb::SBError error = SBDebugger::InitializeWithErrorHandling();
    EXPECT_TRUE(error.Success());
  }
  static void TeatUpTestSuite() { SBDebugger::Terminate(); }

  void TearDown() override {
    if (core)
      ASSERT_THAT_ERROR(core->discard(), Succeeded());
    if (binary)
      ASSERT_THAT_ERROR(binary->discard(), Succeeded());
    if (debugger)
      debugger.Clear();
  }

protected:
  enum : bool { Permanent = true, Temporary = false };
  Log::Mutex mutex;
  Log log;
  VariableReferenceStorage vars;
  lldb::SBDebugger debugger;
  lldb::SBTarget target;
  lldb::SBProcess process;

  static constexpr llvm::StringLiteral k_binary = "linux-x86_64.out.yaml";
  static constexpr llvm::StringLiteral k_core = "linux-x86_64.core.yaml";

  std::optional<llvm::sys::fs::TempFile> core;
  std::optional<llvm::sys::fs::TempFile> binary;

  void CreateDebugger() { debugger = lldb::SBDebugger::Create(); }

  void LoadCore() {
    ASSERT_TRUE(debugger);

    llvm::Expected<lldb_private::TestFile> binary_yaml =
        lldb_private::TestFile::fromYamlFile(k_binary);
    ASSERT_THAT_EXPECTED(binary_yaml, Succeeded());
    llvm::Expected<llvm::sys::fs::TempFile> binary_file =
        binary_yaml->writeToTemporaryFile();
    ASSERT_THAT_EXPECTED(binary_file, Succeeded());
    binary = std::move(*binary_file);
    target = debugger.CreateTarget(binary->TmpName.data());
    ASSERT_TRUE(target);
    debugger.SetSelectedTarget(target);

    llvm::Expected<lldb_private::TestFile> core_yaml =
        lldb_private::TestFile::fromYamlFile(k_core);
    ASSERT_THAT_EXPECTED(core_yaml, Succeeded());
    llvm::Expected<llvm::sys::fs::TempFile> core_file =
        core_yaml->writeToTemporaryFile();
    ASSERT_THAT_EXPECTED(core_file, Succeeded());
    this->core = std::move(*core_file);
    process = target.LoadCore(this->core->TmpName.data());
    ASSERT_TRUE(process);
  }

  static const protocol::Scope *
  FindScope(const std::vector<protocol::Scope> &scopes,
            const protocol::String &name) {
    for (const auto &scope : scopes) {
      if (scope.name == name)
        return &scope;
    }
    return nullptr;
  }
};

TEST_F(VariablesTest, GetNewVariableReference_UniqueAndRanges) {
  CreateDebugger();
  LoadCore();
  auto x15 = target.CreateValueFromExpression("x", "15");
  auto y42 = target.CreateValueFromExpression("y", "42");
  auto gzero = target.CreateValueFromExpression("$0", "42");
  auto gone = target.CreateValueFromExpression("$1", "7");
  const var_ref_t temp1 = vars.Insert(x15, Temporary);
  const var_ref_t temp2 = vars.Insert(y42, Temporary);
  const var_ref_t perm1 = vars.Insert(gzero, Permanent);
  const var_ref_t perm2 = vars.Insert(gone, Permanent);
  EXPECT_NE(temp1.AsUInt32(), temp2.AsUInt32());
  EXPECT_NE(perm1.AsUInt32(), perm2.AsUInt32());
  EXPECT_LT(temp1.AsUInt32(), perm1.AsUInt32());
  EXPECT_LT(temp2.AsUInt32(), perm1.AsUInt32());
}

TEST_F(VariablesTest, InsertAndGetVariable_Temporary) {
  lldb::SBValue dummy;
  const var_ref_t ref = vars.Insert(dummy, Temporary);
  lldb::SBValue out = vars.GetVariable(ref);

  EXPECT_EQ(out.IsValid(), dummy.IsValid());
}

TEST_F(VariablesTest, InsertAndGetVariable_Permanent) {
  lldb::SBValue dummy;
  const var_ref_t ref = vars.Insert(dummy, Permanent);
  lldb::SBValue out = vars.GetVariable(ref);

  EXPECT_EQ(out.IsValid(), dummy.IsValid());
}

TEST_F(VariablesTest, IsPermanentVariableReference) {
  const var_ref_t perm = vars.Insert(lldb::SBValue(), Permanent);
  const var_ref_t temp = vars.Insert(lldb::SBValue(), Temporary);

  EXPECT_EQ(perm.Kind(), eReferenceKindPermanent);
  EXPECT_EQ(temp.Kind(), eReferenceKindTemporary);
}

TEST_F(VariablesTest, Clear_RemovesTemporaryKeepsPermanent) {
  lldb::SBValue dummy;
  const var_ref_t temp = vars.Insert(dummy, Temporary);
  const var_ref_t perm = vars.Insert(dummy, Permanent);
  vars.Clear();

  EXPECT_FALSE(vars.GetVariable(temp).IsValid());
  EXPECT_EQ(vars.GetVariable(perm).IsValid(), dummy.IsValid());
}

TEST_F(VariablesTest, VariablesStore) {
  CreateDebugger();
  LoadCore();
  lldb::SBFrame frame = process.GetSelectedThread().GetSelectedFrame();

  std::vector<protocol::Scope> scopes = vars.Insert(frame);

  const protocol::Scope *locals_scope = FindScope(scopes, "Locals");
  const protocol::Scope *globals_scope = FindScope(scopes, "Globals");
  const protocol::Scope *registers_scope = FindScope(scopes, "Registers");

  ASSERT_NE(locals_scope, nullptr);
  ASSERT_NE(globals_scope, nullptr);
  ASSERT_NE(registers_scope, nullptr);

  auto *locals_store = vars.GetVariableStore(locals_scope->variablesReference);
  auto *globals_store =
      vars.GetVariableStore(globals_scope->variablesReference);
  auto *registers_store =
      vars.GetVariableStore(registers_scope->variablesReference);

  ASSERT_NE(locals_store, nullptr);
  ASSERT_NE(globals_store, nullptr);
  ASSERT_NE(registers_store, nullptr);

  const var_ref_t local_ref = locals_scope->variablesReference;
  const var_ref_t global_ref = globals_scope->variablesReference;
  const var_ref_t register_ref = registers_scope->variablesReference;
  ASSERT_EQ(global_ref.Kind(), eReferenceKindTemporary);
  ASSERT_EQ(local_ref.Kind(), eReferenceKindTemporary);
  ASSERT_EQ(register_ref.Kind(), eReferenceKindTemporary);

  EXPECT_EQ(vars.GetVariableStore(var_ref_t(9999)), nullptr);

  ASSERT_TRUE(vars.FindVariable(local_ref, "rect").IsValid());

  auto variables = locals_store->GetVariables(vars, {}, {});
  ASSERT_THAT_EXPECTED(variables, Succeeded());
  ASSERT_EQ(variables->size(), 1u);
  auto rect = variables->at(0);
  ASSERT_EQ(rect.name, "rect");

  VariablesArguments args;
  args.variablesReference = rect.variablesReference;

  auto *store = vars.GetVariableStore(args.variablesReference);
  ASSERT_NE(store, nullptr);

  variables = store->GetVariables(vars, {}, args);
  ASSERT_THAT_EXPECTED(variables, Succeeded());
  ASSERT_EQ(variables->size(), 4u);
  EXPECT_EQ(variables->at(0).name, "x");
  EXPECT_EQ(variables->at(0).value, "5");
  EXPECT_EQ(variables->at(1).name, "y");
  EXPECT_EQ(variables->at(1).value, "5");
  EXPECT_EQ(variables->at(2).name, "height");
  EXPECT_EQ(variables->at(2).value, "25");
  EXPECT_EQ(variables->at(3).name, "width");
  EXPECT_EQ(variables->at(3).value, "30");
}

TEST_F(VariablesTest, FindVariable_LocalsByName) {
  lldb::SBFrame frame;

  std::vector<protocol::Scope> scopes = vars.Insert(frame);

  const protocol::Scope *locals_scope = FindScope(scopes, "Locals");
  ASSERT_NE(locals_scope, nullptr);

  lldb::SBValue found = vars.FindVariable(locals_scope->variablesReference, "");
  lldb::SBValue dummy;

  EXPECT_EQ(found.IsValid(), dummy.IsValid());
}
