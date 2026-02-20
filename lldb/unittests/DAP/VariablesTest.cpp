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
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBValue.h"
#include "gtest/gtest.h"

using namespace lldb_dap;
using namespace lldb_dap::protocol;

class VariablesTest : public ::testing::Test {

public:
  VariablesTest() : log(llvm::nulls(), mutex), vars(log) {}

protected:
  enum : bool { Permanent = true, Temporary = false };
  Log::Mutex mutex;
  Log log;
  VariableReferenceStorage vars;

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
  const var_ref_t temp1 = vars.InsertVariable(lldb::SBValue(), Temporary);
  const var_ref_t temp2 = vars.InsertVariable(lldb::SBValue(), Temporary);
  const var_ref_t perm1 = vars.InsertVariable(lldb::SBValue(), Permanent);
  const var_ref_t perm2 = vars.InsertVariable(lldb::SBValue(), Permanent);
  EXPECT_NE(temp1.AsUInt32(), temp2.AsUInt32());
  EXPECT_NE(perm1.AsUInt32(), perm2.AsUInt32());
  EXPECT_LT(temp1.AsUInt32(), perm1.AsUInt32());
  EXPECT_LT(temp2.AsUInt32(), perm1.AsUInt32());
}

TEST_F(VariablesTest, InsertAndGetVariable_Temporary) {
  lldb::SBValue dummy;
  const var_ref_t ref = vars.InsertVariable(dummy, Temporary);
  lldb::SBValue out = vars.GetVariable(ref);

  EXPECT_EQ(out.IsValid(), dummy.IsValid());
}

TEST_F(VariablesTest, InsertAndGetVariable_Permanent) {
  lldb::SBValue dummy;
  const var_ref_t ref = vars.InsertVariable(dummy, Permanent);
  lldb::SBValue out = vars.GetVariable(ref);

  EXPECT_EQ(out.IsValid(), dummy.IsValid());
}

TEST_F(VariablesTest, IsPermanentVariableReference) {
  const var_ref_t perm = vars.InsertVariable(lldb::SBValue(), Permanent);
  const var_ref_t temp = vars.InsertVariable(lldb::SBValue(), Temporary);

  EXPECT_EQ(perm.Kind(), eReferenceKindPermanent);
  EXPECT_EQ(temp.Kind(), eReferenceKindTemporary);
}

TEST_F(VariablesTest, Clear_RemovesTemporaryKeepsPermanent) {
  lldb::SBValue dummy;
  const var_ref_t temp = vars.InsertVariable(dummy, Temporary);
  const var_ref_t perm = vars.InsertVariable(dummy, Permanent);
  vars.Clear();

  EXPECT_FALSE(vars.GetVariable(temp).IsValid());
  EXPECT_EQ(vars.GetVariable(perm).IsValid(), dummy.IsValid());
}

TEST_F(VariablesTest, VariablesStore) {
  lldb::SBFrame frame;

  std::vector<protocol::Scope> scopes = vars.CreateScopes(frame);

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
  ASSERT_EQ(global_ref.Kind(), eReferenceKindScope);
  ASSERT_EQ(local_ref.Kind(), eReferenceKindScope);
  ASSERT_EQ(register_ref.Kind(), eReferenceKindScope);

  EXPECT_EQ(vars.GetVariableStore(var_ref_t(9999)), nullptr);
}

TEST_F(VariablesTest, FindVariable_LocalsByName) {
  lldb::SBFrame frame;

  std::vector<protocol::Scope> scopes = vars.CreateScopes(frame);

  const protocol::Scope *locals_scope = FindScope(scopes, "Locals");
  ASSERT_NE(locals_scope, nullptr);

  lldb::SBValue found = vars.FindVariable(locals_scope->variablesReference, "");
  lldb::SBValue dummy;

  EXPECT_EQ(found.IsValid(), dummy.IsValid());
}
