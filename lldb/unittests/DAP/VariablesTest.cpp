//===-- VariablesTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Variables.h"
#include "Protocol/ProtocolTypes.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBValueList.h"
#include "gtest/gtest.h"

using namespace lldb_dap;

class VariablesTest : public ::testing::Test {
protected:
  enum : bool { Permanent = true, Temporary = false };
  Variables vars;

  static const protocol::Scope *
  FindScope(const std::vector<protocol::Scope> &scopes, llvm::StringRef name) {
    for (const auto &scope : scopes) {
      if (scope.name == name)
        return &scope;
    }
    return nullptr;
  }
};

TEST_F(VariablesTest, GetNewVariableReference_UniqueAndRanges) {
  const int64_t temp1 = vars.GetNewVariableReference(Temporary);
  const int64_t temp2 = vars.GetNewVariableReference(Temporary);
  const int64_t perm1 = vars.GetNewVariableReference(Permanent);
  const int64_t perm2 = vars.GetNewVariableReference(Permanent);

  EXPECT_NE(temp1, temp2);
  EXPECT_NE(perm1, perm2);
  EXPECT_LT(temp1, perm1);
  EXPECT_LT(temp2, perm1);
}

TEST_F(VariablesTest, InsertAndGetVariable_Temporary) {
  lldb::SBValue dummy;
  const int64_t ref = vars.InsertVariable(dummy, Temporary);
  lldb::SBValue out = vars.GetVariable(ref);

  EXPECT_EQ(out.IsValid(), dummy.IsValid());
}

TEST_F(VariablesTest, InsertAndGetVariable_Permanent) {
  lldb::SBValue dummy;
  const int64_t ref = vars.InsertVariable(dummy, Permanent);
  lldb::SBValue out = vars.GetVariable(ref);

  EXPECT_EQ(out.IsValid(), dummy.IsValid());
}

TEST_F(VariablesTest, IsPermanentVariableReference) {
  const int64_t perm = vars.GetNewVariableReference(Permanent);
  const int64_t temp = vars.GetNewVariableReference(Temporary);

  EXPECT_TRUE(Variables::IsPermanentVariableReference(perm));
  EXPECT_FALSE(Variables::IsPermanentVariableReference(temp));
}

TEST_F(VariablesTest, Clear_RemovesTemporaryKeepsPermanent) {
  lldb::SBValue dummy;
  const int64_t temp = vars.InsertVariable(dummy, Temporary);
  const int64_t perm = vars.InsertVariable(dummy, Permanent);
  vars.Clear();

  EXPECT_FALSE(vars.GetVariable(temp).IsValid());
  EXPECT_EQ(vars.GetVariable(perm).IsValid(), dummy.IsValid());
}

TEST_F(VariablesTest, GetTopLevelScope_ReturnsCorrectScope) {
  lldb::SBFrame frame;
  uint32_t frame_id = 0;

  std::vector<protocol::Scope> scopes = vars.ReadyFrame(frame_id, frame);

  const protocol::Scope *locals_scope = FindScope(scopes, "Locals");
  const protocol::Scope *globals_scope = FindScope(scopes, "Globals");
  const protocol::Scope *registers_scope = FindScope(scopes, "Registers");

  ASSERT_NE(locals_scope, nullptr);
  ASSERT_NE(globals_scope, nullptr);
  ASSERT_NE(registers_scope, nullptr);

  EXPECT_EQ(vars.GetTopLevelScope(locals_scope->variablesReference),
            vars.GetScope(frame_id, eScopeKindLocals));
  EXPECT_EQ(vars.GetTopLevelScope(globals_scope->variablesReference),
            vars.GetScope(frame_id, eScopeKindGlobals));
  EXPECT_EQ(vars.GetTopLevelScope(registers_scope->variablesReference),
            vars.GetScope(frame_id, eScopeKindRegisters));
  EXPECT_EQ(vars.GetTopLevelScope(9999), nullptr);
}

TEST_F(VariablesTest, FindVariable_LocalsByName) {
  lldb::SBFrame frame;
  uint32_t frame_id = 0;

  std::vector<protocol::Scope> scopes = vars.ReadyFrame(frame_id, frame);

  const protocol::Scope *locals_scope = FindScope(scopes, "Locals");
  ASSERT_NE(locals_scope, nullptr);

  lldb::SBValue found = vars.FindVariable(locals_scope->variablesReference, "");
  lldb::SBValue dummy;

  EXPECT_EQ(found.IsValid(), dummy.IsValid());
}
