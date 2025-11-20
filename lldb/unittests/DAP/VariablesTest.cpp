//===-- VariablesTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Variables.h"
#include "TestFixtures.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBValueList.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_private;
using namespace lldb_dap_tests;

class VariablesTest : public ::testing::Test {
protected:
  SubsystemRAII<SBDebugger> subsystems;

  enum : bool { Permanent = true, Temporary = false };
  TestFixtures fixtures;
  SBValue temp_value;
  SBValue perm_value;
  Variables vars;

  void SetUp() override {
    fixtures.LoadDebugger();
    fixtures.LoadTarget();
    SBTarget &target = fixtures.target;
    temp_value = target.CreateValueFromExpression("temp", "1");
    ASSERT_TRUE(temp_value);
    perm_value = temp_value.Persist();
    ASSERT_TRUE(perm_value);
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
  const int64_t ref = vars.InsertVariable(temp_value);
  SBValue out = vars.GetVariable(ref);

  EXPECT_EQ(out.IsValid(), temp_value.IsValid());
  EXPECT_EQ(out.GetName(), temp_value.GetName());
  EXPECT_EQ(out.GetValue(), temp_value.GetValue());
}

TEST_F(VariablesTest, InsertAndGetVariable_Permanent) {
  const int64_t ref = vars.InsertVariable(perm_value);
  SBValue out = vars.GetVariable(ref);

  EXPECT_EQ(out.IsValid(), perm_value.IsValid());
  EXPECT_EQ(out.GetName(), perm_value.GetName());
  EXPECT_EQ(out.GetValue(), perm_value.GetValue());
}

TEST_F(VariablesTest, IsPermanentVariableReference) {
  const int64_t perm = vars.GetNewVariableReference(Permanent);
  const int64_t temp = vars.GetNewVariableReference(Temporary);

  EXPECT_TRUE(Variables::IsPermanentVariableReference(perm));
  EXPECT_FALSE(Variables::IsPermanentVariableReference(temp));
}

TEST_F(VariablesTest, Clear_RemovesTemporaryKeepsPermanent) {
  const int64_t temp = vars.InsertVariable(temp_value);
  const int64_t perm = vars.InsertVariable(perm_value);
  vars.Clear();

  EXPECT_FALSE(vars.GetVariable(temp).IsValid());
  EXPECT_EQ(vars.GetVariable(perm).IsValid(), perm_value.IsValid());
}

TEST_F(VariablesTest, GetTopLevelScope_ReturnsCorrectScope) {
  vars.locals.Append(SBValue());
  vars.globals.Append(SBValue());
  vars.registers.Append(SBValue());

  EXPECT_EQ(vars.GetTopLevelScope(VARREF_LOCALS), &vars.locals);
  EXPECT_EQ(vars.GetTopLevelScope(VARREF_GLOBALS), &vars.globals);
  EXPECT_EQ(vars.GetTopLevelScope(VARREF_REGS), &vars.registers);
  EXPECT_EQ(vars.GetTopLevelScope(9999), nullptr);
}

TEST_F(VariablesTest, FindVariable_LocalsByName) {
  SBValue dummy;
  vars.locals.Append(dummy);
  SBValue found = vars.FindVariable(VARREF_LOCALS, "");

  EXPECT_EQ(found.IsValid(), dummy.IsValid());
}
