//===- ConstantFoldingTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/ConstantFoldingMIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST_F(GISelMITest, FoldWithBuilder) {
  setUp();
  if (!TM)
    return;
  // Try to use the FoldableInstructionsBuilder to build binary ops.
  ConstantFoldingMIRBuilder CFB(B.getState());
  LLT s32 = LLT::scalar(32);
  int64_t Cst;
  auto MIBCAdd =
      CFB.buildAdd(s32, CFB.buildConstant(s32, 0), CFB.buildConstant(s32, 1));
  // This should be a constant now.
  bool match = mi_match(MIBCAdd->getOperand(0).getReg(), *MRI, m_ICst(Cst));
  EXPECT_TRUE(match);
  EXPECT_EQ(Cst, 1);
  auto MIBCAdd1 =
      CFB.buildInstr(TargetOpcode::G_ADD, {s32},
                     {CFB.buildConstant(s32, 0), CFB.buildConstant(s32, 1)});
  // This should be a constant now.
  match = mi_match(MIBCAdd1->getOperand(0).getReg(), *MRI, m_ICst(Cst));
  EXPECT_TRUE(match);
  EXPECT_EQ(Cst, 1);

  // Try one of the other constructors of MachineIRBuilder to make sure it's
  // compatible.
  ConstantFoldingMIRBuilder CFB1(*MF);
  CFB1.setInsertPt(*EntryMBB, EntryMBB->end());
  auto MIBCSub =
      CFB1.buildInstr(TargetOpcode::G_SUB, {s32},
                      {CFB1.buildConstant(s32, 1), CFB1.buildConstant(s32, 1)});
  // This should be a constant now.
  match = mi_match(MIBCSub->getOperand(0).getReg(), *MRI, m_ICst(Cst));
  EXPECT_TRUE(match);
  EXPECT_EQ(Cst, 0);

  auto MIBCSext1 =
      CFB1.buildInstr(TargetOpcode::G_SEXT_INREG, {s32},
                      {CFB1.buildConstant(s32, 0x01), uint64_t(8)});
  // This should be a constant now.
  match = mi_match(MIBCSext1->getOperand(0).getReg(), *MRI, m_ICst(Cst));
  EXPECT_TRUE(match);
  EXPECT_EQ(1, Cst);

  auto MIBCSext2 =
      CFB1.buildInstr(TargetOpcode::G_SEXT_INREG, {s32},
                      {CFB1.buildConstant(s32, 0x80), uint64_t(8)});
  // This should be a constant now.
  match = mi_match(MIBCSext2->getOperand(0).getReg(), *MRI, m_ICst(Cst));
  EXPECT_TRUE(match);
  EXPECT_EQ(-0x80, Cst);
}

} // namespace