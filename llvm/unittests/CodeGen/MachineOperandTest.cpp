//===- MachineOperandTest.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ilist_node.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(MachineOperandTest, ChangeToTargetIndexTest) {
  // Creating a MachineOperand to change it to TargetIndex
  MachineOperand MO = MachineOperand::CreateImm(50);

  // Checking some precondition on the newly created
  // MachineOperand.
  ASSERT_TRUE(MO.isImm());
  ASSERT_TRUE(MO.getImm() == 50);
  ASSERT_FALSE(MO.isTargetIndex());

  // Changing to TargetIndex with some arbitrary values
  // for index, offset and flags.
  MO.ChangeToTargetIndex(74, 57, 12);

  // Checking that the mutation to TargetIndex happened
  // correctly.
  ASSERT_TRUE(MO.isTargetIndex());
  ASSERT_TRUE(MO.getIndex() == 74);
  ASSERT_TRUE(MO.getOffset() == 57);
  ASSERT_TRUE(MO.getTargetFlags() == 12);
}

TEST(MachineOperandTest, PrintRegisterMask) {
  uint32_t Dummy;
  MachineOperand MO = MachineOperand::CreateRegMask(&Dummy);

  // Checking some preconditions on the newly created
  // MachineOperand.
  ASSERT_TRUE(MO.isRegMask());
  ASSERT_TRUE(MO.getRegMask() == &Dummy);

  // Print a MachineOperand containing a RegMask. Here we check that without a
  // TRI and IntrinsicInfo we still print a less detailed regmask.
  std::string str;
  raw_string_ostream OS(str);
  MO.print(OS, /*TRI=*/nullptr, /*IntrinsicInfo=*/nullptr);
  ASSERT_TRUE(OS.str() == "<regmask ...>");
}

TEST(MachineOperandTest, PrintSubReg) {
  // Create a MachineOperand with RegNum=1 and SubReg=5.
  MachineOperand MO = MachineOperand::CreateReg(
      /*Reg=*/1, /*isDef=*/false, /*isImp=*/false, /*isKill=*/false,
      /*isDead=*/false, /*isUndef=*/false, /*isEarlyClobber=*/false,
      /*SubReg=*/5, /*isDebug=*/false, /*isInternalRead=*/false);

  // Checking some preconditions on the newly created
  // MachineOperand.
  ASSERT_TRUE(MO.isReg());
  ASSERT_TRUE(MO.getReg() == 1);
  ASSERT_TRUE(MO.getSubReg() == 5);

  // Print a MachineOperand containing a SubReg. Here we check that without a
  // TRI and IntrinsicInfo we can still print the subreg index.
  std::string str;
  raw_string_ostream OS(str);
  MO.print(OS, /*TRI=*/nullptr, /*IntrinsicInfo=*/nullptr);
  ASSERT_TRUE(OS.str() == "%physreg1.subreg5");
}

} // end namespace
