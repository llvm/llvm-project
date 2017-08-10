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

} // end namespace
