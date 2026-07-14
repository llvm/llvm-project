//===- AMDGPUNextUseAnalysisTest.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for AMDGPUNextUseAnalysis.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUNextUseAnalysis.h"
#include "AMDGPUUnitTests.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Passes/PassBuilder.h"
#include "gtest/gtest.h"

using namespace llvm;

class AMDGPUNextUseAnalysisTest : public AMDGPUCodeGenTestBase {
public:
  void SetUp() override { setUpImpl("amdgcn-amd-amdhsa", "gfx1310", ""); }
};

// Test that distance calculation works correctly when multiple instructions
// are inserted at different points in the basic block.
TEST_F(AMDGPUNextUseAnalysisTest, DistanceAfterInstructionInsertion) {
  StringRef MIR = R"(
name:            DistanceAfterInstructionInsertion
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    liveins: $vgpr0
    %0:vgpr_32 = COPY $vgpr0
    %1:vgpr_32 = V_MOV_B32_e32 1, implicit $exec
    %2:vgpr_32 = V_MOV_B32_e32 2, implicit $exec
    %3:vgpr_32 = V_ADD_U32_e32 %0, %1, implicit $exec
    %4:vgpr_32 = V_ADD_U32_e32 %3, %2, implicit $exec
    S_NOP 0, implicit %4
    S_ENDPGM 0
...
)";
  EXPECT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("DistanceAfterInstructionInsertion");
  AMDGPUNextUseAnalysis &NUA = MFAM.getResult<AMDGPUNextUseAnalysisPass>(MF);

  MachineBasicBlock &MBB = *MF.begin();
  const auto &TII = *MF.getSubtarget().getInstrInfo();

  auto It = MBB.begin();
  std::advance(It, 3);
  MachineInstr &Add1 = *It++;
  MachineInstr &Add2 = *It++;
  MachineInstr &OrigNop = *It;

  Register Add1DefReg = Add1.getOperand(0).getReg();

  // Add a new multiplication before the first ADD.
  Register NewReg1 =
      MF.getRegInfo().createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  MachineInstr *NewMul1 = BuildMI(MBB, Add1.getIterator(), DebugLoc(),
                                  TII.get(AMDGPU::V_MUL_U32_U24_e64), NewReg1)
                              .addImm(0)
                              .addImm(1)
                              .addImm(2)
                              .getInstr();

  // Add the destination register of the first multiplication as an input
  // operand of the fist ADD.
  Add1.getOperand(1).setReg(NewReg1);

  // Add a second multiplication before the first ADD
  Register NewReg2 =
      MF.getRegInfo().createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  MachineInstr *NewMul2 = BuildMI(MBB, Add1.getIterator(), DebugLoc(),
                                  TII.get(AMDGPU::V_MUL_U32_U24_e64), NewReg2)
                              .addImm(0)
                              .addImm(3)
                              .addImm(4)
                              .getInstr();

  // Add the destination register of the second multiplication as an input
  // operand of the second ADD.
  Add2.getOperand(1).setReg(NewReg2);

  // Add a third multiplication before the second ADD
  Register NewReg3 =
      MF.getRegInfo().createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  MachineInstr *NewMul3 = BuildMI(MBB, Add2.getIterator(), DebugLoc(),
                                  TII.get(AMDGPU::V_MUL_U32_U24_e64), NewReg3)
                              .addImm(0)
                              .addImm(5)
                              .addImm(6)
                              .getInstr();

  // Add the destination register of the third multiplication as an input
  // operand of the first NOP.
  for (MachineOperand &MO : OrigNop.operands()) {
    if (MO.isReg() && MO.isUse())
      MO.setReg(NewReg3);
  }

  Register Add2DefReg = Add2.getOperand(0).getReg();

  MachineInstr &Endpgm = MBB.back();
  MachineInstr *UseOfNewRegs =
      BuildMI(MBB, Endpgm.getIterator(), DebugLoc(), TII.get(AMDGPU::S_NOP))
          .addImm(0)
          .addReg(Add1DefReg, RegState::Implicit)
          .addReg(Add2DefReg, RegState::Implicit)
          .getInstr();

  for (const MachineInstr &MI : MBB)
    MI.dump();

  // Find the uses of the first ADD.
  SmallVector<const MachineOperand *> Add1Uses;
  for (const MachineOperand &MO : UseOfNewRegs->uses()) {
    if (MO.isReg() && MO.getReg() == Add1DefReg)
      Add1Uses.push_back(&MO);
  }

  // Find the uses of the second ADD.
  SmallVector<const MachineOperand *> Add2Uses;
  for (const MachineOperand &MO : UseOfNewRegs->uses()) {
    if (MO.isReg() && MO.getReg() == Add2DefReg)
      Add2Uses.push_back(&MO);
  }

  // Find the uses of the first multiplication.
  SmallVector<const MachineOperand *> Mul1Uses;
  for (const MachineOperand &MO : Add1.uses()) {
    if (MO.isReg() && MO.getReg() == NewReg1)
      Mul1Uses.push_back(&MO);
  }

  // Find the uses of the second multiplication.
  SmallVector<const MachineOperand *> Mul2Uses;
  for (const MachineOperand &MO : Add2.uses()) {
    if (MO.isReg() && MO.getReg() == NewReg2)
      Mul2Uses.push_back(&MO);
  }

  // Find the uses of the third multiplication.
  SmallVector<const MachineOperand *> Mul3Uses;
  for (const MachineOperand &MO : OrigNop.uses()) {
    if (MO.isReg() && MO.getReg() == NewReg3)
      Mul3Uses.push_back(&MO);
  }

  NextUseDistance DistNewMul1 =
      NUA.getShortestDistance(NewReg1, *NewMul1, Mul1Uses);
  EXPECT_EQ(DistNewMul1.getRawValue(), 2);

  NextUseDistance DistNewMul2 =
      NUA.getShortestDistance(NewReg2, *NewMul2, Mul2Uses);
  EXPECT_EQ(DistNewMul2.getRawValue(), 3);

  NextUseDistance DistAdd1 =
      NUA.getShortestDistance(Add1DefReg, Add1, Add1Uses);
  EXPECT_EQ(DistAdd1.getRawValue(), 4);

  NextUseDistance DistNewMul3 =
      NUA.getShortestDistance(NewReg3, *NewMul3, Mul3Uses);
  EXPECT_EQ(DistNewMul3.getRawValue(), 2);

  NextUseDistance DistAdd2 =
      NUA.getShortestDistance(Add2DefReg, Add2, Add2Uses);
  EXPECT_EQ(DistAdd2.getRawValue(), 2);
}
