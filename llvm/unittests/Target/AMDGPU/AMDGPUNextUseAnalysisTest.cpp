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
  MachineInstr &Nop1 = *It;

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

  Register Add2DefReg = Add2.getOperand(0).getReg();

  // Add a third multiplication before the second ADD
  Register NewReg3 =
      MF.getRegInfo().createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  MachineInstr *NewMul3 = BuildMI(MBB, Add2.getIterator(), DebugLoc(),
                                  TII.get(AMDGPU::V_MUL_U32_U24_e64), NewReg3)
                              .addImm(0)
                              .addImm(5)
                              .addImm(6)
                              .getInstr();

  BuildMI(MBB, MBB.back().getIterator(), DebugLoc(), TII.get(AMDGPU::S_NOP))
      .addImm(0)
      .addReg(Add1DefReg, RegState::Implicit)
      .addReg(Add2DefReg, RegState::Implicit);

  // Add the destination register of the third multiplication as an input
  // operand of the first NOP.
  for (MachineOperand &MO : Nop1.operands()) {
    if (MO.isReg() && MO.isUse())
      MO.setReg(NewReg3);
  }

  // Instruction order after the insertion of the new instructions.
  // %0:vgpr_32 = COPY $vgpr0
  // %1:vgpr_32 = V_MOV_B32_e32 1, implicit $exec
  // %2:vgpr_32 = V_MOV_B32_e32 2, implicit $exec
  // %5:vgpr_32 = V_MUL_U32_U24_e64 0, 1, 2, implicit $exec
  // %6:vgpr_32 = V_MUL_U32_U24_e64 0, 3, 4, implicit $exec
  // %3:vgpr_32 = V_ADD_U32_e32 %5:vgpr_32, %1:vgpr_32, implicit $exec
  // %7:vgpr_32 = V_MUL_U32_U24_e64 0, 5, 6, implicit $exec
  // %4:vgpr_32 = V_ADD_U32_e32 %6:vgpr_32, %2:vgpr_32, implicit $exec
  // S_NOP 0, implicit %7:vgpr_32
  // S_NOP 0, implicit %3:vgpr_32, implicit %4:vgpr_32
  // S_ENDPGM 0

  MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<const MachineOperand *> Add1Uses;
  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Add1DefReg))
    for (const MachineOperand &MO : UseMI.uses())
      if (MO.isReg() && MO.getReg() == Add1DefReg)
        Add1Uses.push_back(&MO);

  SmallVector<const MachineOperand *> Add2Uses;
  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Add2DefReg))
    for (const MachineOperand &MO : UseMI.uses())
      if (MO.isReg() && MO.getReg() == Add2DefReg)
        Add2Uses.push_back(&MO);

  SmallVector<const MachineOperand *> Mul1Uses;
  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(NewReg1))
    for (const MachineOperand &MO : UseMI.uses())
      if (MO.isReg() && MO.getReg() == NewReg1)
        Mul1Uses.push_back(&MO);

  SmallVector<const MachineOperand *> Mul2Uses;
  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(NewReg2))
    for (const MachineOperand &MO : UseMI.uses())
      if (MO.isReg() && MO.getReg() == NewReg2)
        Mul2Uses.push_back(&MO);

  SmallVector<const MachineOperand *> Mul3Uses;
  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(NewReg3))
    for (const MachineOperand &MO : UseMI.uses())
      if (MO.isReg() && MO.getReg() == NewReg3)
        Mul3Uses.push_back(&MO);

  NextUseDistance DistNewMul1 =
      NUA.getShortestDistance(NewReg1, *NewMul1, Mul1Uses);
  EXPECT_EQ(DistNewMul1.getRawValue(), 0);

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

  // Add a fourth multiplication after the second ADD.
  Register NewReg4 =
      MF.getRegInfo().createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  MachineInstr *NewMul4 = BuildMI(MBB, Nop1.getIterator(), DebugLoc(),
                                  TII.get(AMDGPU::V_MUL_U32_U24_e64), NewReg4)
                              .addImm(0)
                              .addImm(7)
                              .addImm(8)
                              .getInstr();

  [[maybe_unused]] MachineInstr *Nop3 =
      BuildMI(MBB, MBB.back().getIterator(), DebugLoc(), TII.get(AMDGPU::S_NOP))
          .addImm(0)
          .addReg(NewReg4, RegState::Implicit)
          .getInstr();

  // Instructions after emitting more instructions.
  // %0:vgpr_32 = COPY $vgpr0
  // %1:vgpr_32 = V_MOV_B32_e32 1, implicit $exec
  // %2:vgpr_32 = V_MOV_B32_e32 2, implicit $exec
  // %5:vgpr_32 = V_MUL_U32_U24_e64 0, 1, 2, implicit $exec
  // %6:vgpr_32 = V_MUL_U32_U24_e64 0, 3, 4, implicit $exec
  // %3:vgpr_32 = V_ADD_U32_e32 %5:vgpr_32, %1:vgpr_32, implicit $exec
  // %7:vgpr_32 = V_MUL_U32_U24_e64 0, 5, 6, implicit $exec
  // %4:vgpr_32 = V_ADD_U32_e32 %6:vgpr_32, %2:vgpr_32, implicit $exec
  // %8:vgpr_32 = V_MUL_U32_U24_e64 0, 7, 8, implicit $exec
  // S_NOP 0, implicit %7:vgpr_32
  // S_NOP 0, implicit %3:vgpr_32, implicit %4:vgpr_32
  // S_NOP 0, implicit %8:vgpr_32
  // S_ENDPGM 0

  SmallVector<const MachineOperand *> Mul4Uses;
  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(NewReg4))
    for (const MachineOperand &MO : UseMI.uses())
      if (MO.isReg() && MO.getReg() == NewReg4)
        Mul4Uses.push_back(&MO);

  NextUseDistance DistAdd1b =
      NUA.getShortestDistance(Add1DefReg, Add1, Add1Uses);
  EXPECT_EQ(DistAdd1b.getRawValue(), 4);

  NextUseDistance DistAdd2b =
      NUA.getShortestDistance(Add2DefReg, Add2, Add2Uses);
  EXPECT_EQ(DistAdd2b.getRawValue(), 2);

  NextUseDistance DistNewMul4 =
      NUA.getShortestDistance(NewReg4, *NewMul4, Mul4Uses);
  EXPECT_EQ(DistNewMul4.getRawValue(), 3);
}
