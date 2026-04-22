//===- SparseLiveVariablesTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SparseLiveVariables.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
#include "MFCommon.inc"

TEST(SparseLiveVariablesTest, APITests) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto MBB = MF->CreateMachineBasicBlock();
  MF->push_back(MBB);

  MCInstrDesc MCID = {100, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0};

  MCRegisterClass MRC{
      0, 0, 0, 0, 0, 0, 0, 0, /*Allocatable=*/true, /*BaseClass=*/true};
  TargetRegisterClass RC{&MRC, 0, 0, {}, 0, 0, 0, 0, 0, 0, 0, 0};
  MachineRegisterInfo &MRI = MF->getRegInfo();

  Register Reg0 = MRI.createVirtualRegister(&RC);
  Register Reg1 = MRI.createVirtualRegister(&RC);
  Register Reg2 = MRI.createVirtualRegister(&RC);

  // MI1: %1 = OP %0
  MachineInstr *MI1 = MF->CreateMachineInstr(MCID, DebugLoc());
  MI1->addOperand(*MF, MachineOperand::CreateReg(Reg1, /*isDef*/ true));
  MI1->addOperand(*MF, MachineOperand::CreateReg(Reg0, /*isDef*/ false));
  MBB->insert(MBB->end(), MI1);

  // MI2: %2 = OP %1
  MachineInstr *MI2 = MF->CreateMachineInstr(MCID, DebugLoc());
  MI2->addOperand(*MF, MachineOperand::CreateReg(Reg2, /*isDef*/ true));
  MI2->addOperand(*MF, MachineOperand::CreateReg(Reg1, /*isDef*/ false));
  MBB->insert(MBB->end(), MI2);

  // Run SparseLiveVariables analysis
  SparseLiveVariables LV;
  LV.analyze(*MF);

  // Block Level Sets Check
  const SparseBitVector<> &LiveIn = LV.getLiveInSet(MBB);
  const SparseBitVector<> &LiveOut = LV.getLiveOutSet(MBB);

  // Reg0 is used before being defined -> must be LiveIn
  EXPECT_TRUE(LiveIn.test(Reg0.id()));
  EXPECT_FALSE(LiveIn.test(Reg1.id()));
  EXPECT_FALSE(LiveIn.test(Reg2.id()));

  // No registers are live out (MBB is the only block)
  EXPECT_FALSE(LiveOut.test(Reg0.id()));
  EXPECT_FALSE(LiveOut.test(Reg1.id()));
  EXPECT_FALSE(LiveOut.test(Reg2.id()));

  // Test updateLiveIns (won't add virtual registers, but shouldn't crash)
  LV.updateLiveIns(*MF);
  EXPECT_TRUE(MBB->livein_empty());
}

TEST(SparseLiveVariablesTest, IncrementalLivenessUpdates) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto MBB1 = MF->CreateMachineBasicBlock();
  auto MBB2 = MF->CreateMachineBasicBlock();
  MF->push_back(MBB1);
  MF->push_back(MBB2);
  MBB1->addSuccessor(MBB2);

  MCInstrDesc MCID = {100, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0};

  MCRegisterClass MRC{
      0, 0, 0, 0, 0, 0, 0, 0, /*Allocatable=*/true, /*BaseClass=*/true};
  TargetRegisterClass RC{&MRC, 0, 0, {}, 0, 0, 0, 0, 0, 0, 0, 0};
  MachineRegisterInfo &MRI = MF->getRegInfo();

  Register Reg0 = MRI.createVirtualRegister(&RC);

  // MI1 in MBB1: %0 = OP
  MachineInstr *MI1 = MF->CreateMachineInstr(MCID, DebugLoc());
  MI1->addOperand(*MF, MachineOperand::CreateReg(Reg0, /*isDef*/ true));
  MBB1->insert(MBB1->end(), MI1);

  // Run SparseLiveVariables analysis
  SparseLiveVariables LV;
  LV.analyze(*MF);

  // Initially Reg0 is NOT live-out of MBB1 and NOT live-in to MBB2
  EXPECT_FALSE(LV.getLiveOutSet(MBB1).test(Reg0.id()));
  EXPECT_FALSE(LV.getLiveInSet(MBB2).test(Reg0.id()));

  // Now, dynamically add a use of Reg0 in MBB2!
  MachineInstr *MI2 = MF->CreateMachineInstr(MCID, DebugLoc());
  Register Reg1 = MRI.createVirtualRegister(&RC);
  MI2->addOperand(*MF, MachineOperand::CreateReg(Reg1, /*isDef*/ true));
  MI2->addOperand(*MF, MachineOperand::CreateReg(Reg0, /*isDef*/ false));
  MBB2->insert(MBB2->end(), MI2);

  // Inform LV to incrementally update based on MI2
  LV.addInstruction(*MI2, MBB2);

  // Reg0 should now cleanly flow from MBB1 to MBB2
  EXPECT_TRUE(LV.getLiveOutSet(MBB1).test(Reg0.id()));
  EXPECT_TRUE(LV.getLiveInSet(MBB2).test(Reg0.id()));

  // Now dynamically remove MI2 and update
  LV.removeInstruction(*MI2);
  MBB2->erase(MI2);

  // The phantom liveness should disappear cleanly!
  EXPECT_FALSE(LV.getLiveOutSet(MBB1).test(Reg0.id()));
  EXPECT_FALSE(LV.getLiveInSet(MBB2).test(Reg0.id()));
}

TEST(SparseLiveVariablesTest, RecomputeRegisterLiveness) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto MBB1 = MF->CreateMachineBasicBlock();
  auto MBB2 = MF->CreateMachineBasicBlock();
  MF->push_back(MBB1);
  MF->push_back(MBB2);
  MBB1->addSuccessor(MBB2);

  MCInstrDesc MCID = {100, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0};

  MCRegisterClass MRC{
      0, 0, 0, 0, 0, 0, 0, 0, /*Allocatable=*/true, /*BaseClass=*/true};
  TargetRegisterClass RC{&MRC, 0, 0, {}, 0, 0, 0, 0, 0, 0, 0, 0};
  MachineRegisterInfo &MRI = MF->getRegInfo();

  Register Reg0 = MRI.createVirtualRegister(&RC);
  Register Reg1 = MRI.createVirtualRegister(&RC);

  // MI1 in MBB1: %0 = OP
  MachineInstr *MI1 = MF->CreateMachineInstr(MCID, DebugLoc());
  MI1->addOperand(*MF, MachineOperand::CreateReg(Reg0, /*isDef*/ true));
  MBB1->insert(MBB1->end(), MI1);

  // MI2 in MBB2: %1 = OP %0
  MachineInstr *MI2 = MF->CreateMachineInstr(MCID, DebugLoc());
  MI2->addOperand(*MF, MachineOperand::CreateReg(Reg1, /*isDef*/ true));
  MI2->addOperand(*MF, MachineOperand::CreateReg(Reg0, /*isDef*/ false));
  MBB2->insert(MBB2->end(), MI2);

  SparseLiveVariables LV;
  LV.analyze(*MF);

  EXPECT_TRUE(LV.getLiveOutSet(MBB1).test(Reg0.id()));
  EXPECT_TRUE(LV.getLiveInSet(MBB2).test(Reg0.id()));

  // Manually remove MI2 from MBB2 without informing LV using removeInstruction
  MBB2->erase(MI2);

  // LV still thinks Reg0 is live
  EXPECT_TRUE(LV.getLiveOutSet(MBB1).test(Reg0.id()));
  EXPECT_TRUE(LV.getLiveInSet(MBB2).test(Reg0.id()));

  // Recompute liveness for Reg0
  LV.recomputeRegisterLiveness(Reg0);

  // Now it should be updated
  EXPECT_FALSE(LV.getLiveOutSet(MBB1).test(Reg0.id()));
  EXPECT_FALSE(LV.getLiveInSet(MBB2).test(Reg0.id()));
}

TEST(SparseLiveVariablesTest, HandleMoveUpdatesLiveness) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto MBB1 = MF->CreateMachineBasicBlock();
  auto MBB2 = MF->CreateMachineBasicBlock();
  auto MBB3 = MF->CreateMachineBasicBlock();
  MF->push_back(MBB1);
  MF->push_back(MBB2);
  MF->push_back(MBB3);
  MBB1->addSuccessor(MBB2);
  MBB2->addSuccessor(MBB3);

  MCInstrDesc MCID = {100, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0};

  MCRegisterClass MRC{
      0, 0, 0, 0, 0, 0, 0, 0, /*Allocatable=*/true, /*BaseClass=*/true};
  TargetRegisterClass RC{&MRC, 0, 0, {}, 0, 0, 0, 0, 0, 0, 0, 0};
  MachineRegisterInfo &MRI = MF->getRegInfo();

  Register Reg0 = MRI.createVirtualRegister(&RC);
  Register Reg1 = MRI.createVirtualRegister(&RC);

  // MI1 in MBB1: %0 = OP
  MachineInstr *MI1 = MF->CreateMachineInstr(MCID, DebugLoc());
  MI1->addOperand(*MF, MachineOperand::CreateReg(Reg0, /*isDef*/ true));
  MBB1->insert(MBB1->end(), MI1);

  // MI2 in MBB2: %1 = OP %0
  MachineInstr *MI2 = MF->CreateMachineInstr(MCID, DebugLoc());
  MI2->addOperand(*MF, MachineOperand::CreateReg(Reg1, /*isDef*/ true));
  MI2->addOperand(*MF, MachineOperand::CreateReg(Reg0, /*isDef*/ false));
  MBB2->insert(MBB2->end(), MI2);

  SparseLiveVariables LV;
  LV.analyze(*MF);

  EXPECT_TRUE(LV.getLiveOutSet(MBB1).test(Reg0.id()));
  EXPECT_TRUE(LV.getLiveInSet(MBB2).test(Reg0.id()));
  EXPECT_FALSE(LV.getLiveOutSet(MBB2).test(Reg0.id()));
  EXPECT_FALSE(LV.getLiveInSet(MBB3).test(Reg0.id()));

  // Move MI2 from MBB2 to MBB3
  MBB3->splice(MBB3->end(), MBB2, MI2);
  LV.handleMove(*MI2, MBB2, MBB3);

  // Liveness should now flow through MBB2 to MBB3
  EXPECT_TRUE(LV.getLiveOutSet(MBB1).test(Reg0.id()));
  EXPECT_TRUE(LV.getLiveInSet(MBB2).test(Reg0.id()));
  EXPECT_TRUE(LV.getLiveOutSet(MBB2).test(Reg0.id()));
  EXPECT_TRUE(LV.getLiveInSet(MBB3).test(Reg0.id()));
}

TEST(SparseLiveVariablesTest, PhiNodeLiveness) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto MBB0 = MF->CreateMachineBasicBlock();
  auto MBB1 = MF->CreateMachineBasicBlock();
  auto MBB2 = MF->CreateMachineBasicBlock();
  auto MBB3 = MF->CreateMachineBasicBlock();
  MF->push_back(MBB0);
  MF->push_back(MBB1);
  MF->push_back(MBB2);
  MF->push_back(MBB3);
  MBB0->addSuccessor(MBB1);
  MBB0->addSuccessor(MBB2);
  MBB1->addSuccessor(MBB3);
  MBB2->addSuccessor(MBB3);

  MCInstrDesc MCID = {100, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0};

  MCRegisterClass MRC{
      0, 0, 0, 0, 0, 0, 0, 0, /*Allocatable=*/true, /*BaseClass=*/true};
  TargetRegisterClass RC{&MRC, 0, 0, {}, 0, 0, 0, 0, 0, 0, 0, 0};
  MachineRegisterInfo &MRI = MF->getRegInfo();

  Register Reg1 = MRI.createVirtualRegister(&RC);
  Register Reg2 = MRI.createVirtualRegister(&RC);
  Register Reg3 = MRI.createVirtualRegister(&RC);

  // MBB1: %1 = OP
  MachineInstr *MI1 = MF->CreateMachineInstr(MCID, DebugLoc());
  MI1->addOperand(*MF, MachineOperand::CreateReg(Reg1, /*isDef*/ true));
  MBB1->insert(MBB1->end(), MI1);

  // MBB2: %2 = OP
  MachineInstr *MI2 = MF->CreateMachineInstr(MCID, DebugLoc());
  MI2->addOperand(*MF, MachineOperand::CreateReg(Reg2, /*isDef*/ true));
  MBB2->insert(MBB2->end(), MI2);

  // MBB3: %3 = PHI %1, MBB1, %2, MBB2
  MCInstrDesc PHIMCID = {TargetOpcode::PHI, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  MachineInstr *MI3 = MF->CreateMachineInstr(PHIMCID, DebugLoc());
  MI3->addOperand(*MF, MachineOperand::CreateReg(Reg3, /*isDef*/ true));
  MI3->addOperand(*MF, MachineOperand::CreateReg(Reg1, /*isDef*/ false));
  MI3->addOperand(*MF, MachineOperand::CreateMBB(MBB1));
  MI3->addOperand(*MF, MachineOperand::CreateReg(Reg2, /*isDef*/ false));
  MI3->addOperand(*MF, MachineOperand::CreateMBB(MBB2));
  MBB3->insert(MBB3->end(), MI3);

  SparseLiveVariables LV;
  LV.analyze(*MF);

  // For PHI nodes, Reg1 and Reg2 should be LiveOut of their respective
  // predecessors, but neither should be considered LiveIn to MBB3.
  EXPECT_TRUE(LV.getLiveOutSet(MBB1).test(Reg1.id()));
  EXPECT_FALSE(LV.getLiveInSet(MBB3).test(Reg1.id()));

  EXPECT_TRUE(LV.getLiveOutSet(MBB2).test(Reg2.id()));
  EXPECT_FALSE(LV.getLiveInSet(MBB3).test(Reg2.id()));
}

TEST(SparseLiveVariablesTest, TrackerAPI) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto MBB = MF->CreateMachineBasicBlock();
  MF->push_back(MBB);

  MCInstrDesc MCID = {100, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0};

  MCRegisterClass MRC{
      0, 0, 0, 0, 0, 0, 0, 0, /*Allocatable=*/true, /*BaseClass=*/true};
  TargetRegisterClass RC{&MRC, 0, 0, {}, 0, 0, 0, 0, 0, 0, 0, 0};
  MachineRegisterInfo &MRI = MF->getRegInfo();

  Register Reg0 = MRI.createVirtualRegister(&RC);
  Register Reg1 = MRI.createVirtualRegister(&RC);

  // MI1: %0 = OP
  MachineInstr *MI1 = MF->CreateMachineInstr(MCID, DebugLoc());
  MI1->addOperand(*MF, MachineOperand::CreateReg(Reg0, /*isDef*/ true));
  MBB->insert(MBB->end(), MI1);

  // MI2: %1 = OP %0
  MachineInstr *MI2 = MF->CreateMachineInstr(MCID, DebugLoc());
  MI2->addOperand(*MF, MachineOperand::CreateReg(Reg1, /*isDef*/ true));
  MI2->addOperand(*MF, MachineOperand::CreateReg(Reg0, /*isDef*/ false));
  MBB->insert(MBB->end(), MI2);

  SparseBitVector<> LiveOut;
  // Initialize Tracker with empty LiveOut
  SparseLiveVariables::LivenessTracker Tracker(LiveOut, &MRI);

  EXPECT_FALSE(Tracker.isLive(Reg0));
  EXPECT_FALSE(Tracker.isLive(Reg1));

  Tracker.stepBackward(*MI2);

  // After stepping backward over MI2, Reg1 is killed (def'd), and Reg0 becomes
  // live (used)
  EXPECT_TRUE(Tracker.isLive(Reg0));
  EXPECT_FALSE(Tracker.isLive(Reg1));

  Tracker.stepBackward(*MI1);

  // After stepping backward over MI1, Reg0 is killed (def'd).
  EXPECT_FALSE(Tracker.isLive(Reg0));
  EXPECT_FALSE(Tracker.isLive(Reg1));
}

} // end namespace
