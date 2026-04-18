//===- SparseLiveVariablesTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SparseLiveVariables.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
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
  LV.runOnMachineFunction(*MF);



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
}

} // end namespace
