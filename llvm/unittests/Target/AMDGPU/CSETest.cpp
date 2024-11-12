//===- llvm/unittests/Target/AMDGPU/CSETest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/CSEMIRBuilder.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(AMDGPU, TestCSEForRegisterClassOrBankAndLLT) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx1100", "");
  if (!TM)
    GTEST_SKIP();

  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);

  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  Mod.setDataLayout(TM->createDataLayout());

  auto *Type = FunctionType::get(Type::getVoidTy(Ctx), false);
  auto *F = Function::Create(Type, GlobalValue::ExternalLinkage, "Test", &Mod);

  MachineModuleInfo MMI(TM.get());
  auto MF =
      std::make_unique<MachineFunction>(*F, *TM, ST, MMI.getContext(), 42);
  auto *BB = MF->CreateMachineBasicBlock();
  MF->push_back(BB);

  MachineIRBuilder B(*MF);
  B.setMBB(*BB);

  LLT S32{LLT::scalar(32)};
  Register R0 = B.buildCopy(S32, Register(AMDGPU::SGPR0)).getReg(0);
  Register R1 = B.buildCopy(S32, Register(AMDGPU::SGPR1)).getReg(0);

  GISelCSEInfo CSEInfo;
  CSEInfo.setCSEConfig(std::make_unique<CSEConfigFull>());
  CSEInfo.analyze(*MF);
  B.setCSEInfo(&CSEInfo);
  CSEMIRBuilder CSEB(B.getState());
  CSEB.setInsertPt(B.getMBB(), B.getInsertPt());

  const RegisterBankInfo &RBI = *MF->getSubtarget().getRegBankInfo();

  const TargetRegisterClass *SgprRC = &AMDGPU::SReg_32RegClass;
  const RegisterBank *SgprRB = &RBI.getRegBank(AMDGPU::SGPRRegBankID);
  MachineRegisterInfo::VRegAttrs SgprRCS32 = {SgprRC, S32};
  MachineRegisterInfo::VRegAttrs SgprRBS32 = {SgprRB, S32};

  auto Add = CSEB.buildAdd(S32, R0, R1);
  auto AddRC = CSEB.buildInstr(AMDGPU::G_ADD, {SgprRCS32}, {R0, R1});
  auto AddRB = CSEB.buildInstr(AMDGPU::G_ADD, {{SgprRB, S32}}, {R0, R1});

  EXPECT_NE(Add, AddRC);
  EXPECT_NE(Add, AddRB);
  EXPECT_NE(AddRC, AddRB);

  auto Add_CSE = CSEB.buildAdd(S32, R0, R1);
  auto AddRC_CSE = CSEB.buildInstr(AMDGPU::G_ADD, {{SgprRC, S32}}, {R0, R1});
  auto AddRB_CSE = CSEB.buildInstr(AMDGPU::G_ADD, {SgprRBS32}, {R0, R1});

  EXPECT_EQ(Add, Add_CSE);
  EXPECT_EQ(AddRC, AddRC_CSE);
  EXPECT_EQ(AddRB, AddRB_CSE);
}
