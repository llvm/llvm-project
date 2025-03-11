//===- llvm/unittests/Target/AMDGPU/ExecMayBeModifiedBeforeAnyUse.cpp -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(AMDGPU, IsOperandLegal) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx1200", "");
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

  auto E = BB->end();
  DebugLoc DL;
  const auto &TII = *ST.getInstrInfo();
  const auto &TRI = *ST.getRegisterInfo();
  auto &MRI = MF->getRegInfo();

  Register VReg = MRI.createVirtualRegister(&AMDGPU::CCR_SGPR_64RegClass);
  MachineInstr *Callee =
      BuildMI(*BB, E, DL, TII.get(AMDGPU::S_MOV_B64), VReg).addGlobalAddress(F);
  MachineInstr *Call =
      BuildMI(*BB, E, DL, TII.get(AMDGPU::SI_CALL), AMDGPU::SGPR30_SGPR31)
          .addReg(VReg)
          .addImm(0)
          .addRegMask(TRI.getCallPreservedMask(*MF, CallingConv::AMDGPU_Gfx))
          .addReg(AMDGPU::VGPR0, RegState::Implicit)
          .addReg(AMDGPU::VGPR1, RegState::Implicit);

  // This shouldn't crash.
  ASSERT_FALSE(TII.isOperandLegal(*Call, /*OpIdx=*/0, &Callee->getOperand(1)));
}
