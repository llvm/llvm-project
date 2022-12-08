//===- llvm/unittests/Target/AMDGPU/ExecMayBeModifiedBeforeAnyUse.cpp -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(AMDGPU, ExecMayBeModifiedBeforeAnyUse) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx906", "");
  if (!TM)
    return;

  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);

  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  Mod.setDataLayout(TM->createDataLayout());

  auto *Type = FunctionType::get(Type::getVoidTy(Ctx), false);
  auto *F = Function::Create(Type, GlobalValue::ExternalLinkage, "Test", &Mod);

  MachineModuleInfo MMI(TM.get());
  auto MF = std::make_unique<MachineFunction>(*F, *TM, ST, 42, MMI);
  auto *BB = MF->CreateMachineBasicBlock();
  MF->push_back(BB);

  auto E = BB->end();
  DebugLoc DL;
  const auto &TII = *ST.getInstrInfo();
  auto &MRI = MF->getRegInfo();

  // create machine IR
  Register R = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);

  MachineInstr *DefMI =
      BuildMI(*BB, E, DL, TII.get(AMDGPU::S_MOV_B32), R).addImm(42).getInstr();

  auto First =
      BuildMI(*BB, E, DL, TII.get(AMDGPU::S_NOP)).addReg(R, RegState::Implicit);

  BuildMI(*BB, E, DL, TII.get(AMDGPU::S_NOP)).addReg(R, RegState::Implicit);

  // this violates the continuous sequence of R's uses for the first S_NOP
  First.addReg(R, RegState::Implicit);

#ifdef DEBUG_THIS_TEST
  MF->dump();
  MRI.dumpUses(R);
#endif

  // make sure execMayBeModifiedBeforeAnyUse doesn't crash
  ASSERT_FALSE(execMayBeModifiedBeforeAnyUse(MRI, R, *DefMI));
}
