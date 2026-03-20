//===- SPIRVGlobalRegistryTests.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVGlobalRegistry.h"
#include "SPIRVInstrInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

class SPIRVGlobalRegistryTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    LLVMInitializeSPIRVTargetInfo();
    LLVMInitializeSPIRVTarget();
    LLVMInitializeSPIRVTargetMC();
  }

  void SetUp() override {
    Triple TT("spirv64-unknown-unknown");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget(TT, Error);
    if (!T)
      GTEST_SKIP();
    TargetOptions Options;
    TM.reset(T->createTargetMachine(TT, "", "", Options, std::nullopt,
                                    std::nullopt));
    Ctx = std::make_unique<LLVMContext>();
    Mod = std::make_unique<Module>("M", *Ctx);
    Mod->setDataLayout(TM->createDataLayout());
    auto *F = Function::Create(FunctionType::get(Type::getVoidTy(*Ctx), false),
                               GlobalValue::ExternalLinkage, "f", *Mod);
    MMI = std::make_unique<MachineModuleInfo>(TM.get());
    MF = std::make_unique<MachineFunction>(*F, *TM, *TM->getSubtargetImpl(*F),
                                           MMI->getContext(), 0);
    MBB = MF->CreateMachineBasicBlock();
    MF->push_back(MBB);
  }

  SPIRVTypeInst makeTypeInstr(unsigned Opcode) {
    auto &TII =
        *static_cast<const SPIRVInstrInfo *>(MF->getSubtarget().getInstrInfo());
    Register Reg = MF->getRegInfo().createVirtualRegister(&SPIRV::TYPERegClass);
    return BuildMI(*MBB, MBB->end(), DebugLoc(), TII.get(Opcode))
        .addDef(Reg)
        .getInstr();
  }

  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<Module> Mod;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<MachineFunction> MF;
  MachineBasicBlock *MBB = nullptr;
};

TEST_F(SPIRVGlobalRegistryTest, IsAggregateType) {
  SPIRVGlobalRegistry GR(8);
  EXPECT_TRUE(GR.isAggregateType(makeTypeInstr(SPIRV::OpTypeStruct)));
  EXPECT_TRUE(GR.isAggregateType(makeTypeInstr(SPIRV::OpTypeArray)));
  EXPECT_FALSE(GR.isAggregateType(makeTypeInstr(SPIRV::OpTypeFloat)));
  EXPECT_FALSE(GR.isAggregateType(SPIRVTypeInst(nullptr)));
}
