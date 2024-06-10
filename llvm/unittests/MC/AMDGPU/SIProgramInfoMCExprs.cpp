//===- llvm/unittests/MC/AMDGPU/SIProgramInfoMCExprs.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUHSAMetadataStreamer.h"
#include "SIProgramInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

class SIProgramInfoMCExprsTest : public testing::Test {
protected:
  std::unique_ptr<LLVMTargetMachine> TM;
  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<Module> M;

  SIProgramInfo PI;

  static void SetUpTestSuite() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
  }

  SIProgramInfoMCExprsTest() {
    std::string Triple = "amdgcn-amd-amdhsa";
    std::string CPU = "gfx1010";
    std::string FS = "";

    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(Triple, Error);
    TargetOptions Options;

    TM.reset(static_cast<LLVMTargetMachine *>(TheTarget->createTargetMachine(
        Triple, CPU, FS, Options, std::nullopt, std::nullopt)));

    Ctx = std::make_unique<LLVMContext>();
    M = std::make_unique<Module>("Module", *Ctx);
    M->setDataLayout(TM->createDataLayout());
    auto *FType = FunctionType::get(Type::getVoidTy(*Ctx), false);
    auto *F = Function::Create(FType, GlobalValue::ExternalLinkage, "Test", *M);
    MMI = std::make_unique<MachineModuleInfo>(TM.get());

    auto *ST = TM->getSubtargetImpl(*F);

    MF = std::make_unique<MachineFunction>(*F, *TM, *ST, 1, *MMI);
    MF->initTargetMachineFunctionInfo(*ST);
    PI.reset(*MF.get());
  }
};

TEST_F(SIProgramInfoMCExprsTest, TestDeathHSAKernelEmit) {
  MCContext &Ctx = MF->getContext();
  MCSymbol *Sym = Ctx.getOrCreateSymbol("Unknown");
  PI.ScratchSize = MCSymbolRefExpr::create(Sym, Ctx);

  auto &Func = MF->getFunction();
  Func.setCallingConv(CallingConv::AMDGPU_KERNEL);
  AMDGPU::HSAMD::MetadataStreamerMsgPackV4 MD;

  testing::internal::CaptureStderr();
  MD.emitKernel(*MF, PI);
  std::string err = testing::internal::GetCapturedStderr();
  EXPECT_EQ(
      err, "<unknown>:0: error: could not resolve expression when required.\n");
  EXPECT_TRUE(Ctx.hadError());
}
