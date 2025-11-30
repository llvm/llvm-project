//===- IRTranslator.cpp - IRTranslator unit tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {
struct AArch64IRTranslatorTest : public ::testing::Test {
  LLVMContext C;

public:
  AArch64IRTranslatorTest() {}
  std::unique_ptr<TargetMachine> createTargetMachine() const {
    Triple TargetTriple("aarch64--");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    if (!T)
      return nullptr;

    TargetOptions Options;
    return std::unique_ptr<TargetMachine>(
        T->createTargetMachine(TargetTriple, "", "", Options, std::nullopt,
                               std::nullopt, CodeGenOptLevel::Aggressive));
  }

  std::unique_ptr<Module> parseIR(const char *IR) {
    SMDiagnostic Err;
    std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
    if (!Mod)
      Err.print("Test TargetIRTranslator", errs());
    return Mod;
  }
};
} // namespace

TEST_F(AArch64IRTranslatorTest, IRTranslateBfloat16) {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeGlobalISel(*Registry);

  std::unique_ptr<Module> M = parseIR(R"(
  define void @foo(ptr %p0) {
    %ptr1 = getelementptr bfloat, ptr %p0, i64 0
    %ptr2 = getelementptr bfloat, ptr %p0, i64 1
    %ptr3 = getelementptr bfloat, ptr %p0, i64 2
    %a = load bfloat, ptr %ptr1, align 2
    %b = load bfloat, ptr %ptr2, align 2
    %c = load bfloat, ptr %ptr3, align 2
    %mul = fmul bfloat %a, %b
    %res = fadd bfloat %mul, %c
    %ptr4 = getelementptr bfloat, ptr %p0, i64 3
    store bfloat %res, ptr %ptr4, align 2
    ret void
  }
  )");

  auto TM = createTargetMachine();
  M->setDataLayout(TM->createDataLayout());

  TM->setGlobalISel(true);
  TM->setGlobalISelExtendedLLT(true);
  TM->setGlobalISelAbort(GlobalISelAbortMode::DisableWithDiag);

  legacy::PassManager PM;
  TargetPassConfig *TPC(TM->createPassConfig(PM));

  MachineModuleInfoWrapperPass *MMIWP =
      new MachineModuleInfoWrapperPass(TM.get());
  PM.add(TPC);
  PM.add(MMIWP);
  PM.add(new IRTranslator());
  PM.run(*M);

  auto *MMI = &MMIWP->getMMI();
  Function *F = M->getFunction("foo");
  auto *MF = MMI->getMachineFunction(*F);
  MachineRegisterInfo &MRI = MF->getRegInfo();
  ASSERT_FALSE(MF->getProperties().hasProperty(
      llvm::MachineFunctionProperties::Property::FailedISel));
  for (auto &MI : MF->front()) {
    if (MI.getOpcode() == TargetOpcode::G_LOAD) {
      ASSERT_TRUE(MRI.getType(MI.getOperand(0).getReg()).isBFloat16());
    }

    if (MI.getOpcode() == TargetOpcode::G_FADD ||
        MI.getOpcode() == TargetOpcode::G_FMUL) {
      for (auto &Op : MI.operands()) {
        ASSERT_TRUE(MRI.getType(Op.getReg()).isBFloat16());
      }
    }
  }
  MMI->deleteMachineFunctionFor(*F);

  // Run again without extended LLT
  TM->setGlobalISelExtendedLLT(false);
  PM.run(*M);
  MF = MMI->getMachineFunction(*F);
  ASSERT_TRUE(MF->getProperties().hasProperty(
      llvm::MachineFunctionProperties::Property::FailedISel));
}
