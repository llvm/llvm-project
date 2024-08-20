//===- llvm/unittest/CodeGen/PassManager.cpp - PassManager tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Test MachineModuleInfo.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

std::unique_ptr<Module> parseIR(LLVMContext &Context, const char *IR) {
  SMDiagnostic Err;
  return parseAssemblyString(IR, Err, Context);
}

class MachineModuleInfoTest: public ::testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;
  std::unique_ptr<TargetMachine> TM;

public:
  MachineModuleInfoTest()
      : M(parseIR(Context, "define void @f() {\n"
                           "entry:\n"
                           "  call void @g()\n"
                           "  ret void\n"
                           "}\n"
                           "define void @g() {\n"
                           "  ret void\n"
                           "}\n")) {
    // MachineModuleAnalysis needs a TargetMachine instance.
    llvm::InitializeAllTargets();

    std::string TripleName = Triple::normalize(sys::getDefaultTargetTriple());
    std::string Error;
    const Target *TheTarget =
        TargetRegistry::lookupTarget(TripleName, Error);
    if (!TheTarget)
      return;

    TargetOptions Options;
    TM.reset(TheTarget->createTargetMachine(TripleName, "", "", Options,
                                            std::nullopt));
  }
};

TEST_F(MachineModuleInfoTest, MachineModuleInfoMoveConstructorMovesMCContext) {
  if (!TM)
    GTEST_SKIP();

  LLVMTargetMachine *LLVMTM = static_cast<LLVMTargetMachine *>(TM.get());
  M->setDataLayout(TM->createDataLayout());

  MachineModuleInfo MMI(LLVMTM);

  MCContext* OriginalCtx = &MMI.getContext();

  MachineModuleInfo MovedMMI(std::move(MMI));
  MCContext* MovedCtx = &MovedMMI.getContext();

  EXPECT_EQ(OriginalCtx, MovedCtx);
}

} // end namespace
