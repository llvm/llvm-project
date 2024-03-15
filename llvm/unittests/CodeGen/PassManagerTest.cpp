//===- llvm/unittest/CodeGen/PassManager.cpp - PassManager tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Test that the various MachineFunction pass managers, adaptors, analyses, and
// analysis managers work.
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/Analysis.h"
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

class TestFunctionAnalysis : public AnalysisInfoMixin<TestFunctionAnalysis> {
public:
  struct Result {
    Result(int Count) : InstructionCount(Count) {}
    int InstructionCount;
  };

  /// The number of instructions in the Function.
  Result run(Function &F, FunctionAnalysisManager &AM) {
    return Result(F.getInstructionCount());
  }

private:
  friend AnalysisInfoMixin<TestFunctionAnalysis>;
  static AnalysisKey Key;
};

AnalysisKey TestFunctionAnalysis::Key;

class TestMachineFunctionAnalysis
    : public AnalysisInfoMixin<TestMachineFunctionAnalysis> {
public:
  struct Result {
    Result(int Count) : InstructionCount(Count) {}
    int InstructionCount;
  };

  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &AM) {
    FunctionAnalysisManager &FAM =
        AM.getResult<FunctionAnalysisManagerMachineFunctionProxy>(MF)
            .getManager();
    TestFunctionAnalysis::Result &FAR =
        FAM.getResult<TestFunctionAnalysis>(MF.getFunction());
    return FAR.InstructionCount;
  }

private:
  friend AnalysisInfoMixin<TestMachineFunctionAnalysis>;
  static AnalysisKey Key;
};

AnalysisKey TestMachineFunctionAnalysis::Key;

struct TestMachineFunctionPass : public PassInfoMixin<TestMachineFunctionPass> {
  TestMachineFunctionPass(int &Count, std::vector<int> &Counts)
      : Count(Count), Counts(Counts) {}

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM) {
    FunctionAnalysisManager &FAM =
        MFAM.getResult<FunctionAnalysisManagerMachineFunctionProxy>(MF)
            .getManager();
    TestFunctionAnalysis::Result &FAR =
        FAM.getResult<TestFunctionAnalysis>(MF.getFunction());
    Count += FAR.InstructionCount;

    TestMachineFunctionAnalysis::Result &MFAR =
        MFAM.getResult<TestMachineFunctionAnalysis>(MF);
    Count += MFAR.InstructionCount;

    Counts.push_back(Count);

    return PreservedAnalyses::none();
  }

  int &Count;
  std::vector<int> &Counts;
};

struct TestMachineModulePass : public PassInfoMixin<TestMachineModulePass> {
  TestMachineModulePass(int &Count, std::vector<int> &Counts)
      : Count(Count), Counts(Counts) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    MachineModuleInfo &MMI = MAM.getResult<MachineModuleAnalysis>(M).getMMI();
    FunctionAnalysisManager &FAM =
        MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    MachineFunctionAnalysisManager &MFAM =
        MAM.getResult<MachineFunctionAnalysisManagerModuleProxy>(M)
            .getManager();
    for (Function &F : M) {
      MachineFunction &MF = MMI.getOrCreateMachineFunction(F);
      Count += FAM.getResult<TestFunctionAnalysis>(F).InstructionCount;
      Count += MFAM.getResult<TestMachineFunctionAnalysis>(MF).InstructionCount;
    }
    Counts.push_back(Count);
    return PreservedAnalyses::all();
  }

  int &Count;
  std::vector<int> &Counts;
};

std::unique_ptr<Module> parseIR(LLVMContext &Context, const char *IR) {
  SMDiagnostic Err;
  return parseAssemblyString(IR, Err, Context);
}

class PassManagerTest : public ::testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;
  std::unique_ptr<TargetMachine> TM;

public:
  PassManagerTest()
      : M(parseIR(Context, "define void @f() {\n"
                           "entry:\n"
                           "  call void @g()\n"
                           "  call void @h()\n"
                           "  ret void\n"
                           "}\n"
                           "define void @g() {\n"
                           "  ret void\n"
                           "}\n"
                           "define void @h() {\n"
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

TEST_F(PassManagerTest, Basic) {
  if (!TM)
    GTEST_SKIP();

  LLVMTargetMachine *LLVMTM = static_cast<LLVMTargetMachine *>(TM.get());
  M->setDataLayout(TM->createDataLayout());

  MachineModuleInfo MMI(LLVMTM);

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  MachineFunctionAnalysisManager MFAM;
  PassBuilder PB(TM.get());
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerMachineFunctionAnalyses(MFAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM, &MFAM);

  FAM.registerPass([&] { return TestFunctionAnalysis(); });
  MAM.registerPass([&] { return MachineModuleAnalysis(MMI); });
  MFAM.registerPass([&] { return TestMachineFunctionAnalysis(); });

  int Count = 0;
  std::vector<int> Counts;

  ModulePassManager MPM;
  MachineFunctionPassManager MFPM;
  MPM.addPass(TestMachineModulePass(Count, Counts));
  MPM.addPass(createModuleToMachineFunctionPassAdaptor(
      TestMachineFunctionPass(Count, Counts)));
  MPM.addPass(TestMachineModulePass(Count, Counts));
  MFPM.addPass(TestMachineFunctionPass(Count, Counts));
  MPM.addPass(createModuleToMachineFunctionPassAdaptor(std::move(MFPM)));

  MPM.run(*M, MAM);

  EXPECT_EQ((std::vector<int>{10, 16, 18, 20, 30, 36, 38, 40}), Counts);
  EXPECT_EQ(40, Count);
}

} // namespace
