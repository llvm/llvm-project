//===- llvm/unittest/CodeGen/PassManager.cpp - PassManager tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class TestFunctionAnalysis : public AnalysisInfoMixin<TestFunctionAnalysis> {
public:
  struct Result {
    Result(int Count) : InstructionCount(Count) {}
    int InstructionCount;
  };

  /// Run the analysis pass over the function and return a result.
  Result run(Function &F, FunctionAnalysisManager &AM) {
    int Count = 0;
    for (Function::iterator BBI = F.begin(), BBE = F.end(); BBI != BBE; ++BBI)
      for (BasicBlock::iterator II = BBI->begin(), IE = BBI->end(); II != IE;
           ++II)
        ++Count;
    return Result(Count);
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

  /// Run the analysis pass over the machine function and return a result.
  Result run(MachineFunction &MF, MachineFunctionAnalysisManager::Base &AM) {
    auto &MFAM = static_cast<MachineFunctionAnalysisManager &>(AM);
    // Query function analysis result.
    TestFunctionAnalysis::Result &FAR =
        MFAM.getResult<TestFunctionAnalysis>(MF.getFunction());
    // + 5
    return FAR.InstructionCount;
  }

private:
  friend AnalysisInfoMixin<TestMachineFunctionAnalysis>;
  static AnalysisKey Key;
};

AnalysisKey TestMachineFunctionAnalysis::Key;

const std::string DoInitErrMsg = "doInitialization failed";
const std::string DoFinalErrMsg = "doFinalization failed";

struct TestMachineFunctionPass : public PassInfoMixin<TestMachineFunctionPass> {
  TestMachineFunctionPass(int &Count, std::vector<int> &BeforeInitialization,
                          std::vector<int> &BeforeFinalization,
                          std::vector<int> &MachineFunctionPassCount)
      : Count(Count), BeforeInitialization(BeforeInitialization),
        BeforeFinalization(BeforeFinalization),
        MachineFunctionPassCount(MachineFunctionPassCount) {}

  Error doInitialization(Module &M, MachineFunctionAnalysisManager &MFAM) {
    // Force doInitialization fail by starting with big `Count`.
    if (Count > 10000)
      return make_error<StringError>(DoInitErrMsg, inconvertibleErrorCode());

    // + 1
    ++Count;
    BeforeInitialization.push_back(Count);
    return Error::success();
  }
  Error doFinalization(Module &M, MachineFunctionAnalysisManager &MFAM) {
    // Force doFinalization fail by starting with big `Count`.
    if (Count > 1000)
      return make_error<StringError>(DoFinalErrMsg, inconvertibleErrorCode());

    // + 1
    ++Count;
    BeforeFinalization.push_back(Count);
    return Error::success();
  }

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM) {
    // Query function analysis result.
    TestFunctionAnalysis::Result &FAR =
        MFAM.getResult<TestFunctionAnalysis>(MF.getFunction());
    // 3 + 1 + 1 = 5
    Count += FAR.InstructionCount;

    // Query module analysis result.
    MachineModuleInfo &MMI =
        MFAM.getResult<MachineModuleAnalysis>(*MF.getFunction().getParent());
    // 1 + 1 + 1 = 3
    Count += (MMI.getModule() == MF.getFunction().getParent());

    // Query machine function analysis result.
    TestMachineFunctionAnalysis::Result &MFAR =
        MFAM.getResult<TestMachineFunctionAnalysis>(MF);
    // 3 + 1 + 1 = 5
    Count += MFAR.InstructionCount;

    MachineFunctionPassCount.push_back(Count);

    return PreservedAnalyses::none();
  }

  int &Count;
  std::vector<int> &BeforeInitialization;
  std::vector<int> &BeforeFinalization;
  std::vector<int> &MachineFunctionPassCount;
};

struct TestMachineModulePass : public PassInfoMixin<TestMachineModulePass> {
  TestMachineModulePass(int &Count, std::vector<int> &MachineModulePassCount)
      : Count(Count), MachineModulePassCount(MachineModulePassCount) {}

  Error run(Module &M, MachineFunctionAnalysisManager &MFAM) {
    MachineModuleInfo &MMI = MFAM.getResult<MachineModuleAnalysis>(M);
    // + 1
    Count += (MMI.getModule() == &M);
    MachineModulePassCount.push_back(Count);
    return Error::success();
  }

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &AM) {
    llvm_unreachable(
        "This should never be reached because this is machine module pass");
  }

  int &Count;
  std::vector<int> &MachineModulePassCount;
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
    TM.reset(TheTarget->createTargetMachine(TripleName, "", "",
                                            Options, None));
  }
};

TEST_F(PassManagerTest, Basic) {
  if (!TM)
    GTEST_SKIP();

  LLVMTargetMachine *LLVMTM = static_cast<LLVMTargetMachine *>(TM.get());
  M->setDataLayout(TM->createDataLayout());

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB(TM.get());
  PB.registerModuleAnalyses(MAM);
  PB.registerFunctionAnalyses(FAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  FAM.registerPass([&] { return TestFunctionAnalysis(); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  MAM.registerPass([&] { return MachineModuleAnalysis(LLVMTM); });
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });

  MachineFunctionAnalysisManager MFAM;
  {
    // Test move assignment.
    MachineFunctionAnalysisManager NestedMFAM(FAM, MAM);
    NestedMFAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    NestedMFAM.registerPass([&] { return TestMachineFunctionAnalysis(); });
    MFAM = std::move(NestedMFAM);
  }

  int Count = 0;
  std::vector<int> BeforeInitialization[2];
  std::vector<int> BeforeFinalization[2];
  std::vector<int> TestMachineFunctionCount[2];
  std::vector<int> TestMachineModuleCount[2];

  MachineFunctionPassManager MFPM;
  {
    // Test move assignment.
    MachineFunctionPassManager NestedMFPM;
    NestedMFPM.addPass(TestMachineModulePass(Count, TestMachineModuleCount[0]));
    NestedMFPM.addPass(TestMachineFunctionPass(Count, BeforeInitialization[0],
                                               BeforeFinalization[0],
                                               TestMachineFunctionCount[0]));
    NestedMFPM.addPass(TestMachineModulePass(Count, TestMachineModuleCount[1]));
    NestedMFPM.addPass(TestMachineFunctionPass(Count, BeforeInitialization[1],
                                               BeforeFinalization[1],
                                               TestMachineFunctionCount[1]));
    MFPM = std::move(NestedMFPM);
  }

  ASSERT_FALSE(errorToBool(MFPM.run(*M, MFAM)));

  // Check first machine module pass
  EXPECT_EQ(1u, TestMachineModuleCount[0].size());
  EXPECT_EQ(3, TestMachineModuleCount[0][0]);

  // Check first machine function pass
  EXPECT_EQ(1u, BeforeInitialization[0].size());
  EXPECT_EQ(1, BeforeInitialization[0][0]);
  EXPECT_EQ(3u, TestMachineFunctionCount[0].size());
  EXPECT_EQ(10, TestMachineFunctionCount[0][0]);
  EXPECT_EQ(13, TestMachineFunctionCount[0][1]);
  EXPECT_EQ(16, TestMachineFunctionCount[0][2]);
  EXPECT_EQ(1u, BeforeFinalization[0].size());
  EXPECT_EQ(31, BeforeFinalization[0][0]);

  // Check second machine module pass
  EXPECT_EQ(1u, TestMachineModuleCount[1].size());
  EXPECT_EQ(17, TestMachineModuleCount[1][0]);

  // Check second machine function pass
  EXPECT_EQ(1u, BeforeInitialization[1].size());
  EXPECT_EQ(2, BeforeInitialization[1][0]);
  EXPECT_EQ(3u, TestMachineFunctionCount[1].size());
  EXPECT_EQ(24, TestMachineFunctionCount[1][0]);
  EXPECT_EQ(27, TestMachineFunctionCount[1][1]);
  EXPECT_EQ(30, TestMachineFunctionCount[1][2]);
  EXPECT_EQ(1u, BeforeFinalization[1].size());
  EXPECT_EQ(32, BeforeFinalization[1][0]);

  EXPECT_EQ(32, Count);

  // doInitialization returns error
  Count = 10000;
  MFPM.addPass(TestMachineFunctionPass(Count, BeforeInitialization[1],
                                       BeforeFinalization[1],
                                       TestMachineFunctionCount[1]));
  std::string Message;
  llvm::handleAllErrors(MFPM.run(*M, MFAM), [&](llvm::StringError &Error) {
    Message = Error.getMessage();
  });
  EXPECT_EQ(Message, DoInitErrMsg);

  // doFinalization returns error
  Count = 1000;
  MFPM.addPass(TestMachineFunctionPass(Count, BeforeInitialization[1],
                                       BeforeFinalization[1],
                                       TestMachineFunctionCount[1]));
  llvm::handleAllErrors(MFPM.run(*M, MFAM), [&](llvm::StringError &Error) {
    Message = Error.getMessage();
  });
  EXPECT_EQ(Message, DoFinalErrMsg);
}

} // namespace
