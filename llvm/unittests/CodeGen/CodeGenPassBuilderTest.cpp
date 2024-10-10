//===- llvm/unittest/CodeGen/CodeGenPassBuilderTest.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetMachine.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

class TestTargetMachine : public LLVMTargetMachine {
public:
  TestTargetMachine()
      : LLVMTargetMachine(Target(), "", Triple(""), "", "", TargetOptions(),
                          Reloc::Static, CodeModel::Small,
                          CodeGenOptLevel::Default) {}
};

TestTargetMachine &createTargetMachine() {
  static TestTargetMachine TM;
  return TM;
}

struct DisabledMachineFunctionPass
    : public PassInfoMixin<DisabledMachineFunctionPass> {
  PreservedAnalyses run(MachineFunction &, MachineFunctionAnalysisManager &) {
    return PreservedAnalyses::all();
  }
};

struct ReplacedMachineFunctionPass
    : public PassInfoMixin<ReplacedMachineFunctionPass> {
  PreservedAnalyses run(MachineFunction &, MachineFunctionAnalysisManager &) {
    return PreservedAnalyses::all();
  }
};

class TestCodeGenPassBuilder
    : public CodeGenPassBuilder<TestCodeGenPassBuilder, TestTargetMachine> {

public:
  explicit TestCodeGenPassBuilder(PassBuilder &PB)
      : CodeGenPassBuilder(createTargetMachine(), CGPassBuilderOption(), PB) {
    // Declare disabled passes in constructor.
    disablePass<NoOpModulePass>(3); // Disable the third NoOpModulePass.
    disablePass<DisabledMachineFunctionPass>();
  }

  // Override substitutePass is also OK.
  // template <typename PassT> auto substitutePass() {
  //   if constexpr (std::is_same_v<PassT, ReplacedMachineFunctionPass>)
  //     return NoOpMachineFunctionPass();
  //   else
  //     return;
  // }

  void buildTestPipeline(ModulePassManager &MPM) {
    addModulePass<NoOpModulePass, NoOpModulePass>();
    addFunctionPass<NoOpFunctionPass>();
    addModulePass<NoOpModulePass>();
    addMachineFunctionPass<DisabledMachineFunctionPass>();
    addFunctionPass(NoOpFunctionPass());
    addMachineFunctionPass<NoOpMachineFunctionPass,
                           ReplacedMachineFunctionPass>();
    mergePassManager();
    MPM.addPass(std::move(getMPM()));
    getMPM() = ModulePassManager();
  }
};

class CodeGenPassBuilderTest : public testing::Test {
public:
  CodeGenPassBuilderTest() : PB(&createTargetMachine()) {
    PIC.addClassToPassName(NoOpModulePass::name(), "no-op-module");
    PIC.addClassToPassName(NoOpFunctionPass::name(), "no-op-function");
    PIC.addClassToPassName(NoOpMachineFunctionPass::name(),
                           "no-op-machine-function");
    PIC.addClassToPassName(DisabledMachineFunctionPass::name(), "disabled");
    PIC.addClassToPassName(ReplacedMachineFunctionPass::name(), "replaced");
  }

  void buildPipeline(ModulePassManager &MPM) {
    TestCodeGenPassBuilder CGPB(PB);
    CGPB.buildTestPipeline(MPM);
  }

  std::string getPipelineText(ModulePassManager &MPM) {
    std::string PipelineText;
    raw_string_ostream OS(PipelineText);
    MPM.printPipeline(
        OS, [&](StringRef S) { return PIC.getPassNameForClassName(S); });
    return PipelineText;
  }
  PassInstrumentationCallbacks PIC;
  PassBuilder PB;
};

} // namespace

using PassBuilderBase =
    CodeGenPassBuilder<TestCodeGenPassBuilder, TestTargetMachine>;

// Add a specialization to substitute a pass.
template <>
template <>
auto PassBuilderBase::substitutePass<ReplacedMachineFunctionPass>() {
  return NoOpMachineFunctionPass();
}

TEST_F(CodeGenPassBuilderTest, Basic) {
  ModulePassManager MPM;
  buildPipeline(MPM);
  EXPECT_EQ(getPipelineText(MPM),
            "no-op-module,no-op-module,function(no-op-function,no-op-function,"
            "machine-function(no-op-machine-function,no-op-machine-function))");
}
