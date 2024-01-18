//===- llvm/unittest/CodeGen/CodeGenPassBuilderTest.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CodeGenPassBuilder.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

namespace {

class DummyCodeGenPassBuilder
    : public CodeGenPassBuilder<DummyCodeGenPassBuilder> {
public:
  DummyCodeGenPassBuilder(LLVMTargetMachine &TM, CGPassBuilderOption Opts,
                          PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC){};

  void addPreISel(AddIRPass &addPass) const {
    addPass(NoOpModulePass());
    addPass(NoOpFunctionPass());
    addPass(NoOpFunctionPass());
    addPass(NoOpFunctionPass());
    addPass(NoOpModulePass());
    addPass(NoOpFunctionPass());
  }

  void addAsmPrinter(AddMachinePass &, CreateMCStreamer) const {}

  Error addInstSelector(AddMachinePass &) const { return Error::success(); }
};

class CodeGenPassBuilderTest : public testing::Test {
public:
  std::unique_ptr<LLVMTargetMachine> TM;

  static void SetUpTestCase() {
    InitializeAllTargets();
    InitializeAllTargetMCs();

    // TODO: Move this test to normal lit test when llc supports new pm.
    static const char *argv[] = {
        "test",
        "-print-pipeline-passes",
    };
    int argc = std::size(argv);
    cl::ParseCommandLineOptions(argc, argv);
  }

  void SetUp() override {
    std::string TripleName = Triple::normalize("x86_64-pc-linux-gnu");
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
    if (!TheTarget)
      GTEST_SKIP();

    TargetOptions Options;
    TM = std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine *>(
        TheTarget->createTargetMachine("", "", "", Options, std::nullopt)));
    if (!TM)
      GTEST_SKIP();
  }
};

TEST_F(CodeGenPassBuilderTest, basic) {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassInstrumentationCallbacks PIC;
  DummyCodeGenPassBuilder CGPB(*TM, getCGPassBuilderOption(), &PIC);
  PipelineTuningOptions PTO;
  PassBuilder PB(TM.get(), PTO, std::nullopt, &PIC);

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  MachineFunctionPassManager MFPM;
  Error Err =
      CGPB.buildPipeline(MPM, MFPM, outs(), nullptr, CodeGenFileType::Null);
  EXPECT_FALSE(Err);

  std::string IRPipeline;
  raw_string_ostream IROS(IRPipeline);
  MPM.printPipeline(IROS, [&PIC](StringRef Name) {
    auto PassName = PIC.getPassNameForClassName(Name);
    return PassName.empty() ? Name : PassName;
  });
  const char ExpectedIRPipeline[] =
      "no-op-module,function(no-op-function,"
      "no-op-function,no-op-function),no-op-module";
  // TODO: Move this test to normal lit test when llc supports new pm.
  EXPECT_TRUE(StringRef(IRPipeline).contains(ExpectedIRPipeline));

  std::string MIRPipeline;
  raw_string_ostream MIROS(MIRPipeline);
  MFPM.printPipeline(MIROS, [&PIC](StringRef Name) {
    auto PassName = PIC.getPassNameForClassName(Name);
    return PassName.empty() ? Name : PassName;
  });
  const char ExpectedMIRPipeline[] =
      "FinalizeISelPass,EarlyTailDuplicatePass,OptimizePHIsPass,"
      "StackColoringPass,LocalStackSlotPass,DeadMachineInstructionElimPass,"
      "EarlyMachineLICMPass,MachineCSEPass,MachineSinkingPass,"
      "PeepholeOptimizerPass,DeadMachineInstructionElimPass,"
      "DetectDeadLanesPass,ProcessImplicitDefsPass,PHIEliminationPass,"
      "TwoAddressInstructionPass,RegisterCoalescerPass,"
      "RenameIndependentSubregsPass,MachineSchedulerPass,RAGreedyPass,"
      "VirtRegRewriterPass,StackSlotColoringPass,"
      "RemoveRedundantDebugValuesPass,PostRAMachineSinkingPass,ShrinkWrapPass,"
      "PrologEpilogInserterPass,BranchFolderPass,TailDuplicatePass,"
      "MachineLateInstrsCleanupPass,MachineCopyPropagationPass,"
      "ExpandPostRAPseudosPass,PostRASchedulerPass,MachineBlockPlacementPass,"
      "FEntryInserterPass,XRayInstrumentationPass,PatchableFunctionPass,"
      "FuncletLayoutPass,StackMapLivenessPass,LiveDebugValuesPass,"
      "MachineSanitizerBinaryMetadata,FreeMachineFunctionPass";
  // TODO: Check pipeline string when all pass names are populated.
  // TODO: Move this test to normal lit test when llc supports new pm.
  EXPECT_EQ(MIRPipeline, ExpectedMIRPipeline);
}

} // namespace
