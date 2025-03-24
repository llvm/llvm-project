//===- TargetPassBuilderTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/TargetPassBuilder.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/ExpandFp.h"
#include "llvm/CodeGen/ExpandLargeDivRem.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/PreISelIntrinsicLowering.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"
#include <cassert>

using namespace llvm;

namespace {

class TestTargetMachine : public CodeGenTargetMachineImpl {
public:
  TestTargetMachine()
      : CodeGenTargetMachineImpl(Target(), "", Triple(""), "", "",
                                 TargetOptions(), Reloc::Static,
                                 CodeModel::Small, CodeGenOptLevel::Default) {
    AsmInfo.reset(new MCAsmInfo());
  }

  ~TestTargetMachine() override = default;
};

TestTargetMachine *createTargetMachine() {
  static TestTargetMachine TestTM;
  return &TestTM;
}

template <size_t Tag, typename MemberPtrT> struct PrivateVisitor {
  inline static MemberPtrT Ptr;
};
template <size_t Tag, auto MemberPtrV> struct PrivateVisitorHelper {
  struct Assigner {
    Assigner() { PrivateVisitor<Tag, decltype(MemberPtrV)>::Ptr = MemberPtrV; }
  };
  inline static Assigner A;
};

template <size_t Tag, typename MemberPtrT>
MemberPtrT PrivatePtr = PrivateVisitor<Tag, MemberPtrT>::Ptr;

struct TestDAGISelPass : public PassInfoMixin<TestDAGISelPass> {
  PreservedAnalyses run(MachineFunction &, MachineFunctionAnalysisManager &) {
    return PreservedAnalyses::all();
  }
};

class TestPassBuilder : public TargetPassBuilder {
public:
  using TargetPassBuilder::ModulePassManager;

  TestPassBuilder(PassBuilder &PB) : TargetPassBuilder(PB) {

    registerSelectionDAGISelPass([]() { return TestDAGISelPass(); });
    CGPBO.RequiresCodeGenSCCOrder = true;

    injectBefore<PreISelIntrinsicLoweringPass>([]() {
      ModulePassManager MPM;
      MPM.addPass(NoOpModulePass());
      return MPM;
    });

    injectBefore<ExpandFpPass, ModulePassManager>([] {
      ModulePassManager MPM;
      MPM.addPass(NoOpModulePass());
      return MPM;
    });
  }
};

template struct PrivateVisitorHelper<
    0, &TargetPassBuilder::buildCodeGenIRPipeline>;
template struct PrivateVisitorHelper<
    0, &TargetPassBuilder::invokeInjectionCallbacks>;
template struct PrivateVisitorHelper<
    0, &TargetPassBuilder::ModulePassManager::Passes>;

TEST(TargetPassBuilder, Basic) {
  TargetMachine *TM = createTargetMachine();
  PassInstrumentationCallbacks PIC;
  PassBuilder PB(TM, PipelineTuningOptions(), std::nullopt, &PIC);
  ModulePassManager PM;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager AM;

  /// Register builtin analyses and cross-register the analysis proxies
  PB.registerModuleAnalyses(AM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, AM);

  TestPassBuilder TPB(PB);
  using PassList = TargetPassBuilder::PassList;
  auto MPM = (TPB.*PrivatePtr<0, TestPassBuilder::ModulePassManager (
                                     TargetPassBuilder::*)()>)();
  auto &Passes =
      MPM.*PrivatePtr<0, PassList TestPassBuilder::ModulePassManager::*>;
  (TPB.*PrivatePtr<0, void (TargetPassBuilder::*)(
                          TestPassBuilder::ModulePassManager &)
                          const>)(MPM); // invokeInjectionCallbacks
  auto B = Passes.begin();
  EXPECT_EQ(B->Name, NoOpModulePass::name());
  ++B, ++B, ++B;
  EXPECT_EQ(B->Name, NoOpModulePass::name());
}

template struct PrivateVisitorHelper<1, &TargetPassBuilder::filtPassList>;

TEST(TargetPassBuilder, StartStop) {
  const char *Argv[] = {"CodeGenTests",
                        "--start-after=pre-isel-intrinsic-lowering",
                        "--stop-before=no-op-module,2"};
  cl::ParseCommandLineOptions(std::size(Argv), Argv);

  TargetMachine *TM = createTargetMachine();
  PassInstrumentationCallbacks PIC;
  PassBuilder PB(TM, PipelineTuningOptions(), std::nullopt, &PIC);
  ModulePassManager PM;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager AM;

  /// Register builtin analyses and cross-register the analysis proxies
  PB.registerModuleAnalyses(AM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, AM);

  using PassList = TargetPassBuilder::PassList;

  TestPassBuilder TPB(PB);
  auto MPM = (TPB.*PrivatePtr<0, TestPassBuilder::ModulePassManager (
                                     TargetPassBuilder::*)()>)();
  auto &Passes =
      MPM.*PrivatePtr<0, PassList TestPassBuilder::ModulePassManager::*>;
  (TPB.*PrivatePtr<0, void (TargetPassBuilder::*)(
                          TestPassBuilder::ModulePassManager &)
                          const>)(MPM); // invokeInjectionCallbacks
  (TPB.*PrivatePtr<1, void (TargetPassBuilder::*)(
                          TestPassBuilder::ModulePassManager &)
                          const>)(MPM); // filtPassList
  EXPECT_EQ(Passes.size(), 1u);
  EXPECT_EQ(Passes.begin()->Name, ExpandLargeDivRemPass::name());
}
} // namespace
