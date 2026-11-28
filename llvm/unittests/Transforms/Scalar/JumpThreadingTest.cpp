//===- JumpThreadingTest.cpp - JumpThreading unit tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class NoTwoBBThreadingTTIImpl : public TargetTransformInfoImplBase {
public:
  explicit NoTwoBBThreadingTTIImpl(const DataLayout &DL)
      : TargetTransformInfoImplBase(DL) {}

  bool shouldThreadThroughTwoBasicBlocks() const override { return false; }
};

// @foo is from test/Transforms/JumpThreading/thread-two-bbs.ll; by default the
// pass threads through two blocks and creates bb.cond2.thread. @bar is threaded
// through a single block, which the hook does not control: its condition is
// known in the predecessors, so processThreadableEdges() never reaches
// maybethreadThroughTwoBasicBlocks().
constexpr const char *ThreadingIR = R"(
  @a = global i32 0, align 4

  define void @foo(i32 %cond1, i32 %cond2) {
  entry:
    %tobool = icmp eq i32 %cond1, 0
    br i1 %tobool, label %bb.cond2, label %bb.f1

  bb.f1:
    call void @f1()
    br label %bb.cond2

  bb.cond2:
    %ptr = phi ptr [ null, %bb.f1 ], [ @a, %entry ]
    %tobool1 = icmp eq i32 %cond2, 0
    br i1 %tobool1, label %bb.file, label %bb.f2

  bb.f2:
    call void @f2()
    br label %exit

  bb.file:
    %cmp = icmp eq ptr %ptr, null
    br i1 %cmp, label %bb.f4, label %bb.f3

  bb.f3:
    call void @f3()
    br label %exit

  bb.f4:
    call void @f4()
    br label %exit

  exit:
    ret void
  }

  define void @bar(i1 %cond) {
  entry:
    br i1 %cond, label %bb.t, label %bb.f

  bb.t:
    call void @f1()
    br label %merge

  bb.f:
    call void @f2()
    br label %merge

  merge:
    %p = phi i1 [ true, %bb.t ], [ false, %bb.f ]
    br i1 %p, label %bb.a, label %bb.b

  bb.a:
    call void @f3()
    br label %exit

  bb.b:
    call void @f4()
    br label %exit

  exit:
    ret void
  }

  declare void @f1()
  declare void @f2()
  declare void @f3()
  declare void @f4()
)";

static bool hasBlock(const Function &F, StringRef Name) {
  for (const BasicBlock &BB : F)
    if (BB.getName() == Name)
      return true;
  return false;
}

static std::unique_ptr<Module> runJumpThreading(LLVMContext &Ctx,
                                                bool UseNoTwoBBTTI) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ThreadingIR, Err, Ctx);
  if (!M)
    return nullptr;

  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Must be registered before registerFunctionAnalyses(), which would
  // otherwise install the default TargetIRAnalysis.
  if (UseNoTwoBBTTI)
    FAM.registerPass([] {
      return TargetIRAnalysis([](const Function &F) {
        return TargetTransformInfo(
            std::make_unique<NoTwoBBThreadingTTIImpl>(F.getDataLayout()));
      });
    });

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  FunctionPassManager FPM;
  FPM.addPass(JumpThreadingPass());
  for (Function &F : *M)
    if (!F.isDeclaration())
      FPM.run(F, FAM);
  return M;
}

TEST(JumpThreadingTest, ThreadsThroughTwoBasicBlocksByDefault) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = runJumpThreading(Ctx, /*UseNoTwoBBTTI=*/false);
  ASSERT_TRUE(M);
  EXPECT_TRUE(hasBlock(*M->getFunction("foo"), "bb.cond2.thread"));
  EXPECT_FALSE(hasBlock(*M->getFunction("bar"), "merge"));
}

TEST(JumpThreadingTest, TargetCanDisableThreadingThroughTwoBasicBlocks) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = runJumpThreading(Ctx, /*UseNoTwoBBTTI=*/true);
  ASSERT_TRUE(M);
  EXPECT_FALSE(hasBlock(*M->getFunction("foo"), "bb.cond2.thread"));
  // Only that transformation is disabled; @bar is still threaded, which folds
  // its merge block away.
  EXPECT_FALSE(hasBlock(*M->getFunction("bar"), "merge"));
}

} // end anonymous namespace
