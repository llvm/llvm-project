//===- PassManagerTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/SandboxIR/BasicBlock.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Region.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm::sandboxir;

struct PassManagerTest : public testing::Test {
  llvm::LLVMContext LLVMCtx;
  std::unique_ptr<llvm::Module> LLVMM;
  std::unique_ptr<Context> Ctx;

  Function *parseFunction(const char *IR, const char *FuncName) {
    llvm::SMDiagnostic Err;
    LLVMM = parseAssemblyString(IR, Err, LLVMCtx);

    if (!LLVMM)
      Err.print("PassTest", llvm::errs());
    Ctx = std::make_unique<Context>(LLVMCtx);
    return Ctx->createFunction(LLVMM->getFunction(FuncName));
  }
};

#if !defined(NDEBUG) && defined(EXPENSIVE_CHECKS)
TEST_F(PassManagerTest, FunctionPassReturnValue) {
  auto *F = parseFunction(R"IR(
define void @foo() {
  ret void
}
)IR",
                          "foo");
  llvm::Function *LLVMF = &*LLVMM->getFunction("foo");
  // A pass that deletes an instruction and returns false.
  class DelInstrRetFalsePass final : public FunctionPass {
  public:
    DelInstrRetFalsePass() : FunctionPass("del-instr-ret-false-pass") {}
    bool runOnFunction(Function &F, const Analyses &A) final {
      auto &BB = *F.begin();
      BB.front().eraseFromParent();
      return false;
    }
  };
  // A pass that deletes an instruction and returns true.
  class DelInstrRetTruePass final : public FunctionPass {
  public:
    DelInstrRetTruePass() : FunctionPass("del-instr-ret-true-pass") {}
    bool runOnFunction(Function &F, const Analyses &A) final {
      auto &BB = *F.begin();
      BB.front().eraseFromParent();
      return true;
    }
  };

  // A dummy pass that makes no changes and returns false.
  class DoNothingRetFalsePass final : public FunctionPass {
  public:
    DoNothingRetFalsePass() : FunctionPass("do-nothing-ret-false-pass") {}
    bool runOnFunction(Function &F, const Analyses &A) final { return false; }
  };
  // A dummy pass that makes no changes and returns true.
  class DoNothingRetTruePass final : public FunctionPass {
  public:
    DoNothingRetTruePass() : FunctionPass("do-nothing-ret-true-pass") {}
    bool runOnFunction(Function &F, const Analyses &A) final { return false; }
  };

  Analyses A;

  auto DNRF = std::make_unique<DoNothingRetFalsePass>();
  auto DNRT = std::make_unique<DoNothingRetTruePass>();
  FunctionPassManager DNRF_PM("DNRF_PM");
  DNRF_PM.addPass(std::move(DNRF));
  DNRF_PM.runOnFunction(*F, A);

  FunctionPassManager DNRT_PM("DNRT_PM");
  DNRT_PM.addPass(std::move(DNRT));
  DNRT_PM.runOnFunction(*F, A);

  auto DIRF = std::make_unique<DelInstrRetFalsePass>();
  auto DIRT = std::make_unique<DelInstrRetTruePass>();
  FunctionPassManager DIRF_PM("DIRF_PM");
  DIRF_PM.addPass(std::move(DIRF));
  EXPECT_DEATH(DIRF_PM.runOnFunction(*F, A), ".*changed.*");

  FunctionPassManager DIRT_PM("DIRT_PM");
  DIRT_PM.addPass(std::move(DIRT));
  DIRT_PM.runOnFunction(*F, A);
}

TEST_F(PassManagerTest, RegionPassReturnValue) {
  auto *F = parseFunction(R"IR(
define void @foo() {
  %add0 = add i32 42, 42, !sandboxvec !0
  %add1 = add i32 43, 43, !sandboxvec !0
  ret void, !sandboxvec !0
}
!0 = distinct !{!"sandboxregion"}
)IR",
                          "foo");
  // A pass that erases an instruction from the region and returns false.
  class DelInstrRetFalsePass final : public RegionPass {
  public:
    DelInstrRetFalsePass() : RegionPass("del-instr-ret-false-pass") {}
    bool runOnRegion(Region &R, const Analyses &A) final {
      (*R.begin())->eraseFromParent();
      return false;
    }
  };
  // A pass that erases an instruction from the region and returns true.
  class DelInstrRetTruePass final : public RegionPass {
  public:
    DelInstrRetTruePass() : RegionPass("del-instr-ret-true-pass") {}
    bool runOnRegion(Region &R, const Analyses &A) final {
      (*R.begin())->eraseFromParent();
      return true;
    }
  };
  // A dummy pass that makes no changes and returns false.
  class DoNothingRetFalsePass final : public RegionPass {
  public:
    DoNothingRetFalsePass() : RegionPass("do-nothing-ret-false-pass") {}
    bool runOnRegion(Region &R, const Analyses &A) final { return false; }
  };
  // A dummy pass that makes no changes and returns true.
  class DoNothingRetTruePass final : public RegionPass {
  public:
    DoNothingRetTruePass() : RegionPass("do-nothing-ret-true-pass") {}
    bool runOnRegion(Region &R, const Analyses &A) final { return true; }
  };

  Analyses A;
  auto &BB = *F->begin();
  Instruction *AddI = &BB.front();

  // Build a region containing the add instruction.
  llvm::SmallVector<std::unique_ptr<Region>> Regions =
      Region::createRegionsFromMD(*F);
  Region &Rgn = *Regions[0];

  auto DNRF = std::make_unique<DoNothingRetFalsePass>();
  auto DNRT = std::make_unique<DoNothingRetTruePass>();
  RegionPassManager DNRF_PM("DNRF_PM");
  DNRF_PM.addPass(std::move(DNRF));
  DNRF_PM.runOnRegion(Rgn, A);

  RegionPassManager DNRT_PM("DNRT_PM");
  DNRT_PM.addPass(std::move(DNRT));
  DNRT_PM.runOnRegion(Rgn, A);

  auto DIRF = std::make_unique<DelInstrRetFalsePass>();
  auto DIRT = std::make_unique<DelInstrRetTruePass>();
  RegionPassManager DIRF_PM("DIRF_PM");
  DIRF_PM.addPass(std::move(DIRF));
  EXPECT_DEATH(DIRF_PM.runOnRegion(Rgn, A), ".*changed.*");

  RegionPassManager DIRT_PM("DIRT_PM");
  DIRT_PM.addPass(std::move(DIRT));
  DIRT_PM.runOnRegion(Rgn, A);
}

#endif
