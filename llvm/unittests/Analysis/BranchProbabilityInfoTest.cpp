//===- BranchProbabilityInfoTest.cpp - BranchProbabilityInfo unit tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

struct BranchProbabilityInfoTest : public testing::Test {
  std::unique_ptr<BranchProbabilityInfo> BPI;
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<LoopInfo> LI;
  LLVMContext C;

  BranchProbabilityInfo &buildBPI(Function &F) {
    DT.reset(new DominatorTree(F));
    LI.reset(new LoopInfo(*DT));
    BPI.reset(new BranchProbabilityInfo(F, *LI));
    return *BPI;
  }

  std::unique_ptr<Module> makeLLVMModule() {
    const char *ModuleString = "define void @f() { exit: ret void }\n";
    SMDiagnostic Err;
    return parseAssemblyString(ModuleString, Err, C);
  }
};

TEST_F(BranchProbabilityInfoTest, StressUnreachableHeuristic) {
  auto M = makeLLVMModule();
  Function *F = M->getFunction("f");

  // define void @f() {
  // entry:
  //   switch i32 undef, label %exit, [
  //      i32 0, label %preexit
  //      ...                   ;;< Add lots of cases to stress the heuristic.
  //   ]
  // preexit:
  //   unreachable
  // exit:
  //   ret void
  // }

  auto *ExitBB = &F->back();
  auto *EntryBB = BasicBlock::Create(C, "entry", F, /*insertBefore=*/ExitBB);

  auto *PreExitBB =
      BasicBlock::Create(C, "preexit", F, /*insertBefore=*/ExitBB);
  new UnreachableInst(C, PreExitBB);

  unsigned NumCases = 4096;
  auto *I32 = IntegerType::get(C, 32);
  auto *Undef = UndefValue::get(I32);
  auto *Switch = SwitchInst::Create(Undef, ExitBB, NumCases, EntryBB);
  for (unsigned I = 0; I < NumCases; ++I)
    Switch->addCase(ConstantInt::get(I32, I), PreExitBB);

  BranchProbabilityInfo &BPI = buildBPI(*F);

  // FIXME: This doesn't seem optimal. Since all of the cases handled by the
  // switch have the *same* destination block ("preexit"), shouldn't it be the
  // hot one? I'd expect the results to be reversed here...
  EXPECT_FALSE(BPI.isEdgeHot(EntryBB, PreExitBB));
  EXPECT_TRUE(BPI.isEdgeHot(EntryBB, ExitBB));
}

} // end anonymous namespace
} // end namespace llvm
