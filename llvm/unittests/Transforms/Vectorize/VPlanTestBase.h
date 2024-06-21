//===- llvm/unittest/Transforms/Vectorize/VPlanTestBase.h -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines a VPlanTestBase class, which provides helpers to parse
/// a LLVM IR string and create VPlans given a loop entry block.
//===----------------------------------------------------------------------===//
#ifndef LLVM_UNITTESTS_TRANSFORMS_VECTORIZE_VPLANTESTBASE_H
#define LLVM_UNITTESTS_TRANSFORMS_VECTORIZE_VPLANTESTBASE_H

#include "../lib/Transforms/Vectorize/VPlan.h"
#include "../lib/Transforms/Vectorize/VPlanHCFGBuilder.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

namespace llvm {

/// Helper class to create a module from an assembly string and VPlans for a
/// given loop entry block.
class VPlanTestBase : public testing::Test {
protected:
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;
  DataLayout DL;

  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<Module> M;
  std::unique_ptr<LoopInfo> LI;
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<AssumptionCache> AC;
  std::unique_ptr<ScalarEvolution> SE;

  VPlanTestBase()
      : TLII(), TLI(TLII),
        DL("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-"
           "f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:"
           "16:32:64-S128"),
        Ctx(new LLVMContext) {}

  Module &parseModule(const char *ModuleString) {
    SMDiagnostic Err;
    M = parseAssemblyString(ModuleString, Err, *Ctx);
    EXPECT_TRUE(M);
    return *M;
  }

  void doAnalysis(Function &F) {
    DT.reset(new DominatorTree(F));
    LI.reset(new LoopInfo(*DT));
    AC.reset(new AssumptionCache(F));
    SE.reset(new ScalarEvolution(F, TLI, *AC, *DT, *LI));
  }

  VPlanPtr buildHCFG(BasicBlock *LoopHeader) {
    Function &F = *LoopHeader->getParent();
    assert(!verifyFunction(F) && "input function must be valid");
    doAnalysis(F);

    Loop *L = LI->getLoopFor(LoopHeader);
    auto Plan = VPlan::createInitialVPlan(SE->getBackedgeTakenCount(L), *SE,
                                          L->getLoopPreheader());
    VPlanHCFGBuilder HCFGBuilder(L, LI.get(), *Plan);
    HCFGBuilder.buildHierarchicalCFG();
    return Plan;
  }

  /// Build the VPlan plain CFG for the loop starting from \p LoopHeader.
  VPlanPtr buildPlainCFG(BasicBlock *LoopHeader) {
    Function &F = *LoopHeader->getParent();
    assert(!verifyFunction(F) && "input function must be valid");
    doAnalysis(F);

    Loop *L = LI->getLoopFor(LoopHeader);
    auto Plan = VPlan::createInitialVPlan(SE->getBackedgeTakenCount(L), *SE,
                                          L->getLoopPreheader());
    VPlanHCFGBuilder HCFGBuilder(L, LI.get(), *Plan);
    HCFGBuilder.buildPlainCFG();
    return Plan;
  }
};

} // namespace llvm

#endif // LLVM_UNITTESTS_TRANSFORMS_VECTORIZE_VPLANTESTBASE_H
