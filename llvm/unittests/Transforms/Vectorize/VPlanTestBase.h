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
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

namespace llvm {

/// Helper class to create a module from an assembly string and VPlans for a
/// given loop entry block.
class VPlanTestBase : public testing::Test {
protected:
  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<Module> M;
  std::unique_ptr<LoopInfo> LI;
  std::unique_ptr<DominatorTree> DT;

  VPlanTestBase() : Ctx(new LLVMContext) {}

  Module &parseModule(const char *ModuleString) {
    SMDiagnostic Err;
    M = parseAssemblyString(ModuleString, Err, *Ctx);
    EXPECT_TRUE(M);
    return *M;
  }

  void doAnalysis(Function &F) {
    DT.reset(new DominatorTree(F));
    LI.reset(new LoopInfo(*DT));
  }

  VPlanPtr buildHCFG(BasicBlock *LoopHeader) {
    doAnalysis(*LoopHeader->getParent());

    auto Plan = std::make_unique<VPlan>();
    VPlanHCFGBuilder HCFGBuilder(LI->getLoopFor(LoopHeader), LI.get(), *Plan);
    HCFGBuilder.buildHierarchicalCFG();
    return Plan;
  }

  /// Build the VPlan plain CFG for the loop starting from \p LoopHeader.
  VPlanPtr buildPlainCFG(BasicBlock *LoopHeader) {
    doAnalysis(*LoopHeader->getParent());

    auto Plan = std::make_unique<VPlan>();
    VPlanHCFGBuilder HCFGBuilder(LI->getLoopFor(LoopHeader), LI.get(), *Plan);
    VPRegionBlock *TopRegion = HCFGBuilder.buildPlainCFG();
    Plan->setEntry(TopRegion);
    return Plan;
  }
};

} // namespace llvm

#endif // LLVM_UNITTESTS_TRANSFORMS_VECTORIZE_VPLANTESTBASE_H
