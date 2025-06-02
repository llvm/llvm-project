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
#include "../lib/Transforms/Vectorize/VPlanHelpers.h"
#include "../lib/Transforms/Vectorize/VPlanTransforms.h"
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
class VPlanTestIRBase : public testing::Test {
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

  VPlanTestIRBase()
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

  /// Build the VPlan for the loop starting from \p LoopHeader.
  VPlanPtr buildVPlan(BasicBlock *LoopHeader) {
    Function &F = *LoopHeader->getParent();
    assert(!verifyFunction(F) && "input function must be valid");
    doAnalysis(F);

    Loop *L = LI->getLoopFor(LoopHeader);
    PredicatedScalarEvolution PSE(*SE, *L);
    auto Plan = VPlanTransforms::buildPlainCFG(L, *LI);
    VFRange R(ElementCount::getFixed(1), ElementCount::getFixed(2));
    VPlanTransforms::prepareForVectorization(*Plan, IntegerType::get(*Ctx, 64),
                                             PSE, true, false, L, {}, false, R);
    VPlanTransforms::createLoopRegions(*Plan);
    return Plan;
  }
};

class VPlanTestBase : public testing::Test {
protected:
  LLVMContext C;
  std::unique_ptr<BasicBlock> ScalarHeader;
  SmallVector<std::unique_ptr<VPlan>> Plans;

  VPlanTestBase() : ScalarHeader(BasicBlock::Create(C, "scalar.header")) {
    BranchInst::Create(&*ScalarHeader, &*ScalarHeader);
  }

  VPlan &getPlan(VPValue *TC = nullptr) {
    Plans.push_back(std::make_unique<VPlan>(&*ScalarHeader, TC));
    return *Plans.back();
  }
};

} // namespace llvm

#endif // LLVM_UNITTESTS_TRANSFORMS_VECTORIZE_VPLANTESTBASE_H
