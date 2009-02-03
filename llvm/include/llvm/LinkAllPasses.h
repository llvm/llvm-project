//===- llvm/LinkAllPasses.h ------------ Reference All Passes ---*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all transformation and analysis passes for tools 
// like opt and bugpoint that need this functionality.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LINKALLPASSES_H
#define LLVM_LINKALLPASSES_H

#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Analysis/IntervalPartition.h"
#include "llvm/Analysis/LoopVR.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Function.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include <cstdlib>

namespace {
  struct ForcePassLinking {
    ForcePassLinking() {
      // We must reference the passes in such a way that compilers will not
      // delete it all as dead code, even with whole program optimization,
      // yet is effectively a NO-OP. As the compiler isn't smart enough
      // to know that getenv() never returns -1, this will do the job.
      if (std::getenv("bar") != (char*) -1)
        return;

      (void) llvm::createAAEvalPass();
      (void) llvm::createAggressiveDCEPass();
      (void) llvm::createAliasAnalysisCounterPass();
      (void) llvm::createAliasDebugger();
      (void) llvm::createAndersensPass();
      (void) llvm::createArgumentPromotionPass();
      (void) llvm::createStructRetPromotionPass();
      (void) llvm::createBasicAliasAnalysisPass();
      (void) llvm::createLibCallAliasAnalysisPass(0);
      (void) llvm::createBlockPlacementPass();
      (void) llvm::createBlockProfilerPass();
      (void) llvm::createBreakCriticalEdgesPass();
      (void) llvm::createCFGSimplificationPass();
      (void) llvm::createConstantMergePass();
      (void) llvm::createConstantPropagationPass();
      (void) llvm::createDeadArgEliminationPass();
      (void) llvm::createDeadCodeEliminationPass();
      (void) llvm::createDeadInstEliminationPass();
      (void) llvm::createDeadStoreEliminationPass();
      (void) llvm::createDeadTypeEliminationPass();
      (void) llvm::createEdgeProfilerPass();
      (void) llvm::createFunctionInliningPass();
      (void) llvm::createAlwaysInlinerPass();
      (void) llvm::createFunctionProfilerPass();
      (void) llvm::createGlobalDCEPass();
      (void) llvm::createGlobalOptimizerPass();
      (void) llvm::createGlobalsModRefPass();
      (void) llvm::createGVNPREPass();
      (void) llvm::createIPConstantPropagationPass();
      (void) llvm::createIPSCCPPass();
      (void) llvm::createIndVarSimplifyPass();
      (void) llvm::createInstructionCombiningPass();
      (void) llvm::createInternalizePass(false);
      (void) llvm::createLCSSAPass();
      (void) llvm::createLICMPass();
      (void) llvm::createLoopExtractorPass();
      (void) llvm::createLoopSimplifyPass();
      (void) llvm::createLoopStrengthReducePass();
      (void) llvm::createLoopUnrollPass();
      (void) llvm::createLoopUnswitchPass();
      (void) llvm::createLoopRotatePass();
      (void) llvm::createLoopIndexSplitPass();
      (void) llvm::createLowerAllocationsPass();
      (void) llvm::createLowerInvokePass();
      (void) llvm::createLowerSetJmpPass();
      (void) llvm::createLowerSwitchPass();
      (void) llvm::createNoAAPass();
      (void) llvm::createNoProfileInfoPass();
      (void) llvm::createProfileLoaderPass();
      (void) llvm::createPromoteMemoryToRegisterPass();
      (void) llvm::createDemoteRegisterToMemoryPass();
      (void) llvm::createPruneEHPass();
      (void) llvm::createRaiseAllocationsPass();
      (void) llvm::createReassociatePass();
      (void) llvm::createSCCPPass();
      (void) llvm::createScalarReplAggregatesPass();
      (void) llvm::createSimplifyLibCallsPass();
      (void) llvm::createSimplifyHalfPowrLibCallsPass();
      (void) llvm::createSingleLoopExtractorPass();
      (void) llvm::createStripSymbolsPass();
      (void) llvm::createStripNonDebugSymbolsPass();
      (void) llvm::createStripDeadPrototypesPass();
      (void) llvm::createTailCallEliminationPass();
      (void) llvm::createTailDuplicationPass();
      (void) llvm::createJumpThreadingPass();
      (void) llvm::createUnifyFunctionExitNodesPass();
      (void) llvm::createCondPropagationPass();
      (void) llvm::createNullProfilerRSPass();
      (void) llvm::createRSProfilingPass();
      (void) llvm::createIndMemRemPass();
      (void) llvm::createInstCountPass();
      (void) llvm::createPredicateSimplifierPass();
      (void) llvm::createCodeGenPreparePass();
      (void) llvm::createGVNPass();
      (void) llvm::createMemCpyOptPass();
      (void) llvm::createLoopDeletionPass();
      (void) llvm::createPostDomTree();
      (void) llvm::createPostDomFrontier();
      (void) llvm::createInstructionNamerPass();
      (void) llvm::createPartialSpecializationPass();
      (void) llvm::createFunctionAttrsPass();
      (void) llvm::createMergeFunctionsPass();
      (void) llvm::createPrintModulePass(0);
      (void) llvm::createPrintFunctionPass("", 0);
      (void) llvm::createDbgInfoPrinterPass();

      (void)new llvm::IntervalPartition();
      (void)new llvm::FindUsedTypes();
      (void)new llvm::ScalarEvolution();
      (void)new llvm::LoopVR();
      ((llvm::Function*)0)->viewCFGOnly();
      llvm::AliasSetTracker X(*(llvm::AliasAnalysis*)0);
      X.add((llvm::Value*)0, 0);  // for -print-alias-sets
    }
  } ForcePassLinking; // Force link by creating a global definition.
}

#endif
