//===- llvm/LinkAllPasses.h ------------ Reference All Passes ---*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file was developed by Jeff Cohen and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include "llvm/Analysis/LoadValueNumbering.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
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
      (void) llvm::createAndersensPass();
      (void) llvm::createArgumentPromotionPass();
      (void) llvm::createBasicAliasAnalysisPass();
      (void) llvm::createBasicVNPass();
      (void) llvm::createBlockPlacementPass();
      (void) llvm::createBlockProfilerPass();
      (void) llvm::createBreakCriticalEdgesPass();
      (void) llvm::createCFGSimplificationPass();
      (void) llvm::createConstantMergePass();
      (void) llvm::createConstantPropagationPass();
      (void) llvm::createCorrelatedExpressionEliminationPass();
      (void) llvm::createDeadArgEliminationPass();
      (void) llvm::createDeadCodeEliminationPass();
      (void) llvm::createDeadInstEliminationPass();
      (void) llvm::createDeadStoreEliminationPass();
      (void) llvm::createDeadTypeEliminationPass();
      (void) llvm::createEdgeProfilerPass();
      (void) llvm::createEmitFunctionTablePass();
      (void) llvm::createFunctionInliningPass();
      (void) llvm::createFunctionProfilerPass();
      (void) llvm::createFunctionResolvingPass();
      (void) llvm::createGCSEPass();
      (void) llvm::createGlobalDCEPass();
      (void) llvm::createGlobalOptimizerPass();
      (void) llvm::createGlobalsModRefPass();
      (void) llvm::createIPConstantPropagationPass();
      (void) llvm::createIPSCCPPass();
      (void) llvm::createIndVarSimplifyPass();
      (void) llvm::createInstructionCombiningPass();
      (void) llvm::createInternalizePass(false);
      (void) llvm::createLCSSAPass();
      (void) llvm::createLICMPass();
      (void) llvm::createLoadValueNumberingPass();
      (void) llvm::createLoopExtractorPass();
      (void) llvm::createLoopSimplifyPass();
      (void) llvm::createLoopStrengthReducePass();
      (void) llvm::createLoopUnrollPass();
      (void) llvm::createLoopUnswitchPass();
      (void) llvm::createLowerAllocationsPass();
      (void) llvm::createLowerGCPass();
      (void) llvm::createLowerInvokePass();
      (void) llvm::createLowerPackedPass();
      (void) llvm::createLowerSelectPass();
      (void) llvm::createLowerSetJmpPass();
      (void) llvm::createLowerSwitchPass();
      (void) llvm::createNoAAPass();
      (void) llvm::createNoProfileInfoPass();
      (void) llvm::createProfileLoaderPass();
      (void) llvm::createPromoteMemoryToRegisterPass();
      (void) llvm::createDemoteRegisterToMemoryPass();
      (void) llvm::createPruneEHPass();
      (void) llvm::createRaiseAllocationsPass();
      (void) llvm::createRaisePointerReferencesPass();
      (void) llvm::createReassociatePass();
      (void) llvm::createSCCPPass();
      (void) llvm::createScalarReplAggregatesPass();
      (void) llvm::createSimplifyLibCallsPass();
      (void) llvm::createSingleLoopExtractorPass();
      (void) llvm::createStripSymbolsPass();
      (void) llvm::createTailCallEliminationPass();
      (void) llvm::createTailDuplicationPass();
      (void) llvm::createTraceBasicBlockPass();
      (void) llvm::createTraceValuesPassForBasicBlocks();
      (void) llvm::createTraceValuesPassForFunction();
      (void) llvm::createUnifyFunctionExitNodesPass();
      (void) llvm::createCondPropagationPass();
      (void) llvm::createNullProfilerRSPass();
      (void) llvm::createRSProfilingPass();
      (void) llvm::createIndMemRemPass();
      (void) llvm::createInstCountPass();
      (void) llvm::createPredicateSimplifierPass();

      (void)new llvm::IntervalPartition();
      (void)new llvm::ImmediateDominators();
      (void)new llvm::PostDominatorSet();
      (void)new llvm::FindUsedTypes();
      (void)new llvm::ScalarEvolution();
      ((llvm::Function*)0)->viewCFGOnly();
      llvm::AliasSetTracker X(*(llvm::AliasAnalysis*)0);
      X.add((llvm::Value*)0, 0);  // for -print-alias-sets
    }
  } ForcePassLinking; // Force link by creating a global definition.
}

#endif
