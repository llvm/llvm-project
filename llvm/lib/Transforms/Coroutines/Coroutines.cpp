//===-- Coroutines.cpp ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file implements the common infrastructure for Coroutine Passes.
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;

void llvm::initializeCoroutines(PassRegistry &Registry) {
  initializeCoroEarlyPass(Registry);
  initializeCoroSplitPass(Registry);
  initializeCoroElidePass(Registry);
  initializeCoroCleanupPass(Registry);
}

static void addCoroutineOpt0Passes(const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
  PM.add(createCoroSplitPass());
  PM.add(createCoroElidePass());

  PM.add(createBarrierNoopPass());
  PM.add(createCoroCleanupPass());
}

static void addCoroutineEarlyPasses(const PassManagerBuilder &Builder,
                                    legacy::PassManagerBase &PM) {
  PM.add(createCoroEarlyPass());
}

static void addCoroutineScalarOptimizerPasses(const PassManagerBuilder &Builder,
                                              legacy::PassManagerBase &PM) {
  PM.add(createCoroElidePass());
}

static void addCoroutineSCCPasses(const PassManagerBuilder &Builder,
                                  legacy::PassManagerBase &PM) {
  PM.add(createCoroSplitPass());
}

static void addCoroutineOptimizerLastPasses(const PassManagerBuilder &Builder,
                                            legacy::PassManagerBase &PM) {
  PM.add(createCoroCleanupPass());
}

void llvm::addCoroutinePassesToExtensionPoints(PassManagerBuilder &Builder) {
  Builder.addExtension(PassManagerBuilder::EP_EarlyAsPossible,
                       addCoroutineEarlyPasses);
  Builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                       addCoroutineOpt0Passes);
  Builder.addExtension(PassManagerBuilder::EP_CGSCCOptimizerLate,
                       addCoroutineSCCPasses);
  Builder.addExtension(PassManagerBuilder::EP_ScalarOptimizerLate,
                       addCoroutineScalarOptimizerPasses);
  Builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                       addCoroutineOptimizerLastPasses);
}
