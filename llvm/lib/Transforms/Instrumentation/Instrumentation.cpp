//===-- Instrumentation.cpp - TransformUtils Infrastructure ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the common initialization infrastructure for the
// Instrumentation library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation.h"
#include "llvm-c/Initialization.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"

using namespace llvm;

/// Moves I before IP. Returns new insert point.
static BasicBlock::iterator moveBeforeInsertPoint(BasicBlock::iterator I, BasicBlock::iterator IP) {
  // If I is IP, move the insert point down.
  if (I == IP)
    return ++IP;
  // Otherwise, move I before IP and return IP.
  I->moveBefore(&*IP);
  return IP;
}

/// Instrumentation passes often insert conditional checks into entry blocks.
/// Call this function before splitting the entry block to move instructions
/// that must remain in the entry block up before the split point. Static
/// allocas and llvm.localescape calls, for example, must remain in the entry
/// block.
BasicBlock::iterator llvm::PrepareToSplitEntryBlock(BasicBlock &BB,
                                                    BasicBlock::iterator IP) {
  assert(&BB.getParent()->getEntryBlock() == &BB);
  for (auto I = IP, E = BB.end(); I != E; ++I) {
    bool KeepInEntry = false;
    if (auto *AI = dyn_cast<AllocaInst>(I)) {
      if (AI->isStaticAlloca())
        KeepInEntry = true;
    } else if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      if (II->getIntrinsicID() == llvm::Intrinsic::localescape)
        KeepInEntry = true;
    }
    if (KeepInEntry)
      IP = moveBeforeInsertPoint(I, IP);
  }
  return IP;
}

/// initializeInstrumentation - Initialize all passes in the TransformUtils
/// library.
void llvm::initializeInstrumentation(PassRegistry &Registry) {
  initializeAddressSanitizerPass(Registry);
  initializeAddressSanitizerModulePass(Registry);
  initializeBoundsCheckingPass(Registry);
  initializeGCOVProfilerLegacyPassPass(Registry);
  initializePGOInstrumentationGenLegacyPassPass(Registry);
  initializePGOInstrumentationUseLegacyPassPass(Registry);
  initializePGOIndirectCallPromotionLegacyPassPass(Registry);
  initializeInstrProfilingLegacyPassPass(Registry);
  initializeMemorySanitizerPass(Registry);
  initializeThreadSanitizerPass(Registry);
  initializeSanitizerCoverageModulePass(Registry);
  initializeDataFlowSanitizerPass(Registry);
  initializeEfficiencySanitizerPass(Registry);
}

/// LLVMInitializeInstrumentation - C binding for
/// initializeInstrumentation.
void LLVMInitializeInstrumentation(LLVMPassRegistryRef R) {
  initializeInstrumentation(*unwrap(R));
}
