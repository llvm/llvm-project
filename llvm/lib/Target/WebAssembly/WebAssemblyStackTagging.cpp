//===- WebAssemblyTargetMachine.cpp - Define TargetMachine for WebAssembly -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "WebAssembly.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/Transforms/Utils/MemoryTaggingSupport.h"

using namespace llvm;

#define DEBUG_TYPE "wasm-stack-tagging"

namespace {

static cl::opt<bool> ClMergeInit(
    "stack-tagging-merge-init", cl::Hidden, cl::init(true),
    cl::desc("merge stack variable initializers with tagging when possible"));

static cl::opt<bool>
    ClUseStackSafety("stack-tagging-use-stack-safety", cl::Hidden,
                     cl::init(true),
                     cl::desc("Use Stack Safety analysis results"));

static cl::opt<unsigned> ClScanLimit("stack-tagging-merge-init-scan-limit",
                                     cl::init(40), cl::Hidden);

static cl::opt<unsigned>
    ClMergeInitSizeLimit("stack-tagging-merge-init-size-limit", cl::init(272),
                         cl::Hidden);

static cl::opt<size_t> ClMaxLifetimes(
    "stack-tagging-max-lifetimes-for-alloca", cl::Hidden, cl::init(3),
    cl::ReallyHidden,
    cl::desc("How many lifetime ends to handle for a single alloca."),
    cl::Optional);

struct WebAssemblyStackTagging : public FunctionPass {
  static char ID;
  StackSafetyGlobalInfo const *SSI = nullptr;
  AAResults *AA = nullptr;
  WebAssemblyStackTagging() : FunctionPass(ID) {}

  bool runOnFunction(Function &) override;
private:
#if 1
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<StackSafetyGlobalInfoWrapperPass>();
#if 0
    if (MergeInit)
      AU.addRequired<AAResultsWrapperPass>();
#endif
  }
#endif
}; // end of struct Hello

static const inline Align kTagGranuleSize = Align(16);

}

bool WebAssemblyStackTagging::runOnFunction(Function & Fn) {
  if (!Fn.hasFnAttribute(Attribute::SanitizeMemTag))
    return false;
  auto F = &Fn;
  SSI = &getAnalysis<StackSafetyGlobalInfoWrapperPass>().getResult();
  memtag::StackInfoBuilder SIB(SSI);
  for (Instruction &I : instructions(F))
    SIB.visit(I);
  memtag::StackInfo &SInfo = SIB.get();

  std::unique_ptr<DominatorTree> DeleteDT;
  DominatorTree *DT = nullptr;
  if (auto *P = getAnalysisIfAvailable<DominatorTreeWrapperPass>())
    DT = &P->getDomTree();

  if (DT == nullptr) {
    DeleteDT = std::make_unique<DominatorTree>(*F);
    DT = DeleteDT.get();
  }

#if 0
  std::unique_ptr<PostDominatorTree> DeletePDT;
  PostDominatorTree *PDT = nullptr;
  if (auto *P = getAnalysisIfAvailable<PostDominatorTreeWrapperPass>())
    PDT = &P->getPostDomTree();

  if (PDT == nullptr) {
    DeletePDT = std::make_unique<PostDominatorTree>(*F);
    PDT = DeletePDT.get();
  }
#endif

  std::unique_ptr<LoopInfo> DeleteLI;
  LoopInfo *LI = nullptr;
  if (auto *LIWP = getAnalysisIfAvailable<LoopInfoWrapperPass>()) {
    LI = &LIWP->getLoopInfo();
  } else {
    DeleteLI = std::make_unique<LoopInfo>(*DT);
    LI = DeleteLI.get();
  }


#if 1
  int NextTag = 0;
  for (auto &I : SInfo.AllocasToInstrument) {
    memtag::AllocaInfo &Info = I.second;
    TrackingVH<Instruction> OldAI = Info.AI;
    memtag::alignAndPadAlloca(Info, kTagGranuleSize);
    AllocaInst *AI = Info.AI;
    int Tag = NextTag;
    NextTag = (NextTag + 1) % 16;
    IRBuilder<> IRB(Info.AI->getNextNode());

    // Calls to functions that may return twice (e.g. setjmp) confuse the
    // postdominator analysis, and will leave us to keep memory tagged after
    // function return. Work around this by always untagging at every return
    // statement if return_twice functions are called.
    bool StandardLifetime =
        SInfo.UnrecognizedLifetimes.empty() &&
        memtag::isStandardLifetime(Info.LifetimeStart, Info.LifetimeEnd, DT, LI,
                                   ClMaxLifetimes) &&
        !SInfo.CallsReturnTwice;
//    LLVM_DEBUG(dbgs() << I << "\n");
  }
#endif
  return true;
}

char WebAssemblyStackTagging::ID = 0;
INITIALIZE_PASS_BEGIN(WebAssemblyStackTagging, DEBUG_TYPE, "WebAssembly Stack Tagging",
                      false, false)
#if 0
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
#endif
INITIALIZE_PASS_DEPENDENCY(StackSafetyGlobalInfoWrapperPass)
INITIALIZE_PASS_END(WebAssemblyStackTagging, DEBUG_TYPE, "WebAssembly Stack Tagging",
                    false, false)

FunctionPass *llvm::createWebAssemblyStackTaggingPass() {
  return new WebAssemblyStackTagging();
}
