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
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Transforms/Utils/MemoryTaggingSupport.h"

using namespace llvm;

#define DEBUG_TYPE "wasm-stack-tagging"

namespace {

struct WebAssemblyStackTagging : public FunctionPass {
  static char ID;
  StackSafetyGlobalInfo const *SSI = nullptr;
  AAResults *AA = nullptr;
  WebAssemblyStackTagging() : FunctionPass(ID) {}

  void tagAlloca(AllocaInst *AI, Instruction *InsertBefore, Value *Ptr,
                 uint64_t Size);
  void untagAlloca(AllocaInst *AI, Instruction *InsertBefore, uint64_t Size);

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

void WebAssemblyStackTagging::tagAlloca(AllocaInst *AI, Instruction *InsertBefore,
                                    Value *Ptr, uint64_t Size) {

#if 0
  auto SetTagZeroFunc =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_settag_zero);
  auto StgpFunc =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_stgp);

  InitializerBuilder IB(Size, DL, Ptr, SetTagFunc, SetTagZeroFunc, StgpFunc);
  bool LittleEndian =
      Triple(AI->getModule()->getTargetTriple()).isLittleEndian();
  // Current implementation of initializer merging assumes little endianness.
  if (MergeInit && !F->hasOptNone() && LittleEndian &&
      Size < ClMergeInitSizeLimit) {
    LLVM_DEBUG(dbgs() << "collecting initializers for " << *AI
                      << ", size = " << Size << "\n");
    InsertBefore = collectInitializers(InsertBefore, Ptr, Size, IB);
  }

  IRBuilder<> IRB(InsertBefore);
  IB.generate(IRB);
#endif
}

void WebAssemblyStackTagging::untagAlloca(AllocaInst *AI, Instruction *InsertBefore,
                                      uint64_t Size) {
#if 0
  IRBuilder<> IRB(InsertBefore);
  IRB.CreateCall(SetTagFunc, {IRB.CreatePointerCast(AI, IRB.getPtrTy()),
                              ConstantInt::get(IRB.getInt64Ty(), Size)});
#endif
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

  std::unique_ptr<PostDominatorTree> DeletePDT;
  PostDominatorTree *PDT = nullptr;
  if (auto *P = getAnalysisIfAvailable<PostDominatorTreeWrapperPass>())
    PDT = &P->getPostDomTree();

  if (PDT == nullptr) {
    DeletePDT = std::make_unique<PostDominatorTree>(*F);
    PDT = DeletePDT.get();
  }

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
                                   3) &&
        !SInfo.CallsReturnTwice;
#if 0
    if (StandardLifetime) {
      IntrinsicInst *Start = Info.LifetimeStart[0];
      uint64_t Size =
          cast<ConstantInt>(Start->getArgOperand(0))->getZExtValue();
      Size = alignTo(Size, kTagGranuleSize);
      tagAlloca(AI, Start->getNextNode(), Start->getArgOperand(1), Size);

      auto TagEnd = [&](Instruction *Node) { untagAlloca(AI, Node, Size); };
      if (!DT || !PDT ||
          !memtag::forAllReachableExits(*DT, *PDT, *LI, Start, Info.LifetimeEnd,
                                        SInfo.RetVec, TagEnd)) {
        for (auto *End : Info.LifetimeEnd)
          End->eraseFromParent();
      }
    } else {
      uint64_t Size = *Info.AI->getAllocationSize(*DL);
      Value *Ptr = IRB.CreatePointerCast(TagPCall, IRB.getPtrTy());
      tagAlloca(AI, &*IRB.GetInsertPoint(), Ptr, Size);
      for (auto *RI : SInfo.RetVec) {
        untagAlloca(AI, RI, Size);
      }
      // We may have inserted tag/untag outside of any lifetime interval.
      // Remove all lifetime intrinsics for this alloca.
      for (auto *II : Info.LifetimeStart)
        II->eraseFromParent();
      for (auto *II : Info.LifetimeEnd)
        II->eraseFromParent();
    }
#endif
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
