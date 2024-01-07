//===- WebAssemblyTargetMachine.cpp - Define TargetMachine for WebAssembly -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "WebAssembly.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsWebAssembly.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/MemoryTaggingSupport.h"

using namespace llvm;

#define DEBUG_TYPE "wasm-stack-tagging"

namespace {

struct WebAssemblyStackTagging : public FunctionPass {
  static char ID;
  StackSafetyGlobalInfo const *SSI = nullptr;
  DataLayout const *DL = nullptr;
  AAResults *AA = nullptr;
  WebAssemblyStackTagging() : FunctionPass(ID) {}

  void untagAlloca(AllocaInst *AI, Instruction *InsertBefore, uint64_t Size,
                   Function *StoreTagDecl, Type *ArgOp0Type);

  bool runOnFunction(Function &) override;

private:
  Function *F = nullptr;
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

} // namespace

void WebAssemblyStackTagging::untagAlloca(AllocaInst *AI,
                                          Instruction *InsertBefore,
                                          uint64_t Size, Function *StoreTagDecl,
                                          Type *ArgOp0Type) {

  IRBuilder<> IRB(InsertBefore);
  IRB.CreateCall(StoreTagDecl, {IRB.getInt32(0), AI, ConstantInt::get(ArgOp0Type, Size)});
}

bool WebAssemblyStackTagging::runOnFunction(Function &Fn) {
  if (!Fn.hasFnAttribute(Attribute::SanitizeMemTag))
    return false;

  F = &Fn;
  DL = &Fn.getParent()->getDataLayout();
#if 1
// Unsure of the purpose of the SSI analysis in this context.
// Please place it under further review.
  SSI = &getAnalysis<StackSafetyGlobalInfoWrapperPass>().getResult();
#endif
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
  for (auto &I : SInfo.AllocasToInstrument) {
    memtag::AllocaInfo &Info = I.second;
    TrackingVH<Instruction> OldAI = Info.AI;
    memtag::alignAndPadAlloca(Info, kTagGranuleSize);
    AllocaInst *AI = Info.AI;
    IRBuilder<> IRB(Info.AI->getNextNode());
    Type *Int32Type = IRB.getInt32Ty();
    Type *Int64Type = IRB.getInt64Ty();
    Function *RandomStoreTagDecl = Intrinsic::getDeclaration(
        F->getParent(), Intrinsic::wasm_memory_randomstoretag, {Int64Type});
    Function *StoreTagDecl = Intrinsic::getDeclaration(
        F->getParent(), Intrinsic::wasm_memory_storetag, {Int64Type});

    // Calls to functions that may return twice (e.g. setjmp) confuse the
    // postdominator analysis, and will leave us to keep memory tagged after
    // function return. Work around this by always untagging at every return
    // statement if return_twice functions are called.
    bool StandardLifetime =
        SInfo.UnrecognizedLifetimes.empty() &&
        memtag::isStandardLifetime(Info.LifetimeStart, Info.LifetimeEnd, DT, LI,
                                   3) &&
        !SInfo.CallsReturnTwice;
    if (StandardLifetime) {
      IntrinsicInst *Start = Info.LifetimeStart[0];
      uint64_t Size =
          cast<ConstantInt>(Start->getArgOperand(0))->getZExtValue();
      Size = alignTo(Size, kTagGranuleSize);
      Instruction *RandomStoreTagCall = IRB.CreateCall(
          RandomStoreTagDecl, {ConstantInt::get(Int32Type, 0), Info.AI, ConstantInt::get(Int64Type, Size)});
      if (Info.AI->hasName())
        RandomStoreTagCall->setName(Info.AI->getName() + ".tag");
      Info.AI->replaceAllUsesWith(RandomStoreTagCall);
      RandomStoreTagCall->setOperand(0, Info.AI);

      auto TagEnd = [&](Instruction *Node) {
        untagAlloca(AI, Node, Size, StoreTagDecl, Int64Type);
      };
      if (!DT || !PDT ||
          !memtag::forAllReachableExits(*DT, *PDT, *LI, Start, Info.LifetimeEnd,
                                        SInfo.RetVec, TagEnd)) {
        for (auto *End : Info.LifetimeEnd)
          End->eraseFromParent();
      }
    } else {
      uint64_t Size = *Info.AI->getAllocationSize(*DL);
      Instruction *RandomStoreTagCall = IRB.CreateCall(RandomStoreTagDecl,
		      {ConstantInt::get(Int32Type, 0), AI, ConstantInt::get(Int64Type, Size)});
      if (Info.AI->hasName())
        RandomStoreTagCall->setName(Info.AI->getName() + ".tag");
      Info.AI->replaceAllUsesWith(RandomStoreTagCall);
      RandomStoreTagCall->setOperand(0, Info.AI);
      for (auto *RI : SInfo.RetVec) {
        untagAlloca(AI, RI, Size, StoreTagDecl, Int64Type);
      }
      // We may have inserted tag/untag outside of any lifetime interval.
      // Remove all lifetime intrinsics for this alloca.
      for (auto *II : Info.LifetimeStart)
        II->eraseFromParent();
      for (auto *II : Info.LifetimeEnd)
        II->eraseFromParent();
    }
    // Fixup debug intrinsics to point to the new alloca.
    for (auto *DVI : Info.DbgVariableIntrinsics)
      DVI->replaceVariableLocationOp(OldAI, Info.AI);
  }
  return true;
}

char WebAssemblyStackTagging::ID = 0;
INITIALIZE_PASS_BEGIN(WebAssemblyStackTagging, DEBUG_TYPE,
                      "WebAssembly Stack Tagging", false, false)
#if 0
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
#endif
INITIALIZE_PASS_DEPENDENCY(StackSafetyGlobalInfoWrapperPass)
INITIALIZE_PASS_END(WebAssemblyStackTagging, DEBUG_TYPE,
                    "WebAssembly Stack Tagging", false, false)

FunctionPass *llvm::createWebAssemblyStackTaggingPass() {
  return new WebAssemblyStackTagging();
}
