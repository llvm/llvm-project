//===- WebAssemblyTargetMachine.cpp - Define TargetMachine for WebAssembly -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "WebAssembly.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsWebAssembly.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
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

  Instruction *insertBaseTaggedPointer(
      const MapVector<AllocaInst *, memtag::AllocaInfo> &Allocas,
      const DominatorTree *DT);

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
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
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
  IRB.CreateCall(StoreTagDecl,
                 {IRB.getInt32(0), AI, ConstantInt::get(ArgOp0Type, Size)});
}

Instruction *WebAssemblyStackTagging::insertBaseTaggedPointer(
    const MapVector<AllocaInst *, memtag::AllocaInfo> &AllocasToInstrument,
    const DominatorTree *DT) {
  BasicBlock *PrologueBB = nullptr;
  // Try sinking IRG as deep as possible to avoid hurting shrink wrap.
  for (auto &I : AllocasToInstrument) {
    const memtag::AllocaInfo &Info = I.second;
    AllocaInst *AI = Info.AI;
    if (!PrologueBB) {
      PrologueBB = AI->getParent();
      continue;
    }
    PrologueBB = DT->findNearestCommonDominator(PrologueBB, AI->getParent());
  }
  assert(PrologueBB);

  IRBuilder<> IRB(&PrologueBB->front());
  Function *RdTag = Intrinsic::getOrInsertDeclaration(
      F->getParent(), Intrinsic::wasm_memtag_random);
  Instruction *Base =
      IRB.CreateCall(RdTag, {IRB.getInt32(0),
                             ::llvm::ConstantPointerNull::get(IRB.getPtrTy())});
  Base->setName("basetag");
  return Base;
}

bool WebAssemblyStackTagging::runOnFunction(Function &Fn) {
  if (!Fn.hasFnAttribute(Attribute::SanitizeMemTag))
    return false;

  Triple triplet(Fn.getParent()->getTargetTriple());
  bool iswasm32 = triplet.getArch() == ::llvm::Triple::wasm32;

  F = &Fn;
  DL = &Fn.getParent()->getDataLayout();

  OptimizationRemarkEmitter &ORE =
      getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  SSI = &getAnalysis<StackSafetyGlobalInfoWrapperPass>().getResult();
  memtag::StackInfoBuilder SIB(SSI, "webassembly-stack-tagging");
  for (Instruction &I : instructions(F))
    SIB.visit(ORE, I);
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
  auto &AllocasToInstrument = SInfo.AllocasToInstrument;
  if (AllocasToInstrument.empty()) {
    return true;
  }
  Instruction *Base = insertBaseTaggedPointer(AllocasToInstrument, DT);
  uint64_t NextTag = 0;
  for (auto &I : AllocasToInstrument) {
    memtag::AllocaInfo &Info = I.second;
    memtag::alignAndPadAlloca(Info, kTagGranuleSize);
 
    uint64_t Tag = NextTag;
    if (iswasm32) {
      Tag = static_cast<uint32_t>(Tag);
    }
    ++NextTag;
    AllocaInst *AI = Info.AI;
    IRBuilder<> IRB(Info.AI->getNextNode());
    Type *Int32Type = IRB.getInt32Ty();
    Type *Int64Type = IRB.getInt64Ty();
    Type *IntPtrType = iswasm32 ? Int32Type : Int64Type;
    Function *StoreTagDecl = Intrinsic::getOrInsertDeclaration(
        F->getParent(), Intrinsic::wasm_memtag_store, {IntPtrType});
    // Calls to functions that may return twice (e.g. setjmp) confuse the
    // postdominator analysis, and will leave us to keep memory tagged after
    // function return. Work around this by always untagging at every return
    // statement if return_twice functions are called.
    bool StandardLifetime =
        memtag::isStandardLifetime(Info.LifetimeStart, Info.LifetimeEnd, DT, LI,
                                   3) &&
        !SInfo.CallsReturnTwice;
    if (StandardLifetime) {
      auto *HintTagDecl = Intrinsic::getOrInsertDeclaration(
          F->getParent(), Intrinsic::wasm_memtag_hint, {IntPtrType});
      auto *TagPCall = IRB.CreateCall(
          HintTagDecl, {ConstantInt::get(Int32Type, 0), Info.AI, Base,
                        ConstantInt::get(IntPtrType, Tag)});
      if (Info.AI->hasName())
        TagPCall->setName(Info.AI->getName() + ".tag");
      Info.AI->replaceUsesWithIf(TagPCall, [&](const Use &U) {
        return !isa<LifetimeIntrinsic>(U.getUser());
      });
      TagPCall->setOperand(1, Info.AI);
      IntrinsicInst *Start = Info.LifetimeStart[0];
      uint64_t Size = *Info.AI->getAllocationSize(*DL);
      Size = alignTo(Size, kTagGranuleSize);

      IRBuilder<> IRB2(Start->getNextNode());
      IRB2.CreateCall(StoreTagDecl, {ConstantInt::get(Int32Type, 0), TagPCall,
                                     ConstantInt::get(IntPtrType, Size)});

      auto TagEnd = [&](Instruction *Node) {
        untagAlloca(AI, Node, Size, StoreTagDecl, IntPtrType);
      };
      if (!DT || !PDT ||
          !memtag::forAllReachableExits(*DT, *PDT, *LI, Start, Info.LifetimeEnd,
                                        SInfo.RetVec, TagEnd)) {
        for (auto *End : Info.LifetimeEnd)
          End->eraseFromParent();
      }
    } else {
      uint64_t Size = *Info.AI->getAllocationSize(*DL);
      auto *HintStoreTagDecl = Intrinsic::getOrInsertDeclaration(
          F->getParent(), Intrinsic::wasm_memtag_hintstore,
          {IntPtrType, IntPtrType});
      auto *TagPCall = IRB.CreateCall(HintStoreTagDecl,
                                      {ConstantInt::get(Int32Type, 0), Info.AI,
                                       ConstantInt::get(IntPtrType, Size), Base,
                                       ConstantInt::get(IntPtrType, Tag)});
      if (Info.AI->hasName())
        TagPCall->setName(Info.AI->getName() + ".tag");
      Info.AI->replaceAllUsesWith(TagPCall);
      TagPCall->setOperand(1, Info.AI);
      for (auto *RI : SInfo.RetVec) {
        untagAlloca(AI, RI, Size, StoreTagDecl, IntPtrType);
      }
      // We may have inserted tag/untag outside of any lifetime interval.
      // Remove all lifetime intrinsics for this alloca.
      for (auto *II : Info.LifetimeStart)
        II->eraseFromParent();
      for (auto *II : Info.LifetimeEnd)
        II->eraseFromParent();
    }

    memtag::annotateDebugRecords(Info, static_cast<unsigned long>(Tag));
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
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(WebAssemblyStackTagging, DEBUG_TYPE,
                    "WebAssembly Stack Tagging", false, false)

FunctionPass *llvm::createWebAssemblyStackTaggingPass() {
  return new WebAssemblyStackTagging();
}
