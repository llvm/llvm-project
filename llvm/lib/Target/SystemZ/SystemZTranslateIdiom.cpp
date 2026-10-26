//===-- SystemZTranslateIdiom.cpp - Recongize TR loops --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that recognizes a single-block loop of the form:
// 
//  loop:
//      %p = gep i8, %buf, %i
//      %c = load i8, %p
//      %t = gep i8, %table, (zext i8 %c)
//      %v = load i8, %t
//      store i8 %v, %p
// 
// and replaces it with 
//
//  call void @llvm.s390.translate(ptr %buf, i64 <tripcount>, ptr %table)
//
// Note: <tripcount> must be in the range [1, 256].
//
//===----------------------------------------------------------------------===//

#include "SystemZ.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsS390.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils.h" // LoopSimplifyID, LCSSAID
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;

#define DEBUG_TYPE "systemz-translate-idiom"

namespace {

// Everything the matcher discovers; consumed by legality checks and rewrite.
struct Match {
  StoreInst *Store = nullptr;
  Value *TableBase = nullptr;
  const SCEV *BuffStart = nullptr; // First address written by the loop.
  const SCEV *TripCount = nullptr; // i64 SCEV, exact number of bytes.
};

// Pass-manager-agnostic core. Construct it and call run(). Returns true
// if the loop was replaced (and therefore deleted).
class SystemZTranslateIdiom {
  Loop *L;
  ScalarEvolution &SE;
  AAResults &AA;
  DominatorTree &DT;
  LoopInfo &LI;

public:
  SystemZTranslateIdiom(Loop *L, ScalarEvolution &SE, AAResults &AA,
                        DominatorTree &DT, LoopInfo &LI)
      : L(L), SE(SE), AA(AA), DT(DT), LI(LI) {}

  bool run() {
    Match M;
    if (!matchShape(M) || !matchChain(M) || !isLegal(M))
        return false;
    rewrite(M);
    return true;
  }

private:
  bool matchShape(Match &M);
  bool matchChain(Match &M);
  bool isLegal(const Match &M);
  void rewrite(const Match &M);
};

} // end anonymous namespace

bool SystemZTranslateIdiom::matchShape(Match &M) {
  if(!L->getLoopPreheader() || L->getNumBlocks() != 1 ||
     !L->getUniqueExitBlock() || !L->hasDedicatedExits())
    return false;

  // TR handles 1..256 bytes. Require a provable upper bound in that range.
  unsigned MaxTC = SE.getSmallConstantMaxTripCount(L);
  if (MaxTC < 1 || MaxTC > 256)
    return false;

  const SCEV *BECount = SE.getBackedgeTakenCount(L);
  if(isa<SCEVCouldNotCompute>(BECount))
    return false;

  Type *I64 = Type::getInt64Ty(L->getHeader()->getContext());
  M.TripCount = 
      SE.getTruncateOrZeroExtend(SE.getTripCountFromExitCount(BECount), I64);
  return true;
}

bool SystemZTranslateIdiom::matchChain(Match &M) {
    BasicBlock *BB = L->getHeader();

    // Only one instruction may write memory, and it must be a simple i8 store.
    StoreInst *SI = nullptr;
    for (Instruction &I : *BB) {
      if (!I.mayWriteToMemory())
        continue;
      if (SI)
        return false;
      SI = dyn_cast<StoreInst>(&I);
      if (!SI)
        return false;
    }
    if (!SI || !SI->isSimple() || SI->getPointerAddressSpace() != 0 ||
        !SI->getValueOperand()->getType()->isIntegerTy(8))
      return false;

    // Value being stored: load i8 from table[zext(byte)].
    auto *TableLd = dyn_cast<LoadInst>(SI->getValueOperand());
    if (!TableLd || !TableLd->isSimple() || !TableLd->hasOneUse() ||
        TableLd->getPointerAddressSpace() != 0)
      return false;
    
    auto *GEP = dyn_cast<GetElementPtrInst>(TableLd->getPointerOperand());
    if (!GEP || !GEP->hasOneUse() || GEP->getNumOperands() != 2 || 
        !GEP->getSourceElementType()->isIntegerTy(8) ||
        !L->isLoopInvariant(GEP->getPointerOperand()))
      return false;
    
    // The index must also be an unsigned byte extension (sext is invalid).
    auto *ZE = dyn_cast<ZExtInst>(GEP->getOperand(1));
    if (!ZE || !ZE->hasOneUse() || !ZE->getSrcTy()->isIntegerTy(8))
      return false;
    
    auto *SrcLd = dyn_cast<LoadInst>(ZE->getOperand(0));
    if (!SrcLd || !SrcLd->isSimple() || !SrcLd->hasOneUse())
      return false;

    // The byte that is read must also be the byte that is overwritten, and
    // the address must advance by 1 byte per iteration: {Start,+,1}<L>.
    const SCEV *PtrS = SE.getSCEV(SI->getPointerOperand());
    if (SE.getSCEV(SrcLd->getPointerOperand()) != PtrS)
      return false;
    auto *Rec = dyn_cast<SCEVAddRecExpr>(PtrS);
    if (!Rec || Rec->getLoop() != L || !Rec->isAffine())
        return false;
    auto *Step = dyn_cast<SCEVConstant>(Rec->getStepRecurrence(SE));
    if (!Step || !Step->getValue()->isOne())
      return false;

    // Everything else in the loop must be side-effect free, touch no
    // memory, and be unused outside the loop, so that deleting the
    // loop is safe.
    SmallPtrSet<Instruction *, 8> Chain = {SI, TableLd, GEP, ZE, SrcLd};
    for (Instruction &I : *BB) {
      if (Chain.contains(&I))
        continue;
      if (I.mayReadOrWriteMemory() || I.mayHaveSideEffects())
        return false;
      for (User *U : I.users())
        if (!L->contains(cast<Instruction>(U)))
          return false;
    }

    M.Store = SI;
    M.TableBase = GEP->getPointerOperand();
    M.BuffStart = Rec->getStart();
    return true;
}

bool SystemZTranslateIdiom::isLegal(const Match &M) {
    // The table must not overlap with the buffer.
    MemoryLocation Buf(M.Store->getPointerOperand(),
                       LocationSize::afterPointer());
    MemoryLocation Table(M.TableBase, LocationSize::precise(256));
    if (!AA.isNoAlias(Buf, Table))
      return false;
    
    // The preheader code must be expandable.
    SCEVExpander Expander(SE, "systemz-tr");
    return Expander.isSafeToExpand(M.BuffStart) &&
           Expander.isSafeToExpand(M.TripCount);
}

void SystemZTranslateIdiom::rewrite(const Match &M) {
  BasicBlock *Preheader = L->getLoopPreheader();
  Instruction *InsertPt = Preheader->getTerminator();
  LLVMContext &Ctx = Preheader->getContext();

  SCEVExpander Expander(SE, "systemz-tr");
  Value *Buf = Expander.expandCodeFor(M.BuffStart, PointerType::get(Ctx, 0),
                                      InsertPt);
  Value *Len = 
      Expander.expandCodeFor(M.TripCount, Type::getInt64Ty(Ctx), InsertPt);
  
  IRBuilder<> B(InsertPt);
  B.CreateIntrinsic(Intrinsic::s390_translate, {}, {Buf, Len, M.TableBase});

  LLVM_DEBUG(dbgs() << "systemz-translate-idiom: replaced loop "
                    << L->getName() << " with llvm.s390.translate\n");

  // The loop is now computing unused values, so 
  // delete it and keep DT/SE/LI consistent.
  deleteDeadLoop(L, &DT, &SE, &LI, /*MSSA=*/nullptr);
}

//===----------------------------------------------------------------------===//
// Legacy pass manager wrapper
//===----------------------------------------------------------------------===//
namespace {

class SystemZTranslateIdiomLegacyPass : public LoopPass {
  public:
    static char ID;

    SystemZTranslateIdiomLegacyPass() : LoopPass(ID) {
      initializeSystemZTranslateIdiomLegacyPassPass(
          *PassRegistry::getPassRegistry());
    }

    StringRef getPassName() const override {
      return "SystemZ Recognize TR (translate) loop idiom";
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      // Add required analyses
      AU.addRequired<AAResultsWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<ScalarEvolutionWrapperPass>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);

      // Add preserved analyses
      AU.addPreserved<DominatorTreeWrapperPass>();
      AU.addPreserved<LoopInfoWrapperPass>();
      AU.addPreserved<ScalarEvolutionWrapperPass>();
      AU.addPreservedID(LoopSimplifyID);
      AU.addPreservedID(LCSSAID);
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM) override {
      if(skipLoop(L))
        return false;
      
      auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
      auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
      auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();

      SystemZTranslateIdiom Idiom(L, SE, AA, DT, LI);
      if(!Idiom.run())
        return false;
      
      LPM.markLoopAsDeleted(*L);
      return true;
    }
};

} // end anonymous namespace

char SystemZTranslateIdiomLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(SystemZTranslateIdiomLegacyPass,
                      "systemz-translate-idiom",
                      "SystemZ Recognize TR (translate) loop idiom",
                      false,
                      false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(SystemZTranslateIdiomLegacyPass,
                    "systemz-translate-idiom",
                    "SystemZ Recognize TR (translate) loop idiom",
                    false,
                    false)

Pass *llvm::createSystemZTranslateIdiomPass() {
  return new SystemZTranslateIdiomLegacyPass();
}


