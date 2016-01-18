//===------ PPCLoopPreIncPrep.cpp - Loop Pre-Inc. AM Prep. Pass -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to prepare loops for pre-increment addressing
// modes. Additional PHIs are created for loop induction variables used by
// load/store instructions so that the pre-increment forms can be used.
// Generically, this means transforming loops like this:
//   for (int i = 0; i < n; ++i)
//     array[i] = c;
// to look like this:
//   T *p = array[-1];
//   for (int i = 0; i < n; ++i)
//     *++p = c;
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ppc-loop-preinc-prep"
#include "PPC.h"
#include "PPCTargetMachine.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
using namespace llvm;

// By default, we limit this to creating 16 PHIs (which is a little over half
// of the allocatable register set).
static cl::opt<unsigned> MaxVars("ppc-preinc-prep-max-vars",
                                 cl::Hidden, cl::init(16),
  cl::desc("Potential PHI threshold for PPC preinc loop prep"));

namespace llvm {
  void initializePPCLoopPreIncPrepPass(PassRegistry&);
}

namespace {

  class PPCLoopPreIncPrep : public FunctionPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    PPCLoopPreIncPrep() : FunctionPass(ID), TM(nullptr) {
      initializePPCLoopPreIncPrepPass(*PassRegistry::getPassRegistry());
    }
    PPCLoopPreIncPrep(PPCTargetMachine &TM) : FunctionPass(ID), TM(&TM) {
      initializePPCLoopPreIncPrepPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addPreserved<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addPreserved<LoopInfoWrapperPass>();
      AU.addRequired<ScalarEvolutionWrapperPass>();
    }

    bool runOnFunction(Function &F) override;

    bool runOnLoop(Loop *L);
    void simplifyLoopLatch(Loop *L);
    bool rotateLoop(Loop *L);

  private:
    PPCTargetMachine *TM;
    LoopInfo *LI;
    ScalarEvolution *SE;
  };
}

char PPCLoopPreIncPrep::ID = 0;
static const char *name = "Prepare loop for pre-inc. addressing modes";
INITIALIZE_PASS_BEGIN(PPCLoopPreIncPrep, DEBUG_TYPE, name, false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(PPCLoopPreIncPrep, DEBUG_TYPE, name, false, false)

FunctionPass *llvm::createPPCLoopPreIncPrepPass(PPCTargetMachine &TM) {
  return new PPCLoopPreIncPrep(TM);
}

namespace {
  struct SCEVLess : std::binary_function<const SCEV *, const SCEV *, bool>
  {
    SCEVLess(ScalarEvolution *SE) : SE(SE) {}

    bool operator() (const SCEV *X, const SCEV *Y) const {
      const SCEV *Diff = SE->getMinusSCEV(X, Y);
      return cast<SCEVConstant>(Diff)->getValue()->getSExtValue() < 0;
    }

  protected:
    ScalarEvolution *SE;
  };
}

static bool IsPtrInBounds(Value *BasePtr) {
  Value *StrippedBasePtr = BasePtr;
  while (BitCastInst *BC = dyn_cast<BitCastInst>(StrippedBasePtr))
    StrippedBasePtr = BC->getOperand(0);
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(StrippedBasePtr))
    return GEP->isInBounds();

  return false;
}

static Value *GetPointerOperand(Value *MemI) {
  if (LoadInst *LMemI = dyn_cast<LoadInst>(MemI)) {
    return LMemI->getPointerOperand();
  } else if (StoreInst *SMemI = dyn_cast<StoreInst>(MemI)) {
    return SMemI->getPointerOperand();
  } else if (IntrinsicInst *IMemI = dyn_cast<IntrinsicInst>(MemI)) {
    if (IMemI->getIntrinsicID() == Intrinsic::prefetch)
      return IMemI->getArgOperand(0);
  }

  return 0;
}

bool PPCLoopPreIncPrep::runOnFunction(Function &F) {
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();

  bool MadeChange = false;

  for (auto I = LI->begin(), IE = LI->end(); I != IE; ++I)
    for (auto L = df_begin(*I), LE = df_end(*I); L != LE; ++L)
      MadeChange |= runOnLoop(*L);

  return MadeChange;
}

bool PPCLoopPreIncPrep::runOnLoop(Loop *L) {
  bool MadeChange = false;

  // Only prep. the inner-most loop
  if (!L->empty())
    return MadeChange;

  DEBUG(dbgs() << "PIP: Examining: " << *L << "\n");

  BasicBlock *Header = L->getHeader();

  const PPCSubtarget *ST =
    TM ? TM->getSubtargetImpl(*Header->getParent()) : nullptr;

  unsigned HeaderLoopPredCount =
    std::distance(pred_begin(Header), pred_end(Header));

  // Collect buckets of comparable addresses used by loads and stores.
  typedef std::multimap<const SCEV *, Instruction *, SCEVLess> Bucket;
  SmallVector<Bucket, 16> Buckets;
  for (Loop::block_iterator I = L->block_begin(), IE = L->block_end();
       I != IE; ++I) {
    for (BasicBlock::iterator J = (*I)->begin(), JE = (*I)->end();
        J != JE; ++J) {
      Value *PtrValue;
      Instruction *MemI;

      if (LoadInst *LMemI = dyn_cast<LoadInst>(J)) {
        MemI = LMemI;
        PtrValue = LMemI->getPointerOperand();
      } else if (StoreInst *SMemI = dyn_cast<StoreInst>(J)) {
        MemI = SMemI;
        PtrValue = SMemI->getPointerOperand();
      } else if (IntrinsicInst *IMemI = dyn_cast<IntrinsicInst>(J)) {
        if (IMemI->getIntrinsicID() == Intrinsic::prefetch) {
          MemI = IMemI;
          PtrValue = IMemI->getArgOperand(0);
        } else continue;
      } else continue;

      unsigned PtrAddrSpace = PtrValue->getType()->getPointerAddressSpace();
      if (PtrAddrSpace)
        continue;

      // There are no update forms for Altivec vector load/stores.
      if (ST && ST->hasAltivec() &&
          PtrValue->getType()->getPointerElementType()->isVectorTy())
        continue;

      if (L->isLoopInvariant(PtrValue))
        continue;

      const SCEV *LSCEV = SE->getSCEVAtScope(PtrValue, L);
      if (const SCEVAddRecExpr *LARSCEV = dyn_cast<SCEVAddRecExpr>(LSCEV)) {
        if (LARSCEV->getLoop() != L)
          continue;
      } else {
        continue;
      }

      bool FoundBucket = false;
      for (unsigned i = 0, e = Buckets.size(); i != e; ++i)
        for (Bucket::iterator K = Buckets[i].begin(), KE = Buckets[i].end();
             K != KE; ++K) {
          const SCEV *Diff = SE->getMinusSCEV(K->first, LSCEV);
          if (isa<SCEVConstant>(Diff)) {
            Buckets[i].insert(std::make_pair(LSCEV, MemI));
            FoundBucket = true;
            break;
          }
        }

      if (!FoundBucket) {
        Buckets.push_back(Bucket(SCEVLess(SE)));
        Buckets[Buckets.size()-1].insert(std::make_pair(LSCEV, MemI));
      }
    }
  }

  if (Buckets.empty() || Buckets.size() > MaxVars)
    return MadeChange;

  BasicBlock *LoopPredecessor = L->getLoopPredecessor();
  // If there is no loop predecessor, or the loop predecessor's terminator
  // returns a value (which might contribute to determining the loop's
  // iteration space), insert a new preheader for the loop.
  if (!LoopPredecessor ||
      !LoopPredecessor->getTerminator()->getType()->isVoidTy()) {
    LoopPredecessor = InsertPreheaderForLoop(L, this);
    if (LoopPredecessor)
      MadeChange = true;
  }
  if (!LoopPredecessor)
    return MadeChange;

  DEBUG(dbgs() << "PIP: Found " << Buckets.size() << " buckets\n");

  SmallSet<BasicBlock *, 16> BBChanged;
  for (unsigned i = 0, e = Buckets.size(); i != e; ++i) {
    // The base address of each bucket is transformed into a phi and the others
    // are rewritten as offsets of that variable.

    const SCEVAddRecExpr *BasePtrSCEV =
      cast<SCEVAddRecExpr>(Buckets[i].begin()->first);
    if (!BasePtrSCEV->isAffine())
      continue;

    DEBUG(dbgs() << "PIP: Transforming: " << *BasePtrSCEV << "\n");
    assert(BasePtrSCEV->getLoop() == L &&
           "AddRec for the wrong loop?");

    Instruction *MemI = Buckets[i].begin()->second;
    Value *BasePtr = GetPointerOperand(MemI);
    assert(BasePtr && "No pointer operand");

    Type *I8Ty = Type::getInt8Ty(MemI->getParent()->getContext());
    Type *I8PtrTy = Type::getInt8PtrTy(MemI->getParent()->getContext(),
      BasePtr->getType()->getPointerAddressSpace());

    const SCEV *BasePtrStartSCEV = BasePtrSCEV->getStart();
    if (!SE->isLoopInvariant(BasePtrStartSCEV, L))
      continue;

    const SCEVConstant *BasePtrIncSCEV =
      dyn_cast<SCEVConstant>(BasePtrSCEV->getStepRecurrence(*SE));
    if (!BasePtrIncSCEV)
      continue;
    BasePtrStartSCEV = SE->getMinusSCEV(BasePtrStartSCEV, BasePtrIncSCEV);
    if (!isSafeToExpand(BasePtrStartSCEV, *SE))
      continue;

    DEBUG(dbgs() << "PIP: New start is: " << *BasePtrStartSCEV << "\n");

    PHINode *NewPHI = PHINode::Create(I8PtrTy, HeaderLoopPredCount,
      MemI->hasName() ? MemI->getName() + ".phi" : "",
      Header->getFirstNonPHI());

    SCEVExpander SCEVE(*SE, Header->getModule()->getDataLayout(), "pistart");
    Value *BasePtrStart = SCEVE.expandCodeFor(BasePtrStartSCEV, I8PtrTy,
      LoopPredecessor->getTerminator());

    // Note that LoopPredecessor might occur in the predecessor list multiple
    // times, and we need to add it the right number of times.
    for (pred_iterator PI = pred_begin(Header), PE = pred_end(Header);
         PI != PE; ++PI) {
      if (*PI != LoopPredecessor)
        continue;

      NewPHI->addIncoming(BasePtrStart, LoopPredecessor);
    }

    Instruction *InsPoint = &*Header->getFirstInsertionPt();
    GetElementPtrInst *PtrInc = GetElementPtrInst::Create(
        I8Ty, NewPHI, BasePtrIncSCEV->getValue(),
        MemI->hasName() ? MemI->getName() + ".inc" : "", InsPoint);
    PtrInc->setIsInBounds(IsPtrInBounds(BasePtr));
    for (pred_iterator PI = pred_begin(Header), PE = pred_end(Header);
         PI != PE; ++PI) {
      if (*PI == LoopPredecessor)
        continue;

      NewPHI->addIncoming(PtrInc, *PI);
    }

    Instruction *NewBasePtr;
    if (PtrInc->getType() != BasePtr->getType())
      NewBasePtr = new BitCastInst(PtrInc, BasePtr->getType(),
        PtrInc->hasName() ? PtrInc->getName() + ".cast" : "", InsPoint);
    else
      NewBasePtr = PtrInc;

    if (Instruction *IDel = dyn_cast<Instruction>(BasePtr))
      BBChanged.insert(IDel->getParent());
    BasePtr->replaceAllUsesWith(NewBasePtr);
    RecursivelyDeleteTriviallyDeadInstructions(BasePtr);

    Value *LastNewPtr = NewBasePtr;
    for (Bucket::iterator I = std::next(Buckets[i].begin()),
         IE = Buckets[i].end(); I != IE; ++I) {
      Value *Ptr = GetPointerOperand(I->second);
      assert(Ptr && "No pointer operand");
      if (Ptr == LastNewPtr)
        continue;

      Instruction *RealNewPtr;
      const SCEVConstant *Diff =
        cast<SCEVConstant>(SE->getMinusSCEV(I->first, BasePtrSCEV));
      if (Diff->isZero()) {
        RealNewPtr = NewBasePtr;
      } else {
        Instruction *PtrIP = dyn_cast<Instruction>(Ptr);
        if (PtrIP && isa<Instruction>(NewBasePtr) &&
            cast<Instruction>(NewBasePtr)->getParent() == PtrIP->getParent())
          PtrIP = 0;
        else if (isa<PHINode>(PtrIP))
          PtrIP = &*PtrIP->getParent()->getFirstInsertionPt();
        else if (!PtrIP)
          PtrIP = I->second;

        GetElementPtrInst *NewPtr = GetElementPtrInst::Create(
            I8Ty, PtrInc, Diff->getValue(),
            I->second->hasName() ? I->second->getName() + ".off" : "", PtrIP);
        if (!PtrIP)
          NewPtr->insertAfter(cast<Instruction>(PtrInc));
        NewPtr->setIsInBounds(IsPtrInBounds(Ptr));
        RealNewPtr = NewPtr;
      }

      if (Instruction *IDel = dyn_cast<Instruction>(Ptr))
        BBChanged.insert(IDel->getParent());

      Instruction *ReplNewPtr;
      if (Ptr->getType() != RealNewPtr->getType()) {
        ReplNewPtr = new BitCastInst(RealNewPtr, Ptr->getType(),
          Ptr->hasName() ? Ptr->getName() + ".cast" : "");
        ReplNewPtr->insertAfter(RealNewPtr);
      } else
        ReplNewPtr = RealNewPtr;

      Ptr->replaceAllUsesWith(ReplNewPtr);
      RecursivelyDeleteTriviallyDeadInstructions(Ptr);

      LastNewPtr = RealNewPtr;
    }

    MadeChange = true;
  }

  for (Loop::block_iterator I = L->block_begin(), IE = L->block_end();
       I != IE; ++I) {
    if (BBChanged.count(*I))
      DeleteDeadPHIs(*I);
  }

  return MadeChange;
}

