//===------ PPCLoopPreIncPrep.cpp - Loop Pre-Inc. AM Prep. Pass -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "PPCSubtarget.h"
#include "PPCTargetMachine.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <cassert>
#include <iterator>
#include <utility>

using namespace llvm;

// By default, we limit this to creating 16 PHIs (which is a little over half
// of the allocatable register set).
static cl::opt<unsigned> MaxVars("ppc-preinc-prep-max-vars",
                                 cl::Hidden, cl::init(16),
  cl::desc("Potential PHI threshold for PPC preinc loop prep"));

STATISTIC(PHINodeAlreadyExists, "PHI node already in pre-increment form");
STATISTIC(UpdFormChainRewritten, "Num of update form chain rewritten");

namespace {
  struct BucketElement {
    BucketElement(const SCEVConstant *O, Instruction *I) : Offset(O), Instr(I) {}
    BucketElement(Instruction *I) : Offset(nullptr), Instr(I) {}

    const SCEVConstant *Offset;
    Instruction *Instr;
  };

  struct Bucket {
    Bucket(const SCEV *B, Instruction *I) : BaseSCEV(B),
                                            Elements(1, BucketElement(I)) {}

    const SCEV *BaseSCEV;
    SmallVector<BucketElement, 16> Elements;
  };

  class PPCLoopPreIncPrep : public FunctionPass {
  public:
    static char ID; // Pass ID, replacement for typeid

    PPCLoopPreIncPrep() : FunctionPass(ID) {
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

  private:
    PPCTargetMachine *TM = nullptr;
    const PPCSubtarget *ST; 
    DominatorTree *DT;
    LoopInfo *LI;
    ScalarEvolution *SE;
    bool PreserveLCSSA;

    bool runOnLoop(Loop *L);

    /// Check if required PHI node is already exist in Loop \p L.
    bool alreadyPrepared(Loop *L, Instruction* MemI,
                         const SCEV *BasePtrStartSCEV,
                         const SCEVConstant *BasePtrIncSCEV);

    /// Collect condition matched(\p isValidCandidate() returns true)
    /// candidates in Loop \p L.
    SmallVector<Bucket, 16>
    collectCandidates(Loop *L,
                      std::function<bool(const Instruction *, const Value *)>
                          isValidCandidate,
                      unsigned MaxCandidateNum);

    /// Add a candidate to candidates \p Buckets.
    void addOneCandidate(Instruction *MemI, const SCEV *LSCEV,
                         SmallVector<Bucket, 16> &Buckets,
                         unsigned MaxCandidateNum);

    /// Prepare all candidates in \p Buckets for update form.
    bool updateFormPrep(Loop *L, SmallVector<Bucket, 16> &Buckets);

    /// Prepare for one chain \p BucketChain, find the best base element and
    /// update all other elements in \p BucketChain accordingly.
    bool prepareBaseForUpdateFormChain(Bucket &BucketChain);

    /// Rewrite load/store instructions in \p BucketChain according to
    /// preparation.
    bool rewriteLoadStores(Loop *L, Bucket &BucketChain,
                           SmallSet<BasicBlock *, 16> &BBChanged);
  };

} // end anonymous namespace

char PPCLoopPreIncPrep::ID = 0;
static const char *name = "Prepare loop for pre-inc. addressing modes";
INITIALIZE_PASS_BEGIN(PPCLoopPreIncPrep, DEBUG_TYPE, name, false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(PPCLoopPreIncPrep, DEBUG_TYPE, name, false, false)

static const std::string PHINodeNameSuffix    = ".phi";
static const std::string CastNodeNameSuffix   = ".cast";
static const std::string GEPNodeIncNameSuffix = ".inc";
static const std::string GEPNodeOffNameSuffix = ".off";

FunctionPass *llvm::createPPCLoopPreIncPrepPass(PPCTargetMachine &TM) {
  return new PPCLoopPreIncPrep(TM);
}

static bool IsPtrInBounds(Value *BasePtr) {
  Value *StrippedBasePtr = BasePtr;
  while (BitCastInst *BC = dyn_cast<BitCastInst>(StrippedBasePtr))
    StrippedBasePtr = BC->getOperand(0);
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(StrippedBasePtr))
    return GEP->isInBounds();

  return false;
}

static std::string getInstrName(const Value *I, const std::string Suffix) {
  assert(I && "Invalid paramater!");
  if (I->hasName())
    return (I->getName() + Suffix).str();
  else
    return ""; 
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

  return nullptr;
}

bool PPCLoopPreIncPrep::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  DT = DTWP ? &DTWP->getDomTree() : nullptr;
  PreserveLCSSA = mustPreserveAnalysisID(LCSSAID);
  ST = TM ? TM->getSubtargetImpl(F) : nullptr;

  bool MadeChange = false;

  for (auto I = LI->begin(), IE = LI->end(); I != IE; ++I)
    for (auto L = df_begin(*I), LE = df_end(*I); L != LE; ++L)
      MadeChange |= runOnLoop(*L);

  return MadeChange;
}

void PPCLoopPreIncPrep::addOneCandidate(Instruction *MemI, const SCEV *LSCEV,
                                        SmallVector<Bucket, 16> &Buckets,
                                        unsigned MaxCandidateNum) {
  assert((MemI && GetPointerOperand(MemI)) &&
         "Candidate should be a memory instruction.");
  assert(LSCEV && "Invalid SCEV for Ptr value.");
  bool FoundBucket = false;
  for (auto &B : Buckets) {
    const SCEV *Diff = SE->getMinusSCEV(LSCEV, B.BaseSCEV);
    if (const auto *CDiff = dyn_cast<SCEVConstant>(Diff)) {
      B.Elements.push_back(BucketElement(CDiff, MemI));
      FoundBucket = true;
      break;
    }
  }

  if (!FoundBucket) {
    if (Buckets.size() == MaxCandidateNum)
      return;
    Buckets.push_back(Bucket(LSCEV, MemI));
  }
}

SmallVector<Bucket, 16> PPCLoopPreIncPrep::collectCandidates(
    Loop *L,
    std::function<bool(const Instruction *, const Value *)> isValidCandidate,
    unsigned MaxCandidateNum) {
  SmallVector<Bucket, 16> Buckets;
  for (const auto &BB : L->blocks())
    for (auto &J : *BB) {
      Value *PtrValue;
      Instruction *MemI;

      if (LoadInst *LMemI = dyn_cast<LoadInst>(&J)) {
        MemI = LMemI;
        PtrValue = LMemI->getPointerOperand();
      } else if (StoreInst *SMemI = dyn_cast<StoreInst>(&J)) {
        MemI = SMemI;
        PtrValue = SMemI->getPointerOperand();
      } else if (IntrinsicInst *IMemI = dyn_cast<IntrinsicInst>(&J)) {
        if (IMemI->getIntrinsicID() == Intrinsic::prefetch) {
          MemI = IMemI;
          PtrValue = IMemI->getArgOperand(0);
        } else continue;
      } else continue;

      unsigned PtrAddrSpace = PtrValue->getType()->getPointerAddressSpace();
      if (PtrAddrSpace)
        continue;

      if (L->isLoopInvariant(PtrValue))
        continue;

      const SCEV *LSCEV = SE->getSCEVAtScope(PtrValue, L);
      const SCEVAddRecExpr *LARSCEV = dyn_cast<SCEVAddRecExpr>(LSCEV);
      if (!LARSCEV || LARSCEV->getLoop() != L)
        continue;

      if (isValidCandidate(&J, PtrValue))
        addOneCandidate(MemI, LSCEV, Buckets, MaxCandidateNum);
    }
  return Buckets;
}

// TODO: implement a more clever base choosing policy.
// Currently we always choose an exist load/store offset. This maybe lead to
// suboptimal code sequences. For example, for one DS chain with offsets
// {-32769, 2003, 2007, 2011}, we choose -32769 as base offset, and left disp
// for load/stores are {0, 34772, 34776, 34780}. Though each offset now is a
// multipler of 4, it cannot be represented by sint16.
bool PPCLoopPreIncPrep::prepareBaseForUpdateFormChain(Bucket &BucketChain) {
  // We have a choice now of which instruction's memory operand we use as the
  // base for the generated PHI. Always picking the first instruction in each
  // bucket does not work well, specifically because that instruction might
  // be a prefetch (and there are no pre-increment dcbt variants). Otherwise,
  // the choice is somewhat arbitrary, because the backend will happily
  // generate direct offsets from both the pre-incremented and
  // post-incremented pointer values. Thus, we'll pick the first non-prefetch
  // instruction in each bucket, and adjust the recurrence and other offsets
  // accordingly.
  for (int j = 0, je = BucketChain.Elements.size(); j != je; ++j) {
    if (auto *II = dyn_cast<IntrinsicInst>(BucketChain.Elements[j].Instr))
      if (II->getIntrinsicID() == Intrinsic::prefetch)
        continue;

    // If we'd otherwise pick the first element anyway, there's nothing to do.
    if (j == 0)
      break;

    // If our chosen element has no offset from the base pointer, there's
    // nothing to do.
    if (!BucketChain.Elements[j].Offset ||
        BucketChain.Elements[j].Offset->isZero())
      break;

    const SCEV *Offset = BucketChain.Elements[j].Offset;
    BucketChain.BaseSCEV = SE->getAddExpr(BucketChain.BaseSCEV, Offset);
    for (auto &E : BucketChain.Elements) {
      if (E.Offset)
        E.Offset = cast<SCEVConstant>(SE->getMinusSCEV(E.Offset, Offset));
      else
        E.Offset = cast<SCEVConstant>(SE->getNegativeSCEV(Offset));
    }

    std::swap(BucketChain.Elements[j], BucketChain.Elements[0]);
    break;
  }
  return true;
}

bool PPCLoopPreIncPrep::rewriteLoadStores(
    Loop *L, Bucket &BucketChain, SmallSet<BasicBlock *, 16> &BBChanged) {
  bool MadeChange = false;
  const SCEVAddRecExpr *BasePtrSCEV =
      cast<SCEVAddRecExpr>(BucketChain.BaseSCEV);
  if (!BasePtrSCEV->isAffine())
    return MadeChange;

  LLVM_DEBUG(dbgs() << "PIP: Transforming: " << *BasePtrSCEV << "\n");

  assert(BasePtrSCEV->getLoop() == L && "AddRec for the wrong loop?");

  // The instruction corresponding to the Bucket's BaseSCEV must be the first
  // in the vector of elements.
  Instruction *MemI = BucketChain.Elements.begin()->Instr;
  Value *BasePtr = GetPointerOperand(MemI);
  assert(BasePtr && "No pointer operand");

  Type *I8Ty = Type::getInt8Ty(MemI->getParent()->getContext());
  Type *I8PtrTy = Type::getInt8PtrTy(MemI->getParent()->getContext(),
    BasePtr->getType()->getPointerAddressSpace());

  const SCEV *BasePtrStartSCEV = BasePtrSCEV->getStart();
  if (!SE->isLoopInvariant(BasePtrStartSCEV, L))
    return MadeChange;

  const SCEVConstant *BasePtrIncSCEV =
    dyn_cast<SCEVConstant>(BasePtrSCEV->getStepRecurrence(*SE));
  if (!BasePtrIncSCEV)
    return MadeChange;
  BasePtrStartSCEV = SE->getMinusSCEV(BasePtrStartSCEV, BasePtrIncSCEV);
  if (!isSafeToExpand(BasePtrStartSCEV, *SE))
    return MadeChange;

  if (alreadyPrepared(L, MemI, BasePtrStartSCEV, BasePtrIncSCEV))
    return MadeChange;

  LLVM_DEBUG(dbgs() << "PIP: New start is: " << *BasePtrStartSCEV << "\n");

  BasicBlock *Header = L->getHeader();
  unsigned HeaderLoopPredCount = pred_size(Header);
  BasicBlock *LoopPredecessor = L->getLoopPredecessor();

  PHINode *NewPHI =
      PHINode::Create(I8PtrTy, HeaderLoopPredCount,
                      getInstrName(MemI, PHINodeNameSuffix),
                      Header->getFirstNonPHI());

  SCEVExpander SCEVE(*SE, Header->getModule()->getDataLayout(), "pistart");
  Value *BasePtrStart = SCEVE.expandCodeFor(BasePtrStartSCEV, I8PtrTy,
                                            LoopPredecessor->getTerminator());

  // Note that LoopPredecessor might occur in the predecessor list multiple
  // times, and we need to add it the right number of times.
  for (const auto &PI : predecessors(Header)) {
    if (PI != LoopPredecessor)
      continue;

    NewPHI->addIncoming(BasePtrStart, LoopPredecessor);
  }

  Instruction *InsPoint = &*Header->getFirstInsertionPt();
  GetElementPtrInst *PtrInc = GetElementPtrInst::Create(
      I8Ty, NewPHI, BasePtrIncSCEV->getValue(),
      getInstrName(MemI, GEPNodeIncNameSuffix), InsPoint);
  PtrInc->setIsInBounds(IsPtrInBounds(BasePtr));
  for (const auto &PI : predecessors(Header)) {
    if (PI == LoopPredecessor)
      continue;

    NewPHI->addIncoming(PtrInc, PI);
  }

  Instruction *NewBasePtr;
  if (PtrInc->getType() != BasePtr->getType())
    NewBasePtr = new BitCastInst(PtrInc, BasePtr->getType(),
      getInstrName(PtrInc, CastNodeNameSuffix), InsPoint);
  else
    NewBasePtr = PtrInc;

  if (Instruction *IDel = dyn_cast<Instruction>(BasePtr))
    BBChanged.insert(IDel->getParent());
  BasePtr->replaceAllUsesWith(NewBasePtr);
  RecursivelyDeleteTriviallyDeadInstructions(BasePtr);

  // Keep track of the replacement pointer values we've inserted so that we
  // don't generate more pointer values than necessary.
  SmallPtrSet<Value *, 16> NewPtrs;
  NewPtrs.insert(NewBasePtr);

  for (auto I = std::next(BucketChain.Elements.begin()),
       IE = BucketChain.Elements.end(); I != IE; ++I) {
    Value *Ptr = GetPointerOperand(I->Instr);
    assert(Ptr && "No pointer operand");
    if (NewPtrs.count(Ptr))
      continue;

    Instruction *RealNewPtr;
    if (!I->Offset || I->Offset->getValue()->isZero()) {
      RealNewPtr = NewBasePtr;
    } else {
      Instruction *PtrIP = dyn_cast<Instruction>(Ptr);
      if (PtrIP && isa<Instruction>(NewBasePtr) &&
          cast<Instruction>(NewBasePtr)->getParent() == PtrIP->getParent())
        PtrIP = nullptr;
      else if (PtrIP && isa<PHINode>(PtrIP))
        PtrIP = &*PtrIP->getParent()->getFirstInsertionPt();
      else if (!PtrIP)
        PtrIP = I->Instr;

      GetElementPtrInst *NewPtr = GetElementPtrInst::Create(
          I8Ty, PtrInc, I->Offset->getValue(),
          getInstrName(I->Instr, GEPNodeOffNameSuffix), PtrIP);
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
        getInstrName(Ptr, CastNodeNameSuffix));
      ReplNewPtr->insertAfter(RealNewPtr);
    } else
      ReplNewPtr = RealNewPtr;

    Ptr->replaceAllUsesWith(ReplNewPtr);
    RecursivelyDeleteTriviallyDeadInstructions(Ptr);

    NewPtrs.insert(RealNewPtr);
  }

  MadeChange = true;
  UpdFormChainRewritten++;

  return MadeChange;
}

bool PPCLoopPreIncPrep::updateFormPrep(Loop *L,
                                       SmallVector<Bucket, 16> &Buckets) {
  bool MadeChange = false;
  if (Buckets.empty())
    return MadeChange;
  SmallSet<BasicBlock *, 16> BBChanged;
  for (auto &Bucket : Buckets)
    // The base address of each bucket is transformed into a phi and the others
    // are rewritten based on new base.
    if (prepareBaseForUpdateFormChain(Bucket))
      MadeChange |= rewriteLoadStores(L, Bucket, BBChanged);
  if (MadeChange)
    for (auto &BB : L->blocks())
      if (BBChanged.count(BB))
        DeleteDeadPHIs(BB);
  return MadeChange;
}

// In order to prepare for the pre-increment a PHI is added.
// This function will check to see if that PHI already exists and will return
// true if it found an existing PHI with the same start and increment as the
// one we wanted to create.
bool PPCLoopPreIncPrep::alreadyPrepared(Loop *L, Instruction* MemI,
                                        const SCEV *BasePtrStartSCEV,
                                        const SCEVConstant *BasePtrIncSCEV) {
  BasicBlock *BB = MemI->getParent();
  if (!BB)
    return false;

  BasicBlock *PredBB = L->getLoopPredecessor();
  BasicBlock *LatchBB = L->getLoopLatch();

  if (!PredBB || !LatchBB)
    return false;

  // Run through the PHIs and see if we have some that looks like a preparation
  iterator_range<BasicBlock::phi_iterator> PHIIter = BB->phis();
  for (auto & CurrentPHI : PHIIter) {
    PHINode *CurrentPHINode = dyn_cast<PHINode>(&CurrentPHI);
    if (!CurrentPHINode)
      continue;

    if (!SE->isSCEVable(CurrentPHINode->getType()))
      continue;

    const SCEV *PHISCEV = SE->getSCEVAtScope(CurrentPHINode, L);

    const SCEVAddRecExpr *PHIBasePtrSCEV = dyn_cast<SCEVAddRecExpr>(PHISCEV);
    if (!PHIBasePtrSCEV)
      continue;

    const SCEVConstant *PHIBasePtrIncSCEV =
      dyn_cast<SCEVConstant>(PHIBasePtrSCEV->getStepRecurrence(*SE));
    if (!PHIBasePtrIncSCEV)
      continue;

    if (CurrentPHINode->getNumIncomingValues() == 2) {
      if ((CurrentPHINode->getIncomingBlock(0) == LatchBB &&
           CurrentPHINode->getIncomingBlock(1) == PredBB) ||
          (CurrentPHINode->getIncomingBlock(1) == LatchBB &&
           CurrentPHINode->getIncomingBlock(0) == PredBB)) {
        if (PHIBasePtrSCEV->getStart() == BasePtrStartSCEV &&
            PHIBasePtrIncSCEV == BasePtrIncSCEV) {
          // The existing PHI (CurrentPHINode) has the same start and increment
          //  as the PHI that we wanted to create.
          ++PHINodeAlreadyExists;
          return true;
        }
      }
    }
  }
  return false;
}

bool PPCLoopPreIncPrep::runOnLoop(Loop *L) {
  bool MadeChange = false;

  // Only prep. the inner-most loop
  if (!L->empty())
    return MadeChange;

  LLVM_DEBUG(dbgs() << "PIP: Examining: " << *L << "\n");

  BasicBlock *LoopPredecessor = L->getLoopPredecessor();
  // If there is no loop predecessor, or the loop predecessor's terminator
  // returns a value (which might contribute to determining the loop's
  // iteration space), insert a new preheader for the loop.
  if (!LoopPredecessor ||
      !LoopPredecessor->getTerminator()->getType()->isVoidTy()) {
    LoopPredecessor = InsertPreheaderForLoop(L, DT, LI, nullptr, PreserveLCSSA);
    if (LoopPredecessor)
      MadeChange = true;
  }
  if (!LoopPredecessor) {
    LLVM_DEBUG(dbgs() << "PIP fails since no predecessor for current loop.\n");
    return MadeChange;
  }

  // Check if a load/store has update form. This lambda is used by function
  // collectCandidates which can collect candidates for types defined by lambda.
  auto isUpdateFormCandidate = [&] (const Instruction *I,
                                    const Value *PtrValue) {
    assert((PtrValue && I) && "Invalid parameter!");
    // There are no update forms for Altivec vector load/stores.
    if (ST && ST->hasAltivec() &&
        PtrValue->getType()->getPointerElementType()->isVectorTy())
      return false;
    // See getPreIndexedAddressParts, the displacement for LDU/STDU has to
    // be 4's multiple (DS-form). For i64 loads/stores when the displacement
    // fits in a 16-bit signed field but isn't a multiple of 4, it will be
    // useless and possible to break some original well-form addressing mode
    // to make this pre-inc prep for it.
    if (PtrValue->getType()->getPointerElementType()->isIntegerTy(64)) {
      const SCEV *LSCEV = SE->getSCEVAtScope(const_cast<Value *>(PtrValue), L);
      const SCEVAddRecExpr *LARSCEV = dyn_cast<SCEVAddRecExpr>(LSCEV);
      if (!LARSCEV || LARSCEV->getLoop() != L)
        return false;
      if (const SCEVConstant *StepConst =
              dyn_cast<SCEVConstant>(LARSCEV->getStepRecurrence(*SE))) {
        const APInt &ConstInt = StepConst->getValue()->getValue();
        if (ConstInt.isSignedIntN(16) && ConstInt.srem(4) != 0)
          return false;
      }
    }
    return true;
  };

  // Collect buckets of comparable addresses used by loads, stores and prefetch
  // intrinsic for update form.
  SmallVector<Bucket, 16> UpdateFormBuckets =
      collectCandidates(L, isUpdateFormCandidate, MaxVars);

  // Prepare for update form.
  if (!UpdateFormBuckets.empty())
    MadeChange |= updateFormPrep(L, UpdateFormBuckets);

  return MadeChange;
}
