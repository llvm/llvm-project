//===------ PPCLoopInstrFormPrep.cpp - Loop Instr Form Prep Pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to prepare loops for ppc preferred addressing
// modes, leveraging different instruction form. (eg: DS/DQ form, D/DS form with
// update)
// Additional PHIs are created for loop induction variables used by load/store
// instructions so that preferred addressing modes can be used.
//
// 1: DS/DQ form preparation, prepare the load/store instructions so that they
//    can satisfy the DS/DQ form displacement requirements.
//    Generically, this means transforming loops like this:
//    for (int i = 0; i < n; ++i) {
//      unsigned long x1 = *(unsigned long *)(p + i + 5);
//      unsigned long x2 = *(unsigned long *)(p + i + 9);
//    }
//
//    to look like this:
//
//    unsigned NewP = p + 5;
//    for (int i = 0; i < n; ++i) {
//      unsigned long x1 = *(unsigned long *)(i + NewP);
//      unsigned long x2 = *(unsigned long *)(i + NewP + 4);
//    }
//
// 2: D/DS form with update preparation, prepare the load/store instructions so
//    that we can use update form to do pre-increment.
//    Generically, this means transforming loops like this:
//    for (int i = 0; i < n; ++i)
//      array[i] = c;
//
//    to look like this:
//
//    T *p = array[-1];
//    for (int i = 0; i < n; ++i)
//      *++p = c;
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ppc-loop-instr-form-prep"

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
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <cassert>
#include <iterator>
#include <utility>

using namespace llvm;

// By default, we limit this to creating 16 common bases out of loops per
// function. 16 is a little over half of the allocatable register set.
static cl::opt<unsigned> MaxVarsPrep("ppc-formprep-max-vars",
                                 cl::Hidden, cl::init(16),
  cl::desc("Potential common base number threshold per function for PPC loop "
           "prep"));

static cl::opt<bool> PreferUpdateForm("ppc-formprep-prefer-update",
                                 cl::init(true), cl::Hidden,
  cl::desc("prefer update form when ds form is also a update form"));

// Sum of following 3 per loop thresholds for all loops can not be larger
// than MaxVarsPrep.
// By default, we limit this to creating 9 PHIs for one loop.
// 9 and 3 for each kind prep are exterimental values on Power9.
static cl::opt<unsigned> MaxVarsUpdateForm("ppc-preinc-prep-max-vars",
                                 cl::Hidden, cl::init(3),
  cl::desc("Potential PHI threshold per loop for PPC loop prep of update "
           "form"));

static cl::opt<unsigned> MaxVarsDSForm("ppc-dsprep-max-vars",
                                 cl::Hidden, cl::init(3),
  cl::desc("Potential PHI threshold per loop for PPC loop prep of DS form"));

static cl::opt<unsigned> MaxVarsDQForm("ppc-dqprep-max-vars",
                                 cl::Hidden, cl::init(3),
  cl::desc("Potential PHI threshold per loop for PPC loop prep of DQ form"));


// If would not be profitable if the common base has only one load/store, ISEL
// should already be able to choose best load/store form based on offset for
// single load/store. Set minimal profitable value default to 2 and make it as
// an option.
static cl::opt<unsigned> DispFormPrepMinThreshold("ppc-dispprep-min-threshold",
                                    cl::Hidden, cl::init(2),
  cl::desc("Minimal common base load/store instructions triggering DS/DQ form "
           "preparation"));

STATISTIC(PHINodeAlreadyExistsUpdate, "PHI node already in pre-increment form");
STATISTIC(PHINodeAlreadyExistsDS, "PHI node already in DS form");
STATISTIC(PHINodeAlreadyExistsDQ, "PHI node already in DQ form");
STATISTIC(DSFormChainRewritten, "Num of DS form chain rewritten");
STATISTIC(DQFormChainRewritten, "Num of DQ form chain rewritten");
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

  // "UpdateForm" is not a real PPC instruction form, it stands for dform
  // load/store with update like ldu/stdu, or Prefetch intrinsic.
  // For DS form instructions, their displacements must be multiple of 4.
  // For DQ form instructions, their displacements must be multiple of 16.
  enum InstrForm { UpdateForm = 1, DSForm = 4, DQForm = 16 };

  class PPCLoopInstrFormPrep : public FunctionPass {
  public:
    static char ID; // Pass ID, replacement for typeid

    PPCLoopInstrFormPrep() : FunctionPass(ID) {
      initializePPCLoopInstrFormPrepPass(*PassRegistry::getPassRegistry());
    }

    PPCLoopInstrFormPrep(PPCTargetMachine &TM) : FunctionPass(ID), TM(&TM) {
      initializePPCLoopInstrFormPrepPass(*PassRegistry::getPassRegistry());
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

    /// Successful preparation number for Update/DS/DQ form in all inner most
    /// loops. One successful preparation will put one common base out of loop,
    /// this may leads to register presure like LICM does.
    /// Make sure total preparation number can be controlled by option.
    unsigned SuccPrepCount;

    bool runOnLoop(Loop *L);

    /// Check if required PHI node is already exist in Loop \p L.
    bool alreadyPrepared(Loop *L, Instruction* MemI,
                         const SCEV *BasePtrStartSCEV,
                         const SCEVConstant *BasePtrIncSCEV,
                         InstrForm Form);

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

    /// Prepare all candidates in \p Buckets for displacement form, now for
    /// ds/dq.
    bool dispFormPrep(Loop *L, SmallVector<Bucket, 16> &Buckets,
                      InstrForm Form);

    /// Prepare for one chain \p BucketChain, find the best base element and
    /// update all other elements in \p BucketChain accordingly.
    /// \p Form is used to find the best base element.
    /// If success, best base element must be stored as the first element of
    /// \p BucketChain.
    /// Return false if no base element found, otherwise return true.
    bool prepareBaseForDispFormChain(Bucket &BucketChain,
                                     InstrForm Form);

    /// Prepare for one chain \p BucketChain, find the best base element and
    /// update all other elements in \p BucketChain accordingly.
    /// If success, best base element must be stored as the first element of
    /// \p BucketChain.
    /// Return false if no base element found, otherwise return true.
    bool prepareBaseForUpdateFormChain(Bucket &BucketChain);

    /// Rewrite load/store instructions in \p BucketChain according to
    /// preparation.
    bool rewriteLoadStores(Loop *L, Bucket &BucketChain,
                           SmallSet<BasicBlock *, 16> &BBChanged,
                           InstrForm Form);
  };

} // end anonymous namespace

char PPCLoopInstrFormPrep::ID = 0;
static const char *name = "Prepare loop for ppc preferred instruction forms";
INITIALIZE_PASS_BEGIN(PPCLoopInstrFormPrep, DEBUG_TYPE, name, false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(PPCLoopInstrFormPrep, DEBUG_TYPE, name, false, false)

static constexpr StringRef PHINodeNameSuffix    = ".phi";
static constexpr StringRef CastNodeNameSuffix   = ".cast";
static constexpr StringRef GEPNodeIncNameSuffix = ".inc";
static constexpr StringRef GEPNodeOffNameSuffix = ".off";

FunctionPass *llvm::createPPCLoopInstrFormPrepPass(PPCTargetMachine &TM) {
  return new PPCLoopInstrFormPrep(TM);
}

static bool IsPtrInBounds(Value *BasePtr) {
  Value *StrippedBasePtr = BasePtr;
  while (BitCastInst *BC = dyn_cast<BitCastInst>(StrippedBasePtr))
    StrippedBasePtr = BC->getOperand(0);
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(StrippedBasePtr))
    return GEP->isInBounds();

  return false;
}

static std::string getInstrName(const Value *I, StringRef Suffix) {
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

bool PPCLoopInstrFormPrep::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  DT = DTWP ? &DTWP->getDomTree() : nullptr;
  PreserveLCSSA = mustPreserveAnalysisID(LCSSAID);
  ST = TM ? TM->getSubtargetImpl(F) : nullptr;
  SuccPrepCount = 0;

  bool MadeChange = false;

  for (auto I = LI->begin(), IE = LI->end(); I != IE; ++I)
    for (auto L = df_begin(*I), LE = df_end(*I); L != LE; ++L)
      MadeChange |= runOnLoop(*L);

  return MadeChange;
}

void PPCLoopInstrFormPrep::addOneCandidate(Instruction *MemI, const SCEV *LSCEV,
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

SmallVector<Bucket, 16> PPCLoopInstrFormPrep::collectCandidates(
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

bool PPCLoopInstrFormPrep::prepareBaseForDispFormChain(Bucket &BucketChain,
                                                    InstrForm Form) {
  // RemainderOffsetInfo details:
  // key:            value of (Offset urem DispConstraint). For DSForm, it can
  //                 be [0, 4).
  // first of pair:  the index of first BucketElement whose remainder is equal
  //                 to key. For key 0, this value must be 0.
  // second of pair: number of load/stores with the same remainder.
  DenseMap<unsigned, std::pair<unsigned, unsigned>> RemainderOffsetInfo;

  for (unsigned j = 0, je = BucketChain.Elements.size(); j != je; ++j) {
    if (!BucketChain.Elements[j].Offset)
      RemainderOffsetInfo[0] = std::make_pair(0, 1);
    else {
      unsigned Remainder =
          BucketChain.Elements[j].Offset->getAPInt().urem(Form);
      if (RemainderOffsetInfo.find(Remainder) == RemainderOffsetInfo.end())
        RemainderOffsetInfo[Remainder] = std::make_pair(j, 1);
      else
        RemainderOffsetInfo[Remainder].second++;
    }
  }
  // Currently we choose the most profitable base as the one which has the max
  // number of load/store with same remainder.
  // FIXME: adjust the base selection strategy according to load/store offset
  // distribution.
  // For example, if we have one candidate chain for DS form preparation, which
  // contains following load/stores with different remainders:
  // 1: 10 load/store whose remainder is 1;
  // 2: 9 load/store whose remainder is 2;
  // 3: 1 for remainder 3 and 0 for remainder 0; 
  // Now we will choose the first load/store whose remainder is 1 as base and
  // adjust all other load/stores according to new base, so we will get 10 DS
  // form and 10 X form.
  // But we should be more clever, for this case we could use two bases, one for
  // remainder 1 and the other for remainder 2, thus we could get 19 DS form and 1
  // X form.
  unsigned MaxCountRemainder = 0;
  for (unsigned j = 0; j < (unsigned)Form; j++)
    if ((RemainderOffsetInfo.find(j) != RemainderOffsetInfo.end()) &&
        RemainderOffsetInfo[j].second >
            RemainderOffsetInfo[MaxCountRemainder].second)
      MaxCountRemainder = j;

  // Abort when there are too few insts with common base.
  if (RemainderOffsetInfo[MaxCountRemainder].second < DispFormPrepMinThreshold)
    return false;

  // If the first value is most profitable, no needed to adjust BucketChain
  // elements as they are substracted the first value when collecting.
  if (MaxCountRemainder == 0)
    return true;

  // Adjust load/store to the new chosen base.
  const SCEV *Offset =
      BucketChain.Elements[RemainderOffsetInfo[MaxCountRemainder].first].Offset;
  BucketChain.BaseSCEV = SE->getAddExpr(BucketChain.BaseSCEV, Offset);
  for (auto &E : BucketChain.Elements) {
    if (E.Offset)
      E.Offset = cast<SCEVConstant>(SE->getMinusSCEV(E.Offset, Offset));
    else
      E.Offset = cast<SCEVConstant>(SE->getNegativeSCEV(Offset));
  }

  std::swap(BucketChain.Elements[RemainderOffsetInfo[MaxCountRemainder].first],
            BucketChain.Elements[0]);
  return true;
}

// FIXME: implement a more clever base choosing policy.
// Currently we always choose an exist load/store offset. This maybe lead to
// suboptimal code sequences. For example, for one DS chain with offsets
// {-32769, 2003, 2007, 2011}, we choose -32769 as base offset, and left disp
// for load/stores are {0, 34772, 34776, 34780}. Though each offset now is a
// multipler of 4, it cannot be represented by sint16.
bool PPCLoopInstrFormPrep::prepareBaseForUpdateFormChain(Bucket &BucketChain) {
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

bool PPCLoopInstrFormPrep::rewriteLoadStores(Loop *L, Bucket &BucketChain,
                                          SmallSet<BasicBlock *, 16> &BBChanged,
                                          InstrForm Form) {
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

  if (!SE->isLoopInvariant(BasePtrSCEV->getStart(), L))
    return MadeChange;

  const SCEVConstant *BasePtrIncSCEV =
    dyn_cast<SCEVConstant>(BasePtrSCEV->getStepRecurrence(*SE));
  if (!BasePtrIncSCEV)
    return MadeChange;

  // For some DS form load/store instructions, it can also be an update form,
  // if the stride is a multipler of 4. Use update form if prefer it.
  bool CanPreInc = (Form == UpdateForm ||
                    ((Form == DSForm) && !BasePtrIncSCEV->getAPInt().urem(4) &&
                     PreferUpdateForm));
  const SCEV *BasePtrStartSCEV = nullptr;
  if (CanPreInc)
    BasePtrStartSCEV =
        SE->getMinusSCEV(BasePtrSCEV->getStart(), BasePtrIncSCEV);
  else
    BasePtrStartSCEV = BasePtrSCEV->getStart();

  if (!isSafeToExpand(BasePtrStartSCEV, *SE))
    return MadeChange;

  if (alreadyPrepared(L, MemI, BasePtrStartSCEV, BasePtrIncSCEV, Form))
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
  for (auto PI : predecessors(Header)) {
    if (PI != LoopPredecessor)
      continue;

    NewPHI->addIncoming(BasePtrStart, LoopPredecessor);
  }

  Instruction *PtrInc = nullptr;
  Instruction *NewBasePtr = nullptr;
  if (CanPreInc) {
    Instruction *InsPoint = &*Header->getFirstInsertionPt();
    PtrInc = GetElementPtrInst::Create(
        I8Ty, NewPHI, BasePtrIncSCEV->getValue(),
        getInstrName(MemI, GEPNodeIncNameSuffix), InsPoint);
    cast<GetElementPtrInst>(PtrInc)->setIsInBounds(IsPtrInBounds(BasePtr));
    for (auto PI : predecessors(Header)) {
      if (PI == LoopPredecessor)
        continue;

      NewPHI->addIncoming(PtrInc, PI);
    }
    if (PtrInc->getType() != BasePtr->getType())
      NewBasePtr = new BitCastInst(
          PtrInc, BasePtr->getType(),
          getInstrName(PtrInc, CastNodeNameSuffix), InsPoint);
    else
      NewBasePtr = PtrInc;
  } else {
    // Note that LoopPredecessor might occur in the predecessor list multiple
    // times, and we need to make sure no more incoming value for them in PHI.
    for (auto PI : predecessors(Header)) {
      if (PI == LoopPredecessor)
        continue;

      // For the latch predecessor, we need to insert a GEP just before the
      // terminator to increase the address.
      BasicBlock *BB = PI;
      Instruction *InsPoint = BB->getTerminator();
      PtrInc = GetElementPtrInst::Create(
          I8Ty, NewPHI, BasePtrIncSCEV->getValue(),
          getInstrName(MemI, GEPNodeIncNameSuffix), InsPoint);

      cast<GetElementPtrInst>(PtrInc)->setIsInBounds(IsPtrInBounds(BasePtr));

      NewPHI->addIncoming(PtrInc, PI);
    }
    PtrInc = NewPHI;
    if (NewPHI->getType() != BasePtr->getType())
      NewBasePtr =
          new BitCastInst(NewPHI, BasePtr->getType(),
                          getInstrName(NewPHI, CastNodeNameSuffix),
                          &*Header->getFirstInsertionPt());
    else
      NewBasePtr = NewPHI;
  }

  // Clear the rewriter cache, because values that are in the rewriter's cache
  // can be deleted below, causing the AssertingVH in the cache to trigger.
  SCEVE.clear();

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

  SuccPrepCount++;  

  if (Form == DSForm && !CanPreInc)
    DSFormChainRewritten++;
  else if (Form == DQForm)
    DQFormChainRewritten++;
  else if (Form == UpdateForm || (Form == DSForm && CanPreInc))
    UpdFormChainRewritten++;

  return MadeChange;
}

bool PPCLoopInstrFormPrep::updateFormPrep(Loop *L,
                                       SmallVector<Bucket, 16> &Buckets) {
  bool MadeChange = false;
  if (Buckets.empty())
    return MadeChange;
  SmallSet<BasicBlock *, 16> BBChanged;
  for (auto &Bucket : Buckets)
    // The base address of each bucket is transformed into a phi and the others
    // are rewritten based on new base.
    if (prepareBaseForUpdateFormChain(Bucket))
      MadeChange |= rewriteLoadStores(L, Bucket, BBChanged, UpdateForm);

  if (MadeChange)
    for (auto &BB : L->blocks())
      if (BBChanged.count(BB))
        DeleteDeadPHIs(BB);
  return MadeChange;
}

bool PPCLoopInstrFormPrep::dispFormPrep(Loop *L, SmallVector<Bucket, 16> &Buckets,
                                     InstrForm Form) {
  bool MadeChange = false;

  if (Buckets.empty())
    return MadeChange;

  SmallSet<BasicBlock *, 16> BBChanged;
  for (auto &Bucket : Buckets) {
    if (Bucket.Elements.size() < DispFormPrepMinThreshold)
      continue;
    if (prepareBaseForDispFormChain(Bucket, Form))
      MadeChange |= rewriteLoadStores(L, Bucket, BBChanged, Form);
  }

  if (MadeChange)
    for (auto &BB : L->blocks())
      if (BBChanged.count(BB))
        DeleteDeadPHIs(BB);
  return MadeChange;
}

// In order to prepare for the preferred instruction form, a PHI is added.
// This function will check to see if that PHI already exists and will return
// true if it found an existing PHI with the matched start and increment as the
// one we wanted to create.
bool PPCLoopInstrFormPrep::alreadyPrepared(Loop *L, Instruction* MemI,
                                        const SCEV *BasePtrStartSCEV,
                                        const SCEVConstant *BasePtrIncSCEV,
                                        InstrForm Form) {
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
        if (PHIBasePtrIncSCEV == BasePtrIncSCEV) {
          // The existing PHI (CurrentPHINode) has the same start and increment
          // as the PHI that we wanted to create.
          if (Form == UpdateForm &&
              PHIBasePtrSCEV->getStart() == BasePtrStartSCEV) {
            ++PHINodeAlreadyExistsUpdate;
            return true;
          } 
          if (Form == DSForm || Form == DQForm) {
            const SCEVConstant *Diff = dyn_cast<SCEVConstant>(
                SE->getMinusSCEV(PHIBasePtrSCEV->getStart(), BasePtrStartSCEV));
            if (Diff && !Diff->getAPInt().urem(Form)) {
              if (Form == DSForm)
                ++PHINodeAlreadyExistsDS;
              else
                ++PHINodeAlreadyExistsDQ;
              return true;
            }
          } 
        }
      }
    }
  }
  return false;
}

bool PPCLoopInstrFormPrep::runOnLoop(Loop *L) {
  bool MadeChange = false;

  // Only prep. the inner-most loop
  if (!L->isInnermost())
    return MadeChange;

  // Return if already done enough preparation.
  if (SuccPrepCount >= MaxVarsPrep)
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
  
  // Check if a load/store has DS form.
  auto isDSFormCandidate = [] (const Instruction *I, const Value *PtrValue) {
    assert((PtrValue && I) && "Invalid parameter!");
    if (isa<IntrinsicInst>(I))
      return false;
    Type *PointerElementType = PtrValue->getType()->getPointerElementType();
    return (PointerElementType->isIntegerTy(64)) ||
           (PointerElementType->isFloatTy()) ||
           (PointerElementType->isDoubleTy()) ||
           (PointerElementType->isIntegerTy(32) &&
            llvm::any_of(I->users(),
                         [](const User *U) { return isa<SExtInst>(U); }));
  };

  // Check if a load/store has DQ form.
  auto isDQFormCandidate = [&] (const Instruction *I, const Value *PtrValue) {
    assert((PtrValue && I) && "Invalid parameter!");
    return !isa<IntrinsicInst>(I) && ST && ST->hasP9Vector() &&
           (PtrValue->getType()->getPointerElementType()->isVectorTy());
  };

  // intrinsic for update form.
  SmallVector<Bucket, 16> UpdateFormBuckets =
      collectCandidates(L, isUpdateFormCandidate, MaxVarsUpdateForm);

  // Prepare for update form.
  if (!UpdateFormBuckets.empty())
    MadeChange |= updateFormPrep(L, UpdateFormBuckets);

  // Collect buckets of comparable addresses used by loads and stores for DS
  // form.
  SmallVector<Bucket, 16> DSFormBuckets =
      collectCandidates(L, isDSFormCandidate, MaxVarsDSForm);

  // Prepare for DS form.
  if (!DSFormBuckets.empty())
    MadeChange |= dispFormPrep(L, DSFormBuckets, DSForm);

  // Collect buckets of comparable addresses used by loads and stores for DQ
  // form.
  SmallVector<Bucket, 16> DQFormBuckets =
      collectCandidates(L, isDQFormCandidate, MaxVarsDQForm);

  // Prepare for DQ form.
  if (!DQFormBuckets.empty())
    MadeChange |= dispFormPrep(L, DQFormBuckets, DQForm);

  return MadeChange;
}
