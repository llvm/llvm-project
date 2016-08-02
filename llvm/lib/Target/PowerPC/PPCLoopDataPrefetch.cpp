//===-------- PPCLoopDataPrefetch.cpp - Loop Data Prefetching Pass --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a Loop Data Prefetching Pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ppc-loop-data-prefetch"
#include "PPC.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
using namespace llvm;

// By default, we limit this to creating 16 PHIs (which is a little over half
// of the allocatable register set).
static cl::opt<bool>
PrefetchWrites("ppc-loop-prefetch-writes", cl::Hidden, cl::init(false),
               cl::desc("Prefetch write addresses"));

// This seems like a reasonable default for the BG/Q (this pass is enabled, by
// default, only on the BG/Q).
static cl::opt<unsigned>
PrefDist("ppc-loop-prefetch-distance", cl::Hidden, cl::init(300),
         cl::desc("The loop prefetch distance"));

static cl::opt<unsigned>
CacheLineSize("ppc-loop-prefetch-cache-line", cl::Hidden, cl::init(64),
              cl::desc("The loop prefetch cache line size"));

namespace llvm {
  void initializePPCLoopDataPrefetchPass(PassRegistry&);
}

namespace {

  class PPCLoopDataPrefetch : public FunctionPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    PPCLoopDataPrefetch() : FunctionPass(ID) {
      initializePPCLoopDataPrefetchPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<AssumptionCacheTracker>();
      AU.addPreserved<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addPreserved<LoopInfoWrapperPass>();
      AU.addRequired<ScalarEvolutionWrapperPass>();
      // FIXME: For some reason, preserving SE here breaks LSR (even if
      // this pass changes nothing).
      // AU.addPreserved<ScalarEvolutionWrapperPass>();
      AU.addRequired<TargetTransformInfoWrapperPass>();
    }

    bool runOnFunction(Function &F) override;
    bool runOnLoop(Loop *L);

  private:
    AssumptionCache *AC;
    LoopInfo *LI;
    ScalarEvolution *SE;
    const TargetTransformInfo *TTI;
    const DataLayout *DL;
  };
}

char PPCLoopDataPrefetch::ID = 0;
INITIALIZE_PASS_BEGIN(PPCLoopDataPrefetch, "ppc-loop-data-prefetch",
                      "PPC Loop Data Prefetch", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(PPCLoopDataPrefetch, "ppc-loop-data-prefetch",
                    "PPC Loop Data Prefetch", false, false)

FunctionPass *llvm::createPPCLoopDataPrefetchPass() { return new PPCLoopDataPrefetch(); }

bool PPCLoopDataPrefetch::runOnFunction(Function &F) {
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  DL = &F.getParent()->getDataLayout();
  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

  bool MadeChange = false;

  for (auto I = LI->begin(), IE = LI->end(); I != IE; ++I)
    for (auto L = df_begin(*I), LE = df_end(*I); L != LE; ++L)
      MadeChange |= runOnLoop(*L);

  return MadeChange;
}

bool PPCLoopDataPrefetch::runOnLoop(Loop *L) {
  bool MadeChange = false;

  // Only prefetch in the inner-most loop
  if (!L->empty())
    return MadeChange;

  SmallPtrSet<const Value *, 32> EphValues;
  CodeMetrics::collectEphemeralValues(L, AC, EphValues);

  // Calculate the number of iterations ahead to prefetch
  CodeMetrics Metrics;
  for (Loop::block_iterator I = L->block_begin(), IE = L->block_end();
       I != IE; ++I) {

    // If the loop already has prefetches, then assume that the user knows
    // what he or she is doing and don't add any more.
    for (BasicBlock::iterator J = (*I)->begin(), JE = (*I)->end();
         J != JE; ++J)
      if (CallInst *CI = dyn_cast<CallInst>(J))
        if (Function *F = CI->getCalledFunction())
          if (F->getIntrinsicID() == Intrinsic::prefetch)
            return MadeChange;

    Metrics.analyzeBasicBlock(*I, *TTI, EphValues);
  }
  unsigned LoopSize = Metrics.NumInsts;
  if (!LoopSize)
    LoopSize = 1;

  unsigned ItersAhead = PrefDist/LoopSize;
  if (!ItersAhead)
    ItersAhead = 1;

  SmallVector<std::pair<Instruction *, const SCEVAddRecExpr *>, 16> PrefLoads;
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
        if (!PrefetchWrites) continue;
        MemI = SMemI;
        PtrValue = SMemI->getPointerOperand();
      } else continue;

      unsigned PtrAddrSpace = PtrValue->getType()->getPointerAddressSpace();
      if (PtrAddrSpace)
        continue;

      if (L->isLoopInvariant(PtrValue))
        continue;

      const SCEV *LSCEV = SE->getSCEV(PtrValue);
      const SCEVAddRecExpr *LSCEVAddRec = dyn_cast<SCEVAddRecExpr>(LSCEV);
      if (!LSCEVAddRec)
        continue;

      // We don't want to double prefetch individual cache lines. If this load
      // is known to be within one cache line of some other load that has
      // already been prefetched, then don't prefetch this one as well.
      bool DupPref = false;
      for (SmallVector<std::pair<Instruction *, const SCEVAddRecExpr *>,
             16>::iterator K = PrefLoads.begin(), KE = PrefLoads.end();
           K != KE; ++K) {
        const SCEV *PtrDiff = SE->getMinusSCEV(LSCEVAddRec, K->second);
        if (const SCEVConstant *ConstPtrDiff =
            dyn_cast<SCEVConstant>(PtrDiff)) {
          int64_t PD = std::abs(ConstPtrDiff->getValue()->getSExtValue());
          if (PD < (int64_t) CacheLineSize) {
            DupPref = true;
            break;
          }
        }
      }
      if (DupPref)
        continue;

      const SCEV *NextLSCEV = SE->getAddExpr(LSCEVAddRec, SE->getMulExpr(
        SE->getConstant(LSCEVAddRec->getType(), ItersAhead),
        LSCEVAddRec->getStepRecurrence(*SE)));
      if (!isSafeToExpand(NextLSCEV, *SE))
        continue;

      PrefLoads.push_back(std::make_pair(MemI, LSCEVAddRec));

      Type *I8Ptr = Type::getInt8PtrTy((*I)->getContext(), PtrAddrSpace);
      SCEVExpander SCEVE(*SE, J->getModule()->getDataLayout(), "prefaddr");
      Value *PrefPtrValue = SCEVE.expandCodeFor(NextLSCEV, I8Ptr, MemI);

      IRBuilder<> Builder(MemI);
      Module *M = (*I)->getParent()->getParent();
      Type *I32 = Type::getInt32Ty((*I)->getContext());
      Value *PrefetchFunc = Intrinsic::getDeclaration(M, Intrinsic::prefetch);
      Builder.CreateCall(
          PrefetchFunc,
          {PrefPtrValue,
           ConstantInt::get(I32, MemI->mayReadFromMemory() ? 0 : 1),
           ConstantInt::get(I32, 3), ConstantInt::get(I32, 1)});

      MadeChange = true;
    }
  }

  return MadeChange;
}

