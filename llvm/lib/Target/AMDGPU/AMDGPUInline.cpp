//===- AMDGPUInline.cpp - Code to perform simple function inlining --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This is AMDGPU specific replacement of the standard inliner.
/// The main purpose is to account for the fact that calls not only expensive
/// on the AMDGPU, but much more expensive if a private memory pointer is
/// passed to a function as an argument. In this situation, we are unable to
/// eliminate private memory in the caller unless inlined and end up with slow
/// and expensive scratch access. Thus, we boost the inline threshold for such
/// functions here.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Inliner.h"

using namespace llvm;

#define DEBUG_TYPE "inline"

static cl::opt<int>
ArgAllocaCost("amdgpu-inline-arg-alloca-cost", cl::Hidden, cl::init(4000),
              cl::desc("Cost of alloca argument"));

// If the amount of scratch memory to eliminate exceeds our ability to allocate
// it into registers we gain nothing by aggressively inlining functions for that
// heuristic.
static cl::opt<unsigned>
ArgAllocaCutoff("amdgpu-inline-arg-alloca-cutoff", cl::Hidden, cl::init(256),
                cl::desc("Maximum alloca size to use for inline cost"));

// Inliner constraint to achieve reasonable compilation time
static cl::opt<size_t>
MaxBB("amdgpu-inline-max-bb", cl::Hidden, cl::init(1100),
      cl::desc("Maximum BB number allowed in a function after inlining"
               " (compile time constraint)"));

namespace {

class AMDGPUInliner : public LegacyInlinerBase {

public:
  AMDGPUInliner() : LegacyInlinerBase(ID) {
    initializeAMDGPUInlinerPass(*PassRegistry::getPassRegistry());
    Params = getInlineParams();
  }

  static char ID; // Pass identification, replacement for typeid

  unsigned getInlineThreshold(CallBase &CB) const;

  InlineCost getInlineCost(CallBase &CB) override;

  bool runOnSCC(CallGraphSCC &SCC) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  TargetTransformInfoWrapperPass *TTIWP;

  InlineParams Params;
};

} // end anonymous namespace

char AMDGPUInliner::ID = 0;
INITIALIZE_PASS_BEGIN(AMDGPUInliner, "amdgpu-inline",
                "AMDGPU Function Integration/Inlining", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUInliner, "amdgpu-inline",
                "AMDGPU Function Integration/Inlining", false, false)

Pass *llvm::createAMDGPUFunctionInliningPass() { return new AMDGPUInliner(); }

bool AMDGPUInliner::runOnSCC(CallGraphSCC &SCC) {
  TTIWP = &getAnalysis<TargetTransformInfoWrapperPass>();
  return LegacyInlinerBase::runOnSCC(SCC);
}

void AMDGPUInliner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetTransformInfoWrapperPass>();
  LegacyInlinerBase::getAnalysisUsage(AU);
}

unsigned AMDGPUInliner::getInlineThreshold(CallBase &CB) const {
  int Thres = Params.DefaultThreshold;

  Function *Caller = CB.getCaller();
  // Listen to the inlinehint attribute when it would increase the threshold
  // and the caller does not need to minimize its size.
  Function *Callee = CB.getCalledFunction();
  bool InlineHint = Callee && !Callee->isDeclaration() &&
    Callee->hasFnAttribute(Attribute::InlineHint);
  if (InlineHint && Params.HintThreshold && Params.HintThreshold > Thres
      && !Caller->hasFnAttribute(Attribute::MinSize))
    Thres = Params.HintThreshold.getValue() *
            TTIWP->getTTI(*Callee).getInliningThresholdMultiplier();

  const DataLayout &DL = Caller->getParent()->getDataLayout();
  if (!Callee)
    return (unsigned)Thres;

  // If we have a pointer to private array passed into a function
  // it will not be optimized out, leaving scratch usage.
  // Increase the inline threshold to allow inliniting in this case.
  uint64_t AllocaSize = 0;
  SmallPtrSet<const AllocaInst *, 8> AIVisited;
  for (Value *PtrArg : CB.args()) {
    PointerType *Ty = dyn_cast<PointerType>(PtrArg->getType());
    if (!Ty || (Ty->getAddressSpace() != AMDGPUAS::PRIVATE_ADDRESS &&
                Ty->getAddressSpace() != AMDGPUAS::FLAT_ADDRESS))
      continue;

    PtrArg = getUnderlyingObject(PtrArg);
    if (const AllocaInst *AI = dyn_cast<AllocaInst>(PtrArg)) {
      if (!AI->isStaticAlloca() || !AIVisited.insert(AI).second)
        continue;
      AllocaSize += DL.getTypeAllocSize(AI->getAllocatedType());
      // If the amount of stack memory is excessive we will not be able
      // to get rid of the scratch anyway, bail out.
      if (AllocaSize > ArgAllocaCutoff) {
        AllocaSize = 0;
        break;
      }
    }
  }
  if (AllocaSize)
    Thres += ArgAllocaCost;

  return (unsigned)Thres;
}

// Check if call is just a wrapper around another call.
// In this case we only have call and ret instructions.
static bool isWrapperOnlyCall(CallBase &CB) {
  Function *Callee = CB.getCalledFunction();
  if (!Callee || Callee->size() != 1)
    return false;
  const BasicBlock &BB = Callee->getEntryBlock();
  if (const Instruction *I = BB.getFirstNonPHI()) {
    if (!isa<CallInst>(I)) {
      return false;
    }
    if (isa<ReturnInst>(*std::next(I->getIterator()))) {
      LLVM_DEBUG(dbgs() << "    Wrapper only call detected: "
                        << Callee->getName() << '\n');
      return true;
    }
  }
  return false;
}

InlineCost AMDGPUInliner::getInlineCost(CallBase &CB) {
  Function *Callee = CB.getCalledFunction();
  Function *Caller = CB.getCaller();

  if (!Callee || Callee->isDeclaration())
    return llvm::InlineCost::getNever("undefined callee");

  if (CB.isNoInline())
    return llvm::InlineCost::getNever("noinline");

  TargetTransformInfo &TTI = TTIWP->getTTI(*Callee);
  if (!TTI.areInlineCompatible(Caller, Callee))
    return llvm::InlineCost::getNever("incompatible");

  if (CB.hasFnAttr(Attribute::AlwaysInline)) {
    auto IsViable = isInlineViable(*Callee);
    if (IsViable.isSuccess())
      return llvm::InlineCost::getAlways("alwaysinline viable");
    return llvm::InlineCost::getNever(IsViable.getFailureReason());
  }

  if (isWrapperOnlyCall(CB))
    return llvm::InlineCost::getAlways("wrapper-only call");

  InlineParams LocalParams = Params;
  LocalParams.DefaultThreshold = (int)getInlineThreshold(CB);
  bool RemarksEnabled = false;
  const auto &BBs = Caller->getBasicBlockList();
  if (!BBs.empty()) {
    auto DI = OptimizationRemark(DEBUG_TYPE, "", DebugLoc(), &BBs.front());
    if (DI.isEnabled())
      RemarksEnabled = true;
  }

  OptimizationRemarkEmitter ORE(Caller);
  auto GetAssumptionCache = [this](Function &F) -> AssumptionCache & {
    return ACT->getAssumptionCache(F);
  };

  auto IC = llvm::getInlineCost(CB, Callee, LocalParams, TTI,
                                GetAssumptionCache, GetTLI, nullptr, PSI,
                                RemarksEnabled ? &ORE : nullptr);

  if (IC && !IC.isAlways() && !Callee->hasFnAttribute(Attribute::InlineHint)) {
    // Single BB does not increase total BB amount, thus subtract 1
    size_t Size = Caller->size() + Callee->size() - 1;
    if (MaxBB && Size > MaxBB)
      return llvm::InlineCost::getNever("max number of bb exceeded");
  }
  return IC;
}
