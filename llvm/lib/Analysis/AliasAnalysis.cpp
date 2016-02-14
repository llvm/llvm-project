//===- AliasAnalysis.cpp - Generic Alias Analysis Interface Implementation -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the generic AliasAnalysis interface which is used as the
// common interface used by all clients and implementations of alias analysis.
//
// This file also implements the default version of the AliasAnalysis interface
// that is to be used when no other implementation is specified.  This does some
// simple tests that detect obvious cases: two different global pointers cannot
// alias, a global cannot alias a malloc, two different mallocs cannot alias,
// etc.
//
// This alias analysis implementation really isn't very good for anything, but
// it is very fast, and makes a nice clean default implementation.  Because it
// handles lots of little corner cases, other, more complex, alias analysis
// implementations may choose to rely on this pass to resolve these simple and
// easy cases.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CFLAliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ObjCARCAliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
using namespace llvm;

/// Allow disabling BasicAA from the AA results. This is particularly useful
/// when testing to isolate a single AA implementation.
static cl::opt<bool> DisableBasicAA("disable-basicaa", cl::Hidden,
                                    cl::init(false));

AAResults::AAResults(AAResults &&Arg) : AAs(std::move(Arg.AAs)) {
  for (auto &AA : AAs)
    AA->setAAResults(this);
}

AAResults &AAResults::operator=(AAResults &&Arg) {
  AAs = std::move(Arg.AAs);
  for (auto &AA : AAs)
    AA->setAAResults(this);
  return *this;
}

AAResults::~AAResults() {
// FIXME; It would be nice to at least clear out the pointers back to this
// aggregation here, but we end up with non-nesting lifetimes in the legacy
// pass manager that prevent this from working. In the legacy pass manager
// we'll end up with dangling references here in some cases.
#if 0
  for (auto &AA : AAs)
    AA->setAAResults(nullptr);
#endif
}

//===----------------------------------------------------------------------===//
// Default chaining methods
//===----------------------------------------------------------------------===//

AliasResult AAResults::alias(const MemoryLocation &LocA,
                             const MemoryLocation &LocB) {
  for (const auto &AA : AAs) {
    auto Result = AA->alias(LocA, LocB);
    if (Result != MayAlias)
      return Result;
  }
  return MayAlias;
}

bool AAResults::pointsToConstantMemory(const MemoryLocation &Loc,
                                       bool OrLocal) {
  for (const auto &AA : AAs)
    if (AA->pointsToConstantMemory(Loc, OrLocal))
      return true;

  return false;
}

ModRefInfo AAResults::getArgModRefInfo(ImmutableCallSite CS, unsigned ArgIdx) {
  ModRefInfo Result = MRI_ModRef;

  for (const auto &AA : AAs) {
    Result = ModRefInfo(Result & AA->getArgModRefInfo(CS, ArgIdx));

    // Early-exit the moment we reach the bottom of the lattice.
    if (Result == MRI_NoModRef)
      return Result;
  }

  return Result;
}

ModRefInfo AAResults::getModRefInfo(Instruction *I, ImmutableCallSite Call) {
  // We may have two calls
  if (auto CS = ImmutableCallSite(I)) {
    // Check if the two calls modify the same memory
    return getModRefInfo(Call, CS);
  } else {
    // Otherwise, check if the call modifies or references the
    // location this memory access defines.  The best we can say
    // is that if the call references what this instruction
    // defines, it must be clobbered by this location.
    const MemoryLocation DefLoc = MemoryLocation::get(I);
    if (getModRefInfo(Call, DefLoc) != MRI_NoModRef)
      return MRI_ModRef;
  }
  return MRI_NoModRef;
}

ModRefInfo AAResults::getModRefInfo(ImmutableCallSite CS,
                                    const MemoryLocation &Loc) {
  ModRefInfo Result = MRI_ModRef;

  for (const auto &AA : AAs) {
    Result = ModRefInfo(Result & AA->getModRefInfo(CS, Loc));

    // Early-exit the moment we reach the bottom of the lattice.
    if (Result == MRI_NoModRef)
      return Result;
  }

  return Result;
}

ModRefInfo AAResults::getModRefInfo(ImmutableCallSite CS1,
                                    ImmutableCallSite CS2) {
  ModRefInfo Result = MRI_ModRef;

  for (const auto &AA : AAs) {
    Result = ModRefInfo(Result & AA->getModRefInfo(CS1, CS2));

    // Early-exit the moment we reach the bottom of the lattice.
    if (Result == MRI_NoModRef)
      return Result;
  }

  return Result;
}

FunctionModRefBehavior AAResults::getModRefBehavior(ImmutableCallSite CS) {
  FunctionModRefBehavior Result = FMRB_UnknownModRefBehavior;

  for (const auto &AA : AAs) {
    Result = FunctionModRefBehavior(Result & AA->getModRefBehavior(CS));

    // Early-exit the moment we reach the bottom of the lattice.
    if (Result == FMRB_DoesNotAccessMemory)
      return Result;
  }

  return Result;
}

FunctionModRefBehavior AAResults::getModRefBehavior(const Function *F) {
  FunctionModRefBehavior Result = FMRB_UnknownModRefBehavior;

  for (const auto &AA : AAs) {
    Result = FunctionModRefBehavior(Result & AA->getModRefBehavior(F));

    // Early-exit the moment we reach the bottom of the lattice.
    if (Result == FMRB_DoesNotAccessMemory)
      return Result;
  }

  return Result;
}

//===----------------------------------------------------------------------===//
// Helper method implementation
//===----------------------------------------------------------------------===//

ModRefInfo AAResults::getModRefInfo(const LoadInst *L,
                                    const MemoryLocation &Loc) {
  // Be conservative in the face of volatile/atomic.
  if (!L->isUnordered())
    return MRI_ModRef;

  // If the load address doesn't alias the given address, it doesn't read
  // or write the specified memory.
  if (Loc.Ptr && !alias(MemoryLocation::get(L), Loc))
    return MRI_NoModRef;

  // Otherwise, a load just reads.
  return MRI_Ref;
}

ModRefInfo AAResults::getModRefInfo(const StoreInst *S,
                                    const MemoryLocation &Loc) {
  // Be conservative in the face of volatile/atomic.
  if (!S->isUnordered())
    return MRI_ModRef;

  if (Loc.Ptr) {
    // If the store address cannot alias the pointer in question, then the
    // specified memory cannot be modified by the store.
    if (!alias(MemoryLocation::get(S), Loc))
      return MRI_NoModRef;

    // If the pointer is a pointer to constant memory, then it could not have
    // been modified by this store.
    if (pointsToConstantMemory(Loc))
      return MRI_NoModRef;
  }

  // Otherwise, a store just writes.
  return MRI_Mod;
}

ModRefInfo AAResults::getModRefInfo(const VAArgInst *V,
                                    const MemoryLocation &Loc) {

  if (Loc.Ptr) {
    // If the va_arg address cannot alias the pointer in question, then the
    // specified memory cannot be accessed by the va_arg.
    if (!alias(MemoryLocation::get(V), Loc))
      return MRI_NoModRef;

    // If the pointer is a pointer to constant memory, then it could not have
    // been modified by this va_arg.
    if (pointsToConstantMemory(Loc))
      return MRI_NoModRef;
  }

  // Otherwise, a va_arg reads and writes.
  return MRI_ModRef;
}

ModRefInfo AAResults::getModRefInfo(const CatchPadInst *CatchPad,
                                    const MemoryLocation &Loc) {
  if (Loc.Ptr) {
    // If the pointer is a pointer to constant memory,
    // then it could not have been modified by this catchpad.
    if (pointsToConstantMemory(Loc))
      return MRI_NoModRef;
  }

  // Otherwise, a catchpad reads and writes.
  return MRI_ModRef;
}

ModRefInfo AAResults::getModRefInfo(const CatchReturnInst *CatchRet,
                                    const MemoryLocation &Loc) {
  if (Loc.Ptr) {
    // If the pointer is a pointer to constant memory,
    // then it could not have been modified by this catchpad.
    if (pointsToConstantMemory(Loc))
      return MRI_NoModRef;
  }

  // Otherwise, a catchret reads and writes.
  return MRI_ModRef;
}

ModRefInfo AAResults::getModRefInfo(const AtomicCmpXchgInst *CX,
                                    const MemoryLocation &Loc) {
  // Acquire/Release cmpxchg has properties that matter for arbitrary addresses.
  if (CX->getSuccessOrdering() > Monotonic)
    return MRI_ModRef;

  // If the cmpxchg address does not alias the location, it does not access it.
  if (Loc.Ptr && !alias(MemoryLocation::get(CX), Loc))
    return MRI_NoModRef;

  return MRI_ModRef;
}

ModRefInfo AAResults::getModRefInfo(const AtomicRMWInst *RMW,
                                    const MemoryLocation &Loc) {
  // Acquire/Release atomicrmw has properties that matter for arbitrary addresses.
  if (RMW->getOrdering() > Monotonic)
    return MRI_ModRef;

  // If the atomicrmw address does not alias the location, it does not access it.
  if (Loc.Ptr && !alias(MemoryLocation::get(RMW), Loc))
    return MRI_NoModRef;

  return MRI_ModRef;
}

/// \brief Return information about whether a particular call site modifies
/// or reads the specified memory location \p MemLoc before instruction \p I
/// in a BasicBlock. A ordered basic block \p OBB can be used to speed up
/// instruction-ordering queries inside the BasicBlock containing \p I.
/// FIXME: this is really just shoring-up a deficiency in alias analysis.
/// BasicAA isn't willing to spend linear time determining whether an alloca
/// was captured before or after this particular call, while we are. However,
/// with a smarter AA in place, this test is just wasting compile time.
ModRefInfo AAResults::callCapturesBefore(const Instruction *I,
                                         const MemoryLocation &MemLoc,
                                         DominatorTree *DT,
                                         OrderedBasicBlock *OBB) {
  if (!DT)
    return MRI_ModRef;

  const Value *Object =
      GetUnderlyingObject(MemLoc.Ptr, I->getModule()->getDataLayout());
  if (!isIdentifiedObject(Object) || isa<GlobalValue>(Object) ||
      isa<Constant>(Object))
    return MRI_ModRef;

  ImmutableCallSite CS(I);
  if (!CS.getInstruction() || CS.getInstruction() == Object)
    return MRI_ModRef;

  if (llvm::PointerMayBeCapturedBefore(Object, /* ReturnCaptures */ true,
                                       /* StoreCaptures */ true, I, DT,
                                       /* include Object */ true,
                                       /* OrderedBasicBlock */ OBB))
    return MRI_ModRef;

  unsigned ArgNo = 0;
  ModRefInfo R = MRI_NoModRef;
  for (ImmutableCallSite::arg_iterator CI = CS.arg_begin(), CE = CS.arg_end();
       CI != CE; ++CI, ++ArgNo) {
    // Only look at the no-capture or byval pointer arguments.  If this
    // pointer were passed to arguments that were neither of these, then it
    // couldn't be no-capture.
    if (!(*CI)->getType()->isPointerTy() ||
        (!CS.doesNotCapture(ArgNo) && !CS.isByValArgument(ArgNo)))
      continue;

    // If this is a no-capture pointer argument, see if we can tell that it
    // is impossible to alias the pointer we're checking.  If not, we have to
    // assume that the call could touch the pointer, even though it doesn't
    // escape.
    if (isNoAlias(MemoryLocation(*CI), MemoryLocation(Object)))
      continue;
    if (CS.doesNotAccessMemory(ArgNo))
      continue;
    if (CS.onlyReadsMemory(ArgNo)) {
      R = MRI_Ref;
      continue;
    }
    return MRI_ModRef;
  }
  return R;
}

/// canBasicBlockModify - Return true if it is possible for execution of the
/// specified basic block to modify the location Loc.
///
bool AAResults::canBasicBlockModify(const BasicBlock &BB,
                                    const MemoryLocation &Loc) {
  return canInstructionRangeModRef(BB.front(), BB.back(), Loc, MRI_Mod);
}

/// canInstructionRangeModRef - Return true if it is possible for the
/// execution of the specified instructions to mod\ref (according to the
/// mode) the location Loc. The instructions to consider are all
/// of the instructions in the range of [I1,I2] INCLUSIVE.
/// I1 and I2 must be in the same basic block.
bool AAResults::canInstructionRangeModRef(const Instruction &I1,
                                          const Instruction &I2,
                                          const MemoryLocation &Loc,
                                          const ModRefInfo Mode) {
  assert(I1.getParent() == I2.getParent() &&
         "Instructions not in same basic block!");
  BasicBlock::const_iterator I = I1.getIterator();
  BasicBlock::const_iterator E = I2.getIterator();
  ++E;  // Convert from inclusive to exclusive range.

  for (; I != E; ++I) // Check every instruction in range
    if (getModRefInfo(&*I, Loc) & Mode)
      return true;
  return false;
}

// Provide a definition for the root virtual destructor.
AAResults::Concept::~Concept() {}

// Provide a definition for the static object used to identify passes.
char AAManager::PassID;

namespace {
/// A wrapper pass for external alias analyses. This just squirrels away the
/// callback used to run any analyses and register their results.
struct ExternalAAWrapperPass : ImmutablePass {
  typedef std::function<void(Pass &, Function &, AAResults &)> CallbackT;

  CallbackT CB;

  static char ID;

  ExternalAAWrapperPass() : ImmutablePass(ID) {
    initializeExternalAAWrapperPassPass(*PassRegistry::getPassRegistry());
  }
  explicit ExternalAAWrapperPass(CallbackT CB)
      : ImmutablePass(ID), CB(std::move(CB)) {
    initializeExternalAAWrapperPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
}

char ExternalAAWrapperPass::ID = 0;
INITIALIZE_PASS(ExternalAAWrapperPass, "external-aa", "External Alias Analysis",
                false, true)

ImmutablePass *
llvm::createExternalAAWrapperPass(ExternalAAWrapperPass::CallbackT Callback) {
  return new ExternalAAWrapperPass(std::move(Callback));
}

AAResultsWrapperPass::AAResultsWrapperPass() : FunctionPass(ID) {
  initializeAAResultsWrapperPassPass(*PassRegistry::getPassRegistry());
}

char AAResultsWrapperPass::ID = 0;

INITIALIZE_PASS_BEGIN(AAResultsWrapperPass, "aa",
                      "Function Alias Analysis Results", false, true)
INITIALIZE_PASS_DEPENDENCY(BasicAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(CFLAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ExternalAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ObjCARCAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SCEVAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScopedNoAliasAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TypeBasedAAWrapperPass)
INITIALIZE_PASS_END(AAResultsWrapperPass, "aa",
                    "Function Alias Analysis Results", false, true)

FunctionPass *llvm::createAAResultsWrapperPass() {
  return new AAResultsWrapperPass();
}

/// Run the wrapper pass to rebuild an aggregation over known AA passes.
///
/// This is the legacy pass manager's interface to the new-style AA results
/// aggregation object. Because this is somewhat shoe-horned into the legacy
/// pass manager, we hard code all the specific alias analyses available into
/// it. While the particular set enabled is configured via commandline flags,
/// adding a new alias analysis to LLVM will require adding support for it to
/// this list.
bool AAResultsWrapperPass::runOnFunction(Function &F) {
  // NB! This *must* be reset before adding new AA results to the new
  // AAResults object because in the legacy pass manager, each instance
  // of these will refer to the *same* immutable analyses, registering and
  // unregistering themselves with them. We need to carefully tear down the
  // previous object first, in this case replacing it with an empty one, before
  // registering new results.
  AAR.reset(new AAResults());

  // BasicAA is always available for function analyses. Also, we add it first
  // so that it can trump TBAA results when it proves MustAlias.
  // FIXME: TBAA should have an explicit mode to support this and then we
  // should reconsider the ordering here.
  if (!DisableBasicAA)
    AAR->addAAResult(getAnalysis<BasicAAWrapperPass>().getResult());

  // Populate the results with the currently available AAs.
  if (auto *WrapperPass = getAnalysisIfAvailable<ScopedNoAliasAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<TypeBasedAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass =
          getAnalysisIfAvailable<objcarc::ObjCARCAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<GlobalsAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<SCEVAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<CFLAAWrapperPass>())
    AAR->addAAResult(WrapperPass->getResult());

  // If available, run an external AA providing callback over the results as
  // well.
  if (auto *WrapperPass = getAnalysisIfAvailable<ExternalAAWrapperPass>())
    if (WrapperPass->CB)
      WrapperPass->CB(*this, F, *AAR);

  // Analyses don't mutate the IR, so return false.
  return false;
}

void AAResultsWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<BasicAAWrapperPass>();

  // We also need to mark all the alias analysis passes we will potentially
  // probe in runOnFunction as used here to ensure the legacy pass manager
  // preserves them. This hard coding of lists of alias analyses is specific to
  // the legacy pass manager.
  AU.addUsedIfAvailable<ScopedNoAliasAAWrapperPass>();
  AU.addUsedIfAvailable<TypeBasedAAWrapperPass>();
  AU.addUsedIfAvailable<objcarc::ObjCARCAAWrapperPass>();
  AU.addUsedIfAvailable<GlobalsAAWrapperPass>();
  AU.addUsedIfAvailable<SCEVAAWrapperPass>();
  AU.addUsedIfAvailable<CFLAAWrapperPass>();
}

AAResults llvm::createLegacyPMAAResults(Pass &P, Function &F,
                                        BasicAAResult &BAR) {
  AAResults AAR;

  // Add in our explicitly constructed BasicAA results.
  if (!DisableBasicAA)
    AAR.addAAResult(BAR);

  // Populate the results with the other currently available AAs.
  if (auto *WrapperPass =
          P.getAnalysisIfAvailable<ScopedNoAliasAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = P.getAnalysisIfAvailable<TypeBasedAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass =
          P.getAnalysisIfAvailable<objcarc::ObjCARCAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = P.getAnalysisIfAvailable<GlobalsAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = P.getAnalysisIfAvailable<CFLAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());

  return AAR;
}

bool llvm::isNoAliasCall(const Value *V) {
  if (auto CS = ImmutableCallSite(V))
    return CS.paramHasAttr(0, Attribute::NoAlias);
  return false;
}

bool llvm::isNoAliasArgument(const Value *V) {
  if (const Argument *A = dyn_cast<Argument>(V))
    return A->hasNoAliasAttr();
  return false;
}

bool llvm::isIdentifiedObject(const Value *V) {
  if (isa<AllocaInst>(V))
    return true;
  if (isa<GlobalValue>(V) && !isa<GlobalAlias>(V))
    return true;
  if (isNoAliasCall(V))
    return true;
  if (const Argument *A = dyn_cast<Argument>(V))
    return A->hasNoAliasAttr() || A->hasByValAttr();
  return false;
}

bool llvm::isIdentifiedFunctionLocal(const Value *V) {
  return isa<AllocaInst>(V) || isNoAliasCall(V) || isNoAliasArgument(V);
}

void llvm::addUsedAAAnalyses(AnalysisUsage &AU) {
  // This function needs to be in sync with llvm::createLegacyPMAAResults -- if
  // more alias analyses are added to llvm::createLegacyPMAAResults, they need
  // to be added here also.
  AU.addUsedIfAvailable<ScopedNoAliasAAWrapperPass>();
  AU.addUsedIfAvailable<TypeBasedAAWrapperPass>();
  AU.addUsedIfAvailable<objcarc::ObjCARCAAWrapperPass>();
  AU.addUsedIfAvailable<GlobalsAAWrapperPass>();
  AU.addUsedIfAvailable<CFLAAWrapperPass>();
}
