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
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Pass.h"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Type.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLibraryInfo.h"
using namespace llvm;

// Register the AliasAnalysis interface, providing a nice name to refer to.
INITIALIZE_ANALYSIS_GROUP(AliasAnalysis, "Alias Analysis", NoAA)
char AliasAnalysis::ID = 0;

//===----------------------------------------------------------------------===//
// Default chaining methods
//===----------------------------------------------------------------------===//

AliasAnalysis::AliasResult
AliasAnalysis::alias(const Location &LocA, const Location &LocB) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->alias(LocA, LocB);
}

bool AliasAnalysis::pointsToConstantMemory(const Location &Loc,
                                           bool OrLocal) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->pointsToConstantMemory(Loc, OrLocal);
}

void AliasAnalysis::deleteValue(Value *V) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  AA->deleteValue(V);
}

void AliasAnalysis::copyValue(Value *From, Value *To) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  AA->copyValue(From, To);
}

void AliasAnalysis::addEscapingUse(Use &U) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  AA->addEscapingUse(U);
}


AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(ImmutableCallSite CS,
                             const Location &Loc) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");

  ModRefBehavior MRB = getModRefBehavior(CS);
  if (MRB == DoesNotAccessMemory)
    return NoModRef;

  ModRefResult Mask = ModRef;
  if (onlyReadsMemory(MRB))
    Mask = Ref;

  if (onlyAccessesArgPointees(MRB)) {
    bool doesAlias = false;
    if (doesAccessArgPointees(MRB)) {
      MDNode *CSTag = CS.getInstruction()->getMetadata(LLVMContext::MD_tbaa);
      for (ImmutableCallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
           AI != AE; ++AI) {
        const Value *Arg = *AI;
        if (!Arg->getType()->isPointerTy())
          continue;
        Location CSLoc(Arg, UnknownSize, CSTag);
        if (!isNoAlias(CSLoc, Loc)) {
          doesAlias = true;
          break;
        }
      }
    }
    if (!doesAlias)
      return NoModRef;
  }

  // If Loc is a constant memory location, the call definitely could not
  // modify the memory location.
  if ((Mask & Mod) && pointsToConstantMemory(Loc))
    Mask = ModRefResult(Mask & ~Mod);

  // If this is the end of the chain, don't forward.
  if (!AA) return Mask;

  // Otherwise, fall back to the next AA in the chain. But we can merge
  // in any mask we've managed to compute.
  return ModRefResult(AA->getModRefInfo(CS, Loc) & Mask);
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(ImmutableCallSite CS1, ImmutableCallSite CS2) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");

  // If CS1 or CS2 are readnone, they don't interact.
  ModRefBehavior CS1B = getModRefBehavior(CS1);
  if (CS1B == DoesNotAccessMemory) return NoModRef;

  ModRefBehavior CS2B = getModRefBehavior(CS2);
  if (CS2B == DoesNotAccessMemory) return NoModRef;

  // If they both only read from memory, there is no dependence.
  if (onlyReadsMemory(CS1B) && onlyReadsMemory(CS2B))
    return NoModRef;

  AliasAnalysis::ModRefResult Mask = ModRef;

  // If CS1 only reads memory, the only dependence on CS2 can be
  // from CS1 reading memory written by CS2.
  if (onlyReadsMemory(CS1B))
    Mask = ModRefResult(Mask & Ref);

  // If CS2 only access memory through arguments, accumulate the mod/ref
  // information from CS1's references to the memory referenced by
  // CS2's arguments.
  if (onlyAccessesArgPointees(CS2B)) {
    AliasAnalysis::ModRefResult R = NoModRef;
    if (doesAccessArgPointees(CS2B)) {
      MDNode *CS2Tag = CS2.getInstruction()->getMetadata(LLVMContext::MD_tbaa);
      for (ImmutableCallSite::arg_iterator
           I = CS2.arg_begin(), E = CS2.arg_end(); I != E; ++I) {
        const Value *Arg = *I;
        if (!Arg->getType()->isPointerTy())
          continue;
        Location CS2Loc(Arg, UnknownSize, CS2Tag);
        R = ModRefResult((R | getModRefInfo(CS1, CS2Loc)) & Mask);
        if (R == Mask)
          break;
      }
    }
    return R;
  }

  // If CS1 only accesses memory through arguments, check if CS2 references
  // any of the memory referenced by CS1's arguments. If not, return NoModRef.
  if (onlyAccessesArgPointees(CS1B)) {
    AliasAnalysis::ModRefResult R = NoModRef;
    if (doesAccessArgPointees(CS1B)) {
      MDNode *CS1Tag = CS1.getInstruction()->getMetadata(LLVMContext::MD_tbaa);
      for (ImmutableCallSite::arg_iterator
           I = CS1.arg_begin(), E = CS1.arg_end(); I != E; ++I) {
        const Value *Arg = *I;
        if (!Arg->getType()->isPointerTy())
          continue;
        Location CS1Loc(Arg, UnknownSize, CS1Tag);
        if (getModRefInfo(CS2, CS1Loc) != NoModRef) {
          R = Mask;
          break;
        }
      }
    }
    if (R == NoModRef)
      return R;
  }

  // If this is the end of the chain, don't forward.
  if (!AA) return Mask;

  // Otherwise, fall back to the next AA in the chain. But we can merge
  // in any mask we've managed to compute.
  return ModRefResult(AA->getModRefInfo(CS1, CS2) & Mask);
}

AliasAnalysis::ModRefBehavior
AliasAnalysis::getModRefBehavior(ImmutableCallSite CS) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");

  ModRefBehavior Min = UnknownModRefBehavior;

  // Call back into the alias analysis with the other form of getModRefBehavior
  // to see if it can give a better response.
  if (const Function *F = CS.getCalledFunction())
    Min = getModRefBehavior(F);

  // If this is the end of the chain, don't forward.
  if (!AA) return Min;

  // Otherwise, fall back to the next AA in the chain. But we can merge
  // in any result we've managed to compute.
  return ModRefBehavior(AA->getModRefBehavior(CS) & Min);
}

AliasAnalysis::ModRefBehavior
AliasAnalysis::getModRefBehavior(const Function *F) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->getModRefBehavior(F);
}

//===----------------------------------------------------------------------===//
// AliasAnalysis non-virtual helper method implementation
//===----------------------------------------------------------------------===//

AliasAnalysis::Location AliasAnalysis::getLocation(const LoadInst *LI) {
  return Location(LI->getPointerOperand(),
                  getTypeStoreSize(LI->getType()),
                  LI->getMetadata(LLVMContext::MD_tbaa));
}

AliasAnalysis::Location AliasAnalysis::getLocation(const StoreInst *SI) {
  return Location(SI->getPointerOperand(),
                  getTypeStoreSize(SI->getValueOperand()->getType()),
                  SI->getMetadata(LLVMContext::MD_tbaa));
}

AliasAnalysis::Location AliasAnalysis::getLocation(const VAArgInst *VI) {
  return Location(VI->getPointerOperand(),
                  UnknownSize,
                  VI->getMetadata(LLVMContext::MD_tbaa));
}

AliasAnalysis::Location
AliasAnalysis::getLocation(const AtomicCmpXchgInst *CXI) {
  return Location(CXI->getPointerOperand(),
                  getTypeStoreSize(CXI->getCompareOperand()->getType()),
                  CXI->getMetadata(LLVMContext::MD_tbaa));
}

AliasAnalysis::Location
AliasAnalysis::getLocation(const AtomicRMWInst *RMWI) {
  return Location(RMWI->getPointerOperand(),
                  getTypeStoreSize(RMWI->getValOperand()->getType()),
                  RMWI->getMetadata(LLVMContext::MD_tbaa));
}

AliasAnalysis::Location 
AliasAnalysis::getLocationForSource(const MemTransferInst *MTI) {
  uint64_t Size = UnknownSize;
  if (ConstantInt *C = dyn_cast<ConstantInt>(MTI->getLength()))
    Size = C->getValue().getZExtValue();

  // memcpy/memmove can have TBAA tags. For memcpy, they apply
  // to both the source and the destination.
  MDNode *TBAATag = MTI->getMetadata(LLVMContext::MD_tbaa);

  return Location(MTI->getRawSource(), Size, TBAATag);
}

AliasAnalysis::Location 
AliasAnalysis::getLocationForDest(const MemIntrinsic *MTI) {
  uint64_t Size = UnknownSize;
  if (ConstantInt *C = dyn_cast<ConstantInt>(MTI->getLength()))
    Size = C->getValue().getZExtValue();

  // memcpy/memmove can have TBAA tags. For memcpy, they apply
  // to both the source and the destination.
  MDNode *TBAATag = MTI->getMetadata(LLVMContext::MD_tbaa);
  
  return Location(MTI->getRawDest(), Size, TBAATag);
}



AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(const LoadInst *L, const Location &Loc) {
  // Be conservative in the face of volatile/atomic.
  if (!L->isUnordered())
    return ModRef;

  // If the load address doesn't alias the given address, it doesn't read
  // or write the specified memory.
  if (!alias(getLocation(L), Loc))
    return NoModRef;

  // Otherwise, a load just reads.
  return Ref;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(const StoreInst *S, const Location &Loc) {
  // Be conservative in the face of volatile/atomic.
  if (!S->isUnordered())
    return ModRef;

  // If the store address cannot alias the pointer in question, then the
  // specified memory cannot be modified by the store.
  if (!alias(getLocation(S), Loc))
    return NoModRef;

  // If the pointer is a pointer to constant memory, then it could not have been
  // modified by this store.
  if (pointsToConstantMemory(Loc))
    return NoModRef;

  // Otherwise, a store just writes.
  return Mod;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(const VAArgInst *V, const Location &Loc) {
  // If the va_arg address cannot alias the pointer in question, then the
  // specified memory cannot be accessed by the va_arg.
  if (!alias(getLocation(V), Loc))
    return NoModRef;

  // If the pointer is a pointer to constant memory, then it could not have been
  // modified by this va_arg.
  if (pointsToConstantMemory(Loc))
    return NoModRef;

  // Otherwise, a va_arg reads and writes.
  return ModRef;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(const AtomicCmpXchgInst *CX, const Location &Loc) {
  // Acquire/Release cmpxchg has properties that matter for arbitrary addresses.
  if (CX->getOrdering() > Monotonic)
    return ModRef;

  // If the cmpxchg address does not alias the location, it does not access it.
  if (!alias(getLocation(CX), Loc))
    return NoModRef;

  return ModRef;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(const AtomicRMWInst *RMW, const Location &Loc) {
  // Acquire/Release atomicrmw has properties that matter for arbitrary addresses.
  if (RMW->getOrdering() > Monotonic)
    return ModRef;

  // If the atomicrmw address does not alias the location, it does not access it.
  if (!alias(getLocation(RMW), Loc))
    return NoModRef;

  return ModRef;
}

namespace {
  /// Only find pointer captures which happen before the given instruction. Uses
  /// the dominator tree to determine whether one instruction is before another.
  struct CapturesBefore : public CaptureTracker {
    CapturesBefore(const Instruction *I, DominatorTree *DT)
      : BeforeHere(I), DT(DT), Captured(false) {}

    void tooManyUses() { Captured = true; }

    bool shouldExplore(Use *U) {
      Instruction *I = cast<Instruction>(U->getUser());
      BasicBlock *BB = I->getParent();
      if (BeforeHere != I &&
          (!DT->isReachableFromEntry(BB) || DT->dominates(BeforeHere, I)))
        return false;
      return true;
    }

    bool captured(Use *U) {
      Instruction *I = cast<Instruction>(U->getUser());
      BasicBlock *BB = I->getParent();
      if (BeforeHere != I &&
          (!DT->isReachableFromEntry(BB) || DT->dominates(BeforeHere, I)))
        return false;
      Captured = true;
      return true;
    }

    const Instruction *BeforeHere;
    DominatorTree *DT;

    bool Captured;
  };
}

// FIXME: this is really just shoring-up a deficiency in alias analysis.
// BasicAA isn't willing to spend linear time determining whether an alloca
// was captured before or after this particular call, while we are. However,
// with a smarter AA in place, this test is just wasting compile time.
AliasAnalysis::ModRefResult
AliasAnalysis::callCapturesBefore(const Instruction *I,
                                  const AliasAnalysis::Location &MemLoc,
                                  DominatorTree *DT) {
  if (!DT || !TD) return AliasAnalysis::ModRef;

  const Value *Object = GetUnderlyingObject(MemLoc.Ptr, TD);
  if (!isIdentifiedObject(Object) || isa<GlobalValue>(Object) ||
      isa<Constant>(Object))
    return AliasAnalysis::ModRef;

  ImmutableCallSite CS(I);
  if (!CS.getInstruction() || CS.getInstruction() == Object)
    return AliasAnalysis::ModRef;

  CapturesBefore CB(I, DT);
  llvm::PointerMayBeCaptured(Object, &CB);
  if (CB.Captured)
    return AliasAnalysis::ModRef;

  unsigned ArgNo = 0;
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
    if (!isNoAlias(AliasAnalysis::Location(*CI),
                   AliasAnalysis::Location(Object))) {
      return AliasAnalysis::ModRef;
    }
  }
  return AliasAnalysis::NoModRef;
}

// AliasAnalysis destructor: DO NOT move this to the header file for
// AliasAnalysis or else clients of the AliasAnalysis class may not depend on
// the AliasAnalysis.o file in the current .a file, causing alias analysis
// support to not be included in the tool correctly!
//
AliasAnalysis::~AliasAnalysis() {}

/// InitializeAliasAnalysis - Subclasses must call this method to initialize the
/// AliasAnalysis interface before any other methods are called.
///
void AliasAnalysis::InitializeAliasAnalysis(Pass *P) {
  TD = P->getAnalysisIfAvailable<DataLayout>();
  TLI = P->getAnalysisIfAvailable<TargetLibraryInfo>();
  AA = &P->getAnalysis<AliasAnalysis>();
}

// getAnalysisUsage - All alias analysis implementations should invoke this
// directly (using AliasAnalysis::getAnalysisUsage(AU)).
void AliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();         // All AA's chain
}

/// getTypeStoreSize - Return the DataLayout store size for the given type,
/// if known, or a conservative value otherwise.
///
uint64_t AliasAnalysis::getTypeStoreSize(Type *Ty) {
  return TD ? TD->getTypeStoreSize(Ty) : UnknownSize;
}

/// canBasicBlockModify - Return true if it is possible for execution of the
/// specified basic block to modify the value pointed to by Ptr.
///
bool AliasAnalysis::canBasicBlockModify(const BasicBlock &BB,
                                        const Location &Loc) {
  return canInstructionRangeModify(BB.front(), BB.back(), Loc);
}

/// canInstructionRangeModify - Return true if it is possible for the execution
/// of the specified instructions to modify the value pointed to by Ptr.  The
/// instructions to consider are all of the instructions in the range of [I1,I2]
/// INCLUSIVE.  I1 and I2 must be in the same basic block.
///
bool AliasAnalysis::canInstructionRangeModify(const Instruction &I1,
                                              const Instruction &I2,
                                              const Location &Loc) {
  assert(I1.getParent() == I2.getParent() &&
         "Instructions not in same basic block!");
  BasicBlock::const_iterator I = &I1;
  BasicBlock::const_iterator E = &I2;
  ++E;  // Convert from inclusive to exclusive range.

  for (; I != E; ++I) // Check every instruction in range
    if (getModRefInfo(I, Loc) & Mod)
      return true;
  return false;
}

/// isNoAliasCall - Return true if this pointer is returned by a noalias
/// function.
bool llvm::isNoAliasCall(const Value *V) {
  if (isa<CallInst>(V) || isa<InvokeInst>(V))
    return ImmutableCallSite(cast<Instruction>(V))
      .paramHasAttr(0, Attributes::NoAlias);
  return false;
}

/// isIdentifiedObject - Return true if this pointer refers to a distinct and
/// identifiable object.  This returns true for:
///    Global Variables and Functions (but not Global Aliases)
///    Allocas and Mallocs
///    ByVal and NoAlias Arguments
///    NoAlias returns
///
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

/// isKnownNonNull - Return true if we know that the specified value is never
/// null.
bool llvm::isKnownNonNull(const Value *V) {
  // Alloca never returns null, malloc might.
  if (isa<AllocaInst>(V)) return true;

  // A byval argument is never null.
  if (const Argument *A = dyn_cast<Argument>(V))
    return A->hasByValAttr();

  // Global values are not null unless extern weak.
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(V))
    return !GV->hasExternalWeakLinkage();
  return false;
}
