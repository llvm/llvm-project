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
#include "llvm/Pass.h"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Target/TargetData.h"
using namespace llvm;

// Register the AliasAnalysis interface, providing a nice name to refer to.
static RegisterAnalysisGroup<AliasAnalysis> Z("Alias Analysis");
char AliasAnalysis::ID = 0;

//===----------------------------------------------------------------------===//
// Default chaining methods
//===----------------------------------------------------------------------===//

AliasAnalysis::AliasResult
AliasAnalysis::alias(const Value *V1, unsigned V1Size,
                     const Value *V2, unsigned V2Size) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->alias(V1, V1Size, V2, V2Size);
}

void AliasAnalysis::getMustAliases(Value *P, std::vector<Value*> &RetVals) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->getMustAliases(P, RetVals);
}

bool AliasAnalysis::pointsToConstantMemory(const Value *P) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->pointsToConstantMemory(P);
}

AliasAnalysis::ModRefBehavior
AliasAnalysis::getModRefBehavior(Function *F, CallSite CS,
                                 std::vector<PointerAccessInfo> *Info) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->getModRefBehavior(F, CS, Info);
}

bool AliasAnalysis::hasNoModRefInfoForCalls() const {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->hasNoModRefInfoForCalls();
}

void AliasAnalysis::deleteValue(Value *V) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  AA->deleteValue(V);
}

void AliasAnalysis::copyValue(Value *From, Value *To) {
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  AA->copyValue(From, To);
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(CallSite CS1, CallSite CS2) {
  // FIXME: we can do better.
  assert(AA && "AA didn't call InitializeAliasAnalysis in its run method!");
  return AA->getModRefInfo(CS1, CS2);
}


//===----------------------------------------------------------------------===//
// AliasAnalysis non-virtual helper method implementation
//===----------------------------------------------------------------------===//

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(LoadInst *L, Value *P, unsigned Size) {
  return alias(L->getOperand(0), TD->getTypeStoreSize(L->getType()),
               P, Size) ? Ref : NoModRef;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(StoreInst *S, Value *P, unsigned Size) {
  // If the stored address cannot alias the pointer in question, then the
  // pointer cannot be modified by the store.
  if (!alias(S->getOperand(1),
             TD->getTypeStoreSize(S->getOperand(0)->getType()), P, Size))
    return NoModRef;

  // If the pointer is a pointer to constant memory, then it could not have been
  // modified by this store.
  return pointsToConstantMemory(P) ? NoModRef : Mod;
}

AliasAnalysis::ModRefBehavior
AliasAnalysis::getModRefBehavior(CallSite CS,
                                 std::vector<PointerAccessInfo> *Info) {
  if (CS.doesNotAccessMemory())
    // Can't do better than this.
    return DoesNotAccessMemory;
  ModRefBehavior MRB = UnknownModRefBehavior;
  if (Function *F = CS.getCalledFunction())
    MRB = getModRefBehavior(F, CS, Info);
  if (MRB != DoesNotAccessMemory && CS.onlyReadsMemory())
    return OnlyReadsMemory;
  return MRB;
}

AliasAnalysis::ModRefBehavior
AliasAnalysis::getModRefBehavior(Function *F,
                                 std::vector<PointerAccessInfo> *Info) {
  if (F->doesNotAccessMemory())
    // Can't do better than this.
    return DoesNotAccessMemory;
  ModRefBehavior MRB = getModRefBehavior(F, CallSite(), Info);
  if (MRB != DoesNotAccessMemory && F->onlyReadsMemory())
    return OnlyReadsMemory;
  return MRB;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(CallSite CS, Value *P, unsigned Size) {
  ModRefResult Mask = ModRef;
  ModRefBehavior MRB = getModRefBehavior(CS);
  if (MRB == OnlyReadsMemory)
    Mask = Ref;
  else if (MRB == DoesNotAccessMemory)
    return NoModRef;

  if (!AA) return Mask;

  // If P points to a constant memory location, the call definitely could not
  // modify the memory location.
  if ((Mask & Mod) && AA->pointsToConstantMemory(P))
    Mask = ModRefResult(Mask & ~Mod);

  return ModRefResult(Mask & AA->getModRefInfo(CS, P, Size));
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
  TD = &P->getAnalysis<TargetData>();
  AA = &P->getAnalysis<AliasAnalysis>();
}

// getAnalysisUsage - All alias analysis implementations should invoke this
// directly (using AliasAnalysis::getAnalysisUsage(AU)) to make sure that
// TargetData is required by the pass.
void AliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetData>();            // All AA's need TargetData.
  AU.addRequired<AliasAnalysis>();         // All AA's chain
}

/// canBasicBlockModify - Return true if it is possible for execution of the
/// specified basic block to modify the value pointed to by Ptr.
///
bool AliasAnalysis::canBasicBlockModify(const BasicBlock &BB,
                                        const Value *Ptr, unsigned Size) {
  return canInstructionRangeModify(BB.front(), BB.back(), Ptr, Size);
}

/// canInstructionRangeModify - Return true if it is possible for the execution
/// of the specified instructions to modify the value pointed to by Ptr.  The
/// instructions to consider are all of the instructions in the range of [I1,I2]
/// INCLUSIVE.  I1 and I2 must be in the same basic block.
///
bool AliasAnalysis::canInstructionRangeModify(const Instruction &I1,
                                              const Instruction &I2,
                                              const Value *Ptr, unsigned Size) {
  assert(I1.getParent() == I2.getParent() &&
         "Instructions not in same basic block!");
  BasicBlock::iterator I = const_cast<Instruction*>(&I1);
  BasicBlock::iterator E = const_cast<Instruction*>(&I2);
  ++E;  // Convert from inclusive to exclusive range.

  for (; I != E; ++I) // Check every instruction in range
    if (getModRefInfo(I, const_cast<Value*>(Ptr), Size) & Mod)
      return true;
  return false;
}

/// isNoAliasCall - Return true if this pointer is returned by a noalias
/// function.
bool llvm::isNoAliasCall(const Value *V) {
  if (isa<CallInst>(V) || isa<InvokeInst>(V))
    return CallSite(const_cast<Instruction*>(cast<Instruction>(V)))
      .paramHasAttr(0, Attribute::NoAlias);
  return false;
}

/// isIdentifiedObject - Return true if this pointer refers to a distinct and
/// identifiable object.  This returns true for:
///    Global Variables and Functions
///    Allocas and Mallocs
///    ByVal and NoAlias Arguments
///    NoAlias returns
///
bool llvm::isIdentifiedObject(const Value *V) {
  if (isa<GlobalValue>(V) || isa<AllocationInst>(V) || isNoAliasCall(V))
    return true;
  if (const Argument *A = dyn_cast<Argument>(V))
    return A->hasNoAliasAttr() || A->hasByValAttr();
  return false;
}

// Because of the way .a files work, we must force the BasicAA implementation to
// be pulled in if the AliasAnalysis classes are pulled in.  Otherwise we run
// the risk of AliasAnalysis being used, but the default implementation not
// being linked into the tool that uses it.
DEFINING_FILE_FOR(AliasAnalysis)
