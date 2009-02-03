//===- AliasSetTracker.cpp - Alias Sets Tracker implementation-------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AliasSetTracker and AliasSet classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/Streams.h"
using namespace llvm;

/// mergeSetIn - Merge the specified alias set into this alias set.
///
void AliasSet::mergeSetIn(AliasSet &AS, AliasSetTracker &AST) {
  assert(!AS.Forward && "Alias set is already forwarding!");
  assert(!Forward && "This set is a forwarding set!!");

  // Update the alias and access types of this set...
  AccessTy |= AS.AccessTy;
  AliasTy  |= AS.AliasTy;

  if (AliasTy == MustAlias) {
    // Check that these two merged sets really are must aliases.  Since both
    // used to be must-alias sets, we can just check any pointer from each set
    // for aliasing.
    AliasAnalysis &AA = AST.getAliasAnalysis();
    HashNodePair *L = getSomePointer();
    HashNodePair *R = AS.getSomePointer();

    // If the pointers are not a must-alias pair, this set becomes a may alias.
    if (AA.alias(L->first, L->second.getSize(), R->first, R->second.getSize())
        != AliasAnalysis::MustAlias)
      AliasTy = MayAlias;
  }

  if (CallSites.empty()) {            // Merge call sites...
    if (!AS.CallSites.empty())
      std::swap(CallSites, AS.CallSites);
  } else if (!AS.CallSites.empty()) {
    CallSites.insert(CallSites.end(), AS.CallSites.begin(), AS.CallSites.end());
    AS.CallSites.clear();
  }

  AS.Forward = this;  // Forward across AS now...
  addRef();           // AS is now pointing to us...

  // Merge the list of constituent pointers...
  if (AS.PtrList) {
    *PtrListEnd = AS.PtrList;
    AS.PtrList->second.setPrevInList(PtrListEnd);
    PtrListEnd = AS.PtrListEnd;

    AS.PtrList = 0;
    AS.PtrListEnd = &AS.PtrList;
    assert(*AS.PtrListEnd == 0 && "End of list is not null?");
  }
}

void AliasSetTracker::removeAliasSet(AliasSet *AS) {
  if (AliasSet *Fwd = AS->Forward) {
    Fwd->dropRef(*this);
    AS->Forward = 0;
  }
  AliasSets.erase(AS);
}

void AliasSet::removeFromTracker(AliasSetTracker &AST) {
  assert(RefCount == 0 && "Cannot remove non-dead alias set from tracker!");
  AST.removeAliasSet(this);
}

void AliasSet::addPointer(AliasSetTracker &AST, HashNodePair &Entry,
                          unsigned Size, bool KnownMustAlias) {
  assert(!Entry.second.hasAliasSet() && "Entry already in set!");

  // Check to see if we have to downgrade to _may_ alias.
  if (isMustAlias() && !KnownMustAlias)
    if (HashNodePair *P = getSomePointer()) {
      AliasAnalysis &AA = AST.getAliasAnalysis();
      AliasAnalysis::AliasResult Result =
        AA.alias(P->first, P->second.getSize(), Entry.first, Size);
      if (Result == AliasAnalysis::MayAlias)
        AliasTy = MayAlias;
      else                  // First entry of must alias must have maximum size!
        P->second.updateSize(Size);
      assert(Result != AliasAnalysis::NoAlias && "Cannot be part of must set!");
    }

  Entry.second.setAliasSet(this);
  Entry.second.updateSize(Size);

  // Add it to the end of the list...
  assert(*PtrListEnd == 0 && "End of list is not null?");
  *PtrListEnd = &Entry;
  PtrListEnd = Entry.second.setPrevInList(PtrListEnd);
  assert(*PtrListEnd == 0 && "End of list is not null?");
  addRef();               // Entry points to alias set...
}

void AliasSet::addCallSite(CallSite CS, AliasAnalysis &AA) {
  CallSites.push_back(CS);

  AliasAnalysis::ModRefBehavior Behavior = AA.getModRefBehavior(CS);
  if (Behavior == AliasAnalysis::DoesNotAccessMemory)
    return;
  else if (Behavior == AliasAnalysis::OnlyReadsMemory) {
    AliasTy = MayAlias;
    AccessTy |= Refs;
    return;
  }

  // FIXME: This should use mod/ref information to make this not suck so bad
  AliasTy = MayAlias;
  AccessTy = ModRef;
}

/// aliasesPointer - Return true if the specified pointer "may" (or must)
/// alias one of the members in the set.
///
bool AliasSet::aliasesPointer(const Value *Ptr, unsigned Size,
                              AliasAnalysis &AA) const {
  if (AliasTy == MustAlias) {
    assert(CallSites.empty() && "Illegal must alias set!");

    // If this is a set of MustAliases, only check to see if the pointer aliases
    // SOME value in the set...
    HashNodePair *SomePtr = getSomePointer();
    assert(SomePtr && "Empty must-alias set??");
    return AA.alias(SomePtr->first, SomePtr->second.getSize(), Ptr, Size);
  }

  // If this is a may-alias set, we have to check all of the pointers in the set
  // to be sure it doesn't alias the set...
  for (iterator I = begin(), E = end(); I != E; ++I)
    if (AA.alias(Ptr, Size, I.getPointer(), I.getSize()))
      return true;

  // Check the call sites list and invoke list...
  if (!CallSites.empty()) {
    if (AA.hasNoModRefInfoForCalls())
      return true;

    for (unsigned i = 0, e = CallSites.size(); i != e; ++i)
      if (AA.getModRefInfo(CallSites[i], const_cast<Value*>(Ptr), Size)
                   != AliasAnalysis::NoModRef)
        return true;
  }

  return false;
}

bool AliasSet::aliasesCallSite(CallSite CS, AliasAnalysis &AA) const {
  if (AA.doesNotAccessMemory(CS))
    return false;

  if (AA.hasNoModRefInfoForCalls())
    return true;

  for (unsigned i = 0, e = CallSites.size(); i != e; ++i)
    if (AA.getModRefInfo(CallSites[i], CS) != AliasAnalysis::NoModRef ||
        AA.getModRefInfo(CS, CallSites[i]) != AliasAnalysis::NoModRef)
      return true;

  for (iterator I = begin(), E = end(); I != E; ++I)
    if (AA.getModRefInfo(CS, I.getPointer(), I.getSize()) !=
           AliasAnalysis::NoModRef)
      return true;

  return false;
}


/// findAliasSetForPointer - Given a pointer, find the one alias set to put the
/// instruction referring to the pointer into.  If there are multiple alias sets
/// that may alias the pointer, merge them together and return the unified set.
///
AliasSet *AliasSetTracker::findAliasSetForPointer(const Value *Ptr,
                                                  unsigned Size) {
  AliasSet *FoundSet = 0;
  for (iterator I = begin(), E = end(); I != E; ++I)
    if (!I->Forward && I->aliasesPointer(Ptr, Size, AA)) {
      if (FoundSet == 0) {  // If this is the first alias set ptr can go into.
        FoundSet = I;       // Remember it.
      } else {              // Otherwise, we must merge the sets.
        FoundSet->mergeSetIn(*I, *this);     // Merge in contents.
      }
    }

  return FoundSet;
}

/// containsPointer - Return true if the specified location is represented by
/// this alias set, false otherwise.  This does not modify the AST object or
/// alias sets.
bool AliasSetTracker::containsPointer(Value *Ptr, unsigned Size) const {
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    if (!I->Forward && I->aliasesPointer(Ptr, Size, AA))
      return true;
  return false;
}



AliasSet *AliasSetTracker::findAliasSetForCallSite(CallSite CS) {
  AliasSet *FoundSet = 0;
  for (iterator I = begin(), E = end(); I != E; ++I)
    if (!I->Forward && I->aliasesCallSite(CS, AA)) {
      if (FoundSet == 0) {  // If this is the first alias set ptr can go into.
        FoundSet = I;       // Remember it.
      } else if (!I->Forward) {     // Otherwise, we must merge the sets.
        FoundSet->mergeSetIn(*I, *this);     // Merge in contents.
      }
    }

  return FoundSet;
}




/// getAliasSetForPointer - Return the alias set that the specified pointer
/// lives in...
AliasSet &AliasSetTracker::getAliasSetForPointer(Value *Pointer, unsigned Size,
                                                 bool *New) {
  AliasSet::HashNodePair &Entry = getEntryFor(Pointer);

  // Check to see if the pointer is already known...
  if (Entry.second.hasAliasSet()) {
    Entry.second.updateSize(Size);
    // Return the set!
    return *Entry.second.getAliasSet(*this)->getForwardedTarget(*this);
  } else if (AliasSet *AS = findAliasSetForPointer(Pointer, Size)) {
    // Add it to the alias set it aliases...
    AS->addPointer(*this, Entry, Size);
    return *AS;
  } else {
    if (New) *New = true;
    // Otherwise create a new alias set to hold the loaded pointer...
    AliasSets.push_back(new AliasSet());
    AliasSets.back().addPointer(*this, Entry, Size);
    return AliasSets.back();
  }
}

bool AliasSetTracker::add(Value *Ptr, unsigned Size) {
  bool NewPtr;
  addPointer(Ptr, Size, AliasSet::NoModRef, NewPtr);
  return NewPtr;
}


bool AliasSetTracker::add(LoadInst *LI) {
  bool NewPtr;
  AliasSet &AS = addPointer(LI->getOperand(0),
                            AA.getTargetData().getTypeStoreSize(LI->getType()),
                            AliasSet::Refs, NewPtr);
  if (LI->isVolatile()) AS.setVolatile();
  return NewPtr;
}

bool AliasSetTracker::add(StoreInst *SI) {
  bool NewPtr;
  Value *Val = SI->getOperand(0);
  AliasSet &AS = addPointer(SI->getOperand(1),
                            AA.getTargetData().getTypeStoreSize(Val->getType()),
                            AliasSet::Mods, NewPtr);
  if (SI->isVolatile()) AS.setVolatile();
  return NewPtr;
}

bool AliasSetTracker::add(FreeInst *FI) {
  bool NewPtr;
  addPointer(FI->getOperand(0), ~0, AliasSet::Mods, NewPtr);
  return NewPtr;
}

bool AliasSetTracker::add(VAArgInst *VAAI) {
  bool NewPtr;
  addPointer(VAAI->getOperand(0), ~0, AliasSet::ModRef, NewPtr);
  return NewPtr;
}


bool AliasSetTracker::add(CallSite CS) {
  if (AA.doesNotAccessMemory(CS))
    return true; // doesn't alias anything

  AliasSet *AS = findAliasSetForCallSite(CS);
  if (!AS) {
    AliasSets.push_back(new AliasSet());
    AS = &AliasSets.back();
    AS->addCallSite(CS, AA);
    return true;
  } else {
    AS->addCallSite(CS, AA);
    return false;
  }
}

bool AliasSetTracker::add(Instruction *I) {
  // Dispatch to one of the other add methods...
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return add(LI);
  else if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return add(SI);
  else if (CallInst *CI = dyn_cast<CallInst>(I))
    return add(CI);
  else if (InvokeInst *II = dyn_cast<InvokeInst>(I))
    return add(II);
  else if (FreeInst *FI = dyn_cast<FreeInst>(I))
    return add(FI);
  else if (VAArgInst *VAAI = dyn_cast<VAArgInst>(I))
    return add(VAAI);
  return true;
}

void AliasSetTracker::add(BasicBlock &BB) {
  for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I)
    add(I);
}

void AliasSetTracker::add(const AliasSetTracker &AST) {
  assert(&AA == &AST.AA &&
         "Merging AliasSetTracker objects with different Alias Analyses!");

  // Loop over all of the alias sets in AST, adding the pointers contained
  // therein into the current alias sets.  This can cause alias sets to be
  // merged together in the current AST.
  for (const_iterator I = AST.begin(), E = AST.end(); I != E; ++I)
    if (!I->Forward) {   // Ignore forwarding alias sets
      AliasSet &AS = const_cast<AliasSet&>(*I);

      // If there are any call sites in the alias set, add them to this AST.
      for (unsigned i = 0, e = AS.CallSites.size(); i != e; ++i)
        add(AS.CallSites[i]);

      // Loop over all of the pointers in this alias set...
      AliasSet::iterator I = AS.begin(), E = AS.end();
      bool X;
      for (; I != E; ++I) {
        AliasSet &NewAS = addPointer(I.getPointer(), I.getSize(),
                                     (AliasSet::AccessType)AS.AccessTy, X);
        if (AS.isVolatile()) NewAS.setVolatile();
      }
    }
}

/// remove - Remove the specified (potentially non-empty) alias set from the
/// tracker.
void AliasSetTracker::remove(AliasSet &AS) {
  // Drop all call sites.
  AS.CallSites.clear();
  
  // Clear the alias set.
  unsigned NumRefs = 0;
  while (!AS.empty()) {
    AliasSet::HashNodePair *P = AS.PtrList;
    
    // Unlink from the list of values.
    P->second.removeFromList();
    
    // Remember how many references need to be dropped.
    ++NumRefs;

    // Finally, remove the entry.
    Value *Remove = P->first;   // Take a copy because it is invalid to pass
    PointerMap.erase(Remove);   // a reference to the data being erased.
  }
  
  // Stop using the alias set, removing it.
  AS.RefCount -= NumRefs;
  if (AS.RefCount == 0)
    AS.removeFromTracker(*this);
}

bool AliasSetTracker::remove(Value *Ptr, unsigned Size) {
  AliasSet *AS = findAliasSetForPointer(Ptr, Size);
  if (!AS) return false;
  remove(*AS);
  return true;
}

bool AliasSetTracker::remove(LoadInst *LI) {
  unsigned Size = AA.getTargetData().getTypeStoreSize(LI->getType());
  AliasSet *AS = findAliasSetForPointer(LI->getOperand(0), Size);
  if (!AS) return false;
  remove(*AS);
  return true;
}

bool AliasSetTracker::remove(StoreInst *SI) {
  unsigned Size =
    AA.getTargetData().getTypeStoreSize(SI->getOperand(0)->getType());
  AliasSet *AS = findAliasSetForPointer(SI->getOperand(1), Size);
  if (!AS) return false;
  remove(*AS);
  return true;
}

bool AliasSetTracker::remove(FreeInst *FI) {
  AliasSet *AS = findAliasSetForPointer(FI->getOperand(0), ~0);
  if (!AS) return false;
  remove(*AS);
  return true;
}

bool AliasSetTracker::remove(VAArgInst *VAAI) {
  AliasSet *AS = findAliasSetForPointer(VAAI->getOperand(0), ~0);
  if (!AS) return false;
  remove(*AS);
  return true;
}

bool AliasSetTracker::remove(CallSite CS) {
  if (AA.doesNotAccessMemory(CS))
    return false; // doesn't alias anything

  AliasSet *AS = findAliasSetForCallSite(CS);
  if (!AS) return false;
  remove(*AS);
  return true;
}

bool AliasSetTracker::remove(Instruction *I) {
  // Dispatch to one of the other remove methods...
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return remove(LI);
  else if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return remove(SI);
  else if (CallInst *CI = dyn_cast<CallInst>(I))
    return remove(CI);
  else if (FreeInst *FI = dyn_cast<FreeInst>(I))
    return remove(FI);
  else if (VAArgInst *VAAI = dyn_cast<VAArgInst>(I))
    return remove(VAAI);
  return true;
}


// deleteValue method - This method is used to remove a pointer value from the
// AliasSetTracker entirely.  It should be used when an instruction is deleted
// from the program to update the AST.  If you don't use this, you would have
// dangling pointers to deleted instructions.
//
void AliasSetTracker::deleteValue(Value *PtrVal) {
  // Notify the alias analysis implementation that this value is gone.
  AA.deleteValue(PtrVal);

  // If this is a call instruction, remove the callsite from the appropriate
  // AliasSet.
  CallSite CS = CallSite::get(PtrVal);
  if (CS.getInstruction())
    if (!AA.doesNotAccessMemory(CS))
      if (AliasSet *AS = findAliasSetForCallSite(CS))
        AS->removeCallSite(CS);

  // First, look up the PointerRec for this pointer.
  hash_map<Value*, AliasSet::PointerRec>::iterator I = PointerMap.find(PtrVal);
  if (I == PointerMap.end()) return;  // Noop

  // If we found one, remove the pointer from the alias set it is in.
  AliasSet::HashNodePair &PtrValEnt = *I;
  AliasSet *AS = PtrValEnt.second.getAliasSet(*this);

  // Unlink from the list of values...
  PtrValEnt.second.removeFromList();
  // Stop using the alias set
  AS->dropRef(*this);
  PointerMap.erase(I);
}

// copyValue - This method should be used whenever a preexisting value in the
// program is copied or cloned, introducing a new value.  Note that it is ok for
// clients that use this method to introduce the same value multiple times: if
// the tracker already knows about a value, it will ignore the request.
//
void AliasSetTracker::copyValue(Value *From, Value *To) {
  // Notify the alias analysis implementation that this value is copied.
  AA.copyValue(From, To);

  // First, look up the PointerRec for this pointer.
  hash_map<Value*, AliasSet::PointerRec>::iterator I = PointerMap.find(From);
  if (I == PointerMap.end() || !I->second.hasAliasSet())
    return;  // Noop

  AliasSet::HashNodePair &Entry = getEntryFor(To);
  if (Entry.second.hasAliasSet()) return;    // Already in the tracker!

  // Add it to the alias set it aliases...
  AliasSet *AS = I->second.getAliasSet(*this);
  AS->addPointer(*this, Entry, I->second.getSize(), true);
}



//===----------------------------------------------------------------------===//
//               AliasSet/AliasSetTracker Printing Support
//===----------------------------------------------------------------------===//

void AliasSet::print(std::ostream &OS) const {
  OS << "  AliasSet[" << (void*)this << "," << RefCount << "] ";
  OS << (AliasTy == MustAlias ? "must" : "may") << " alias, ";
  switch (AccessTy) {
  case NoModRef: OS << "No access "; break;
  case Refs    : OS << "Ref       "; break;
  case Mods    : OS << "Mod       "; break;
  case ModRef  : OS << "Mod/Ref   "; break;
  default: assert(0 && "Bad value for AccessTy!");
  }
  if (isVolatile()) OS << "[volatile] ";
  if (Forward)
    OS << " forwarding to " << (void*)Forward;


  if (!empty()) {
    OS << "Pointers: ";
    for (iterator I = begin(), E = end(); I != E; ++I) {
      if (I != begin()) OS << ", ";
      WriteAsOperand(OS << "(", I.getPointer());
      OS << ", " << I.getSize() << ")";
    }
  }
  if (!CallSites.empty()) {
    OS << "\n    " << CallSites.size() << " Call Sites: ";
    for (unsigned i = 0, e = CallSites.size(); i != e; ++i) {
      if (i) OS << ", ";
      WriteAsOperand(OS, CallSites[i].getCalledValue());
    }
  }
  OS << "\n";
}

void AliasSetTracker::print(std::ostream &OS) const {
  OS << "Alias Set Tracker: " << AliasSets.size() << " alias sets for "
     << PointerMap.size() << " pointer values.\n";
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    I->print(OS);
  OS << "\n";
}

void AliasSet::dump() const { print (cerr); }
void AliasSetTracker::dump() const { print(cerr); }

//===----------------------------------------------------------------------===//
//                            AliasSetPrinter Pass
//===----------------------------------------------------------------------===//

namespace {
  class VISIBILITY_HIDDEN AliasSetPrinter : public FunctionPass {
    AliasSetTracker *Tracker;
  public:
    static char ID; // Pass identification, replacement for typeid
    AliasSetPrinter() : FunctionPass(&ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<AliasAnalysis>();
    }

    virtual bool runOnFunction(Function &F) {
      Tracker = new AliasSetTracker(getAnalysis<AliasAnalysis>());

      for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
        Tracker->add(&*I);
      Tracker->print(cerr);
      delete Tracker;
      return false;
    }
  };
}

char AliasSetPrinter::ID = 0;
static RegisterPass<AliasSetPrinter>
X("print-alias-sets", "Alias Set Printer", false, true);
