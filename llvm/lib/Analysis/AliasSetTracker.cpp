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
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/DataLayout.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

/// mergeSetIn - Merge the specified alias set into this alias set.
///
void AliasSet::mergeSetIn(AliasSet &AS, AliasSetTracker &AST) {
  assert(!AS.Forward && "Alias set is already forwarding!");
  assert(!Forward && "This set is a forwarding set!!");

  // Update the alias and access types of this set...
  AccessTy |= AS.AccessTy;
  AliasTy  |= AS.AliasTy;
  Volatile |= AS.Volatile;

  if (AliasTy == MustAlias) {
    // Check that these two merged sets really are must aliases.  Since both
    // used to be must-alias sets, we can just check any pointer from each set
    // for aliasing.
    AliasAnalysis &AA = AST.getAliasAnalysis();
    PointerRec *L = getSomePointer();
    PointerRec *R = AS.getSomePointer();

    // If the pointers are not a must-alias pair, this set becomes a may alias.
    if (AA.alias(AliasAnalysis::Location(L->getValue(),
                                         L->getSize(),
                                         L->getTBAAInfo()),
                 AliasAnalysis::Location(R->getValue(),
                                         R->getSize(),
                                         R->getTBAAInfo()))
        != AliasAnalysis::MustAlias)
      AliasTy = MayAlias;
  }

  if (UnknownInsts.empty()) {            // Merge call sites...
    if (!AS.UnknownInsts.empty())
      std::swap(UnknownInsts, AS.UnknownInsts);
  } else if (!AS.UnknownInsts.empty()) {
    UnknownInsts.insert(UnknownInsts.end(), AS.UnknownInsts.begin(), AS.UnknownInsts.end());
    AS.UnknownInsts.clear();
  }

  AS.Forward = this;  // Forward across AS now...
  addRef();           // AS is now pointing to us...

  // Merge the list of constituent pointers...
  if (AS.PtrList) {
    *PtrListEnd = AS.PtrList;
    AS.PtrList->setPrevInList(PtrListEnd);
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

void AliasSet::addPointer(AliasSetTracker &AST, PointerRec &Entry,
                          uint64_t Size, const MDNode *TBAAInfo,
                          bool KnownMustAlias) {
  assert(!Entry.hasAliasSet() && "Entry already in set!");

  // Check to see if we have to downgrade to _may_ alias.
  if (isMustAlias() && !KnownMustAlias)
    if (PointerRec *P = getSomePointer()) {
      AliasAnalysis &AA = AST.getAliasAnalysis();
      AliasAnalysis::AliasResult Result =
        AA.alias(AliasAnalysis::Location(P->getValue(), P->getSize(),
                                         P->getTBAAInfo()),
                 AliasAnalysis::Location(Entry.getValue(), Size, TBAAInfo));
      if (Result != AliasAnalysis::MustAlias)
        AliasTy = MayAlias;
      else                  // First entry of must alias must have maximum size!
        P->updateSizeAndTBAAInfo(Size, TBAAInfo);
      assert(Result != AliasAnalysis::NoAlias && "Cannot be part of must set!");
    }

  Entry.setAliasSet(this);
  Entry.updateSizeAndTBAAInfo(Size, TBAAInfo);

  // Add it to the end of the list...
  assert(*PtrListEnd == 0 && "End of list is not null?");
  *PtrListEnd = &Entry;
  PtrListEnd = Entry.setPrevInList(PtrListEnd);
  assert(*PtrListEnd == 0 && "End of list is not null?");
  addRef();               // Entry points to alias set.
}

void AliasSet::addUnknownInst(Instruction *I, AliasAnalysis &AA) {
  UnknownInsts.push_back(I);

  if (!I->mayWriteToMemory()) {
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
bool AliasSet::aliasesPointer(const Value *Ptr, uint64_t Size,
                              const MDNode *TBAAInfo,
                              AliasAnalysis &AA) const {
  if (AliasTy == MustAlias) {
    assert(UnknownInsts.empty() && "Illegal must alias set!");

    // If this is a set of MustAliases, only check to see if the pointer aliases
    // SOME value in the set.
    PointerRec *SomePtr = getSomePointer();
    assert(SomePtr && "Empty must-alias set??");
    return AA.alias(AliasAnalysis::Location(SomePtr->getValue(),
                                            SomePtr->getSize(),
                                            SomePtr->getTBAAInfo()),
                    AliasAnalysis::Location(Ptr, Size, TBAAInfo));
  }

  // If this is a may-alias set, we have to check all of the pointers in the set
  // to be sure it doesn't alias the set...
  for (iterator I = begin(), E = end(); I != E; ++I)
    if (AA.alias(AliasAnalysis::Location(Ptr, Size, TBAAInfo),
                 AliasAnalysis::Location(I.getPointer(), I.getSize(),
                                         I.getTBAAInfo())))
      return true;

  // Check the unknown instructions...
  if (!UnknownInsts.empty()) {
    for (unsigned i = 0, e = UnknownInsts.size(); i != e; ++i)
      if (AA.getModRefInfo(UnknownInsts[i],
                           AliasAnalysis::Location(Ptr, Size, TBAAInfo)) !=
            AliasAnalysis::NoModRef)
        return true;
  }

  return false;
}

bool AliasSet::aliasesUnknownInst(Instruction *Inst, AliasAnalysis &AA) const {
  if (!Inst->mayReadOrWriteMemory())
    return false;

  for (unsigned i = 0, e = UnknownInsts.size(); i != e; ++i) {
    CallSite C1 = getUnknownInst(i), C2 = Inst;
    if (!C1 || !C2 ||
        AA.getModRefInfo(C1, C2) != AliasAnalysis::NoModRef ||
        AA.getModRefInfo(C2, C1) != AliasAnalysis::NoModRef)
      return true;
  }

  for (iterator I = begin(), E = end(); I != E; ++I)
    if (AA.getModRefInfo(Inst, AliasAnalysis::Location(I.getPointer(),
                                                       I.getSize(),
                                                       I.getTBAAInfo())) !=
           AliasAnalysis::NoModRef)
      return true;

  return false;
}

void AliasSetTracker::clear() {
  // Delete all the PointerRec entries.
  for (PointerMapType::iterator I = PointerMap.begin(), E = PointerMap.end();
       I != E; ++I)
    I->second->eraseFromList();
  
  PointerMap.clear();
  
  // The alias sets should all be clear now.
  AliasSets.clear();
}


/// findAliasSetForPointer - Given a pointer, find the one alias set to put the
/// instruction referring to the pointer into.  If there are multiple alias sets
/// that may alias the pointer, merge them together and return the unified set.
///
AliasSet *AliasSetTracker::findAliasSetForPointer(const Value *Ptr,
                                                  uint64_t Size,
                                                  const MDNode *TBAAInfo) {
  AliasSet *FoundSet = 0;
  for (iterator I = begin(), E = end(); I != E; ++I) {
    if (I->Forward || !I->aliasesPointer(Ptr, Size, TBAAInfo, AA)) continue;
    
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
bool AliasSetTracker::containsPointer(Value *Ptr, uint64_t Size,
                                      const MDNode *TBAAInfo) const {
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    if (!I->Forward && I->aliasesPointer(Ptr, Size, TBAAInfo, AA))
      return true;
  return false;
}



AliasSet *AliasSetTracker::findAliasSetForUnknownInst(Instruction *Inst) {
  AliasSet *FoundSet = 0;
  for (iterator I = begin(), E = end(); I != E; ++I) {
    if (I->Forward || !I->aliasesUnknownInst(Inst, AA))
      continue;
    
    if (FoundSet == 0)        // If this is the first alias set ptr can go into.
      FoundSet = I;           // Remember it.
    else if (!I->Forward)     // Otherwise, we must merge the sets.
      FoundSet->mergeSetIn(*I, *this);     // Merge in contents.
  }
  return FoundSet;
}




/// getAliasSetForPointer - Return the alias set that the specified pointer
/// lives in.
AliasSet &AliasSetTracker::getAliasSetForPointer(Value *Pointer, uint64_t Size,
                                                 const MDNode *TBAAInfo,
                                                 bool *New) {
  AliasSet::PointerRec &Entry = getEntryFor(Pointer);

  // Check to see if the pointer is already known.
  if (Entry.hasAliasSet()) {
    Entry.updateSizeAndTBAAInfo(Size, TBAAInfo);
    // Return the set!
    return *Entry.getAliasSet(*this)->getForwardedTarget(*this);
  }
  
  if (AliasSet *AS = findAliasSetForPointer(Pointer, Size, TBAAInfo)) {
    // Add it to the alias set it aliases.
    AS->addPointer(*this, Entry, Size, TBAAInfo);
    return *AS;
  }
  
  if (New) *New = true;
  // Otherwise create a new alias set to hold the loaded pointer.
  AliasSets.push_back(new AliasSet());
  AliasSets.back().addPointer(*this, Entry, Size, TBAAInfo);
  return AliasSets.back();
}

bool AliasSetTracker::add(Value *Ptr, uint64_t Size, const MDNode *TBAAInfo) {
  bool NewPtr;
  addPointer(Ptr, Size, TBAAInfo, AliasSet::NoModRef, NewPtr);
  return NewPtr;
}


bool AliasSetTracker::add(LoadInst *LI) {
  if (LI->getOrdering() > Monotonic) return addUnknown(LI);
  AliasSet::AccessType ATy = AliasSet::Refs;
  if (!LI->isUnordered()) ATy = AliasSet::ModRef;
  bool NewPtr;
  AliasSet &AS = addPointer(LI->getOperand(0),
                            AA.getTypeStoreSize(LI->getType()),
                            LI->getMetadata(LLVMContext::MD_tbaa),
                            ATy, NewPtr);
  if (LI->isVolatile()) AS.setVolatile();
  return NewPtr;
}

bool AliasSetTracker::add(StoreInst *SI) {
  if (SI->getOrdering() > Monotonic) return addUnknown(SI);
  AliasSet::AccessType ATy = AliasSet::Mods;
  if (!SI->isUnordered()) ATy = AliasSet::ModRef;
  bool NewPtr;
  Value *Val = SI->getOperand(0);
  AliasSet &AS = addPointer(SI->getOperand(1),
                            AA.getTypeStoreSize(Val->getType()),
                            SI->getMetadata(LLVMContext::MD_tbaa),
                            ATy, NewPtr);
  if (SI->isVolatile()) AS.setVolatile();
  return NewPtr;
}

bool AliasSetTracker::add(VAArgInst *VAAI) {
  bool NewPtr;
  addPointer(VAAI->getOperand(0), AliasAnalysis::UnknownSize, 
             VAAI->getMetadata(LLVMContext::MD_tbaa),
             AliasSet::ModRef, NewPtr);
  return NewPtr;
}


bool AliasSetTracker::addUnknown(Instruction *Inst) {
  if (isa<DbgInfoIntrinsic>(Inst)) 
    return true; // Ignore DbgInfo Intrinsics.
  if (!Inst->mayReadOrWriteMemory())
    return true; // doesn't alias anything

  AliasSet *AS = findAliasSetForUnknownInst(Inst);
  if (AS) {
    AS->addUnknownInst(Inst, AA);
    return false;
  }
  AliasSets.push_back(new AliasSet());
  AS = &AliasSets.back();
  AS->addUnknownInst(Inst, AA);
  return true;
}

bool AliasSetTracker::add(Instruction *I) {
  // Dispatch to one of the other add methods.
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return add(LI);
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return add(SI);
  if (VAArgInst *VAAI = dyn_cast<VAArgInst>(I))
    return add(VAAI);
  return addUnknown(I);
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
  for (const_iterator I = AST.begin(), E = AST.end(); I != E; ++I) {
    if (I->Forward) continue;   // Ignore forwarding alias sets
    
    AliasSet &AS = const_cast<AliasSet&>(*I);

    // If there are any call sites in the alias set, add them to this AST.
    for (unsigned i = 0, e = AS.UnknownInsts.size(); i != e; ++i)
      add(AS.UnknownInsts[i]);

    // Loop over all of the pointers in this alias set.
    bool X;
    for (AliasSet::iterator ASI = AS.begin(), E = AS.end(); ASI != E; ++ASI) {
      AliasSet &NewAS = addPointer(ASI.getPointer(), ASI.getSize(),
                                   ASI.getTBAAInfo(),
                                   (AliasSet::AccessType)AS.AccessTy, X);
      if (AS.isVolatile()) NewAS.setVolatile();
    }
  }
}

/// remove - Remove the specified (potentially non-empty) alias set from the
/// tracker.
void AliasSetTracker::remove(AliasSet &AS) {
  // Drop all call sites.
  AS.UnknownInsts.clear();
  
  // Clear the alias set.
  unsigned NumRefs = 0;
  while (!AS.empty()) {
    AliasSet::PointerRec *P = AS.PtrList;

    Value *ValToRemove = P->getValue();
    
    // Unlink and delete entry from the list of values.
    P->eraseFromList();
    
    // Remember how many references need to be dropped.
    ++NumRefs;

    // Finally, remove the entry.
    PointerMap.erase(ValToRemove);
  }
  
  // Stop using the alias set, removing it.
  AS.RefCount -= NumRefs;
  if (AS.RefCount == 0)
    AS.removeFromTracker(*this);
}

bool
AliasSetTracker::remove(Value *Ptr, uint64_t Size, const MDNode *TBAAInfo) {
  AliasSet *AS = findAliasSetForPointer(Ptr, Size, TBAAInfo);
  if (!AS) return false;
  remove(*AS);
  return true;
}

bool AliasSetTracker::remove(LoadInst *LI) {
  uint64_t Size = AA.getTypeStoreSize(LI->getType());
  const MDNode *TBAAInfo = LI->getMetadata(LLVMContext::MD_tbaa);
  AliasSet *AS = findAliasSetForPointer(LI->getOperand(0), Size, TBAAInfo);
  if (!AS) return false;
  remove(*AS);
  return true;
}

bool AliasSetTracker::remove(StoreInst *SI) {
  uint64_t Size = AA.getTypeStoreSize(SI->getOperand(0)->getType());
  const MDNode *TBAAInfo = SI->getMetadata(LLVMContext::MD_tbaa);
  AliasSet *AS = findAliasSetForPointer(SI->getOperand(1), Size, TBAAInfo);
  if (!AS) return false;
  remove(*AS);
  return true;
}

bool AliasSetTracker::remove(VAArgInst *VAAI) {
  AliasSet *AS = findAliasSetForPointer(VAAI->getOperand(0),
                                        AliasAnalysis::UnknownSize,
                                        VAAI->getMetadata(LLVMContext::MD_tbaa));
  if (!AS) return false;
  remove(*AS);
  return true;
}

bool AliasSetTracker::removeUnknown(Instruction *I) {
  if (!I->mayReadOrWriteMemory())
    return false; // doesn't alias anything

  AliasSet *AS = findAliasSetForUnknownInst(I);
  if (!AS) return false;
  remove(*AS);
  return true;
}

bool AliasSetTracker::remove(Instruction *I) {
  // Dispatch to one of the other remove methods...
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return remove(LI);
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return remove(SI);
  if (VAArgInst *VAAI = dyn_cast<VAArgInst>(I))
    return remove(VAAI);
  return removeUnknown(I);
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
  // AliasSet (if present).
  if (Instruction *Inst = dyn_cast<Instruction>(PtrVal)) {
    if (Inst->mayReadOrWriteMemory()) {
      // Scan all the alias sets to see if this call site is contained.
      for (iterator I = begin(), E = end(); I != E; ++I) {
        if (I->Forward) continue;
        
        I->removeUnknownInst(Inst);
      }
    }
  }

  // First, look up the PointerRec for this pointer.
  PointerMapType::iterator I = PointerMap.find_as(PtrVal);
  if (I == PointerMap.end()) return;  // Noop

  // If we found one, remove the pointer from the alias set it is in.
  AliasSet::PointerRec *PtrValEnt = I->second;
  AliasSet *AS = PtrValEnt->getAliasSet(*this);

  // Unlink and delete from the list of values.
  PtrValEnt->eraseFromList();
  
  // Stop using the alias set.
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
  PointerMapType::iterator I = PointerMap.find_as(From);
  if (I == PointerMap.end())
    return;  // Noop
  assert(I->second->hasAliasSet() && "Dead entry?");

  AliasSet::PointerRec &Entry = getEntryFor(To);
  if (Entry.hasAliasSet()) return;    // Already in the tracker!

  // Add it to the alias set it aliases...
  I = PointerMap.find_as(From);
  AliasSet *AS = I->second->getAliasSet(*this);
  AS->addPointer(*this, Entry, I->second->getSize(),
                 I->second->getTBAAInfo(),
                 true);
}



//===----------------------------------------------------------------------===//
//               AliasSet/AliasSetTracker Printing Support
//===----------------------------------------------------------------------===//

void AliasSet::print(raw_ostream &OS) const {
  OS << "  AliasSet[" << (const void*)this << ", " << RefCount << "] ";
  OS << (AliasTy == MustAlias ? "must" : "may") << " alias, ";
  switch (AccessTy) {
  case NoModRef: OS << "No access "; break;
  case Refs    : OS << "Ref       "; break;
  case Mods    : OS << "Mod       "; break;
  case ModRef  : OS << "Mod/Ref   "; break;
  default: llvm_unreachable("Bad value for AccessTy!");
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
  if (!UnknownInsts.empty()) {
    OS << "\n    " << UnknownInsts.size() << " Unknown instructions: ";
    for (unsigned i = 0, e = UnknownInsts.size(); i != e; ++i) {
      if (i) OS << ", ";
      WriteAsOperand(OS, UnknownInsts[i]);
    }
  }
  OS << "\n";
}

void AliasSetTracker::print(raw_ostream &OS) const {
  OS << "Alias Set Tracker: " << AliasSets.size() << " alias sets for "
     << PointerMap.size() << " pointer values.\n";
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    I->print(OS);
  OS << "\n";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void AliasSet::dump() const { print(dbgs()); }
void AliasSetTracker::dump() const { print(dbgs()); }
#endif

//===----------------------------------------------------------------------===//
//                     ASTCallbackVH Class Implementation
//===----------------------------------------------------------------------===//

void AliasSetTracker::ASTCallbackVH::deleted() {
  assert(AST && "ASTCallbackVH called with a null AliasSetTracker!");
  AST->deleteValue(getValPtr());
  // this now dangles!
}

void AliasSetTracker::ASTCallbackVH::allUsesReplacedWith(Value *V) {
  AST->copyValue(getValPtr(), V);
}

AliasSetTracker::ASTCallbackVH::ASTCallbackVH(Value *V, AliasSetTracker *ast)
  : CallbackVH(V), AST(ast) {}

AliasSetTracker::ASTCallbackVH &
AliasSetTracker::ASTCallbackVH::operator=(Value *V) {
  return *this = ASTCallbackVH(V, AST);
}

//===----------------------------------------------------------------------===//
//                            AliasSetPrinter Pass
//===----------------------------------------------------------------------===//

namespace {
  class AliasSetPrinter : public FunctionPass {
    AliasSetTracker *Tracker;
  public:
    static char ID; // Pass identification, replacement for typeid
    AliasSetPrinter() : FunctionPass(ID) {
      initializeAliasSetPrinterPass(*PassRegistry::getPassRegistry());
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<AliasAnalysis>();
    }

    virtual bool runOnFunction(Function &F) {
      Tracker = new AliasSetTracker(getAnalysis<AliasAnalysis>());

      for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
        Tracker->add(&*I);
      Tracker->print(errs());
      delete Tracker;
      return false;
    }
  };
}

char AliasSetPrinter::ID = 0;
INITIALIZE_PASS_BEGIN(AliasSetPrinter, "print-alias-sets",
                "Alias Set Printer", false, true)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(AliasSetPrinter, "print-alias-sets",
                "Alias Set Printer", false, true)
