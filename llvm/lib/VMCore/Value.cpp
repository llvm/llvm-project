//===-- Value.cpp - Implement the Value class -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Value, ValueHandle, and User classes.
//
//===----------------------------------------------------------------------===//

#include "LLVMContextImpl.h"
#include "llvm/Constant.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InstrTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Operator.h"
#include "llvm/Module.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LeakDetector.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                                Value Class
//===----------------------------------------------------------------------===//

static inline Type *checkType(Type *Ty) {
  assert(Ty && "Value defined with a null type: Error!");
  return const_cast<Type*>(Ty);
}

Value::Value(Type *ty, unsigned scid)
  : SubclassID(scid), HasValueHandle(0),
    SubclassOptionalData(0), SubclassData(0), VTy((Type*)checkType(ty)),
    UseList(0), Name(0) {
  // FIXME: Why isn't this in the subclass gunk??
  if (isa<CallInst>(this) || isa<InvokeInst>(this))
    assert((VTy->isFirstClassType() || VTy->isVoidTy() || VTy->isStructTy()) &&
           "invalid CallInst type!");
  else if (!isa<Constant>(this) && !isa<BasicBlock>(this))
    assert((VTy->isFirstClassType() || VTy->isVoidTy()) &&
           "Cannot create non-first-class values except for constants!");
}

Value::~Value() {
  // Notify all ValueHandles (if present) that this value is going away.
  if (HasValueHandle)
    ValueHandleBase::ValueIsDeleted(this);

#ifndef NDEBUG      // Only in -g mode...
  // Check to make sure that there are no uses of this value that are still
  // around when the value is destroyed.  If there are, then we have a dangling
  // reference and something is wrong.  This code is here to print out what is
  // still being referenced.  The value in question should be printed as
  // a <badref>
  //
  if (!use_empty()) {
    dbgs() << "While deleting: " << *VTy << " %" << getName() << "\n";
    for (use_iterator I = use_begin(), E = use_end(); I != E; ++I)
      dbgs() << "Use still stuck around after Def is destroyed:"
           << **I << "\n";
  }
#endif
  assert(use_empty() && "Uses remain when a value is destroyed!");

  // If this value is named, destroy the name.  This should not be in a symtab
  // at this point.
  if (Name && SubclassID != MDStringVal)
    Name->Destroy();

  // There should be no uses of this object anymore, remove it.
  LeakDetector::removeGarbageObject(this);
}

/// hasNUses - Return true if this Value has exactly N users.
///
bool Value::hasNUses(unsigned N) const {
  const_use_iterator UI = use_begin(), E = use_end();

  for (; N; --N, ++UI)
    if (UI == E) return false;  // Too few.
  return UI == E;
}

/// hasNUsesOrMore - Return true if this value has N users or more.  This is
/// logically equivalent to getNumUses() >= N.
///
bool Value::hasNUsesOrMore(unsigned N) const {
  const_use_iterator UI = use_begin(), E = use_end();

  for (; N; --N, ++UI)
    if (UI == E) return false;  // Too few.

  return true;
}

/// isUsedInBasicBlock - Return true if this value is used in the specified
/// basic block.
bool Value::isUsedInBasicBlock(const BasicBlock *BB) const {
  // Start by scanning over the instructions looking for a use before we start
  // the expensive use iteration.
  unsigned MaxBlockSize = 3;
  for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
    if (std::find(I->op_begin(), I->op_end(), this) != I->op_end())
      return true;
    if (MaxBlockSize-- == 0) // If the block is larger fall back to use_iterator
      break;
  }

  if (MaxBlockSize != 0) // We scanned the entire block and found no use.
    return false;

  for (const_use_iterator I = use_begin(), E = use_end(); I != E; ++I) {
    const Instruction *User = dyn_cast<Instruction>(*I);
    if (User && User->getParent() == BB)
      return true;
  }
  return false;
}


/// getNumUses - This method computes the number of uses of this Value.  This
/// is a linear time operation.  Use hasOneUse or hasNUses to check for specific
/// values.
unsigned Value::getNumUses() const {
  return (unsigned)std::distance(use_begin(), use_end());
}

static bool getSymTab(Value *V, ValueSymbolTable *&ST) {
  ST = 0;
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    if (BasicBlock *P = I->getParent())
      if (Function *PP = P->getParent())
        ST = &PP->getValueSymbolTable();
  } else if (BasicBlock *BB = dyn_cast<BasicBlock>(V)) {
    if (Function *P = BB->getParent())
      ST = &P->getValueSymbolTable();
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    if (Module *P = GV->getParent())
      ST = &P->getValueSymbolTable();
  } else if (Argument *A = dyn_cast<Argument>(V)) {
    if (Function *P = A->getParent())
      ST = &P->getValueSymbolTable();
  } else if (isa<MDString>(V))
    return true;
  else {
    assert(isa<Constant>(V) && "Unknown value type!");
    return true;  // no name is setable for this.
  }
  return false;
}

StringRef Value::getName() const {
  // Make sure the empty string is still a C string. For historical reasons,
  // some clients want to call .data() on the result and expect it to be null
  // terminated.
  if (!Name) return StringRef("", 0);
  return Name->getKey();
}

void Value::setName(const Twine &NewName) {
  assert(SubclassID != MDStringVal &&
         "Cannot set the name of MDString with this method!");

  // Fast path for common IRBuilder case of setName("") when there is no name.
  if (NewName.isTriviallyEmpty() && !hasName())
    return;

  SmallString<256> NameData;
  StringRef NameRef = NewName.toStringRef(NameData);

  // Name isn't changing?
  if (getName() == NameRef)
    return;

  assert(!getType()->isVoidTy() && "Cannot assign a name to void values!");

  // Get the symbol table to update for this object.
  ValueSymbolTable *ST;
  if (getSymTab(this, ST))
    return;  // Cannot set a name on this value (e.g. constant).

  if (!ST) { // No symbol table to update?  Just do the change.
    if (NameRef.empty()) {
      // Free the name for this value.
      Name->Destroy();
      Name = 0;
      return;
    }

    if (Name)
      Name->Destroy();

    // NOTE: Could optimize for the case the name is shrinking to not deallocate
    // then reallocated.

    // Create the new name.
    Name = ValueName::Create(NameRef.begin(), NameRef.end());
    Name->setValue(this);
    return;
  }

  // NOTE: Could optimize for the case the name is shrinking to not deallocate
  // then reallocated.
  if (hasName()) {
    // Remove old name.
    ST->removeValueName(Name);
    Name->Destroy();
    Name = 0;

    if (NameRef.empty())
      return;
  }

  // Name is changing to something new.
  Name = ST->createValueName(NameRef, this);
}


/// takeName - transfer the name from V to this value, setting V's name to
/// empty.  It is an error to call V->takeName(V).
void Value::takeName(Value *V) {
  assert(SubclassID != MDStringVal && "Cannot take the name of an MDString!");

  ValueSymbolTable *ST = 0;
  // If this value has a name, drop it.
  if (hasName()) {
    // Get the symtab this is in.
    if (getSymTab(this, ST)) {
      // We can't set a name on this value, but we need to clear V's name if
      // it has one.
      if (V->hasName()) V->setName("");
      return;  // Cannot set a name on this value (e.g. constant).
    }

    // Remove old name.
    if (ST)
      ST->removeValueName(Name);
    Name->Destroy();
    Name = 0;
  }

  // Now we know that this has no name.

  // If V has no name either, we're done.
  if (!V->hasName()) return;

  // Get this's symtab if we didn't before.
  if (!ST) {
    if (getSymTab(this, ST)) {
      // Clear V's name.
      V->setName("");
      return;  // Cannot set a name on this value (e.g. constant).
    }
  }

  // Get V's ST, this should always succed, because V has a name.
  ValueSymbolTable *VST;
  bool Failure = getSymTab(V, VST);
  assert(!Failure && "V has a name, so it should have a ST!"); (void)Failure;

  // If these values are both in the same symtab, we can do this very fast.
  // This works even if both values have no symtab yet.
  if (ST == VST) {
    // Take the name!
    Name = V->Name;
    V->Name = 0;
    Name->setValue(this);
    return;
  }

  // Otherwise, things are slightly more complex.  Remove V's name from VST and
  // then reinsert it into ST.

  if (VST)
    VST->removeValueName(V->Name);
  Name = V->Name;
  V->Name = 0;
  Name->setValue(this);

  if (ST)
    ST->reinsertValue(this);
}


void Value::replaceAllUsesWith(Value *New) {
  assert(New && "Value::replaceAllUsesWith(<null>) is invalid!");
  assert(New != this && "this->replaceAllUsesWith(this) is NOT valid!");
  assert(New->getType() == getType() &&
         "replaceAllUses of value with new value of different type!");

  // Notify all ValueHandles (if present) that this value is going away.
  if (HasValueHandle)
    ValueHandleBase::ValueIsRAUWd(this, New);
  
  while (!use_empty()) {
    Use &U = *UseList;
    // Must handle Constants specially, we cannot call replaceUsesOfWith on a
    // constant because they are uniqued.
    if (Constant *C = dyn_cast<Constant>(U.getUser())) {
      if (!isa<GlobalValue>(C)) {
        C->replaceUsesOfWithOnConstant(this, New, &U);
        continue;
      }
    }
    
    U.set(New);
  }
  
  if (BasicBlock *BB = dyn_cast<BasicBlock>(this))
    BB->replaceSuccessorsPhiUsesWith(cast<BasicBlock>(New));
}

namespace {
// Various metrics for how much to strip off of pointers.
enum PointerStripKind {
  PSK_ZeroIndices,
  PSK_InBoundsConstantIndices,
  PSK_InBounds
};

template <PointerStripKind StripKind>
static Value *stripPointerCastsAndOffsets(Value *V) {
  if (!V->getType()->isPointerTy())
    return V;

  // Even though we don't look through PHI nodes, we could be called on an
  // instruction in an unreachable block, which may be on a cycle.
  SmallPtrSet<Value *, 4> Visited;

  Visited.insert(V);
  do {
    if (GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
      switch (StripKind) {
      case PSK_ZeroIndices:
        if (!GEP->hasAllZeroIndices())
          return V;
        break;
      case PSK_InBoundsConstantIndices:
        if (!GEP->hasAllConstantIndices())
          return V;
        // fallthrough
      case PSK_InBounds:
        if (!GEP->isInBounds())
          return V;
        break;
      }
      V = GEP->getPointerOperand();
    } else if (Operator::getOpcode(V) == Instruction::BitCast) {
      V = cast<Operator>(V)->getOperand(0);
    } else if (GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
      if (GA->mayBeOverridden())
        return V;
      V = GA->getAliasee();
    } else {
      return V;
    }
    assert(V->getType()->isPointerTy() && "Unexpected operand type!");
  } while (Visited.insert(V));

  return V;
}
} // namespace

Value *Value::stripPointerCasts() {
  return stripPointerCastsAndOffsets<PSK_ZeroIndices>(this);
}

Value *Value::stripInBoundsConstantOffsets() {
  return stripPointerCastsAndOffsets<PSK_InBoundsConstantIndices>(this);
}

Value *Value::stripInBoundsOffsets() {
  return stripPointerCastsAndOffsets<PSK_InBounds>(this);
}

/// isDereferenceablePointer - Test if this value is always a pointer to
/// allocated and suitably aligned memory for a simple load or store.
static bool isDereferenceablePointer(const Value *V,
                                     SmallPtrSet<const Value *, 32> &Visited) {
  // Note that it is not safe to speculate into a malloc'd region because
  // malloc may return null.
  // It's also not always safe to follow a bitcast, for example:
  //   bitcast i8* (alloca i8) to i32*
  // would result in a 4-byte load from a 1-byte alloca. Some cases could
  // be handled using DataLayout to check sizes and alignments though.

  // These are obviously ok.
  if (isa<AllocaInst>(V)) return true;

  // Global variables which can't collapse to null are ok.
  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    return !GV->hasExternalWeakLinkage();

  // byval arguments are ok.
  if (const Argument *A = dyn_cast<Argument>(V))
    return A->hasByValAttr();

  // For GEPs, determine if the indexing lands within the allocated object.
  if (const GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
    // Conservatively require that the base pointer be fully dereferenceable.
    if (!Visited.insert(GEP->getOperand(0)))
      return false;
    if (!isDereferenceablePointer(GEP->getOperand(0), Visited))
      return false;
    // Check the indices.
    gep_type_iterator GTI = gep_type_begin(GEP);
    for (User::const_op_iterator I = GEP->op_begin()+1,
         E = GEP->op_end(); I != E; ++I) {
      Value *Index = *I;
      Type *Ty = *GTI++;
      // Struct indices can't be out of bounds.
      if (isa<StructType>(Ty))
        continue;
      ConstantInt *CI = dyn_cast<ConstantInt>(Index);
      if (!CI)
        return false;
      // Zero is always ok.
      if (CI->isZero())
        continue;
      // Check to see that it's within the bounds of an array.
      ArrayType *ATy = dyn_cast<ArrayType>(Ty);
      if (!ATy)
        return false;
      if (CI->getValue().getActiveBits() > 64)
        return false;
      if (CI->getZExtValue() >= ATy->getNumElements())
        return false;
    }
    // Indices check out; this is dereferenceable.
    return true;
  }

  // If we don't know, assume the worst.
  return false;
}

/// isDereferenceablePointer - Test if this value is always a pointer to
/// allocated and suitably aligned memory for a simple load or store.
bool Value::isDereferenceablePointer() const {
  SmallPtrSet<const Value *, 32> Visited;
  return ::isDereferenceablePointer(this, Visited);
}

/// DoPHITranslation - If this value is a PHI node with CurBB as its parent,
/// return the value in the PHI node corresponding to PredBB.  If not, return
/// ourself.  This is useful if you want to know the value something has in a
/// predecessor block.
Value *Value::DoPHITranslation(const BasicBlock *CurBB,
                               const BasicBlock *PredBB) {
  PHINode *PN = dyn_cast<PHINode>(this);
  if (PN && PN->getParent() == CurBB)
    return PN->getIncomingValueForBlock(PredBB);
  return this;
}

LLVMContext &Value::getContext() const { return VTy->getContext(); }

//===----------------------------------------------------------------------===//
//                             ValueHandleBase Class
//===----------------------------------------------------------------------===//

/// AddToExistingUseList - Add this ValueHandle to the use list for VP, where
/// List is known to point into the existing use list.
void ValueHandleBase::AddToExistingUseList(ValueHandleBase **List) {
  assert(List && "Handle list is null?");

  // Splice ourselves into the list.
  Next = *List;
  *List = this;
  setPrevPtr(List);
  if (Next) {
    Next->setPrevPtr(&Next);
    assert(VP.getPointer() == Next->VP.getPointer() && "Added to wrong list?");
  }
}

void ValueHandleBase::AddToExistingUseListAfter(ValueHandleBase *List) {
  assert(List && "Must insert after existing node");

  Next = List->Next;
  setPrevPtr(&List->Next);
  List->Next = this;
  if (Next)
    Next->setPrevPtr(&Next);
}

/// AddToUseList - Add this ValueHandle to the use list for VP.
void ValueHandleBase::AddToUseList() {
  assert(VP.getPointer() && "Null pointer doesn't have a use list!");

  LLVMContextImpl *pImpl = VP.getPointer()->getContext().pImpl;

  if (VP.getPointer()->HasValueHandle) {
    // If this value already has a ValueHandle, then it must be in the
    // ValueHandles map already.
    ValueHandleBase *&Entry = pImpl->ValueHandles[VP.getPointer()];
    assert(Entry != 0 && "Value doesn't have any handles?");
    AddToExistingUseList(&Entry);
    return;
  }

  // Ok, it doesn't have any handles yet, so we must insert it into the
  // DenseMap.  However, doing this insertion could cause the DenseMap to
  // reallocate itself, which would invalidate all of the PrevP pointers that
  // point into the old table.  Handle this by checking for reallocation and
  // updating the stale pointers only if needed.
  DenseMap<Value*, ValueHandleBase*> &Handles = pImpl->ValueHandles;
  const void *OldBucketPtr = Handles.getPointerIntoBucketsArray();

  ValueHandleBase *&Entry = Handles[VP.getPointer()];
  assert(Entry == 0 && "Value really did already have handles?");
  AddToExistingUseList(&Entry);
  VP.getPointer()->HasValueHandle = true;

  // If reallocation didn't happen or if this was the first insertion, don't
  // walk the table.
  if (Handles.isPointerIntoBucketsArray(OldBucketPtr) ||
      Handles.size() == 1) {
    return;
  }

  // Okay, reallocation did happen.  Fix the Prev Pointers.
  for (DenseMap<Value*, ValueHandleBase*>::iterator I = Handles.begin(),
       E = Handles.end(); I != E; ++I) {
    assert(I->second && I->first == I->second->VP.getPointer() &&
           "List invariant broken!");
    I->second->setPrevPtr(&I->second);
  }
}

/// RemoveFromUseList - Remove this ValueHandle from its current use list.
void ValueHandleBase::RemoveFromUseList() {
  assert(VP.getPointer() && VP.getPointer()->HasValueHandle &&
         "Pointer doesn't have a use list!");

  // Unlink this from its use list.
  ValueHandleBase **PrevPtr = getPrevPtr();
  assert(*PrevPtr == this && "List invariant broken");

  *PrevPtr = Next;
  if (Next) {
    assert(Next->getPrevPtr() == &Next && "List invariant broken");
    Next->setPrevPtr(PrevPtr);
    return;
  }

  // If the Next pointer was null, then it is possible that this was the last
  // ValueHandle watching VP.  If so, delete its entry from the ValueHandles
  // map.
  LLVMContextImpl *pImpl = VP.getPointer()->getContext().pImpl;
  DenseMap<Value*, ValueHandleBase*> &Handles = pImpl->ValueHandles;
  if (Handles.isPointerIntoBucketsArray(PrevPtr)) {
    Handles.erase(VP.getPointer());
    VP.getPointer()->HasValueHandle = false;
  }
}


void ValueHandleBase::ValueIsDeleted(Value *V) {
  assert(V->HasValueHandle && "Should only be called if ValueHandles present");

  // Get the linked list base, which is guaranteed to exist since the
  // HasValueHandle flag is set.
  LLVMContextImpl *pImpl = V->getContext().pImpl;
  ValueHandleBase *Entry = pImpl->ValueHandles[V];
  assert(Entry && "Value bit set but no entries exist");

  // We use a local ValueHandleBase as an iterator so that ValueHandles can add
  // and remove themselves from the list without breaking our iteration.  This
  // is not really an AssertingVH; we just have to give ValueHandleBase a kind.
  // Note that we deliberately do not the support the case when dropping a value
  // handle results in a new value handle being permanently added to the list
  // (as might occur in theory for CallbackVH's): the new value handle will not
  // be processed and the checking code will mete out righteous punishment if
  // the handle is still present once we have finished processing all the other
  // value handles (it is fine to momentarily add then remove a value handle).
  for (ValueHandleBase Iterator(Assert, *Entry); Entry; Entry = Iterator.Next) {
    Iterator.RemoveFromUseList();
    Iterator.AddToExistingUseListAfter(Entry);
    assert(Entry->Next == &Iterator && "Loop invariant broken.");

    switch (Entry->getKind()) {
    case Assert:
      break;
    case Tracking:
      // Mark that this value has been deleted by setting it to an invalid Value
      // pointer.
      Entry->operator=(DenseMapInfo<Value *>::getTombstoneKey());
      break;
    case Weak:
      // Weak just goes to null, which will unlink it from the list.
      Entry->operator=(0);
      break;
    case Callback:
      // Forward to the subclass's implementation.
      static_cast<CallbackVH*>(Entry)->deleted();
      break;
    }
  }

  // All callbacks, weak references, and assertingVHs should be dropped by now.
  if (V->HasValueHandle) {
#ifndef NDEBUG      // Only in +Asserts mode...
    dbgs() << "While deleting: " << *V->getType() << " %" << V->getName()
           << "\n";
    if (pImpl->ValueHandles[V]->getKind() == Assert)
      llvm_unreachable("An asserting value handle still pointed to this"
                       " value!");

#endif
    llvm_unreachable("All references to V were not removed?");
  }
}


void ValueHandleBase::ValueIsRAUWd(Value *Old, Value *New) {
  assert(Old->HasValueHandle &&"Should only be called if ValueHandles present");
  assert(Old != New && "Changing value into itself!");

  // Get the linked list base, which is guaranteed to exist since the
  // HasValueHandle flag is set.
  LLVMContextImpl *pImpl = Old->getContext().pImpl;
  ValueHandleBase *Entry = pImpl->ValueHandles[Old];

  assert(Entry && "Value bit set but no entries exist");

  // We use a local ValueHandleBase as an iterator so that
  // ValueHandles can add and remove themselves from the list without
  // breaking our iteration.  This is not really an AssertingVH; we
  // just have to give ValueHandleBase some kind.
  for (ValueHandleBase Iterator(Assert, *Entry); Entry; Entry = Iterator.Next) {
    Iterator.RemoveFromUseList();
    Iterator.AddToExistingUseListAfter(Entry);
    assert(Entry->Next == &Iterator && "Loop invariant broken.");

    switch (Entry->getKind()) {
    case Assert:
      // Asserting handle does not follow RAUW implicitly.
      break;
    case Tracking:
      // Tracking goes to new value like a WeakVH. Note that this may make it
      // something incompatible with its templated type. We don't want to have a
      // virtual (or inline) interface to handle this though, so instead we make
      // the TrackingVH accessors guarantee that a client never sees this value.

      // FALLTHROUGH
    case Weak:
      // Weak goes to the new value, which will unlink it from Old's list.
      Entry->operator=(New);
      break;
    case Callback:
      // Forward to the subclass's implementation.
      static_cast<CallbackVH*>(Entry)->allUsesReplacedWith(New);
      break;
    }
  }

#ifndef NDEBUG
  // If any new tracking or weak value handles were added while processing the
  // list, then complain about it now.
  if (Old->HasValueHandle)
    for (Entry = pImpl->ValueHandles[Old]; Entry; Entry = Entry->Next)
      switch (Entry->getKind()) {
      case Tracking:
      case Weak:
        dbgs() << "After RAUW from " << *Old->getType() << " %"
               << Old->getName() << " to " << *New->getType() << " %"
               << New->getName() << "\n";
        llvm_unreachable("A tracking or weak value handle still pointed to the"
                         " old value!\n");
      default:
        break;
      }
#endif
}

// Default implementation for CallbackVH.
void CallbackVH::allUsesReplacedWith(Value *) {}

void CallbackVH::deleted() {
  setValPtr(NULL);
}
