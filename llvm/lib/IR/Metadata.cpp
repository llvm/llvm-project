//===-- Metadata.cpp - Implement Metadata classes -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Metadata classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Metadata.h"
#include "LLVMContextImpl.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"

using namespace llvm;

MetadataAsValue::MetadataAsValue(Type *Ty, Metadata *MD)
    : Value(Ty, MetadataAsValueVal), MD(MD) {
  track();
}

MetadataAsValue::~MetadataAsValue() {
  getType()->getContext().pImpl->MetadataAsValues.erase(MD);
  untrack();
}

/// \brief Canonicalize metadata arguments to intrinsics.
///
/// To support bitcode upgrades (and assembly semantic sugar) for \a
/// MetadataAsValue, we need to canonicalize certain metadata.
///
///   - nullptr is replaced by an empty MDNode.
///   - An MDNode with a single null operand is replaced by an empty MDNode.
///   - An MDNode whose only operand is a \a ConstantAsMetadata gets skipped.
///
/// This maintains readability of bitcode from when metadata was a type of
/// value, and these bridges were unnecessary.
static Metadata *canonicalizeMetadataForValue(LLVMContext &Context,
                                              Metadata *MD) {
  if (!MD)
    // !{}
    return MDNode::get(Context, None);

  // Return early if this isn't a single-operand MDNode.
  auto *N = dyn_cast<MDNode>(MD);
  if (!N || N->getNumOperands() != 1)
    return MD;

  if (!N->getOperand(0))
    // !{}
    return MDNode::get(Context, None);

  if (auto *C = dyn_cast<ConstantAsMetadata>(N->getOperand(0)))
    // Look through the MDNode.
    return C;

  return MD;
}

MetadataAsValue *MetadataAsValue::get(LLVMContext &Context, Metadata *MD) {
  MD = canonicalizeMetadataForValue(Context, MD);
  auto *&Entry = Context.pImpl->MetadataAsValues[MD];
  if (!Entry)
    Entry = new MetadataAsValue(Type::getMetadataTy(Context), MD);
  return Entry;
}

MetadataAsValue *MetadataAsValue::getIfExists(LLVMContext &Context,
                                              Metadata *MD) {
  MD = canonicalizeMetadataForValue(Context, MD);
  auto &Store = Context.pImpl->MetadataAsValues;
  auto I = Store.find(MD);
  return I == Store.end() ? nullptr : I->second;
}

void MetadataAsValue::handleChangedMetadata(Metadata *MD) {
  LLVMContext &Context = getContext();
  MD = canonicalizeMetadataForValue(Context, MD);
  auto &Store = Context.pImpl->MetadataAsValues;

  // Stop tracking the old metadata.
  Store.erase(this->MD);
  untrack();
  this->MD = nullptr;

  // Start tracking MD, or RAUW if necessary.
  auto *&Entry = Store[MD];
  if (Entry) {
    replaceAllUsesWith(Entry);
    delete this;
    return;
  }

  this->MD = MD;
  track();
  Entry = this;
}

void MetadataAsValue::track() {
  if (MD)
    MetadataTracking::track(&MD, *MD, *this);
}

void MetadataAsValue::untrack() {
  if (MD)
    MetadataTracking::untrack(MD);
}

void ReplaceableMetadataImpl::addRef(void *Ref, OwnerTy Owner) {
  bool WasInserted =
      UseMap.insert(std::make_pair(Ref, std::make_pair(Owner, NextIndex)))
          .second;
  (void)WasInserted;
  assert(WasInserted && "Expected to add a reference");

  ++NextIndex;
  assert(NextIndex != 0 && "Unexpected overflow");
}

void ReplaceableMetadataImpl::dropRef(void *Ref) {
  bool WasErased = UseMap.erase(Ref);
  (void)WasErased;
  assert(WasErased && "Expected to drop a reference");
}

void ReplaceableMetadataImpl::moveRef(void *Ref, void *New,
                                      const Metadata &MD) {
  auto I = UseMap.find(Ref);
  assert(I != UseMap.end() && "Expected to move a reference");
  auto OwnerAndIndex = I->second;
  UseMap.erase(I);
  bool WasInserted = UseMap.insert(std::make_pair(New, OwnerAndIndex)).second;
  (void)WasInserted;
  assert(WasInserted && "Expected to add a reference");

  // Check that the references are direct if there's no owner.
  (void)MD;
  assert((OwnerAndIndex.first || *static_cast<Metadata **>(Ref) == &MD) &&
         "Reference without owner must be direct");
  assert((OwnerAndIndex.first || *static_cast<Metadata **>(New) == &MD) &&
         "Reference without owner must be direct");
}

void ReplaceableMetadataImpl::replaceAllUsesWith(Metadata *MD) {
  assert(!(MD && isa<MDNodeFwdDecl>(MD)) && "Expected non-temp node");

  if (UseMap.empty())
    return;

  // Copy out uses since UseMap will get touched below.
  typedef std::pair<void *, std::pair<OwnerTy, uint64_t>> UseTy;
  SmallVector<UseTy, 8> Uses(UseMap.begin(), UseMap.end());
  std::sort(Uses.begin(), Uses.end(), [](const UseTy &L, const UseTy &R) {
    return L.second.second < R.second.second;
  });
  for (const auto &Pair : Uses) {
    // Check that this Ref hasn't disappeared after RAUW (when updating a
    // previous Ref).
    if (!UseMap.count(Pair.first))
      continue;

    OwnerTy Owner = Pair.second.first;
    if (!Owner) {
      // Update unowned tracking references directly.
      Metadata *&Ref = *static_cast<Metadata **>(Pair.first);
      Ref = MD;
      if (MD)
        MetadataTracking::track(Ref);
      UseMap.erase(Pair.first);
      continue;
    }

    // Check for MetadataAsValue.
    if (Owner.is<MetadataAsValue *>()) {
      Owner.get<MetadataAsValue *>()->handleChangedMetadata(MD);
      continue;
    }

    // There's a Metadata owner -- dispatch.
    Metadata *OwnerMD = Owner.get<Metadata *>();
    switch (OwnerMD->getMetadataID()) {
#define HANDLE_METADATA_LEAF(CLASS)                                            \
  case Metadata::CLASS##Kind:                                                  \
    cast<CLASS>(OwnerMD)->handleChangedOperand(Pair.first, MD);                \
    continue;
#include "llvm/IR/Metadata.def"
    default:
      llvm_unreachable("Invalid metadata subclass");
    }
  }
  assert(UseMap.empty() && "Expected all uses to be replaced");
}

void ReplaceableMetadataImpl::resolveAllUses(bool ResolveUsers) {
  if (UseMap.empty())
    return;

  if (!ResolveUsers) {
    UseMap.clear();
    return;
  }

  // Copy out uses since UseMap could get touched below.
  typedef std::pair<void *, std::pair<OwnerTy, uint64_t>> UseTy;
  SmallVector<UseTy, 8> Uses(UseMap.begin(), UseMap.end());
  std::sort(Uses.begin(), Uses.end(), [](const UseTy &L, const UseTy &R) {
    return L.second.second < R.second.second;
  });
  UseMap.clear();
  for (const auto &Pair : Uses) {
    auto Owner = Pair.second.first;
    if (!Owner)
      continue;
    if (Owner.is<MetadataAsValue *>())
      continue;

    // Resolve UniquableMDNodes that point at this.
    auto *OwnerMD = dyn_cast<UniquableMDNode>(Owner.get<Metadata *>());
    if (!OwnerMD)
      continue;
    if (OwnerMD->isResolved())
      continue;
    OwnerMD->decrementUnresolvedOperandCount();
  }
}

static Function *getLocalFunction(Value *V) {
  assert(V && "Expected value");
  if (auto *A = dyn_cast<Argument>(V))
    return A->getParent();
  if (BasicBlock *BB = cast<Instruction>(V)->getParent())
    return BB->getParent();
  return nullptr;
}

ValueAsMetadata *ValueAsMetadata::get(Value *V) {
  assert(V && "Unexpected null Value");

  auto &Context = V->getContext();
  auto *&Entry = Context.pImpl->ValuesAsMetadata[V];
  if (!Entry) {
    assert((isa<Constant>(V) || isa<Argument>(V) || isa<Instruction>(V)) &&
           "Expected constant or function-local value");
    assert(!V->NameAndIsUsedByMD.getInt() &&
           "Expected this to be the only metadata use");
    V->NameAndIsUsedByMD.setInt(true);
    if (auto *C = dyn_cast<Constant>(V))
      Entry = new ConstantAsMetadata(C);
    else
      Entry = new LocalAsMetadata(V);
  }

  return Entry;
}

ValueAsMetadata *ValueAsMetadata::getIfExists(Value *V) {
  assert(V && "Unexpected null Value");
  return V->getContext().pImpl->ValuesAsMetadata.lookup(V);
}

void ValueAsMetadata::handleDeletion(Value *V) {
  assert(V && "Expected valid value");

  auto &Store = V->getType()->getContext().pImpl->ValuesAsMetadata;
  auto I = Store.find(V);
  if (I == Store.end())
    return;

  // Remove old entry from the map.
  ValueAsMetadata *MD = I->second;
  assert(MD && "Expected valid metadata");
  assert(MD->getValue() == V && "Expected valid mapping");
  Store.erase(I);

  // Delete the metadata.
  MD->replaceAllUsesWith(nullptr);
  delete MD;
}

void ValueAsMetadata::handleRAUW(Value *From, Value *To) {
  assert(From && "Expected valid value");
  assert(To && "Expected valid value");
  assert(From != To && "Expected changed value");
  assert(From->getType() == To->getType() && "Unexpected type change");

  LLVMContext &Context = From->getType()->getContext();
  auto &Store = Context.pImpl->ValuesAsMetadata;
  auto I = Store.find(From);
  if (I == Store.end()) {
    assert(!From->NameAndIsUsedByMD.getInt() &&
           "Expected From not to be used by metadata");
    return;
  }

  // Remove old entry from the map.
  assert(From->NameAndIsUsedByMD.getInt() &&
         "Expected From to be used by metadata");
  From->NameAndIsUsedByMD.setInt(false);
  ValueAsMetadata *MD = I->second;
  assert(MD && "Expected valid metadata");
  assert(MD->getValue() == From && "Expected valid mapping");
  Store.erase(I);

  if (isa<LocalAsMetadata>(MD)) {
    if (auto *C = dyn_cast<Constant>(To)) {
      // Local became a constant.
      MD->replaceAllUsesWith(ConstantAsMetadata::get(C));
      delete MD;
      return;
    }
    if (getLocalFunction(From) && getLocalFunction(To) &&
        getLocalFunction(From) != getLocalFunction(To)) {
      // Function changed.
      MD->replaceAllUsesWith(nullptr);
      delete MD;
      return;
    }
  } else if (!isa<Constant>(To)) {
    // Changed to function-local value.
    MD->replaceAllUsesWith(nullptr);
    delete MD;
    return;
  }

  auto *&Entry = Store[To];
  if (Entry) {
    // The target already exists.
    MD->replaceAllUsesWith(Entry);
    delete MD;
    return;
  }

  // Update MD in place (and update the map entry).
  assert(!To->NameAndIsUsedByMD.getInt() &&
         "Expected this to be the only metadata use");
  To->NameAndIsUsedByMD.setInt(true);
  MD->V = To;
  Entry = MD;
}

//===----------------------------------------------------------------------===//
// MDString implementation.
//

MDString *MDString::get(LLVMContext &Context, StringRef Str) {
  auto &Store = Context.pImpl->MDStringCache;
  auto I = Store.find(Str);
  if (I != Store.end())
    return &I->second;

  auto *Entry =
      StringMapEntry<MDString>::Create(Str, Store.getAllocator(), MDString());
  bool WasInserted = Store.insert(Entry);
  (void)WasInserted;
  assert(WasInserted && "Expected entry to be inserted");
  Entry->second.Entry = Entry;
  return &Entry->second;
}

StringRef MDString::getString() const {
  assert(Entry && "Expected to find string map entry");
  return Entry->first();
}

//===----------------------------------------------------------------------===//
// MDNode implementation.
//

void *MDNode::operator new(size_t Size, unsigned NumOps) {
  void *Ptr = ::operator new(Size + NumOps * sizeof(MDOperand));
  MDOperand *O = static_cast<MDOperand *>(Ptr);
  for (MDOperand *E = O + NumOps; O != E; ++O)
    (void)new (O) MDOperand;
  return O;
}

void MDNode::operator delete(void *Mem) {
  MDNode *N = static_cast<MDNode *>(Mem);
  MDOperand *O = static_cast<MDOperand *>(Mem);
  for (MDOperand *E = O - N->NumOperands; O != E; --O)
    (O - 1)->~MDOperand();
  ::operator delete(O);
}

MDNode::MDNode(LLVMContext &Context, unsigned ID, ArrayRef<Metadata *> MDs)
    : Metadata(ID), Context(Context), NumOperands(MDs.size()),
      MDNodeSubclassData(0) {
  for (unsigned I = 0, E = MDs.size(); I != E; ++I)
    setOperand(I, MDs[I]);
}

bool MDNode::isResolved() const {
  if (isa<MDNodeFwdDecl>(this))
    return false;
  return cast<UniquableMDNode>(this)->isResolved();
}

static bool isOperandUnresolved(Metadata *Op) {
  if (auto *N = dyn_cast_or_null<MDNode>(Op))
    return !N->isResolved();
  return false;
}

UniquableMDNode::UniquableMDNode(LLVMContext &C, unsigned ID,
                                 ArrayRef<Metadata *> Vals, bool AllowRAUW)
    : MDNode(C, ID, Vals) {
  if (!AllowRAUW)
    return;

  // Check whether any operands are unresolved, requiring re-uniquing.
  unsigned NumUnresolved = 0;
  for (const auto &Op : operands())
    NumUnresolved += unsigned(isOperandUnresolved(Op));

  if (!NumUnresolved)
    return;

  ReplaceableUses.reset(new ReplaceableMetadataImpl);
  SubclassData32 = NumUnresolved;
}

void UniquableMDNode::resolve() {
  assert(!isResolved() && "Expected this to be unresolved");

  // Move the map, so that this immediately looks resolved.
  auto Uses = std::move(ReplaceableUses);
  SubclassData32 = 0;
  assert(isResolved() && "Expected this to be resolved");

  // Drop RAUW support.
  Uses->resolveAllUses();
}

void UniquableMDNode::resolveAfterOperandChange(Metadata *Old, Metadata *New) {
  assert(SubclassData32 != 0 && "Expected unresolved operands");

  // Check if an operand was resolved.
  if (!isOperandUnresolved(Old)) {
    if (isOperandUnresolved(New))
      // An operand was un-resolved!
      ++SubclassData32;
  } else if (!isOperandUnresolved(New))
    decrementUnresolvedOperandCount();
}

void UniquableMDNode::decrementUnresolvedOperandCount() {
  if (!--SubclassData32)
    // Last unresolved operand has just been resolved.
    resolve();
}

void UniquableMDNode::resolveCycles() {
  if (isResolved())
    return;

  // Resolve this node immediately.
  resolve();

  // Resolve all operands.
  for (const auto &Op : operands()) {
    if (!Op)
      continue;
    assert(!isa<MDNodeFwdDecl>(Op) &&
           "Expected all forward declarations to be resolved");
    if (auto *N = dyn_cast<UniquableMDNode>(Op))
      if (!N->isResolved())
        N->resolveCycles();
  }
}

void MDTuple::recalculateHash() {
  setHash(hash_combine_range(op_begin(), op_end()));
#ifndef NDEBUG
  {
    SmallVector<Metadata *, 8> MDs(op_begin(), op_end());
    unsigned RawHash = hash_combine_range(MDs.begin(), MDs.end());
    assert(getHash() == RawHash &&
           "Expected hash of MDOperand to equal hash of Metadata*");
  }
#endif
}

void MDNode::dropAllReferences() {
  for (unsigned I = 0, E = NumOperands; I != E; ++I)
    setOperand(I, nullptr);
  if (auto *N = dyn_cast<UniquableMDNode>(this))
    if (!N->isResolved()) {
      N->ReplaceableUses->resolveAllUses(/* ResolveUsers */ false);
      N->ReplaceableUses.reset();
    }
}

namespace llvm {
/// \brief Make MDOperand transparent for hashing.
///
/// This overload of an implementation detail of the hashing library makes
/// MDOperand hash to the same value as a \a Metadata pointer.
///
/// Note that overloading \a hash_value() as follows:
///
/// \code
///     size_t hash_value(const MDOperand &X) { return hash_value(X.get()); }
/// \endcode
///
/// does not cause MDOperand to be transparent.  In particular, a bare pointer
/// doesn't get hashed before it's combined, whereas \a MDOperand would.
static const Metadata *get_hashable_data(const MDOperand &X) { return X.get(); }
}

void UniquableMDNode::handleChangedOperand(void *Ref, Metadata *New) {
  unsigned Op = static_cast<MDOperand *>(Ref) - op_begin();
  assert(Op < getNumOperands() && "Expected valid operand");

  if (isStoredDistinctInContext()) {
    assert(isResolved() && "Expected distinct node to be resolved");

    // This node is not uniqued.  Just set the operand and be done with it.
    setOperand(Op, New);
    return;
  }

  // This node is uniqued.
  eraseFromStore();

  Metadata *Old = getOperand(Op);
  setOperand(Op, New);

  // Drop uniquing for self-reference cycles.
  if (New == this) {
    storeDistinctInContext();
    if (!isResolved())
      resolve();
    return;
  }

  // Re-unique the node.
  auto *Uniqued = uniquify();
  if (Uniqued == this) {
    if (!isResolved())
      resolveAfterOperandChange(Old, New);
    return;
  }

  // Collision.
  if (!isResolved()) {
    // Still unresolved, so RAUW.
    //
    // First, clear out all operands to prevent any recursion (similar to
    // dropAllReferences(), but we still need the use-list).
    for (unsigned O = 0, E = getNumOperands(); O != E; ++O)
      setOperand(O, nullptr);
    ReplaceableUses->replaceAllUsesWith(Uniqued);
    deleteAsSubclass();
    return;
  }

  // Store in non-uniqued form if RAUW isn't possible.
  storeDistinctInContext();
}

void UniquableMDNode::deleteAsSubclass() {
  switch (getMetadataID()) {
  default:
    llvm_unreachable("Invalid subclass of UniquableMDNode");
#define HANDLE_UNIQUABLE_LEAF(CLASS)                                           \
  case CLASS##Kind:                                                            \
    delete cast<CLASS>(this);                                                  \
    break;
#include "llvm/IR/Metadata.def"
  }
}

UniquableMDNode *UniquableMDNode::uniquify() {
  switch (getMetadataID()) {
  default:
    llvm_unreachable("Invalid subclass of UniquableMDNode");
#define HANDLE_UNIQUABLE_LEAF(CLASS)                                           \
  case CLASS##Kind:                                                            \
    return cast<CLASS>(this)->uniquifyImpl();
#include "llvm/IR/Metadata.def"
  }
}

void UniquableMDNode::eraseFromStore() {
  switch (getMetadataID()) {
  default:
    llvm_unreachable("Invalid subclass of UniquableMDNode");
#define HANDLE_UNIQUABLE_LEAF(CLASS)                                           \
  case CLASS##Kind:                                                            \
    cast<CLASS>(this)->eraseFromStoreImpl();                                   \
    break;
#include "llvm/IR/Metadata.def"
  }
}

MDTuple *MDTuple::getImpl(LLVMContext &Context, ArrayRef<Metadata *> MDs,
                          bool ShouldCreate) {
  MDTupleInfo::KeyTy Key(MDs);

  auto &Store = Context.pImpl->MDTuples;
  auto I = Store.find_as(Key);
  if (I != Store.end())
    return *I;
  if (!ShouldCreate)
    return nullptr;

  // Coallocate space for the node and Operands together, then placement new.
  auto *N = new (MDs.size()) MDTuple(Context, MDs, /* AllowRAUW */ true);
  N->setHash(Key.Hash);
  Store.insert(N);
  return N;
}

MDTuple *MDTuple::getDistinct(LLVMContext &Context, ArrayRef<Metadata *> MDs) {
  auto *N = new (MDs.size()) MDTuple(Context, MDs, /* AllowRAUW */ false);
  N->storeDistinctInContext();
  return N;
}

MDTuple *MDTuple::uniquifyImpl() {
  recalculateHash();
  MDTupleInfo::KeyTy Key(this);

  auto &Store = getContext().pImpl->MDTuples;
  auto I = Store.find_as(Key);
  if (I == Store.end()) {
    Store.insert(this);
    return this;
  }
  return *I;
}

void MDTuple::eraseFromStoreImpl() { getContext().pImpl->MDTuples.erase(this); }

MDLocation::MDLocation(LLVMContext &C, unsigned Line, unsigned Column,
                       ArrayRef<Metadata *> MDs, bool AllowRAUW)
    : UniquableMDNode(C, MDLocationKind, MDs, AllowRAUW) {
  assert((MDs.size() == 1 || MDs.size() == 2) &&
         "Expected a scope and optional inlined-at");

  // Set line and column.
  assert(Line < (1u << 24) && "Expected 24-bit line");
  assert(Column < (1u << 8) && "Expected 8-bit column");

  MDNodeSubclassData = Line;
  SubclassData16 = Column;
}

MDLocation *MDLocation::constructHelper(LLVMContext &Context, unsigned Line,
                                        unsigned Column, Metadata *Scope,
                                        Metadata *InlinedAt, bool AllowRAUW) {
  SmallVector<Metadata *, 2> Ops;
  Ops.push_back(Scope);
  if (InlinedAt)
    Ops.push_back(InlinedAt);
  return new (Ops.size()) MDLocation(Context, Line, Column, Ops, AllowRAUW);
}

static void adjustLine(unsigned &Line) {
  // Set to unknown on overflow.  Still use 24 bits for now.
  if (Line >= (1u << 24))
    Line = 0;
}

static void adjustColumn(unsigned &Column) {
  // Set to unknown on overflow.  Still use 8 bits for now.
  if (Column >= (1u << 8))
    Column = 0;
}

MDLocation *MDLocation::getImpl(LLVMContext &Context, unsigned Line,
                                unsigned Column, Metadata *Scope,
                                Metadata *InlinedAt, bool ShouldCreate) {
  // Fixup line/column.
  adjustLine(Line);
  adjustColumn(Column);

  MDLocationInfo::KeyTy Key(Line, Column, Scope, InlinedAt);

  auto &Store = Context.pImpl->MDLocations;
  auto I = Store.find_as(Key);
  if (I != Store.end())
    return *I;
  if (!ShouldCreate)
    return nullptr;

  auto *N = constructHelper(Context, Line, Column, Scope, InlinedAt,
                            /* AllowRAUW */ true);
  Store.insert(N);
  return N;
}

MDLocation *MDLocation::getDistinct(LLVMContext &Context, unsigned Line,
                                    unsigned Column, Metadata *Scope,
                                    Metadata *InlinedAt) {
  // Fixup line/column.
  adjustLine(Line);
  adjustColumn(Column);

  auto *N = constructHelper(Context, Line, Column, Scope, InlinedAt,
                            /* AllowRAUW */ false);
  N->storeDistinctInContext();
  return N;
}

MDLocation *MDLocation::uniquifyImpl() {
  MDLocationInfo::KeyTy Key(this);

  auto &Store = getContext().pImpl->MDLocations;
  auto I = Store.find_as(Key);
  if (I == Store.end()) {
    Store.insert(this);
    return this;
  }
  return *I;
}

void MDLocation::eraseFromStoreImpl() {
  getContext().pImpl->MDLocations.erase(this);
}

MDNodeFwdDecl *MDNode::getTemporary(LLVMContext &Context,
                                    ArrayRef<Metadata *> MDs) {
  return MDNodeFwdDecl::get(Context, MDs);
}

void MDNode::deleteTemporary(MDNode *N) { delete cast<MDNodeFwdDecl>(N); }

void UniquableMDNode::storeDistinctInContext() {
  assert(!IsDistinctInContext && "Expected newly distinct metadata");
  IsDistinctInContext = true;
  if (auto *T = dyn_cast<MDTuple>(this))
    T->setHash(0);
  getContext().pImpl->DistinctMDNodes.insert(this);
}

void MDNode::replaceOperandWith(unsigned I, Metadata *New) {
  if (getOperand(I) == New)
    return;

  if (isDistinct()) {
    setOperand(I, New);
    return;
  }

  cast<UniquableMDNode>(this)->handleChangedOperand(mutable_begin() + I, New);
}

void MDNode::setOperand(unsigned I, Metadata *New) {
  assert(I < NumOperands);
  if (isStoredDistinctInContext() || isa<MDNodeFwdDecl>(this))
    // No need for a callback, this isn't uniqued.
    mutable_begin()[I].reset(New, nullptr);
  else
    mutable_begin()[I].reset(New, this);
}

/// \brief Get a node, or a self-reference that looks like it.
///
/// Special handling for finding self-references, for use by \a
/// MDNode::concatenate() and \a MDNode::intersect() to maintain behaviour from
/// when self-referencing nodes were still uniqued.  If the first operand has
/// the same operands as \c Ops, return the first operand instead.
static MDNode *getOrSelfReference(LLVMContext &Context,
                                  ArrayRef<Metadata *> Ops) {
  if (!Ops.empty())
    if (MDNode *N = dyn_cast_or_null<MDNode>(Ops[0]))
      if (N->getNumOperands() == Ops.size() && N == N->getOperand(0)) {
        for (unsigned I = 1, E = Ops.size(); I != E; ++I)
          if (Ops[I] != N->getOperand(I))
            return MDNode::get(Context, Ops);
        return N;
      }

  return MDNode::get(Context, Ops);
}

MDNode *MDNode::concatenate(MDNode *A, MDNode *B) {
  if (!A)
    return B;
  if (!B)
    return A;

  SmallVector<Metadata *, 4> MDs(A->getNumOperands() + B->getNumOperands());

  unsigned j = 0;
  for (unsigned i = 0, ie = A->getNumOperands(); i != ie; ++i)
    MDs[j++] = A->getOperand(i);
  for (unsigned i = 0, ie = B->getNumOperands(); i != ie; ++i)
    MDs[j++] = B->getOperand(i);

  // FIXME: This preserves long-standing behaviour, but is it really the right
  // behaviour?  Or was that an unintended side-effect of node uniquing?
  return getOrSelfReference(A->getContext(), MDs);
}

MDNode *MDNode::intersect(MDNode *A, MDNode *B) {
  if (!A || !B)
    return nullptr;

  SmallVector<Metadata *, 4> MDs;
  for (unsigned i = 0, ie = A->getNumOperands(); i != ie; ++i) {
    Metadata *MD = A->getOperand(i);
    for (unsigned j = 0, je = B->getNumOperands(); j != je; ++j)
      if (MD == B->getOperand(j)) {
        MDs.push_back(MD);
        break;
      }
  }

  // FIXME: This preserves long-standing behaviour, but is it really the right
  // behaviour?  Or was that an unintended side-effect of node uniquing?
  return getOrSelfReference(A->getContext(), MDs);
}

MDNode *MDNode::getMostGenericAliasScope(MDNode *A, MDNode *B) {
  if (!A || !B)
    return nullptr;

  SmallVector<Metadata *, 4> MDs(B->op_begin(), B->op_end());
  for (unsigned i = 0, ie = A->getNumOperands(); i != ie; ++i) {
    Metadata *MD = A->getOperand(i);
    bool insert = true;
    for (unsigned j = 0, je = B->getNumOperands(); j != je; ++j)
      if (MD == B->getOperand(j)) {
        insert = false;
        break;
      }
    if (insert)
        MDs.push_back(MD);
  }

  // FIXME: This preserves long-standing behaviour, but is it really the right
  // behaviour?  Or was that an unintended side-effect of node uniquing?
  return getOrSelfReference(A->getContext(), MDs);
}

MDNode *MDNode::getMostGenericFPMath(MDNode *A, MDNode *B) {
  if (!A || !B)
    return nullptr;

  APFloat AVal = mdconst::extract<ConstantFP>(A->getOperand(0))->getValueAPF();
  APFloat BVal = mdconst::extract<ConstantFP>(B->getOperand(0))->getValueAPF();
  if (AVal.compare(BVal) == APFloat::cmpLessThan)
    return A;
  return B;
}

static bool isContiguous(const ConstantRange &A, const ConstantRange &B) {
  return A.getUpper() == B.getLower() || A.getLower() == B.getUpper();
}

static bool canBeMerged(const ConstantRange &A, const ConstantRange &B) {
  return !A.intersectWith(B).isEmptySet() || isContiguous(A, B);
}

static bool tryMergeRange(SmallVectorImpl<ConstantInt *> &EndPoints,
                          ConstantInt *Low, ConstantInt *High) {
  ConstantRange NewRange(Low->getValue(), High->getValue());
  unsigned Size = EndPoints.size();
  APInt LB = EndPoints[Size - 2]->getValue();
  APInt LE = EndPoints[Size - 1]->getValue();
  ConstantRange LastRange(LB, LE);
  if (canBeMerged(NewRange, LastRange)) {
    ConstantRange Union = LastRange.unionWith(NewRange);
    Type *Ty = High->getType();
    EndPoints[Size - 2] =
        cast<ConstantInt>(ConstantInt::get(Ty, Union.getLower()));
    EndPoints[Size - 1] =
        cast<ConstantInt>(ConstantInt::get(Ty, Union.getUpper()));
    return true;
  }
  return false;
}

static void addRange(SmallVectorImpl<ConstantInt *> &EndPoints,
                     ConstantInt *Low, ConstantInt *High) {
  if (!EndPoints.empty())
    if (tryMergeRange(EndPoints, Low, High))
      return;

  EndPoints.push_back(Low);
  EndPoints.push_back(High);
}

MDNode *MDNode::getMostGenericRange(MDNode *A, MDNode *B) {
  // Given two ranges, we want to compute the union of the ranges. This
  // is slightly complitade by having to combine the intervals and merge
  // the ones that overlap.

  if (!A || !B)
    return nullptr;

  if (A == B)
    return A;

  // First, walk both lists in older of the lower boundary of each interval.
  // At each step, try to merge the new interval to the last one we adedd.
  SmallVector<ConstantInt *, 4> EndPoints;
  int AI = 0;
  int BI = 0;
  int AN = A->getNumOperands() / 2;
  int BN = B->getNumOperands() / 2;
  while (AI < AN && BI < BN) {
    ConstantInt *ALow = mdconst::extract<ConstantInt>(A->getOperand(2 * AI));
    ConstantInt *BLow = mdconst::extract<ConstantInt>(B->getOperand(2 * BI));

    if (ALow->getValue().slt(BLow->getValue())) {
      addRange(EndPoints, ALow,
               mdconst::extract<ConstantInt>(A->getOperand(2 * AI + 1)));
      ++AI;
    } else {
      addRange(EndPoints, BLow,
               mdconst::extract<ConstantInt>(B->getOperand(2 * BI + 1)));
      ++BI;
    }
  }
  while (AI < AN) {
    addRange(EndPoints, mdconst::extract<ConstantInt>(A->getOperand(2 * AI)),
             mdconst::extract<ConstantInt>(A->getOperand(2 * AI + 1)));
    ++AI;
  }
  while (BI < BN) {
    addRange(EndPoints, mdconst::extract<ConstantInt>(B->getOperand(2 * BI)),
             mdconst::extract<ConstantInt>(B->getOperand(2 * BI + 1)));
    ++BI;
  }

  // If we have more than 2 ranges (4 endpoints) we have to try to merge
  // the last and first ones.
  unsigned Size = EndPoints.size();
  if (Size > 4) {
    ConstantInt *FB = EndPoints[0];
    ConstantInt *FE = EndPoints[1];
    if (tryMergeRange(EndPoints, FB, FE)) {
      for (unsigned i = 0; i < Size - 2; ++i) {
        EndPoints[i] = EndPoints[i + 2];
      }
      EndPoints.resize(Size - 2);
    }
  }

  // If in the end we have a single range, it is possible that it is now the
  // full range. Just drop the metadata in that case.
  if (EndPoints.size() == 2) {
    ConstantRange Range(EndPoints[0]->getValue(), EndPoints[1]->getValue());
    if (Range.isFullSet())
      return nullptr;
  }

  SmallVector<Metadata *, 4> MDs;
  MDs.reserve(EndPoints.size());
  for (auto *I : EndPoints)
    MDs.push_back(ConstantAsMetadata::get(I));
  return MDNode::get(A->getContext(), MDs);
}

//===----------------------------------------------------------------------===//
// NamedMDNode implementation.
//

static SmallVector<TrackingMDRef, 4> &getNMDOps(void *Operands) {
  return *(SmallVector<TrackingMDRef, 4> *)Operands;
}

NamedMDNode::NamedMDNode(const Twine &N)
    : Name(N.str()), Parent(nullptr),
      Operands(new SmallVector<TrackingMDRef, 4>()) {}

NamedMDNode::~NamedMDNode() {
  dropAllReferences();
  delete &getNMDOps(Operands);
}

unsigned NamedMDNode::getNumOperands() const {
  return (unsigned)getNMDOps(Operands).size();
}

MDNode *NamedMDNode::getOperand(unsigned i) const {
  assert(i < getNumOperands() && "Invalid Operand number!");
  auto *N = getNMDOps(Operands)[i].get();
  return cast_or_null<MDNode>(N);
}

void NamedMDNode::addOperand(MDNode *M) { getNMDOps(Operands).emplace_back(M); }

void NamedMDNode::setOperand(unsigned I, MDNode *New) {
  assert(I < getNumOperands() && "Invalid operand number");
  getNMDOps(Operands)[I].reset(New);
}

void NamedMDNode::eraseFromParent() {
  getParent()->eraseNamedMetadata(this);
}

void NamedMDNode::dropAllReferences() {
  getNMDOps(Operands).clear();
}

StringRef NamedMDNode::getName() const {
  return StringRef(Name);
}

//===----------------------------------------------------------------------===//
// Instruction Metadata method implementations.
//

void Instruction::setMetadata(StringRef Kind, MDNode *Node) {
  if (!Node && !hasMetadata())
    return;
  setMetadata(getContext().getMDKindID(Kind), Node);
}

MDNode *Instruction::getMetadataImpl(StringRef Kind) const {
  return getMetadataImpl(getContext().getMDKindID(Kind));
}

void Instruction::dropUnknownMetadata(ArrayRef<unsigned> KnownIDs) {
  SmallSet<unsigned, 5> KnownSet;
  KnownSet.insert(KnownIDs.begin(), KnownIDs.end());

  // Drop debug if needed
  if (KnownSet.erase(LLVMContext::MD_dbg))
    DbgLoc = DebugLoc();

  if (!hasMetadataHashEntry())
    return; // Nothing to remove!

  DenseMap<const Instruction *, LLVMContextImpl::MDMapTy> &MetadataStore =
      getContext().pImpl->MetadataStore;

  if (KnownSet.empty()) {
    // Just drop our entry at the store.
    MetadataStore.erase(this);
    setHasMetadataHashEntry(false);
    return;
  }

  LLVMContextImpl::MDMapTy &Info = MetadataStore[this];
  unsigned I;
  unsigned E;
  // Walk the array and drop any metadata we don't know.
  for (I = 0, E = Info.size(); I != E;) {
    if (KnownSet.count(Info[I].first)) {
      ++I;
      continue;
    }

    Info[I] = std::move(Info.back());
    Info.pop_back();
    --E;
  }
  assert(E == Info.size());

  if (E == 0) {
    // Drop our entry at the store.
    MetadataStore.erase(this);
    setHasMetadataHashEntry(false);
  }
}

/// setMetadata - Set the metadata of of the specified kind to the specified
/// node.  This updates/replaces metadata if already present, or removes it if
/// Node is null.
void Instruction::setMetadata(unsigned KindID, MDNode *Node) {
  if (!Node && !hasMetadata())
    return;

  // Handle 'dbg' as a special case since it is not stored in the hash table.
  if (KindID == LLVMContext::MD_dbg) {
    DbgLoc = DebugLoc::getFromDILocation(Node);
    return;
  }
  
  // Handle the case when we're adding/updating metadata on an instruction.
  if (Node) {
    LLVMContextImpl::MDMapTy &Info = getContext().pImpl->MetadataStore[this];
    assert(!Info.empty() == hasMetadataHashEntry() &&
           "HasMetadata bit is wonked");
    if (Info.empty()) {
      setHasMetadataHashEntry(true);
    } else {
      // Handle replacement of an existing value.
      for (auto &P : Info)
        if (P.first == KindID) {
          P.second.reset(Node);
          return;
        }
    }

    // No replacement, just add it to the list.
    Info.emplace_back(std::piecewise_construct, std::make_tuple(KindID),
                      std::make_tuple(Node));
    return;
  }

  // Otherwise, we're removing metadata from an instruction.
  assert((hasMetadataHashEntry() ==
          (getContext().pImpl->MetadataStore.count(this) > 0)) &&
         "HasMetadata bit out of date!");
  if (!hasMetadataHashEntry())
    return;  // Nothing to remove!
  LLVMContextImpl::MDMapTy &Info = getContext().pImpl->MetadataStore[this];

  // Common case is removing the only entry.
  if (Info.size() == 1 && Info[0].first == KindID) {
    getContext().pImpl->MetadataStore.erase(this);
    setHasMetadataHashEntry(false);
    return;
  }

  // Handle removal of an existing value.
  for (unsigned i = 0, e = Info.size(); i != e; ++i)
    if (Info[i].first == KindID) {
      Info[i] = std::move(Info.back());
      Info.pop_back();
      assert(!Info.empty() && "Removing last entry should be handled above");
      return;
    }
  // Otherwise, removing an entry that doesn't exist on the instruction.
}

void Instruction::setAAMetadata(const AAMDNodes &N) {
  setMetadata(LLVMContext::MD_tbaa, N.TBAA);
  setMetadata(LLVMContext::MD_alias_scope, N.Scope);
  setMetadata(LLVMContext::MD_noalias, N.NoAlias);
}

MDNode *Instruction::getMetadataImpl(unsigned KindID) const {
  // Handle 'dbg' as a special case since it is not stored in the hash table.
  if (KindID == LLVMContext::MD_dbg)
    return DbgLoc.getAsMDNode();

  if (!hasMetadataHashEntry()) return nullptr;
  
  LLVMContextImpl::MDMapTy &Info = getContext().pImpl->MetadataStore[this];
  assert(!Info.empty() && "bit out of sync with hash table");

  for (const auto &I : Info)
    if (I.first == KindID)
      return I.second;
  return nullptr;
}

void Instruction::getAllMetadataImpl(
    SmallVectorImpl<std::pair<unsigned, MDNode *>> &Result) const {
  Result.clear();
  
  // Handle 'dbg' as a special case since it is not stored in the hash table.
  if (!DbgLoc.isUnknown()) {
    Result.push_back(
        std::make_pair((unsigned)LLVMContext::MD_dbg, DbgLoc.getAsMDNode()));
    if (!hasMetadataHashEntry()) return;
  }
  
  assert(hasMetadataHashEntry() &&
         getContext().pImpl->MetadataStore.count(this) &&
         "Shouldn't have called this");
  const LLVMContextImpl::MDMapTy &Info =
    getContext().pImpl->MetadataStore.find(this)->second;
  assert(!Info.empty() && "Shouldn't have called this");

  Result.reserve(Result.size() + Info.size());
  for (auto &I : Info)
    Result.push_back(std::make_pair(I.first, cast<MDNode>(I.second.get())));

  // Sort the resulting array so it is stable.
  if (Result.size() > 1)
    array_pod_sort(Result.begin(), Result.end());
}

void Instruction::getAllMetadataOtherThanDebugLocImpl(
    SmallVectorImpl<std::pair<unsigned, MDNode *>> &Result) const {
  Result.clear();
  assert(hasMetadataHashEntry() &&
         getContext().pImpl->MetadataStore.count(this) &&
         "Shouldn't have called this");
  const LLVMContextImpl::MDMapTy &Info =
    getContext().pImpl->MetadataStore.find(this)->second;
  assert(!Info.empty() && "Shouldn't have called this");
  Result.reserve(Result.size() + Info.size());
  for (auto &I : Info)
    Result.push_back(std::make_pair(I.first, cast<MDNode>(I.second.get())));

  // Sort the resulting array so it is stable.
  if (Result.size() > 1)
    array_pod_sort(Result.begin(), Result.end());
}

/// clearMetadataHashEntries - Clear all hashtable-based metadata from
/// this instruction.
void Instruction::clearMetadataHashEntries() {
  assert(hasMetadataHashEntry() && "Caller should check");
  getContext().pImpl->MetadataStore.erase(this);
  setHasMetadataHashEntry(false);
}
