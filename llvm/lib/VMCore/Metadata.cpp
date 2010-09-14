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

#include "llvm/Metadata.h"
#include "LLVMContextImpl.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Instruction.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallString.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/Support/LeakDetector.h"
#include "llvm/Support/ValueHandle.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// MDString implementation.
//

MDString::MDString(LLVMContext &C, StringRef S)
  : Value(Type::getMetadataTy(C), Value::MDStringVal), Str(S) {}

MDString *MDString::get(LLVMContext &Context, StringRef Str) {
  LLVMContextImpl *pImpl = Context.pImpl;
  StringMapEntry<MDString *> &Entry =
    pImpl->MDStringCache.GetOrCreateValue(Str);
  MDString *&S = Entry.getValue();
  if (!S) S = new MDString(Context, Entry.getKey());
  return S;
}

//===----------------------------------------------------------------------===//
// MDNodeOperand implementation.
//

// Use CallbackVH to hold MDNode operands.
namespace llvm {
class MDNodeOperand : public CallbackVH {
  MDNode *Parent;
public:
  MDNodeOperand(Value *V, MDNode *P) : CallbackVH(V), Parent(P) {}
  ~MDNodeOperand() {}

  void set(Value *V) {
    setValPtr(V);
  }

  virtual void deleted();
  virtual void allUsesReplacedWith(Value *NV);
};
} // end namespace llvm.


void MDNodeOperand::deleted() {
  Parent->replaceOperand(this, 0);
}

void MDNodeOperand::allUsesReplacedWith(Value *NV) {
  Parent->replaceOperand(this, NV);
}



//===----------------------------------------------------------------------===//
// MDNode implementation.
//

/// getOperandPtr - Helper function to get the MDNodeOperand's coallocated on
/// the end of the MDNode.
static MDNodeOperand *getOperandPtr(MDNode *N, unsigned Op) {
  // Use <= instead of < to permit a one-past-the-end address.
  assert(Op <= N->getNumOperands() && "Invalid operand number");
  return reinterpret_cast<MDNodeOperand*>(N+1)+Op;
}

MDNode::MDNode(LLVMContext &C, Value *const *Vals, unsigned NumVals,
               bool isFunctionLocal)
: Value(Type::getMetadataTy(C), Value::MDNodeVal) {
  NumOperands = NumVals;

  if (isFunctionLocal)
    setValueSubclassData(getSubclassDataFromValue() | FunctionLocalBit);

  // Initialize the operand list, which is co-allocated on the end of the node.
  for (MDNodeOperand *Op = getOperandPtr(this, 0), *E = Op+NumOperands;
       Op != E; ++Op, ++Vals)
    new (Op) MDNodeOperand(*Vals, this);
}


/// ~MDNode - Destroy MDNode.
MDNode::~MDNode() {
  assert((getSubclassDataFromValue() & DestroyFlag) != 0 &&
         "Not being destroyed through destroy()?");
  LLVMContextImpl *pImpl = getType()->getContext().pImpl;
  if (isNotUniqued()) {
    pImpl->NonUniquedMDNodes.erase(this);
  } else {
    pImpl->MDNodeSet.RemoveNode(this);
  }

  // Destroy the operands.
  for (MDNodeOperand *Op = getOperandPtr(this, 0), *E = Op+NumOperands;
       Op != E; ++Op)
    Op->~MDNodeOperand();
}

static const Function *getFunctionForValue(Value *V) {
  if (!V) return NULL;
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    BasicBlock *BB = I->getParent();
    return BB ? BB->getParent() : 0;
  }
  if (Argument *A = dyn_cast<Argument>(V))
    return A->getParent();
  if (BasicBlock *BB = dyn_cast<BasicBlock>(V))
    return BB->getParent();
  if (MDNode *MD = dyn_cast<MDNode>(V))
    return MD->getFunction();
  return NULL;
}

#ifndef NDEBUG
static const Function *assertLocalFunction(const MDNode *N) {
  if (!N->isFunctionLocal()) return 0;

  // FIXME: This does not handle cyclic function local metadata.
  const Function *F = 0, *NewF = 0;
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    if (Value *V = N->getOperand(i)) {
      if (MDNode *MD = dyn_cast<MDNode>(V))
        NewF = assertLocalFunction(MD);
      else
        NewF = getFunctionForValue(V);
    }
    if (F == 0)
      F = NewF;
    else 
      assert((NewF == 0 || F == NewF) &&"inconsistent function-local metadata");
  }
  return F;
}
#endif

// getFunction - If this metadata is function-local and recursively has a
// function-local operand, return the first such operand's parent function.
// Otherwise, return null. getFunction() should not be used for performance-
// critical code because it recursively visits all the MDNode's operands.  
const Function *MDNode::getFunction() const {
#ifndef NDEBUG
  return assertLocalFunction(this);
#endif
  if (!isFunctionLocal()) return NULL;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (const Function *F = getFunctionForValue(getOperand(i)))
      return F;
  return NULL;
}

// destroy - Delete this node.  Only when there are no uses.
void MDNode::destroy() {
  setValueSubclassData(getSubclassDataFromValue() | DestroyFlag);
  // Placement delete, the free the memory.
  this->~MDNode();
  free(this);
}

/// isFunctionLocalValue - Return true if this is a value that would require a
/// function-local MDNode.
static bool isFunctionLocalValue(Value *V) {
  return isa<Instruction>(V) || isa<Argument>(V) || isa<BasicBlock>(V) ||
         (isa<MDNode>(V) && cast<MDNode>(V)->isFunctionLocal());
}

MDNode *MDNode::getMDNode(LLVMContext &Context, Value *const *Vals,
                          unsigned NumVals, FunctionLocalness FL,
                          bool Insert) {
  LLVMContextImpl *pImpl = Context.pImpl;

  // Add all the operand pointers. Note that we don't have to add the
  // isFunctionLocal bit because that's implied by the operands.
  // Note that if the operands are later nulled out, the node will be
  // removed from the uniquing map.
  FoldingSetNodeID ID;
  for (unsigned i = 0; i != NumVals; ++i)
    ID.AddPointer(Vals[i]);

  void *InsertPoint;
  MDNode *N = NULL;
  
  if ((N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint)))
    return N;
    
  bool isFunctionLocal = false;
  switch (FL) {
  case FL_Unknown:
    for (unsigned i = 0; i != NumVals; ++i) {
      Value *V = Vals[i];
      if (!V) continue;
      if (isFunctionLocalValue(V)) {
        isFunctionLocal = true;
        break;
      }
    }
    break;
  case FL_No:
    isFunctionLocal = false;
    break;
  case FL_Yes:
    isFunctionLocal = true;
    break;
  }

  // Coallocate space for the node and Operands together, then placement new.
  void *Ptr = malloc(sizeof(MDNode)+NumVals*sizeof(MDNodeOperand));
  N = new (Ptr) MDNode(Context, Vals, NumVals, isFunctionLocal);

  // InsertPoint will have been set by the FindNodeOrInsertPos call.
  pImpl->MDNodeSet.InsertNode(N, InsertPoint);

  return N;
}

MDNode *MDNode::get(LLVMContext &Context, Value*const* Vals, unsigned NumVals) {
  return getMDNode(Context, Vals, NumVals, FL_Unknown);
}

MDNode *MDNode::getWhenValsUnresolved(LLVMContext &Context, Value *const *Vals,
                                      unsigned NumVals, bool isFunctionLocal) {
  return getMDNode(Context, Vals, NumVals, isFunctionLocal ? FL_Yes : FL_No);
}

MDNode *MDNode::getIfExists(LLVMContext &Context, Value *const *Vals,
                            unsigned NumVals) {
  return getMDNode(Context, Vals, NumVals, FL_Unknown, false);
}

MDNode *MDNode::getTemporary(LLVMContext &Context, Value *const *Vals,
                             unsigned NumVals) {
  MDNode *N = (MDNode *)malloc(sizeof(MDNode)+NumVals*sizeof(MDNodeOperand));
  N = new (N) MDNode(Context, Vals, NumVals, FL_No);
  N->setValueSubclassData(N->getSubclassDataFromValue() |
                          NotUniquedBit);
  LeakDetector::addGarbageObject(N);
  return N;
}

void MDNode::deleteTemporary(MDNode *N) {
  assert(N->use_empty() && "Temporary MDNode has uses!");
  assert(!N->getContext().pImpl->MDNodeSet.RemoveNode(N) &&
         "Deleting a non-temporary uniqued node!");
  assert(!N->getContext().pImpl->NonUniquedMDNodes.erase(N) &&
         "Deleting a non-temporary non-uniqued node!");
  assert((N->getSubclassDataFromValue() & NotUniquedBit) &&
         "Temporary MDNode does not have NotUniquedBit set!");
  assert((N->getSubclassDataFromValue() & DestroyFlag) == 0 &&
         "Temporary MDNode has DestroyFlag set!");
  LeakDetector::removeGarbageObject(N);
  N->destroy();
}

/// getOperand - Return specified operand.
Value *MDNode::getOperand(unsigned i) const {
  return *getOperandPtr(const_cast<MDNode*>(this), i);
}

void MDNode::Profile(FoldingSetNodeID &ID) const {
  // Add all the operand pointers. Note that we don't have to add the
  // isFunctionLocal bit because that's implied by the operands.
  // Note that if the operands are later nulled out, the node will be
  // removed from the uniquing map.
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    ID.AddPointer(getOperand(i));
}

void MDNode::setIsNotUniqued() {
  setValueSubclassData(getSubclassDataFromValue() | NotUniquedBit);
  LLVMContextImpl *pImpl = getType()->getContext().pImpl;
  pImpl->NonUniquedMDNodes.insert(this);
}

// Replace value from this node's operand list.
void MDNode::replaceOperand(MDNodeOperand *Op, Value *To) {
  Value *From = *Op;

  // If is possible that someone did GV->RAUW(inst), replacing a global variable
  // with an instruction or some other function-local object.  If this is a
  // non-function-local MDNode, it can't point to a function-local object.
  // Handle this case by implicitly dropping the MDNode reference to null.
  // Likewise if the MDNode is function-local but for a different function.
  if (To && isFunctionLocalValue(To)) {
    if (!isFunctionLocal())
      To = 0;
    else {
      const Function *F = getFunction();
      const Function *FV = getFunctionForValue(To);
      // Metadata can be function-local without having an associated function.
      // So only consider functions to have changed if non-null.
      if (F && FV && F != FV)
        To = 0;
    }
  }
  
  if (From == To)
    return;

  // Update the operand.
  Op->set(To);

  // If this node is already not being uniqued (because one of the operands
  // already went to null), then there is nothing else to do here.
  if (isNotUniqued()) return;

  LLVMContextImpl *pImpl = getType()->getContext().pImpl;

  // Remove "this" from the context map.  FoldingSet doesn't have to reprofile
  // this node to remove it, so we don't care what state the operands are in.
  pImpl->MDNodeSet.RemoveNode(this);

  // If we are dropping an argument to null, we choose to not unique the MDNode
  // anymore.  This commonly occurs during destruction, and uniquing these
  // brings little reuse.  Also, this means we don't need to include
  // isFunctionLocal bits in FoldingSetNodeIDs for MDNodes.
  if (To == 0) {
    setIsNotUniqued();
    return;
  }

  // Now that the node is out of the folding set, get ready to reinsert it.
  // First, check to see if another node with the same operands already exists
  // in the set.  If it doesn't exist, this returns the position to insert it.
  FoldingSetNodeID ID;
  Profile(ID);
  void *InsertPoint;
  MDNode *N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (N) {
    N->replaceAllUsesWith(this);
    N->destroy();
    N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
    assert(N == 0 && "shouldn't be in the map now!"); (void)N;
  }

  // InsertPoint will have been set by the FindNodeOrInsertPos call.
  pImpl->MDNodeSet.InsertNode(this, InsertPoint);

  // If this MDValue was previously function-local but no longer is, clear
  // its function-local flag.
  if (isFunctionLocal() && !isFunctionLocalValue(To)) {
    bool isStillFunctionLocal = false;
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
      Value *V = getOperand(i);
      if (!V) continue;
      if (isFunctionLocalValue(V)) {
        isStillFunctionLocal = true;
        break;
      }
    }
    if (!isStillFunctionLocal)
      setValueSubclassData(getSubclassDataFromValue() & ~FunctionLocalBit);
  }
}

//===----------------------------------------------------------------------===//
// NamedMDNode implementation.
//

static SmallVector<TrackingVH<MDNode>, 4> &getNMDOps(void *Operands) {
  return *(SmallVector<TrackingVH<MDNode>, 4>*)Operands;
}

NamedMDNode::NamedMDNode(const Twine &N)
  : Name(N.str()), Parent(0),
    Operands(new SmallVector<TrackingVH<MDNode>, 4>()) {
}

NamedMDNode::~NamedMDNode() {
  dropAllReferences();
  delete &getNMDOps(Operands);
}

/// getNumOperands - Return number of NamedMDNode operands.
unsigned NamedMDNode::getNumOperands() const {
  return (unsigned)getNMDOps(Operands).size();
}

/// getOperand - Return specified operand.
MDNode *NamedMDNode::getOperand(unsigned i) const {
  assert(i < getNumOperands() && "Invalid Operand number!");
  return dyn_cast<MDNode>(&*getNMDOps(Operands)[i]);
}

/// addOperand - Add metadata Operand.
void NamedMDNode::addOperand(MDNode *M) {
  assert(!M->isFunctionLocal() &&
         "NamedMDNode operands must not be function-local!");
  getNMDOps(Operands).push_back(TrackingVH<MDNode>(M));
}

/// eraseFromParent - Drop all references and remove the node from parent
/// module.
void NamedMDNode::eraseFromParent() {
  getParent()->eraseNamedMetadata(this);
}

/// dropAllReferences - Remove all uses and clear node vector.
void NamedMDNode::dropAllReferences() {
  getNMDOps(Operands).clear();
}

/// getName - Return a constant reference to this named metadata's name.
StringRef NamedMDNode::getName() const {
  return StringRef(Name);
}

//===----------------------------------------------------------------------===//
// Instruction Metadata method implementations.
//

void Instruction::setMetadata(const char *Kind, MDNode *Node) {
  if (Node == 0 && !hasMetadata()) return;
  setMetadata(getContext().getMDKindID(Kind), Node);
}

MDNode *Instruction::getMetadataImpl(const char *Kind) const {
  return getMetadataImpl(getContext().getMDKindID(Kind));
}

/// setMetadata - Set the metadata of of the specified kind to the specified
/// node.  This updates/replaces metadata if already present, or removes it if
/// Node is null.
void Instruction::setMetadata(unsigned KindID, MDNode *Node) {
  if (Node == 0 && !hasMetadata()) return;

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
      for (unsigned i = 0, e = Info.size(); i != e; ++i)
        if (Info[i].first == KindID) {
          Info[i].second = Node;
          return;
        }
    }

    // No replacement, just add it to the list.
    Info.push_back(std::make_pair(KindID, Node));
    return;
  }

  // Otherwise, we're removing metadata from an instruction.
  assert(hasMetadataHashEntry() &&
         getContext().pImpl->MetadataStore.count(this) &&
         "HasMetadata bit out of date!");
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
      Info[i] = Info.back();
      Info.pop_back();
      assert(!Info.empty() && "Removing last entry should be handled above");
      return;
    }
  // Otherwise, removing an entry that doesn't exist on the instruction.
}

MDNode *Instruction::getMetadataImpl(unsigned KindID) const {
  // Handle 'dbg' as a special case since it is not stored in the hash table.
  if (KindID == LLVMContext::MD_dbg)
    return DbgLoc.getAsMDNode(getContext());
  
  if (!hasMetadataHashEntry()) return 0;
  
  LLVMContextImpl::MDMapTy &Info = getContext().pImpl->MetadataStore[this];
  assert(!Info.empty() && "bit out of sync with hash table");

  for (LLVMContextImpl::MDMapTy::iterator I = Info.begin(), E = Info.end();
       I != E; ++I)
    if (I->first == KindID)
      return I->second;
  return 0;
}

void Instruction::getAllMetadataImpl(SmallVectorImpl<std::pair<unsigned,
                                       MDNode*> > &Result) const {
  Result.clear();
  
  // Handle 'dbg' as a special case since it is not stored in the hash table.
  if (!DbgLoc.isUnknown()) {
    Result.push_back(std::make_pair((unsigned)LLVMContext::MD_dbg,
                                    DbgLoc.getAsMDNode(getContext())));
    if (!hasMetadataHashEntry()) return;
  }
  
  assert(hasMetadataHashEntry() &&
         getContext().pImpl->MetadataStore.count(this) &&
         "Shouldn't have called this");
  const LLVMContextImpl::MDMapTy &Info =
    getContext().pImpl->MetadataStore.find(this)->second;
  assert(!Info.empty() && "Shouldn't have called this");

  Result.append(Info.begin(), Info.end());

  // Sort the resulting array so it is stable.
  if (Result.size() > 1)
    array_pod_sort(Result.begin(), Result.end());
}

void Instruction::
getAllMetadataOtherThanDebugLocImpl(SmallVectorImpl<std::pair<unsigned,
                                    MDNode*> > &Result) const {
  Result.clear();
  assert(hasMetadataHashEntry() &&
         getContext().pImpl->MetadataStore.count(this) &&
         "Shouldn't have called this");
  const LLVMContextImpl::MDMapTy &Info =
  getContext().pImpl->MetadataStore.find(this)->second;
  assert(!Info.empty() && "Shouldn't have called this");
  
  Result.append(Info.begin(), Info.end());
  
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

