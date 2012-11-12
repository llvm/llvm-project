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
#include "llvm/ADT/STLExtras.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/LeakDetector.h"
#include "llvm/Support/ValueHandle.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// MDString implementation.
//

void MDString::anchor() { }

MDString::MDString(LLVMContext &C)
  : Value(Type::getMetadataTy(C), Value::MDStringVal) {}

MDString *MDString::get(LLVMContext &Context, StringRef Str) {
  LLVMContextImpl *pImpl = Context.pImpl;
  StringMapEntry<Value*> &Entry =
    pImpl->MDStringCache.GetOrCreateValue(Str);
  Value *&S = Entry.getValue();
  if (!S) S = new MDString(Context);
  S->setValueName(&Entry);
  return cast<MDString>(S);
}

//===----------------------------------------------------------------------===//
// MDNodeOperand implementation.
//

// Use CallbackVH to hold MDNode operands.
namespace llvm {
class MDNodeOperand : public CallbackVH {
  MDNode *getParent() {
    MDNodeOperand *Cur = this;

    while (Cur->getValPtrInt() != 1)
      --Cur;

    assert(Cur->getValPtrInt() == 1 &&
           "Couldn't find the beginning of the operand list!");
    return reinterpret_cast<MDNode*>(Cur) - 1;
  }

public:
  MDNodeOperand(Value *V) : CallbackVH(V) {}
  ~MDNodeOperand() {}

  void set(Value *V) {
    unsigned IsFirst = this->getValPtrInt();
    this->setValPtr(V);
    this->setAsFirstOperand(IsFirst);
  }

  /// setAsFirstOperand - Accessor method to mark the operand as the first in
  /// the list.
  void setAsFirstOperand(unsigned V) { this->setValPtrInt(V); }

  virtual void deleted();
  virtual void allUsesReplacedWith(Value *NV);
};
} // end namespace llvm.


void MDNodeOperand::deleted() {
  getParent()->replaceOperand(this, 0);
}

void MDNodeOperand::allUsesReplacedWith(Value *NV) {
  getParent()->replaceOperand(this, NV);
}

//===----------------------------------------------------------------------===//
// MDNode implementation.
//

/// getOperandPtr - Helper function to get the MDNodeOperand's coallocated on
/// the end of the MDNode.
static MDNodeOperand *getOperandPtr(MDNode *N, unsigned Op) {
  // Use <= instead of < to permit a one-past-the-end address.
  assert(Op <= N->getNumOperands() && "Invalid operand number");
  return reinterpret_cast<MDNodeOperand*>(N + 1) + Op;
}

void MDNode::replaceOperandWith(unsigned i, Value *Val) {
  MDNodeOperand *Op = getOperandPtr(this, i);
  replaceOperand(Op, Val);
}

MDNode::MDNode(LLVMContext &C, ArrayRef<Value*> Vals, bool isFunctionLocal)
: Value(Type::getMetadataTy(C), Value::MDNodeVal) {
  NumOperands = Vals.size();

  if (isFunctionLocal)
    setValueSubclassData(getSubclassDataFromValue() | FunctionLocalBit);

  // Initialize the operand list, which is co-allocated on the end of the node.
  unsigned i = 0;
  for (MDNodeOperand *Op = getOperandPtr(this, 0), *E = Op+NumOperands;
       Op != E; ++Op, ++i) {
    new (Op) MDNodeOperand(Vals[i]);

    // Mark the first MDNodeOperand as being the first in the list of operands.
    if (i == 0)
      Op->setAsFirstOperand(1);
  }
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
#else
  if (!isFunctionLocal()) return NULL;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (const Function *F = getFunctionForValue(getOperand(i)))
      return F;
  return NULL;
#endif
}

// destroy - Delete this node.  Only when there are no uses.
void MDNode::destroy() {
  setValueSubclassData(getSubclassDataFromValue() | DestroyFlag);
  // Placement delete, then free the memory.
  this->~MDNode();
  free(this);
}

/// isFunctionLocalValue - Return true if this is a value that would require a
/// function-local MDNode.
static bool isFunctionLocalValue(Value *V) {
  return isa<Instruction>(V) || isa<Argument>(V) || isa<BasicBlock>(V) ||
         (isa<MDNode>(V) && cast<MDNode>(V)->isFunctionLocal());
}

MDNode *MDNode::getMDNode(LLVMContext &Context, ArrayRef<Value*> Vals,
                          FunctionLocalness FL, bool Insert) {
  LLVMContextImpl *pImpl = Context.pImpl;

  // Add all the operand pointers. Note that we don't have to add the
  // isFunctionLocal bit because that's implied by the operands.
  // Note that if the operands are later nulled out, the node will be
  // removed from the uniquing map.
  FoldingSetNodeID ID;
  for (unsigned i = 0; i != Vals.size(); ++i)
    ID.AddPointer(Vals[i]);

  void *InsertPoint;
  MDNode *N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (N || !Insert)
    return N;

  bool isFunctionLocal = false;
  switch (FL) {
  case FL_Unknown:
    for (unsigned i = 0; i != Vals.size(); ++i) {
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
  void *Ptr = malloc(sizeof(MDNode) + Vals.size() * sizeof(MDNodeOperand));
  N = new (Ptr) MDNode(Context, Vals, isFunctionLocal);

  // Cache the operand hash.
  N->Hash = ID.ComputeHash();

  // InsertPoint will have been set by the FindNodeOrInsertPos call.
  pImpl->MDNodeSet.InsertNode(N, InsertPoint);

  return N;
}

MDNode *MDNode::get(LLVMContext &Context, ArrayRef<Value*> Vals) {
  return getMDNode(Context, Vals, FL_Unknown);
}

MDNode *MDNode::getWhenValsUnresolved(LLVMContext &Context,
                                      ArrayRef<Value*> Vals,
                                      bool isFunctionLocal) {
  return getMDNode(Context, Vals, isFunctionLocal ? FL_Yes : FL_No);
}

MDNode *MDNode::getIfExists(LLVMContext &Context, ArrayRef<Value*> Vals) {
  return getMDNode(Context, Vals, FL_Unknown, false);
}

MDNode *MDNode::getTemporary(LLVMContext &Context, ArrayRef<Value*> Vals) {
  MDNode *N =
    (MDNode *)malloc(sizeof(MDNode) + Vals.size() * sizeof(MDNodeOperand));
  N = new (N) MDNode(Context, Vals, FL_No);
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
  // in the set.  If so, then this node is redundant.
  FoldingSetNodeID ID;
  Profile(ID);
  void *InsertPoint;
  if (MDNode *N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint)) {
    replaceAllUsesWith(N);
    destroy();
    return;
  }

  // Cache the operand hash.
  Hash = ID.ComputeHash();
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

MDNode *MDNode::getMostGenericTBAA(MDNode *A, MDNode *B) {
  if (!A || !B)
    return NULL;

  if (A == B)
    return A;

  SmallVector<MDNode *, 4> PathA;
  MDNode *T = A;
  while (T) {
    PathA.push_back(T);
    T = T->getNumOperands() >= 2 ? cast_or_null<MDNode>(T->getOperand(1)) : 0;
  }

  SmallVector<MDNode *, 4> PathB;
  T = B;
  while (T) {
    PathB.push_back(T);
    T = T->getNumOperands() >= 2 ? cast_or_null<MDNode>(T->getOperand(1)) : 0;
  }

  int IA = PathA.size() - 1;
  int IB = PathB.size() - 1;

  MDNode *Ret = 0;
  while (IA >= 0 && IB >=0) {
    if (PathA[IA] == PathB[IB])
      Ret = PathA[IA];
    else
      break;
    --IA;
    --IB;
  }
  return Ret;
}

MDNode *MDNode::getMostGenericFPMath(MDNode *A, MDNode *B) {
  if (!A || !B)
    return NULL;

  APFloat AVal = cast<ConstantFP>(A->getOperand(0))->getValueAPF();
  APFloat BVal = cast<ConstantFP>(B->getOperand(0))->getValueAPF();
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

static bool tryMergeRange(SmallVector<Value*, 4> &EndPoints, ConstantInt *Low,
                          ConstantInt *High) {
  ConstantRange NewRange(Low->getValue(), High->getValue());
  unsigned Size = EndPoints.size();
  APInt LB = cast<ConstantInt>(EndPoints[Size - 2])->getValue();
  APInt LE = cast<ConstantInt>(EndPoints[Size - 1])->getValue();
  ConstantRange LastRange(LB, LE);
  if (canBeMerged(NewRange, LastRange)) {
    ConstantRange Union = LastRange.unionWith(NewRange);
    Type *Ty = High->getType();
    EndPoints[Size - 2] = ConstantInt::get(Ty, Union.getLower());
    EndPoints[Size - 1] = ConstantInt::get(Ty, Union.getUpper());
    return true;
  }
  return false;
}

static void addRange(SmallVector<Value*, 4> &EndPoints, ConstantInt *Low,
                     ConstantInt *High) {
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
    return NULL;

  if (A == B)
    return A;

  // First, walk both lists in older of the lower boundary of each interval.
  // At each step, try to merge the new interval to the last one we adedd.
  SmallVector<Value*, 4> EndPoints;
  int AI = 0;
  int BI = 0;
  int AN = A->getNumOperands() / 2;
  int BN = B->getNumOperands() / 2;
  while (AI < AN && BI < BN) {
    ConstantInt *ALow = cast<ConstantInt>(A->getOperand(2 * AI));
    ConstantInt *BLow = cast<ConstantInt>(B->getOperand(2 * BI));

    if (ALow->getValue().slt(BLow->getValue())) {
      addRange(EndPoints, ALow, cast<ConstantInt>(A->getOperand(2 * AI + 1)));
      ++AI;
    } else {
      addRange(EndPoints, BLow, cast<ConstantInt>(B->getOperand(2 * BI + 1)));
      ++BI;
    }
  }
  while (AI < AN) {
    addRange(EndPoints, cast<ConstantInt>(A->getOperand(2 * AI)),
             cast<ConstantInt>(A->getOperand(2 * AI + 1)));
    ++AI;
  }
  while (BI < BN) {
    addRange(EndPoints, cast<ConstantInt>(B->getOperand(2 * BI)),
             cast<ConstantInt>(B->getOperand(2 * BI + 1)));
    ++BI;
  }

  // If we have more than 2 ranges (4 endpoints) we have to try to merge
  // the last and first ones.
  unsigned Size = EndPoints.size();
  if (Size > 4) {
    ConstantInt *FB = cast<ConstantInt>(EndPoints[0]);
    ConstantInt *FE = cast<ConstantInt>(EndPoints[1]);
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
    ConstantRange Range(cast<ConstantInt>(EndPoints[0])->getValue(),
                        cast<ConstantInt>(EndPoints[1])->getValue());
    if (Range.isFullSet())
      return NULL;
  }

  return MDNode::get(A->getContext(), EndPoints);
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

void Instruction::setMetadata(StringRef Kind, MDNode *Node) {
  if (Node == 0 && !hasMetadata()) return;
  setMetadata(getContext().getMDKindID(Kind), Node);
}

MDNode *Instruction::getMetadataImpl(StringRef Kind) const {
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
  assert((hasMetadataHashEntry() ==
          getContext().pImpl->MetadataStore.count(this)) &&
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

