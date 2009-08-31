//===-- Instructions.cpp - Implement the LLVM instructions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements all of the non-inline methods for the LLVM instruction
// classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Operator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                            CallSite Class
//===----------------------------------------------------------------------===//

#define CALLSITE_DELEGATE_GETTER(METHOD) \
  Instruction *II(getInstruction());     \
  return isCall()                        \
    ? cast<CallInst>(II)->METHOD         \
    : cast<InvokeInst>(II)->METHOD

#define CALLSITE_DELEGATE_SETTER(METHOD) \
  Instruction *II(getInstruction());     \
  if (isCall())                          \
    cast<CallInst>(II)->METHOD;          \
  else                                   \
    cast<InvokeInst>(II)->METHOD

CallSite::CallSite(Instruction *C) {
  assert((isa<CallInst>(C) || isa<InvokeInst>(C)) && "Not a call!");
  I.setPointer(C);
  I.setInt(isa<CallInst>(C));
}
unsigned CallSite::getCallingConv() const {
  CALLSITE_DELEGATE_GETTER(getCallingConv());
}
void CallSite::setCallingConv(unsigned CC) {
  CALLSITE_DELEGATE_SETTER(setCallingConv(CC));
}
const AttrListPtr &CallSite::getAttributes() const {
  CALLSITE_DELEGATE_GETTER(getAttributes());
}
void CallSite::setAttributes(const AttrListPtr &PAL) {
  CALLSITE_DELEGATE_SETTER(setAttributes(PAL));
}
bool CallSite::paramHasAttr(uint16_t i, Attributes attr) const {
  CALLSITE_DELEGATE_GETTER(paramHasAttr(i, attr));
}
uint16_t CallSite::getParamAlignment(uint16_t i) const {
  CALLSITE_DELEGATE_GETTER(getParamAlignment(i));
}
bool CallSite::doesNotAccessMemory() const {
  CALLSITE_DELEGATE_GETTER(doesNotAccessMemory());
}
void CallSite::setDoesNotAccessMemory(bool doesNotAccessMemory) {
  CALLSITE_DELEGATE_SETTER(setDoesNotAccessMemory(doesNotAccessMemory));
}
bool CallSite::onlyReadsMemory() const {
  CALLSITE_DELEGATE_GETTER(onlyReadsMemory());
}
void CallSite::setOnlyReadsMemory(bool onlyReadsMemory) {
  CALLSITE_DELEGATE_SETTER(setOnlyReadsMemory(onlyReadsMemory));
}
bool CallSite::doesNotReturn() const {
 CALLSITE_DELEGATE_GETTER(doesNotReturn());
}
void CallSite::setDoesNotReturn(bool doesNotReturn) {
  CALLSITE_DELEGATE_SETTER(setDoesNotReturn(doesNotReturn));
}
bool CallSite::doesNotThrow() const {
  CALLSITE_DELEGATE_GETTER(doesNotThrow());
}
void CallSite::setDoesNotThrow(bool doesNotThrow) {
  CALLSITE_DELEGATE_SETTER(setDoesNotThrow(doesNotThrow));
}

bool CallSite::hasArgument(const Value *Arg) const {
  for (arg_iterator AI = this->arg_begin(), E = this->arg_end(); AI != E; ++AI)
    if (AI->get() == Arg)
      return true;
  return false;
}

#undef CALLSITE_DELEGATE_GETTER
#undef CALLSITE_DELEGATE_SETTER

//===----------------------------------------------------------------------===//
//                            TerminatorInst Class
//===----------------------------------------------------------------------===//

// Out of line virtual method, so the vtable, etc has a home.
TerminatorInst::~TerminatorInst() {
}

//===----------------------------------------------------------------------===//
//                           UnaryInstruction Class
//===----------------------------------------------------------------------===//

// Out of line virtual method, so the vtable, etc has a home.
UnaryInstruction::~UnaryInstruction() {
}

//===----------------------------------------------------------------------===//
//                              SelectInst Class
//===----------------------------------------------------------------------===//

/// areInvalidOperands - Return a string if the specified operands are invalid
/// for a select operation, otherwise return null.
const char *SelectInst::areInvalidOperands(Value *Op0, Value *Op1, Value *Op2) {
  if (Op1->getType() != Op2->getType())
    return "both values to select must have same type";
  
  if (const VectorType *VT = dyn_cast<VectorType>(Op0->getType())) {
    // Vector select.
    if (VT->getElementType() != Type::getInt1Ty(Op0->getContext()))
      return "vector select condition element type must be i1";
    const VectorType *ET = dyn_cast<VectorType>(Op1->getType());
    if (ET == 0)
      return "selected values for vector select must be vectors";
    if (ET->getNumElements() != VT->getNumElements())
      return "vector select requires selected vectors to have "
                   "the same vector length as select condition";
  } else if (Op0->getType() != Type::getInt1Ty(Op0->getContext())) {
    return "select condition must be i1 or <n x i1>";
  }
  return 0;
}


//===----------------------------------------------------------------------===//
//                               PHINode Class
//===----------------------------------------------------------------------===//

PHINode::PHINode(const PHINode &PN)
  : Instruction(PN.getType(), Instruction::PHI,
                allocHungoffUses(PN.getNumOperands()), PN.getNumOperands()),
    ReservedSpace(PN.getNumOperands()) {
  Use *OL = OperandList;
  for (unsigned i = 0, e = PN.getNumOperands(); i != e; i+=2) {
    OL[i] = PN.getOperand(i);
    OL[i+1] = PN.getOperand(i+1);
  }
  SubclassOptionalData = PN.SubclassOptionalData;
}

PHINode::~PHINode() {
  if (OperandList)
    dropHungoffUses(OperandList);
}

// removeIncomingValue - Remove an incoming value.  This is useful if a
// predecessor basic block is deleted.
Value *PHINode::removeIncomingValue(unsigned Idx, bool DeletePHIIfEmpty) {
  unsigned NumOps = getNumOperands();
  Use *OL = OperandList;
  assert(Idx*2 < NumOps && "BB not in PHI node!");
  Value *Removed = OL[Idx*2];

  // Move everything after this operand down.
  //
  // FIXME: we could just swap with the end of the list, then erase.  However,
  // client might not expect this to happen.  The code as it is thrashes the
  // use/def lists, which is kinda lame.
  for (unsigned i = (Idx+1)*2; i != NumOps; i += 2) {
    OL[i-2] = OL[i];
    OL[i-2+1] = OL[i+1];
  }

  // Nuke the last value.
  OL[NumOps-2].set(0);
  OL[NumOps-2+1].set(0);
  NumOperands = NumOps-2;

  // If the PHI node is dead, because it has zero entries, nuke it now.
  if (NumOps == 2 && DeletePHIIfEmpty) {
    // If anyone is using this PHI, make them use a dummy value instead...
    replaceAllUsesWith(UndefValue::get(getType()));
    eraseFromParent();
  }
  return Removed;
}

/// resizeOperands - resize operands - This adjusts the length of the operands
/// list according to the following behavior:
///   1. If NumOps == 0, grow the operand list in response to a push_back style
///      of operation.  This grows the number of ops by 1.5 times.
///   2. If NumOps > NumOperands, reserve space for NumOps operands.
///   3. If NumOps == NumOperands, trim the reserved space.
///
void PHINode::resizeOperands(unsigned NumOps) {
  unsigned e = getNumOperands();
  if (NumOps == 0) {
    NumOps = e*3/2;
    if (NumOps < 4) NumOps = 4;      // 4 op PHI nodes are VERY common.
  } else if (NumOps*2 > NumOperands) {
    // No resize needed.
    if (ReservedSpace >= NumOps) return;
  } else if (NumOps == NumOperands) {
    if (ReservedSpace == NumOps) return;
  } else {
    return;
  }

  ReservedSpace = NumOps;
  Use *OldOps = OperandList;
  Use *NewOps = allocHungoffUses(NumOps);
  std::copy(OldOps, OldOps + e, NewOps);
  OperandList = NewOps;
  if (OldOps) Use::zap(OldOps, OldOps + e, true);
}

/// hasConstantValue - If the specified PHI node always merges together the same
/// value, return the value, otherwise return null.
///
Value *PHINode::hasConstantValue(bool AllowNonDominatingInstruction) const {
  // If the PHI node only has one incoming value, eliminate the PHI node...
  if (getNumIncomingValues() == 1) {
    if (getIncomingValue(0) != this)   // not  X = phi X
      return getIncomingValue(0);
    else
      return UndefValue::get(getType());  // Self cycle is dead.
  }
      
  // Otherwise if all of the incoming values are the same for the PHI, replace
  // the PHI node with the incoming value.
  //
  Value *InVal = 0;
  bool HasUndefInput = false;
  for (unsigned i = 0, e = getNumIncomingValues(); i != e; ++i)
    if (isa<UndefValue>(getIncomingValue(i))) {
      HasUndefInput = true;
    } else if (getIncomingValue(i) != this) { // Not the PHI node itself...
      if (InVal && getIncomingValue(i) != InVal)
        return 0;  // Not the same, bail out.
      else
        InVal = getIncomingValue(i);
    }
  
  // The only case that could cause InVal to be null is if we have a PHI node
  // that only has entries for itself.  In this case, there is no entry into the
  // loop, so kill the PHI.
  //
  if (InVal == 0) InVal = UndefValue::get(getType());
  
  // If we have a PHI node like phi(X, undef, X), where X is defined by some
  // instruction, we cannot always return X as the result of the PHI node.  Only
  // do this if X is not an instruction (thus it must dominate the PHI block),
  // or if the client is prepared to deal with this possibility.
  if (HasUndefInput && !AllowNonDominatingInstruction)
    if (Instruction *IV = dyn_cast<Instruction>(InVal))
      // If it's in the entry block, it dominates everything.
      if (IV->getParent() != &IV->getParent()->getParent()->getEntryBlock() ||
          isa<InvokeInst>(IV))
        return 0;   // Cannot guarantee that InVal dominates this PHINode.

  // All of the incoming values are the same, return the value now.
  return InVal;
}


//===----------------------------------------------------------------------===//
//                        CallInst Implementation
//===----------------------------------------------------------------------===//

CallInst::~CallInst() {
}

void CallInst::init(Value *Func, Value* const *Params, unsigned NumParams) {
  assert(NumOperands == NumParams+1 && "NumOperands not set up?");
  Use *OL = OperandList;
  OL[0] = Func;

  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());
  FTy = FTy;  // silence warning.

  assert((NumParams == FTy->getNumParams() ||
          (FTy->isVarArg() && NumParams > FTy->getNumParams())) &&
         "Calling a function with bad signature!");
  for (unsigned i = 0; i != NumParams; ++i) {
    assert((i >= FTy->getNumParams() || 
            FTy->getParamType(i) == Params[i]->getType()) &&
           "Calling a function with a bad signature!");
    OL[i+1] = Params[i];
  }
}

void CallInst::init(Value *Func, Value *Actual1, Value *Actual2) {
  assert(NumOperands == 3 && "NumOperands not set up?");
  Use *OL = OperandList;
  OL[0] = Func;
  OL[1] = Actual1;
  OL[2] = Actual2;

  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());
  FTy = FTy;  // silence warning.

  assert((FTy->getNumParams() == 2 ||
          (FTy->isVarArg() && FTy->getNumParams() < 2)) &&
         "Calling a function with bad signature");
  assert((0 >= FTy->getNumParams() || 
          FTy->getParamType(0) == Actual1->getType()) &&
         "Calling a function with a bad signature!");
  assert((1 >= FTy->getNumParams() || 
          FTy->getParamType(1) == Actual2->getType()) &&
         "Calling a function with a bad signature!");
}

void CallInst::init(Value *Func, Value *Actual) {
  assert(NumOperands == 2 && "NumOperands not set up?");
  Use *OL = OperandList;
  OL[0] = Func;
  OL[1] = Actual;

  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());
  FTy = FTy;  // silence warning.

  assert((FTy->getNumParams() == 1 ||
          (FTy->isVarArg() && FTy->getNumParams() == 0)) &&
         "Calling a function with bad signature");
  assert((0 == FTy->getNumParams() || 
          FTy->getParamType(0) == Actual->getType()) &&
         "Calling a function with a bad signature!");
}

void CallInst::init(Value *Func) {
  assert(NumOperands == 1 && "NumOperands not set up?");
  Use *OL = OperandList;
  OL[0] = Func;

  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());
  FTy = FTy;  // silence warning.

  assert(FTy->getNumParams() == 0 && "Calling a function with bad signature");
}

CallInst::CallInst(Value *Func, Value* Actual, const Twine &Name,
                   Instruction *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call,
                OperandTraits<CallInst>::op_end(this) - 2,
                2, InsertBefore) {
  init(Func, Actual);
  setName(Name);
}

CallInst::CallInst(Value *Func, Value* Actual, const Twine &Name,
                   BasicBlock  *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call,
                OperandTraits<CallInst>::op_end(this) - 2,
                2, InsertAtEnd) {
  init(Func, Actual);
  setName(Name);
}
CallInst::CallInst(Value *Func, const Twine &Name,
                   Instruction *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call,
                OperandTraits<CallInst>::op_end(this) - 1,
                1, InsertBefore) {
  init(Func);
  setName(Name);
}

CallInst::CallInst(Value *Func, const Twine &Name,
                   BasicBlock *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call,
                OperandTraits<CallInst>::op_end(this) - 1,
                1, InsertAtEnd) {
  init(Func);
  setName(Name);
}

CallInst::CallInst(const CallInst &CI)
  : Instruction(CI.getType(), Instruction::Call,
                OperandTraits<CallInst>::op_end(this) - CI.getNumOperands(),
                CI.getNumOperands()) {
  setAttributes(CI.getAttributes());
  SubclassData = CI.SubclassData;
  Use *OL = OperandList;
  Use *InOL = CI.OperandList;
  for (unsigned i = 0, e = CI.getNumOperands(); i != e; ++i)
    OL[i] = InOL[i];
  SubclassOptionalData = CI.SubclassOptionalData;
}

void CallInst::addAttribute(unsigned i, Attributes attr) {
  AttrListPtr PAL = getAttributes();
  PAL = PAL.addAttr(i, attr);
  setAttributes(PAL);
}

void CallInst::removeAttribute(unsigned i, Attributes attr) {
  AttrListPtr PAL = getAttributes();
  PAL = PAL.removeAttr(i, attr);
  setAttributes(PAL);
}

bool CallInst::paramHasAttr(unsigned i, Attributes attr) const {
  if (AttributeList.paramHasAttr(i, attr))
    return true;
  if (const Function *F = getCalledFunction())
    return F->paramHasAttr(i, attr);
  return false;
}


//===----------------------------------------------------------------------===//
//                        InvokeInst Implementation
//===----------------------------------------------------------------------===//

void InvokeInst::init(Value *Fn, BasicBlock *IfNormal, BasicBlock *IfException,
                      Value* const *Args, unsigned NumArgs) {
  assert(NumOperands == 3+NumArgs && "NumOperands not set up?");
  Use *OL = OperandList;
  OL[0] = Fn;
  OL[1] = IfNormal;
  OL[2] = IfException;
  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Fn->getType())->getElementType());
  FTy = FTy;  // silence warning.

  assert(((NumArgs == FTy->getNumParams()) ||
          (FTy->isVarArg() && NumArgs > FTy->getNumParams())) &&
         "Calling a function with bad signature");

  for (unsigned i = 0, e = NumArgs; i != e; i++) {
    assert((i >= FTy->getNumParams() || 
            FTy->getParamType(i) == Args[i]->getType()) &&
           "Invoking a function with a bad signature!");
    
    OL[i+3] = Args[i];
  }
}

InvokeInst::InvokeInst(const InvokeInst &II)
  : TerminatorInst(II.getType(), Instruction::Invoke,
                   OperandTraits<InvokeInst>::op_end(this)
                   - II.getNumOperands(),
                   II.getNumOperands()) {
  setAttributes(II.getAttributes());
  SubclassData = II.SubclassData;
  Use *OL = OperandList, *InOL = II.OperandList;
  for (unsigned i = 0, e = II.getNumOperands(); i != e; ++i)
    OL[i] = InOL[i];
  SubclassOptionalData = II.SubclassOptionalData;
}

BasicBlock *InvokeInst::getSuccessorV(unsigned idx) const {
  return getSuccessor(idx);
}
unsigned InvokeInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}
void InvokeInst::setSuccessorV(unsigned idx, BasicBlock *B) {
  return setSuccessor(idx, B);
}

bool InvokeInst::paramHasAttr(unsigned i, Attributes attr) const {
  if (AttributeList.paramHasAttr(i, attr))
    return true;
  if (const Function *F = getCalledFunction())
    return F->paramHasAttr(i, attr);
  return false;
}

void InvokeInst::addAttribute(unsigned i, Attributes attr) {
  AttrListPtr PAL = getAttributes();
  PAL = PAL.addAttr(i, attr);
  setAttributes(PAL);
}

void InvokeInst::removeAttribute(unsigned i, Attributes attr) {
  AttrListPtr PAL = getAttributes();
  PAL = PAL.removeAttr(i, attr);
  setAttributes(PAL);
}


//===----------------------------------------------------------------------===//
//                        ReturnInst Implementation
//===----------------------------------------------------------------------===//

ReturnInst::ReturnInst(const ReturnInst &RI)
  : TerminatorInst(Type::getVoidTy(RI.getContext()), Instruction::Ret,
                   OperandTraits<ReturnInst>::op_end(this) -
                     RI.getNumOperands(),
                   RI.getNumOperands()) {
  if (RI.getNumOperands())
    Op<0>() = RI.Op<0>();
  SubclassOptionalData = RI.SubclassOptionalData;
}

ReturnInst::ReturnInst(LLVMContext &C, Value *retVal, Instruction *InsertBefore)
  : TerminatorInst(Type::getVoidTy(C), Instruction::Ret,
                   OperandTraits<ReturnInst>::op_end(this) - !!retVal, !!retVal,
                   InsertBefore) {
  if (retVal)
    Op<0>() = retVal;
}
ReturnInst::ReturnInst(LLVMContext &C, Value *retVal, BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::getVoidTy(C), Instruction::Ret,
                   OperandTraits<ReturnInst>::op_end(this) - !!retVal, !!retVal,
                   InsertAtEnd) {
  if (retVal)
    Op<0>() = retVal;
}
ReturnInst::ReturnInst(LLVMContext &Context, BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::getVoidTy(Context), Instruction::Ret,
                   OperandTraits<ReturnInst>::op_end(this), 0, InsertAtEnd) {
}

unsigned ReturnInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}

/// Out-of-line ReturnInst method, put here so the C++ compiler can choose to
/// emit the vtable for the class in this translation unit.
void ReturnInst::setSuccessorV(unsigned idx, BasicBlock *NewSucc) {
  llvm_unreachable("ReturnInst has no successors!");
}

BasicBlock *ReturnInst::getSuccessorV(unsigned idx) const {
  llvm_unreachable("ReturnInst has no successors!");
  return 0;
}

ReturnInst::~ReturnInst() {
}

//===----------------------------------------------------------------------===//
//                        UnwindInst Implementation
//===----------------------------------------------------------------------===//

UnwindInst::UnwindInst(LLVMContext &Context, Instruction *InsertBefore)
  : TerminatorInst(Type::getVoidTy(Context), Instruction::Unwind,
                   0, 0, InsertBefore) {
}
UnwindInst::UnwindInst(LLVMContext &Context, BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::getVoidTy(Context), Instruction::Unwind,
                   0, 0, InsertAtEnd) {
}


unsigned UnwindInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}

void UnwindInst::setSuccessorV(unsigned idx, BasicBlock *NewSucc) {
  llvm_unreachable("UnwindInst has no successors!");
}

BasicBlock *UnwindInst::getSuccessorV(unsigned idx) const {
  llvm_unreachable("UnwindInst has no successors!");
  return 0;
}

//===----------------------------------------------------------------------===//
//                      UnreachableInst Implementation
//===----------------------------------------------------------------------===//

UnreachableInst::UnreachableInst(LLVMContext &Context, 
                                 Instruction *InsertBefore)
  : TerminatorInst(Type::getVoidTy(Context), Instruction::Unreachable,
                   0, 0, InsertBefore) {
}
UnreachableInst::UnreachableInst(LLVMContext &Context, BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::getVoidTy(Context), Instruction::Unreachable,
                   0, 0, InsertAtEnd) {
}

unsigned UnreachableInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}

void UnreachableInst::setSuccessorV(unsigned idx, BasicBlock *NewSucc) {
  llvm_unreachable("UnwindInst has no successors!");
}

BasicBlock *UnreachableInst::getSuccessorV(unsigned idx) const {
  llvm_unreachable("UnwindInst has no successors!");
  return 0;
}

//===----------------------------------------------------------------------===//
//                        BranchInst Implementation
//===----------------------------------------------------------------------===//

void BranchInst::AssertOK() {
  if (isConditional())
    assert(getCondition()->getType() == Type::getInt1Ty(getContext()) &&
           "May only branch on boolean predicates!");
}

BranchInst::BranchInst(BasicBlock *IfTrue, Instruction *InsertBefore)
  : TerminatorInst(Type::getVoidTy(IfTrue->getContext()), Instruction::Br,
                   OperandTraits<BranchInst>::op_end(this) - 1,
                   1, InsertBefore) {
  assert(IfTrue != 0 && "Branch destination may not be null!");
  Op<-1>() = IfTrue;
}
BranchInst::BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
                       Instruction *InsertBefore)
  : TerminatorInst(Type::getVoidTy(IfTrue->getContext()), Instruction::Br,
                   OperandTraits<BranchInst>::op_end(this) - 3,
                   3, InsertBefore) {
  Op<-1>() = IfTrue;
  Op<-2>() = IfFalse;
  Op<-3>() = Cond;
#ifndef NDEBUG
  AssertOK();
#endif
}

BranchInst::BranchInst(BasicBlock *IfTrue, BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::getVoidTy(IfTrue->getContext()), Instruction::Br,
                   OperandTraits<BranchInst>::op_end(this) - 1,
                   1, InsertAtEnd) {
  assert(IfTrue != 0 && "Branch destination may not be null!");
  Op<-1>() = IfTrue;
}

BranchInst::BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
           BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::getVoidTy(IfTrue->getContext()), Instruction::Br,
                   OperandTraits<BranchInst>::op_end(this) - 3,
                   3, InsertAtEnd) {
  Op<-1>() = IfTrue;
  Op<-2>() = IfFalse;
  Op<-3>() = Cond;
#ifndef NDEBUG
  AssertOK();
#endif
}


BranchInst::BranchInst(const BranchInst &BI) :
  TerminatorInst(Type::getVoidTy(BI.getContext()), Instruction::Br,
                 OperandTraits<BranchInst>::op_end(this) - BI.getNumOperands(),
                 BI.getNumOperands()) {
  Op<-1>() = BI.Op<-1>();
  if (BI.getNumOperands() != 1) {
    assert(BI.getNumOperands() == 3 && "BR can have 1 or 3 operands!");
    Op<-3>() = BI.Op<-3>();
    Op<-2>() = BI.Op<-2>();
  }
  SubclassOptionalData = BI.SubclassOptionalData;
}


Use* Use::getPrefix() {
  PointerIntPair<Use**, 2, PrevPtrTag> &PotentialPrefix(this[-1].Prev);
  if (PotentialPrefix.getOpaqueValue())
    return 0;

  return reinterpret_cast<Use*>((char*)&PotentialPrefix + 1);
}

BranchInst::~BranchInst() {
  if (NumOperands == 1) {
    if (Use *Prefix = OperandList->getPrefix()) {
      Op<-1>() = 0;
      //
      // mark OperandList to have a special value for scrutiny
      // by baseclass destructors and operator delete
      OperandList = Prefix;
    } else {
      NumOperands = 3;
      OperandList = op_begin();
    }
  }
}


BasicBlock *BranchInst::getSuccessorV(unsigned idx) const {
  return getSuccessor(idx);
}
unsigned BranchInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}
void BranchInst::setSuccessorV(unsigned idx, BasicBlock *B) {
  setSuccessor(idx, B);
}


//===----------------------------------------------------------------------===//
//                        AllocationInst Implementation
//===----------------------------------------------------------------------===//

static Value *getAISize(LLVMContext &Context, Value *Amt) {
  if (!Amt)
    Amt = ConstantInt::get(Type::getInt32Ty(Context), 1);
  else {
    assert(!isa<BasicBlock>(Amt) &&
           "Passed basic block into allocation size parameter! Use other ctor");
    assert(Amt->getType() == Type::getInt32Ty(Context) &&
           "Malloc/Allocation array size is not a 32-bit integer!");
  }
  return Amt;
}

AllocationInst::AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy,
                               unsigned Align, const Twine &Name,
                               Instruction *InsertBefore)
  : UnaryInstruction(PointerType::getUnqual(Ty), iTy,
                     getAISize(Ty->getContext(), ArraySize), InsertBefore) {
  setAlignment(Align);
  assert(Ty != Type::getVoidTy(Ty->getContext()) && "Cannot allocate void!");
  setName(Name);
}

AllocationInst::AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy,
                               unsigned Align, const Twine &Name,
                               BasicBlock *InsertAtEnd)
  : UnaryInstruction(PointerType::getUnqual(Ty), iTy,
                     getAISize(Ty->getContext(), ArraySize), InsertAtEnd) {
  setAlignment(Align);
  assert(Ty != Type::getVoidTy(Ty->getContext()) && "Cannot allocate void!");
  setName(Name);
}

// Out of line virtual method, so the vtable, etc has a home.
AllocationInst::~AllocationInst() {
}

void AllocationInst::setAlignment(unsigned Align) {
  assert((Align & (Align-1)) == 0 && "Alignment is not a power of 2!");
  SubclassData = Log2_32(Align) + 1;
  assert(getAlignment() == Align && "Alignment representation error!");
}

bool AllocationInst::isArrayAllocation() const {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(getOperand(0)))
    return CI->getZExtValue() != 1;
  return true;
}

const Type *AllocationInst::getAllocatedType() const {
  return getType()->getElementType();
}

/// isStaticAlloca - Return true if this alloca is in the entry block of the
/// function and is a constant size.  If so, the code generator will fold it
/// into the prolog/epilog code, so it is basically free.
bool AllocaInst::isStaticAlloca() const {
  // Must be constant size.
  if (!isa<ConstantInt>(getArraySize())) return false;
  
  // Must be in the entry block.
  const BasicBlock *Parent = getParent();
  return Parent == &Parent->getParent()->front();
}

//===----------------------------------------------------------------------===//
//                             FreeInst Implementation
//===----------------------------------------------------------------------===//

void FreeInst::AssertOK() {
  assert(isa<PointerType>(getOperand(0)->getType()) &&
         "Can not free something of nonpointer type!");
}

FreeInst::FreeInst(Value *Ptr, Instruction *InsertBefore)
  : UnaryInstruction(Type::getVoidTy(Ptr->getContext()),
                     Free, Ptr, InsertBefore) {
  AssertOK();
}

FreeInst::FreeInst(Value *Ptr, BasicBlock *InsertAtEnd)
  : UnaryInstruction(Type::getVoidTy(Ptr->getContext()),
                     Free, Ptr, InsertAtEnd) {
  AssertOK();
}


//===----------------------------------------------------------------------===//
//                           LoadInst Implementation
//===----------------------------------------------------------------------===//

void LoadInst::AssertOK() {
  assert(isa<PointerType>(getOperand(0)->getType()) &&
         "Ptr must have pointer type.");
}

LoadInst::LoadInst(Value *Ptr, const Twine &Name, Instruction *InsertBef)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertBef) {
  setVolatile(false);
  setAlignment(0);
  AssertOK();
  setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const Twine &Name, BasicBlock *InsertAE)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertAE) {
  setVolatile(false);
  setAlignment(0);
  AssertOK();
  setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const Twine &Name, bool isVolatile,
                   Instruction *InsertBef)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertBef) {
  setVolatile(isVolatile);
  setAlignment(0);
  AssertOK();
  setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const Twine &Name, bool isVolatile, 
                   unsigned Align, Instruction *InsertBef)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertBef) {
  setVolatile(isVolatile);
  setAlignment(Align);
  AssertOK();
  setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const Twine &Name, bool isVolatile, 
                   unsigned Align, BasicBlock *InsertAE)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertAE) {
  setVolatile(isVolatile);
  setAlignment(Align);
  AssertOK();
  setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const Twine &Name, bool isVolatile,
                   BasicBlock *InsertAE)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertAE) {
  setVolatile(isVolatile);
  setAlignment(0);
  AssertOK();
  setName(Name);
}



LoadInst::LoadInst(Value *Ptr, const char *Name, Instruction *InsertBef)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertBef) {
  setVolatile(false);
  setAlignment(0);
  AssertOK();
  if (Name && Name[0]) setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const char *Name, BasicBlock *InsertAE)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertAE) {
  setVolatile(false);
  setAlignment(0);
  AssertOK();
  if (Name && Name[0]) setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const char *Name, bool isVolatile,
                   Instruction *InsertBef)
: UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                   Load, Ptr, InsertBef) {
  setVolatile(isVolatile);
  setAlignment(0);
  AssertOK();
  if (Name && Name[0]) setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const char *Name, bool isVolatile,
                   BasicBlock *InsertAE)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertAE) {
  setVolatile(isVolatile);
  setAlignment(0);
  AssertOK();
  if (Name && Name[0]) setName(Name);
}

void LoadInst::setAlignment(unsigned Align) {
  assert((Align & (Align-1)) == 0 && "Alignment is not a power of 2!");
  SubclassData = (SubclassData & 1) | ((Log2_32(Align)+1)<<1);
}

//===----------------------------------------------------------------------===//
//                           StoreInst Implementation
//===----------------------------------------------------------------------===//

void StoreInst::AssertOK() {
  assert(getOperand(0) && getOperand(1) && "Both operands must be non-null!");
  assert(isa<PointerType>(getOperand(1)->getType()) &&
         "Ptr must have pointer type!");
  assert(getOperand(0)->getType() ==
                 cast<PointerType>(getOperand(1)->getType())->getElementType()
         && "Ptr must be a pointer to Val type!");
}


StoreInst::StoreInst(Value *val, Value *addr, Instruction *InsertBefore)
  : Instruction(Type::getVoidTy(val->getContext()), Store,
                OperandTraits<StoreInst>::op_begin(this),
                OperandTraits<StoreInst>::operands(this),
                InsertBefore) {
  Op<0>() = val;
  Op<1>() = addr;
  setVolatile(false);
  setAlignment(0);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, BasicBlock *InsertAtEnd)
  : Instruction(Type::getVoidTy(val->getContext()), Store,
                OperandTraits<StoreInst>::op_begin(this),
                OperandTraits<StoreInst>::operands(this),
                InsertAtEnd) {
  Op<0>() = val;
  Op<1>() = addr;
  setVolatile(false);
  setAlignment(0);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile,
                     Instruction *InsertBefore)
  : Instruction(Type::getVoidTy(val->getContext()), Store,
                OperandTraits<StoreInst>::op_begin(this),
                OperandTraits<StoreInst>::operands(this),
                InsertBefore) {
  Op<0>() = val;
  Op<1>() = addr;
  setVolatile(isVolatile);
  setAlignment(0);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile,
                     unsigned Align, Instruction *InsertBefore)
  : Instruction(Type::getVoidTy(val->getContext()), Store,
                OperandTraits<StoreInst>::op_begin(this),
                OperandTraits<StoreInst>::operands(this),
                InsertBefore) {
  Op<0>() = val;
  Op<1>() = addr;
  setVolatile(isVolatile);
  setAlignment(Align);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile,
                     unsigned Align, BasicBlock *InsertAtEnd)
  : Instruction(Type::getVoidTy(val->getContext()), Store,
                OperandTraits<StoreInst>::op_begin(this),
                OperandTraits<StoreInst>::operands(this),
                InsertAtEnd) {
  Op<0>() = val;
  Op<1>() = addr;
  setVolatile(isVolatile);
  setAlignment(Align);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile,
                     BasicBlock *InsertAtEnd)
  : Instruction(Type::getVoidTy(val->getContext()), Store,
                OperandTraits<StoreInst>::op_begin(this),
                OperandTraits<StoreInst>::operands(this),
                InsertAtEnd) {
  Op<0>() = val;
  Op<1>() = addr;
  setVolatile(isVolatile);
  setAlignment(0);
  AssertOK();
}

void StoreInst::setAlignment(unsigned Align) {
  assert((Align & (Align-1)) == 0 && "Alignment is not a power of 2!");
  SubclassData = (SubclassData & 1) | ((Log2_32(Align)+1)<<1);
}

//===----------------------------------------------------------------------===//
//                       GetElementPtrInst Implementation
//===----------------------------------------------------------------------===//

static unsigned retrieveAddrSpace(const Value *Val) {
  return cast<PointerType>(Val->getType())->getAddressSpace();
}

void GetElementPtrInst::init(Value *Ptr, Value* const *Idx, unsigned NumIdx,
                             const Twine &Name) {
  assert(NumOperands == 1+NumIdx && "NumOperands not initialized?");
  Use *OL = OperandList;
  OL[0] = Ptr;

  for (unsigned i = 0; i != NumIdx; ++i)
    OL[i+1] = Idx[i];

  setName(Name);
}

void GetElementPtrInst::init(Value *Ptr, Value *Idx, const Twine &Name) {
  assert(NumOperands == 2 && "NumOperands not initialized?");
  Use *OL = OperandList;
  OL[0] = Ptr;
  OL[1] = Idx;

  setName(Name);
}

GetElementPtrInst::GetElementPtrInst(const GetElementPtrInst &GEPI)
  : Instruction(GEPI.getType(), GetElementPtr,
                OperandTraits<GetElementPtrInst>::op_end(this)
                - GEPI.getNumOperands(),
                GEPI.getNumOperands()) {
  Use *OL = OperandList;
  Use *GEPIOL = GEPI.OperandList;
  for (unsigned i = 0, E = NumOperands; i != E; ++i)
    OL[i] = GEPIOL[i];
  SubclassOptionalData = GEPI.SubclassOptionalData;
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, Value *Idx,
                                     const Twine &Name, Instruction *InBe)
  : Instruction(PointerType::get(
      checkType(getIndexedType(Ptr->getType(),Idx)), retrieveAddrSpace(Ptr)),
                GetElementPtr,
                OperandTraits<GetElementPtrInst>::op_end(this) - 2,
                2, InBe) {
  init(Ptr, Idx, Name);
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, Value *Idx,
                                     const Twine &Name, BasicBlock *IAE)
  : Instruction(PointerType::get(
            checkType(getIndexedType(Ptr->getType(),Idx)),  
                retrieveAddrSpace(Ptr)),
                GetElementPtr,
                OperandTraits<GetElementPtrInst>::op_end(this) - 2,
                2, IAE) {
  init(Ptr, Idx, Name);
}

/// getIndexedType - Returns the type of the element that would be accessed with
/// a gep instruction with the specified parameters.
///
/// The Idxs pointer should point to a continuous piece of memory containing the
/// indices, either as Value* or uint64_t.
///
/// A null type is returned if the indices are invalid for the specified
/// pointer type.
///
template <typename IndexTy>
static const Type* getIndexedTypeInternal(const Type *Ptr, IndexTy const *Idxs,
                                          unsigned NumIdx) {
  const PointerType *PTy = dyn_cast<PointerType>(Ptr);
  if (!PTy) return 0;   // Type isn't a pointer type!
  const Type *Agg = PTy->getElementType();

  // Handle the special case of the empty set index set, which is always valid.
  if (NumIdx == 0)
    return Agg;
  
  // If there is at least one index, the top level type must be sized, otherwise
  // it cannot be 'stepped over'.  We explicitly allow abstract types (those
  // that contain opaque types) under the assumption that it will be resolved to
  // a sane type later.
  if (!Agg->isSized() && !Agg->isAbstract())
    return 0;

  unsigned CurIdx = 1;
  for (; CurIdx != NumIdx; ++CurIdx) {
    const CompositeType *CT = dyn_cast<CompositeType>(Agg);
    if (!CT || isa<PointerType>(CT)) return 0;
    IndexTy Index = Idxs[CurIdx];
    if (!CT->indexValid(Index)) return 0;
    Agg = CT->getTypeAtIndex(Index);

    // If the new type forwards to another type, then it is in the middle
    // of being refined to another type (and hence, may have dropped all
    // references to what it was using before).  So, use the new forwarded
    // type.
    if (const Type *Ty = Agg->getForwardedType())
      Agg = Ty;
  }
  return CurIdx == NumIdx ? Agg : 0;
}

const Type* GetElementPtrInst::getIndexedType(const Type *Ptr,
                                              Value* const *Idxs,
                                              unsigned NumIdx) {
  return getIndexedTypeInternal(Ptr, Idxs, NumIdx);
}

const Type* GetElementPtrInst::getIndexedType(const Type *Ptr,
                                              uint64_t const *Idxs,
                                              unsigned NumIdx) {
  return getIndexedTypeInternal(Ptr, Idxs, NumIdx);
}

const Type* GetElementPtrInst::getIndexedType(const Type *Ptr, Value *Idx) {
  const PointerType *PTy = dyn_cast<PointerType>(Ptr);
  if (!PTy) return 0;   // Type isn't a pointer type!

  // Check the pointer index.
  if (!PTy->indexValid(Idx)) return 0;

  return PTy->getElementType();
}


/// hasAllZeroIndices - Return true if all of the indices of this GEP are
/// zeros.  If so, the result pointer and the first operand have the same
/// value, just potentially different types.
bool GetElementPtrInst::hasAllZeroIndices() const {
  for (unsigned i = 1, e = getNumOperands(); i != e; ++i) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(getOperand(i))) {
      if (!CI->isZero()) return false;
    } else {
      return false;
    }
  }
  return true;
}

/// hasAllConstantIndices - Return true if all of the indices of this GEP are
/// constant integers.  If so, the result pointer and the first operand have
/// a constant offset between them.
bool GetElementPtrInst::hasAllConstantIndices() const {
  for (unsigned i = 1, e = getNumOperands(); i != e; ++i) {
    if (!isa<ConstantInt>(getOperand(i)))
      return false;
  }
  return true;
}


//===----------------------------------------------------------------------===//
//                           ExtractElementInst Implementation
//===----------------------------------------------------------------------===//

ExtractElementInst::ExtractElementInst(Value *Val, Value *Index,
                                       const Twine &Name,
                                       Instruction *InsertBef)
  : Instruction(cast<VectorType>(Val->getType())->getElementType(),
                ExtractElement,
                OperandTraits<ExtractElementInst>::op_begin(this),
                2, InsertBef) {
  assert(isValidOperands(Val, Index) &&
         "Invalid extractelement instruction operands!");
  Op<0>() = Val;
  Op<1>() = Index;
  setName(Name);
}

ExtractElementInst::ExtractElementInst(Value *Val, Value *Index,
                                       const Twine &Name,
                                       BasicBlock *InsertAE)
  : Instruction(cast<VectorType>(Val->getType())->getElementType(),
                ExtractElement,
                OperandTraits<ExtractElementInst>::op_begin(this),
                2, InsertAE) {
  assert(isValidOperands(Val, Index) &&
         "Invalid extractelement instruction operands!");

  Op<0>() = Val;
  Op<1>() = Index;
  setName(Name);
}


bool ExtractElementInst::isValidOperands(const Value *Val, const Value *Index) {
  if (!isa<VectorType>(Val->getType()) ||
      Index->getType() != Type::getInt32Ty(Val->getContext()))
    return false;
  return true;
}


//===----------------------------------------------------------------------===//
//                           InsertElementInst Implementation
//===----------------------------------------------------------------------===//

InsertElementInst::InsertElementInst(Value *Vec, Value *Elt, Value *Index,
                                     const Twine &Name,
                                     Instruction *InsertBef)
  : Instruction(Vec->getType(), InsertElement,
                OperandTraits<InsertElementInst>::op_begin(this),
                3, InsertBef) {
  assert(isValidOperands(Vec, Elt, Index) &&
         "Invalid insertelement instruction operands!");
  Op<0>() = Vec;
  Op<1>() = Elt;
  Op<2>() = Index;
  setName(Name);
}

InsertElementInst::InsertElementInst(Value *Vec, Value *Elt, Value *Index,
                                     const Twine &Name,
                                     BasicBlock *InsertAE)
  : Instruction(Vec->getType(), InsertElement,
                OperandTraits<InsertElementInst>::op_begin(this),
                3, InsertAE) {
  assert(isValidOperands(Vec, Elt, Index) &&
         "Invalid insertelement instruction operands!");

  Op<0>() = Vec;
  Op<1>() = Elt;
  Op<2>() = Index;
  setName(Name);
}

bool InsertElementInst::isValidOperands(const Value *Vec, const Value *Elt, 
                                        const Value *Index) {
  if (!isa<VectorType>(Vec->getType()))
    return false;   // First operand of insertelement must be vector type.
  
  if (Elt->getType() != cast<VectorType>(Vec->getType())->getElementType())
    return false;// Second operand of insertelement must be vector element type.
    
  if (Index->getType() != Type::getInt32Ty(Vec->getContext()))
    return false;  // Third operand of insertelement must be i32.
  return true;
}


//===----------------------------------------------------------------------===//
//                      ShuffleVectorInst Implementation
//===----------------------------------------------------------------------===//

ShuffleVectorInst::ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                                     const Twine &Name,
                                     Instruction *InsertBefore)
: Instruction(VectorType::get(cast<VectorType>(V1->getType())->getElementType(),
                cast<VectorType>(Mask->getType())->getNumElements()),
              ShuffleVector,
              OperandTraits<ShuffleVectorInst>::op_begin(this),
              OperandTraits<ShuffleVectorInst>::operands(this),
              InsertBefore) {
  assert(isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector instruction operands!");
  Op<0>() = V1;
  Op<1>() = V2;
  Op<2>() = Mask;
  setName(Name);
}

ShuffleVectorInst::ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                                     const Twine &Name,
                                     BasicBlock *InsertAtEnd)
  : Instruction(V1->getType(), ShuffleVector,
                OperandTraits<ShuffleVectorInst>::op_begin(this),
                OperandTraits<ShuffleVectorInst>::operands(this),
                InsertAtEnd) {
  assert(isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector instruction operands!");

  Op<0>() = V1;
  Op<1>() = V2;
  Op<2>() = Mask;
  setName(Name);
}

bool ShuffleVectorInst::isValidOperands(const Value *V1, const Value *V2,
                                        const Value *Mask) {
  if (!isa<VectorType>(V1->getType()) || V1->getType() != V2->getType())
    return false;
  
  const VectorType *MaskTy = dyn_cast<VectorType>(Mask->getType());
  if (!isa<Constant>(Mask) || MaskTy == 0 ||
      MaskTy->getElementType() != Type::getInt32Ty(V1->getContext()))
    return false;
  return true;
}

/// getMaskValue - Return the index from the shuffle mask for the specified
/// output result.  This is either -1 if the element is undef or a number less
/// than 2*numelements.
int ShuffleVectorInst::getMaskValue(unsigned i) const {
  const Constant *Mask = cast<Constant>(getOperand(2));
  if (isa<UndefValue>(Mask)) return -1;
  if (isa<ConstantAggregateZero>(Mask)) return 0;
  const ConstantVector *MaskCV = cast<ConstantVector>(Mask);
  assert(i < MaskCV->getNumOperands() && "Index out of range");

  if (isa<UndefValue>(MaskCV->getOperand(i)))
    return -1;
  return cast<ConstantInt>(MaskCV->getOperand(i))->getZExtValue();
}

//===----------------------------------------------------------------------===//
//                             InsertValueInst Class
//===----------------------------------------------------------------------===//

void InsertValueInst::init(Value *Agg, Value *Val, const unsigned *Idx, 
                           unsigned NumIdx, const Twine &Name) {
  assert(NumOperands == 2 && "NumOperands not initialized?");
  Op<0>() = Agg;
  Op<1>() = Val;

  Indices.insert(Indices.end(), Idx, Idx + NumIdx);
  setName(Name);
}

void InsertValueInst::init(Value *Agg, Value *Val, unsigned Idx, 
                           const Twine &Name) {
  assert(NumOperands == 2 && "NumOperands not initialized?");
  Op<0>() = Agg;
  Op<1>() = Val;

  Indices.push_back(Idx);
  setName(Name);
}

InsertValueInst::InsertValueInst(const InsertValueInst &IVI)
  : Instruction(IVI.getType(), InsertValue,
                OperandTraits<InsertValueInst>::op_begin(this), 2),
    Indices(IVI.Indices) {
  Op<0>() = IVI.getOperand(0);
  Op<1>() = IVI.getOperand(1);
  SubclassOptionalData = IVI.SubclassOptionalData;
}

InsertValueInst::InsertValueInst(Value *Agg,
                                 Value *Val,
                                 unsigned Idx, 
                                 const Twine &Name,
                                 Instruction *InsertBefore)
  : Instruction(Agg->getType(), InsertValue,
                OperandTraits<InsertValueInst>::op_begin(this),
                2, InsertBefore) {
  init(Agg, Val, Idx, Name);
}

InsertValueInst::InsertValueInst(Value *Agg,
                                 Value *Val,
                                 unsigned Idx, 
                                 const Twine &Name,
                                 BasicBlock *InsertAtEnd)
  : Instruction(Agg->getType(), InsertValue,
                OperandTraits<InsertValueInst>::op_begin(this),
                2, InsertAtEnd) {
  init(Agg, Val, Idx, Name);
}

//===----------------------------------------------------------------------===//
//                             ExtractValueInst Class
//===----------------------------------------------------------------------===//

void ExtractValueInst::init(const unsigned *Idx, unsigned NumIdx,
                            const Twine &Name) {
  assert(NumOperands == 1 && "NumOperands not initialized?");

  Indices.insert(Indices.end(), Idx, Idx + NumIdx);
  setName(Name);
}

void ExtractValueInst::init(unsigned Idx, const Twine &Name) {
  assert(NumOperands == 1 && "NumOperands not initialized?");

  Indices.push_back(Idx);
  setName(Name);
}

ExtractValueInst::ExtractValueInst(const ExtractValueInst &EVI)
  : UnaryInstruction(EVI.getType(), ExtractValue, EVI.getOperand(0)),
    Indices(EVI.Indices) {
  SubclassOptionalData = EVI.SubclassOptionalData;
}

// getIndexedType - Returns the type of the element that would be extracted
// with an extractvalue instruction with the specified parameters.
//
// A null type is returned if the indices are invalid for the specified
// pointer type.
//
const Type* ExtractValueInst::getIndexedType(const Type *Agg,
                                             const unsigned *Idxs,
                                             unsigned NumIdx) {
  unsigned CurIdx = 0;
  for (; CurIdx != NumIdx; ++CurIdx) {
    const CompositeType *CT = dyn_cast<CompositeType>(Agg);
    if (!CT || isa<PointerType>(CT) || isa<VectorType>(CT)) return 0;
    unsigned Index = Idxs[CurIdx];
    if (!CT->indexValid(Index)) return 0;
    Agg = CT->getTypeAtIndex(Index);

    // If the new type forwards to another type, then it is in the middle
    // of being refined to another type (and hence, may have dropped all
    // references to what it was using before).  So, use the new forwarded
    // type.
    if (const Type *Ty = Agg->getForwardedType())
      Agg = Ty;
  }
  return CurIdx == NumIdx ? Agg : 0;
}

const Type* ExtractValueInst::getIndexedType(const Type *Agg,
                                             unsigned Idx) {
  return getIndexedType(Agg, &Idx, 1);
}

//===----------------------------------------------------------------------===//
//                             BinaryOperator Class
//===----------------------------------------------------------------------===//

/// AdjustIType - Map Add, Sub, and Mul to FAdd, FSub, and FMul when the
/// type is floating-point, to help provide compatibility with an older API.
///
static BinaryOperator::BinaryOps AdjustIType(BinaryOperator::BinaryOps iType,
                                             const Type *Ty) {
  // API compatibility: Adjust integer opcodes to floating-point opcodes.
  if (Ty->isFPOrFPVector()) {
    if (iType == BinaryOperator::Add) iType = BinaryOperator::FAdd;
    else if (iType == BinaryOperator::Sub) iType = BinaryOperator::FSub;
    else if (iType == BinaryOperator::Mul) iType = BinaryOperator::FMul;
  }
  return iType;
}

BinaryOperator::BinaryOperator(BinaryOps iType, Value *S1, Value *S2,
                               const Type *Ty, const Twine &Name,
                               Instruction *InsertBefore)
  : Instruction(Ty, AdjustIType(iType, Ty),
                OperandTraits<BinaryOperator>::op_begin(this),
                OperandTraits<BinaryOperator>::operands(this),
                InsertBefore) {
  Op<0>() = S1;
  Op<1>() = S2;
  init(AdjustIType(iType, Ty));
  setName(Name);
}

BinaryOperator::BinaryOperator(BinaryOps iType, Value *S1, Value *S2, 
                               const Type *Ty, const Twine &Name,
                               BasicBlock *InsertAtEnd)
  : Instruction(Ty, AdjustIType(iType, Ty),
                OperandTraits<BinaryOperator>::op_begin(this),
                OperandTraits<BinaryOperator>::operands(this),
                InsertAtEnd) {
  Op<0>() = S1;
  Op<1>() = S2;
  init(AdjustIType(iType, Ty));
  setName(Name);
}


void BinaryOperator::init(BinaryOps iType) {
  Value *LHS = getOperand(0), *RHS = getOperand(1);
  LHS = LHS; RHS = RHS; // Silence warnings.
  assert(LHS->getType() == RHS->getType() &&
         "Binary operator operand types must match!");
#ifndef NDEBUG
  switch (iType) {
  case Add: case Sub:
  case Mul:
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert(getType()->isIntOrIntVector() &&
           "Tried to create an integer operation on a non-integer type!");
    break;
  case FAdd: case FSub:
  case FMul:
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert(getType()->isFPOrFPVector() &&
           "Tried to create a floating-point operation on a "
           "non-floating-point type!");
    break;
  case UDiv: 
  case SDiv: 
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert((getType()->isInteger() || (isa<VectorType>(getType()) && 
            cast<VectorType>(getType())->getElementType()->isInteger())) &&
           "Incorrect operand type (not integer) for S/UDIV");
    break;
  case FDiv:
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert(getType()->isFPOrFPVector() &&
           "Incorrect operand type (not floating point) for FDIV");
    break;
  case URem: 
  case SRem: 
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert((getType()->isInteger() || (isa<VectorType>(getType()) && 
            cast<VectorType>(getType())->getElementType()->isInteger())) &&
           "Incorrect operand type (not integer) for S/UREM");
    break;
  case FRem:
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert(getType()->isFPOrFPVector() &&
           "Incorrect operand type (not floating point) for FREM");
    break;
  case Shl:
  case LShr:
  case AShr:
    assert(getType() == LHS->getType() &&
           "Shift operation should return same type as operands!");
    assert((getType()->isInteger() ||
            (isa<VectorType>(getType()) && 
             cast<VectorType>(getType())->getElementType()->isInteger())) &&
           "Tried to create a shift operation on a non-integral type!");
    break;
  case And: case Or:
  case Xor:
    assert(getType() == LHS->getType() &&
           "Logical operation should return same type as operands!");
    assert((getType()->isInteger() ||
            (isa<VectorType>(getType()) && 
             cast<VectorType>(getType())->getElementType()->isInteger())) &&
           "Tried to create a logical operation on a non-integral type!");
    break;
  default:
    break;
  }
#endif
}

BinaryOperator *BinaryOperator::Create(BinaryOps Op, Value *S1, Value *S2,
                                       const Twine &Name,
                                       Instruction *InsertBefore) {
  assert(S1->getType() == S2->getType() &&
         "Cannot create binary operator with two operands of differing type!");
  return new BinaryOperator(Op, S1, S2, S1->getType(), Name, InsertBefore);
}

BinaryOperator *BinaryOperator::Create(BinaryOps Op, Value *S1, Value *S2,
                                       const Twine &Name,
                                       BasicBlock *InsertAtEnd) {
  BinaryOperator *Res = Create(Op, S1, S2, Name);
  InsertAtEnd->getInstList().push_back(Res);
  return Res;
}

BinaryOperator *BinaryOperator::CreateNeg(Value *Op, const Twine &Name,
                                          Instruction *InsertBefore) {
  Value *zero = ConstantFP::getZeroValueForNegation(Op->getType());
  return new BinaryOperator(Instruction::Sub,
                            zero, Op,
                            Op->getType(), Name, InsertBefore);
}

BinaryOperator *BinaryOperator::CreateNeg(Value *Op, const Twine &Name,
                                          BasicBlock *InsertAtEnd) {
  Value *zero = ConstantFP::getZeroValueForNegation(Op->getType());
  return new BinaryOperator(Instruction::Sub,
                            zero, Op,
                            Op->getType(), Name, InsertAtEnd);
}

BinaryOperator *BinaryOperator::CreateFNeg(Value *Op, const Twine &Name,
                                           Instruction *InsertBefore) {
  Value *zero = ConstantFP::getZeroValueForNegation(Op->getType());
  return new BinaryOperator(Instruction::FSub,
                            zero, Op,
                            Op->getType(), Name, InsertBefore);
}

BinaryOperator *BinaryOperator::CreateFNeg(Value *Op, const Twine &Name,
                                           BasicBlock *InsertAtEnd) {
  Value *zero = ConstantFP::getZeroValueForNegation(Op->getType());
  return new BinaryOperator(Instruction::FSub,
                            zero, Op,
                            Op->getType(), Name, InsertAtEnd);
}

BinaryOperator *BinaryOperator::CreateNot(Value *Op, const Twine &Name,
                                          Instruction *InsertBefore) {
  Constant *C;
  if (const VectorType *PTy = dyn_cast<VectorType>(Op->getType())) {
    C = Constant::getAllOnesValue(PTy->getElementType());
    C = ConstantVector::get(
                              std::vector<Constant*>(PTy->getNumElements(), C));
  } else {
    C = Constant::getAllOnesValue(Op->getType());
  }
  
  return new BinaryOperator(Instruction::Xor, Op, C,
                            Op->getType(), Name, InsertBefore);
}

BinaryOperator *BinaryOperator::CreateNot(Value *Op, const Twine &Name,
                                          BasicBlock *InsertAtEnd) {
  Constant *AllOnes;
  if (const VectorType *PTy = dyn_cast<VectorType>(Op->getType())) {
    // Create a vector of all ones values.
    Constant *Elt = Constant::getAllOnesValue(PTy->getElementType());
    AllOnes = ConstantVector::get(
                            std::vector<Constant*>(PTy->getNumElements(), Elt));
  } else {
    AllOnes = Constant::getAllOnesValue(Op->getType());
  }
  
  return new BinaryOperator(Instruction::Xor, Op, AllOnes,
                            Op->getType(), Name, InsertAtEnd);
}


// isConstantAllOnes - Helper function for several functions below
static inline bool isConstantAllOnes(const Value *V) {
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(V))
    return CI->isAllOnesValue();
  if (const ConstantVector *CV = dyn_cast<ConstantVector>(V))
    return CV->isAllOnesValue();
  return false;
}

bool BinaryOperator::isNeg(const Value *V) {
  if (const BinaryOperator *Bop = dyn_cast<BinaryOperator>(V))
    if (Bop->getOpcode() == Instruction::Sub)
      if (Constant* C = dyn_cast<Constant>(Bop->getOperand(0)))
        return C->isNegativeZeroValue();
  return false;
}

bool BinaryOperator::isFNeg(const Value *V) {
  if (const BinaryOperator *Bop = dyn_cast<BinaryOperator>(V))
    if (Bop->getOpcode() == Instruction::FSub)
      if (Constant* C = dyn_cast<Constant>(Bop->getOperand(0)))
      return C->isNegativeZeroValue();
  return false;
}

bool BinaryOperator::isNot(const Value *V) {
  if (const BinaryOperator *Bop = dyn_cast<BinaryOperator>(V))
    return (Bop->getOpcode() == Instruction::Xor &&
            (isConstantAllOnes(Bop->getOperand(1)) ||
             isConstantAllOnes(Bop->getOperand(0))));
  return false;
}

Value *BinaryOperator::getNegArgument(Value *BinOp) {
  return cast<BinaryOperator>(BinOp)->getOperand(1);
}

const Value *BinaryOperator::getNegArgument(const Value *BinOp) {
  return getNegArgument(const_cast<Value*>(BinOp));
}

Value *BinaryOperator::getFNegArgument(Value *BinOp) {
  return cast<BinaryOperator>(BinOp)->getOperand(1);
}

const Value *BinaryOperator::getFNegArgument(const Value *BinOp) {
  return getFNegArgument(const_cast<Value*>(BinOp));
}

Value *BinaryOperator::getNotArgument(Value *BinOp) {
  assert(isNot(BinOp) && "getNotArgument on non-'not' instruction!");
  BinaryOperator *BO = cast<BinaryOperator>(BinOp);
  Value *Op0 = BO->getOperand(0);
  Value *Op1 = BO->getOperand(1);
  if (isConstantAllOnes(Op0)) return Op1;

  assert(isConstantAllOnes(Op1));
  return Op0;
}

const Value *BinaryOperator::getNotArgument(const Value *BinOp) {
  return getNotArgument(const_cast<Value*>(BinOp));
}


// swapOperands - Exchange the two operands to this instruction.  This
// instruction is safe to use on any binary instruction and does not
// modify the semantics of the instruction.  If the instruction is
// order dependent (SetLT f.e.) the opcode is changed.
//
bool BinaryOperator::swapOperands() {
  if (!isCommutative())
    return true; // Can't commute operands
  Op<0>().swap(Op<1>());
  return false;
}

//===----------------------------------------------------------------------===//
//                                CastInst Class
//===----------------------------------------------------------------------===//

// Just determine if this cast only deals with integral->integral conversion.
bool CastInst::isIntegerCast() const {
  switch (getOpcode()) {
    default: return false;
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::Trunc:
      return true;
    case Instruction::BitCast:
      return getOperand(0)->getType()->isInteger() && getType()->isInteger();
  }
}

bool CastInst::isLosslessCast() const {
  // Only BitCast can be lossless, exit fast if we're not BitCast
  if (getOpcode() != Instruction::BitCast)
    return false;

  // Identity cast is always lossless
  const Type* SrcTy = getOperand(0)->getType();
  const Type* DstTy = getType();
  if (SrcTy == DstTy)
    return true;
  
  // Pointer to pointer is always lossless.
  if (isa<PointerType>(SrcTy))
    return isa<PointerType>(DstTy);
  return false;  // Other types have no identity values
}

/// This function determines if the CastInst does not require any bits to be
/// changed in order to effect the cast. Essentially, it identifies cases where
/// no code gen is necessary for the cast, hence the name no-op cast.  For 
/// example, the following are all no-op casts:
/// # bitcast i32* %x to i8*
/// # bitcast <2 x i32> %x to <4 x i16> 
/// # ptrtoint i32* %x to i32     ; on 32-bit plaforms only
/// @brief Determine if a cast is a no-op.
bool CastInst::isNoopCast(const Type *IntPtrTy) const {
  switch (getOpcode()) {
    default:
      assert(!"Invalid CastOp");
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt: 
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
      return false; // These always modify bits
    case Instruction::BitCast:
      return true;  // BitCast never modifies bits.
    case Instruction::PtrToInt:
      return IntPtrTy->getScalarSizeInBits() ==
             getType()->getScalarSizeInBits();
    case Instruction::IntToPtr:
      return IntPtrTy->getScalarSizeInBits() ==
             getOperand(0)->getType()->getScalarSizeInBits();
  }
}

/// This function determines if a pair of casts can be eliminated and what 
/// opcode should be used in the elimination. This assumes that there are two 
/// instructions like this:
/// *  %F = firstOpcode SrcTy %x to MidTy
/// *  %S = secondOpcode MidTy %F to DstTy
/// The function returns a resultOpcode so these two casts can be replaced with:
/// *  %Replacement = resultOpcode %SrcTy %x to DstTy
/// If no such cast is permited, the function returns 0.
unsigned CastInst::isEliminableCastPair(
  Instruction::CastOps firstOp, Instruction::CastOps secondOp,
  const Type *SrcTy, const Type *MidTy, const Type *DstTy, const Type *IntPtrTy)
{
  // Define the 144 possibilities for these two cast instructions. The values
  // in this matrix determine what to do in a given situation and select the
  // case in the switch below.  The rows correspond to firstOp, the columns 
  // correspond to secondOp.  In looking at the table below, keep in  mind
  // the following cast properties:
  //
  //          Size Compare       Source               Destination
  // Operator  Src ? Size   Type       Sign         Type       Sign
  // -------- ------------ -------------------   ---------------------
  // TRUNC         >       Integer      Any        Integral     Any
  // ZEXT          <       Integral   Unsigned     Integer      Any
  // SEXT          <       Integral    Signed      Integer      Any
  // FPTOUI       n/a      FloatPt      n/a        Integral   Unsigned
  // FPTOSI       n/a      FloatPt      n/a        Integral    Signed 
  // UITOFP       n/a      Integral   Unsigned     FloatPt      n/a   
  // SITOFP       n/a      Integral    Signed      FloatPt      n/a   
  // FPTRUNC       >       FloatPt      n/a        FloatPt      n/a   
  // FPEXT         <       FloatPt      n/a        FloatPt      n/a   
  // PTRTOINT     n/a      Pointer      n/a        Integral   Unsigned
  // INTTOPTR     n/a      Integral   Unsigned     Pointer      n/a
  // BITCONVERT    =       FirstClass   n/a       FirstClass    n/a   
  //
  // NOTE: some transforms are safe, but we consider them to be non-profitable.
  // For example, we could merge "fptoui double to i32" + "zext i32 to i64",
  // into "fptoui double to i64", but this loses information about the range
  // of the produced value (we no longer know the top-part is all zeros). 
  // Further this conversion is often much more expensive for typical hardware,
  // and causes issues when building libgcc.  We disallow fptosi+sext for the 
  // same reason.
  const unsigned numCastOps = 
    Instruction::CastOpsEnd - Instruction::CastOpsBegin;
  static const uint8_t CastResults[numCastOps][numCastOps] = {
    // T        F  F  U  S  F  F  P  I  B   -+
    // R  Z  S  P  P  I  I  T  P  2  N  T    |
    // U  E  E  2  2  2  2  R  E  I  T  C    +- secondOp
    // N  X  X  U  S  F  F  N  X  N  2  V    |
    // C  T  T  I  I  P  P  C  T  T  P  T   -+
    {  1, 0, 0,99,99, 0, 0,99,99,99, 0, 3 }, // Trunc      -+
    {  8, 1, 9,99,99, 2, 0,99,99,99, 2, 3 }, // ZExt        |
    {  8, 0, 1,99,99, 0, 2,99,99,99, 0, 3 }, // SExt        |
    {  0, 0, 0,99,99, 0, 0,99,99,99, 0, 3 }, // FPToUI      |
    {  0, 0, 0,99,99, 0, 0,99,99,99, 0, 3 }, // FPToSI      |
    { 99,99,99, 0, 0,99,99, 0, 0,99,99, 4 }, // UIToFP      +- firstOp
    { 99,99,99, 0, 0,99,99, 0, 0,99,99, 4 }, // SIToFP      |
    { 99,99,99, 0, 0,99,99, 1, 0,99,99, 4 }, // FPTrunc     |
    { 99,99,99, 2, 2,99,99,10, 2,99,99, 4 }, // FPExt       |
    {  1, 0, 0,99,99, 0, 0,99,99,99, 7, 3 }, // PtrToInt    |
    { 99,99,99,99,99,99,99,99,99,13,99,12 }, // IntToPtr    |
    {  5, 5, 5, 6, 6, 5, 5, 6, 6,11, 5, 1 }, // BitCast    -+
  };

  int ElimCase = CastResults[firstOp-Instruction::CastOpsBegin]
                            [secondOp-Instruction::CastOpsBegin];
  switch (ElimCase) {
    case 0: 
      // categorically disallowed
      return 0;
    case 1: 
      // allowed, use first cast's opcode
      return firstOp;
    case 2: 
      // allowed, use second cast's opcode
      return secondOp;
    case 3: 
      // no-op cast in second op implies firstOp as long as the DestTy 
      // is integer
      if (DstTy->isInteger())
        return firstOp;
      return 0;
    case 4:
      // no-op cast in second op implies firstOp as long as the DestTy
      // is floating point
      if (DstTy->isFloatingPoint())
        return firstOp;
      return 0;
    case 5: 
      // no-op cast in first op implies secondOp as long as the SrcTy
      // is an integer
      if (SrcTy->isInteger())
        return secondOp;
      return 0;
    case 6:
      // no-op cast in first op implies secondOp as long as the SrcTy
      // is a floating point
      if (SrcTy->isFloatingPoint())
        return secondOp;
      return 0;
    case 7: { 
      // ptrtoint, inttoptr -> bitcast (ptr -> ptr) if int size is >= ptr size
      if (!IntPtrTy)
        return 0;
      unsigned PtrSize = IntPtrTy->getScalarSizeInBits();
      unsigned MidSize = MidTy->getScalarSizeInBits();
      if (MidSize >= PtrSize)
        return Instruction::BitCast;
      return 0;
    }
    case 8: {
      // ext, trunc -> bitcast,    if the SrcTy and DstTy are same size
      // ext, trunc -> ext,        if sizeof(SrcTy) < sizeof(DstTy)
      // ext, trunc -> trunc,      if sizeof(SrcTy) > sizeof(DstTy)
      unsigned SrcSize = SrcTy->getScalarSizeInBits();
      unsigned DstSize = DstTy->getScalarSizeInBits();
      if (SrcSize == DstSize)
        return Instruction::BitCast;
      else if (SrcSize < DstSize)
        return firstOp;
      return secondOp;
    }
    case 9: // zext, sext -> zext, because sext can't sign extend after zext
      return Instruction::ZExt;
    case 10:
      // fpext followed by ftrunc is allowed if the bit size returned to is
      // the same as the original, in which case its just a bitcast
      if (SrcTy == DstTy)
        return Instruction::BitCast;
      return 0; // If the types are not the same we can't eliminate it.
    case 11:
      // bitcast followed by ptrtoint is allowed as long as the bitcast
      // is a pointer to pointer cast.
      if (isa<PointerType>(SrcTy) && isa<PointerType>(MidTy))
        return secondOp;
      return 0;
    case 12:
      // inttoptr, bitcast -> intptr  if bitcast is a ptr to ptr cast
      if (isa<PointerType>(MidTy) && isa<PointerType>(DstTy))
        return firstOp;
      return 0;
    case 13: {
      // inttoptr, ptrtoint -> bitcast if SrcSize<=PtrSize and SrcSize==DstSize
      if (!IntPtrTy)
        return 0;
      unsigned PtrSize = IntPtrTy->getScalarSizeInBits();
      unsigned SrcSize = SrcTy->getScalarSizeInBits();
      unsigned DstSize = DstTy->getScalarSizeInBits();
      if (SrcSize <= PtrSize && SrcSize == DstSize)
        return Instruction::BitCast;
      return 0;
    }
    case 99: 
      // cast combination can't happen (error in input). This is for all cases
      // where the MidTy is not the same for the two cast instructions.
      assert(!"Invalid Cast Combination");
      return 0;
    default:
      assert(!"Error in CastResults table!!!");
      return 0;
  }
  return 0;
}

CastInst *CastInst::Create(Instruction::CastOps op, Value *S, const Type *Ty, 
  const Twine &Name, Instruction *InsertBefore) {
  // Construct and return the appropriate CastInst subclass
  switch (op) {
    case Trunc:    return new TruncInst    (S, Ty, Name, InsertBefore);
    case ZExt:     return new ZExtInst     (S, Ty, Name, InsertBefore);
    case SExt:     return new SExtInst     (S, Ty, Name, InsertBefore);
    case FPTrunc:  return new FPTruncInst  (S, Ty, Name, InsertBefore);
    case FPExt:    return new FPExtInst    (S, Ty, Name, InsertBefore);
    case UIToFP:   return new UIToFPInst   (S, Ty, Name, InsertBefore);
    case SIToFP:   return new SIToFPInst   (S, Ty, Name, InsertBefore);
    case FPToUI:   return new FPToUIInst   (S, Ty, Name, InsertBefore);
    case FPToSI:   return new FPToSIInst   (S, Ty, Name, InsertBefore);
    case PtrToInt: return new PtrToIntInst (S, Ty, Name, InsertBefore);
    case IntToPtr: return new IntToPtrInst (S, Ty, Name, InsertBefore);
    case BitCast:  return new BitCastInst  (S, Ty, Name, InsertBefore);
    default:
      assert(!"Invalid opcode provided");
  }
  return 0;
}

CastInst *CastInst::Create(Instruction::CastOps op, Value *S, const Type *Ty,
  const Twine &Name, BasicBlock *InsertAtEnd) {
  // Construct and return the appropriate CastInst subclass
  switch (op) {
    case Trunc:    return new TruncInst    (S, Ty, Name, InsertAtEnd);
    case ZExt:     return new ZExtInst     (S, Ty, Name, InsertAtEnd);
    case SExt:     return new SExtInst     (S, Ty, Name, InsertAtEnd);
    case FPTrunc:  return new FPTruncInst  (S, Ty, Name, InsertAtEnd);
    case FPExt:    return new FPExtInst    (S, Ty, Name, InsertAtEnd);
    case UIToFP:   return new UIToFPInst   (S, Ty, Name, InsertAtEnd);
    case SIToFP:   return new SIToFPInst   (S, Ty, Name, InsertAtEnd);
    case FPToUI:   return new FPToUIInst   (S, Ty, Name, InsertAtEnd);
    case FPToSI:   return new FPToSIInst   (S, Ty, Name, InsertAtEnd);
    case PtrToInt: return new PtrToIntInst (S, Ty, Name, InsertAtEnd);
    case IntToPtr: return new IntToPtrInst (S, Ty, Name, InsertAtEnd);
    case BitCast:  return new BitCastInst  (S, Ty, Name, InsertAtEnd);
    default:
      assert(!"Invalid opcode provided");
  }
  return 0;
}

CastInst *CastInst::CreateZExtOrBitCast(Value *S, const Type *Ty, 
                                        const Twine &Name,
                                        Instruction *InsertBefore) {
  if (S->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return Create(Instruction::BitCast, S, Ty, Name, InsertBefore);
  return Create(Instruction::ZExt, S, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreateZExtOrBitCast(Value *S, const Type *Ty, 
                                        const Twine &Name,
                                        BasicBlock *InsertAtEnd) {
  if (S->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return Create(Instruction::BitCast, S, Ty, Name, InsertAtEnd);
  return Create(Instruction::ZExt, S, Ty, Name, InsertAtEnd);
}

CastInst *CastInst::CreateSExtOrBitCast(Value *S, const Type *Ty, 
                                        const Twine &Name,
                                        Instruction *InsertBefore) {
  if (S->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return Create(Instruction::BitCast, S, Ty, Name, InsertBefore);
  return Create(Instruction::SExt, S, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreateSExtOrBitCast(Value *S, const Type *Ty, 
                                        const Twine &Name,
                                        BasicBlock *InsertAtEnd) {
  if (S->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return Create(Instruction::BitCast, S, Ty, Name, InsertAtEnd);
  return Create(Instruction::SExt, S, Ty, Name, InsertAtEnd);
}

CastInst *CastInst::CreateTruncOrBitCast(Value *S, const Type *Ty,
                                         const Twine &Name,
                                         Instruction *InsertBefore) {
  if (S->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return Create(Instruction::BitCast, S, Ty, Name, InsertBefore);
  return Create(Instruction::Trunc, S, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreateTruncOrBitCast(Value *S, const Type *Ty,
                                         const Twine &Name, 
                                         BasicBlock *InsertAtEnd) {
  if (S->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return Create(Instruction::BitCast, S, Ty, Name, InsertAtEnd);
  return Create(Instruction::Trunc, S, Ty, Name, InsertAtEnd);
}

CastInst *CastInst::CreatePointerCast(Value *S, const Type *Ty,
                                      const Twine &Name,
                                      BasicBlock *InsertAtEnd) {
  assert(isa<PointerType>(S->getType()) && "Invalid cast");
  assert((Ty->isInteger() || isa<PointerType>(Ty)) &&
         "Invalid cast");

  if (Ty->isInteger())
    return Create(Instruction::PtrToInt, S, Ty, Name, InsertAtEnd);
  return Create(Instruction::BitCast, S, Ty, Name, InsertAtEnd);
}

/// @brief Create a BitCast or a PtrToInt cast instruction
CastInst *CastInst::CreatePointerCast(Value *S, const Type *Ty, 
                                      const Twine &Name, 
                                      Instruction *InsertBefore) {
  assert(isa<PointerType>(S->getType()) && "Invalid cast");
  assert((Ty->isInteger() || isa<PointerType>(Ty)) &&
         "Invalid cast");

  if (Ty->isInteger())
    return Create(Instruction::PtrToInt, S, Ty, Name, InsertBefore);
  return Create(Instruction::BitCast, S, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreateIntegerCast(Value *C, const Type *Ty, 
                                      bool isSigned, const Twine &Name,
                                      Instruction *InsertBefore) {
  assert(C->getType()->isInteger() && Ty->isInteger() && "Invalid cast");
  unsigned SrcBits = C->getType()->getScalarSizeInBits();
  unsigned DstBits = Ty->getScalarSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::Trunc :
      (isSigned ? Instruction::SExt : Instruction::ZExt)));
  return Create(opcode, C, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreateIntegerCast(Value *C, const Type *Ty, 
                                      bool isSigned, const Twine &Name,
                                      BasicBlock *InsertAtEnd) {
  assert(C->getType()->isIntOrIntVector() && Ty->isIntOrIntVector() &&
         "Invalid cast");
  unsigned SrcBits = C->getType()->getScalarSizeInBits();
  unsigned DstBits = Ty->getScalarSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::Trunc :
      (isSigned ? Instruction::SExt : Instruction::ZExt)));
  return Create(opcode, C, Ty, Name, InsertAtEnd);
}

CastInst *CastInst::CreateFPCast(Value *C, const Type *Ty, 
                                 const Twine &Name, 
                                 Instruction *InsertBefore) {
  assert(C->getType()->isFPOrFPVector() && Ty->isFPOrFPVector() &&
         "Invalid cast");
  unsigned SrcBits = C->getType()->getScalarSizeInBits();
  unsigned DstBits = Ty->getScalarSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::FPTrunc : Instruction::FPExt));
  return Create(opcode, C, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreateFPCast(Value *C, const Type *Ty, 
                                 const Twine &Name, 
                                 BasicBlock *InsertAtEnd) {
  assert(C->getType()->isFPOrFPVector() && Ty->isFPOrFPVector() &&
         "Invalid cast");
  unsigned SrcBits = C->getType()->getScalarSizeInBits();
  unsigned DstBits = Ty->getScalarSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::FPTrunc : Instruction::FPExt));
  return Create(opcode, C, Ty, Name, InsertAtEnd);
}

// Check whether it is valid to call getCastOpcode for these types.
// This routine must be kept in sync with getCastOpcode.
bool CastInst::isCastable(const Type *SrcTy, const Type *DestTy) {
  if (!SrcTy->isFirstClassType() || !DestTy->isFirstClassType())
    return false;

  if (SrcTy == DestTy)
    return true;

  // Get the bit sizes, we'll need these
  unsigned SrcBits = SrcTy->getScalarSizeInBits();   // 0 for ptr
  unsigned DestBits = DestTy->getScalarSizeInBits(); // 0 for ptr

  // Run through the possibilities ...
  if (DestTy->isInteger()) {                   // Casting to integral
    if (SrcTy->isInteger()) {                  // Casting from integral
        return true;
    } else if (SrcTy->isFloatingPoint()) {     // Casting from floating pt
      return true;
    } else if (const VectorType *PTy = dyn_cast<VectorType>(SrcTy)) {
                                               // Casting from vector
      return DestBits == PTy->getBitWidth();
    } else {                                   // Casting from something else
      return isa<PointerType>(SrcTy);
    }
  } else if (DestTy->isFloatingPoint()) {      // Casting to floating pt
    if (SrcTy->isInteger()) {                  // Casting from integral
      return true;
    } else if (SrcTy->isFloatingPoint()) {     // Casting from floating pt
      return true;
    } else if (const VectorType *PTy = dyn_cast<VectorType>(SrcTy)) {
                                               // Casting from vector
      return DestBits == PTy->getBitWidth();
    } else {                                   // Casting from something else
      return false;
    }
  } else if (const VectorType *DestPTy = dyn_cast<VectorType>(DestTy)) {
                                                // Casting to vector
    if (const VectorType *SrcPTy = dyn_cast<VectorType>(SrcTy)) {
                                                // Casting from vector
      return DestPTy->getBitWidth() == SrcPTy->getBitWidth();
    } else {                                    // Casting from something else
      return DestPTy->getBitWidth() == SrcBits;
    }
  } else if (isa<PointerType>(DestTy)) {        // Casting to pointer
    if (isa<PointerType>(SrcTy)) {              // Casting from pointer
      return true;
    } else if (SrcTy->isInteger()) {            // Casting from integral
      return true;
    } else {                                    // Casting from something else
      return false;
    }
  } else {                                      // Casting to something else
    return false;
  }
}

// Provide a way to get a "cast" where the cast opcode is inferred from the 
// types and size of the operand. This, basically, is a parallel of the 
// logic in the castIsValid function below.  This axiom should hold:
//   castIsValid( getCastOpcode(Val, Ty), Val, Ty)
// should not assert in castIsValid. In other words, this produces a "correct"
// casting opcode for the arguments passed to it.
// This routine must be kept in sync with isCastable.
Instruction::CastOps
CastInst::getCastOpcode(
  const Value *Src, bool SrcIsSigned, const Type *DestTy, bool DestIsSigned) {
  // Get the bit sizes, we'll need these
  const Type *SrcTy = Src->getType();
  unsigned SrcBits = SrcTy->getScalarSizeInBits();   // 0 for ptr
  unsigned DestBits = DestTy->getScalarSizeInBits(); // 0 for ptr

  assert(SrcTy->isFirstClassType() && DestTy->isFirstClassType() &&
         "Only first class types are castable!");

  // Run through the possibilities ...
  if (DestTy->isInteger()) {                       // Casting to integral
    if (SrcTy->isInteger()) {                      // Casting from integral
      if (DestBits < SrcBits)
        return Trunc;                               // int -> smaller int
      else if (DestBits > SrcBits) {                // its an extension
        if (SrcIsSigned)
          return SExt;                              // signed -> SEXT
        else
          return ZExt;                              // unsigned -> ZEXT
      } else {
        return BitCast;                             // Same size, No-op cast
      }
    } else if (SrcTy->isFloatingPoint()) {          // Casting from floating pt
      if (DestIsSigned) 
        return FPToSI;                              // FP -> sint
      else
        return FPToUI;                              // FP -> uint 
    } else if (const VectorType *PTy = dyn_cast<VectorType>(SrcTy)) {
      assert(DestBits == PTy->getBitWidth() &&
               "Casting vector to integer of different width");
      PTy = NULL;
      return BitCast;                             // Same size, no-op cast
    } else {
      assert(isa<PointerType>(SrcTy) &&
             "Casting from a value that is not first-class type");
      return PtrToInt;                              // ptr -> int
    }
  } else if (DestTy->isFloatingPoint()) {           // Casting to floating pt
    if (SrcTy->isInteger()) {                      // Casting from integral
      if (SrcIsSigned)
        return SIToFP;                              // sint -> FP
      else
        return UIToFP;                              // uint -> FP
    } else if (SrcTy->isFloatingPoint()) {          // Casting from floating pt
      if (DestBits < SrcBits) {
        return FPTrunc;                             // FP -> smaller FP
      } else if (DestBits > SrcBits) {
        return FPExt;                               // FP -> larger FP
      } else  {
        return BitCast;                             // same size, no-op cast
      }
    } else if (const VectorType *PTy = dyn_cast<VectorType>(SrcTy)) {
      assert(DestBits == PTy->getBitWidth() &&
             "Casting vector to floating point of different width");
      PTy = NULL;
      return BitCast;                             // same size, no-op cast
    } else {
      llvm_unreachable("Casting pointer or non-first class to float");
    }
  } else if (const VectorType *DestPTy = dyn_cast<VectorType>(DestTy)) {
    if (const VectorType *SrcPTy = dyn_cast<VectorType>(SrcTy)) {
      assert(DestPTy->getBitWidth() == SrcPTy->getBitWidth() &&
             "Casting vector to vector of different widths");
      SrcPTy = NULL;
      return BitCast;                             // vector -> vector
    } else if (DestPTy->getBitWidth() == SrcBits) {
      return BitCast;                               // float/int -> vector
    } else {
      assert(!"Illegal cast to vector (wrong type or size)");
    }
  } else if (isa<PointerType>(DestTy)) {
    if (isa<PointerType>(SrcTy)) {
      return BitCast;                               // ptr -> ptr
    } else if (SrcTy->isInteger()) {
      return IntToPtr;                              // int -> ptr
    } else {
      assert(!"Casting pointer to other than pointer or int");
    }
  } else {
    assert(!"Casting to type that is not first-class");
  }

  // If we fall through to here we probably hit an assertion cast above
  // and assertions are not turned on. Anything we return is an error, so
  // BitCast is as good a choice as any.
  return BitCast;
}

//===----------------------------------------------------------------------===//
//                    CastInst SubClass Constructors
//===----------------------------------------------------------------------===//

/// Check that the construction parameters for a CastInst are correct. This
/// could be broken out into the separate constructors but it is useful to have
/// it in one place and to eliminate the redundant code for getting the sizes
/// of the types involved.
bool 
CastInst::castIsValid(Instruction::CastOps op, Value *S, const Type *DstTy) {

  // Check for type sanity on the arguments
  const Type *SrcTy = S->getType();
  if (!SrcTy->isFirstClassType() || !DstTy->isFirstClassType())
    return false;

  // Get the size of the types in bits, we'll need this later
  unsigned SrcBitSize = SrcTy->getScalarSizeInBits();
  unsigned DstBitSize = DstTy->getScalarSizeInBits();

  // Switch on the opcode provided
  switch (op) {
  default: return false; // This is an input error
  case Instruction::Trunc:
    return SrcTy->isIntOrIntVector() &&
           DstTy->isIntOrIntVector()&& SrcBitSize > DstBitSize;
  case Instruction::ZExt:
    return SrcTy->isIntOrIntVector() &&
           DstTy->isIntOrIntVector()&& SrcBitSize < DstBitSize;
  case Instruction::SExt: 
    return SrcTy->isIntOrIntVector() &&
           DstTy->isIntOrIntVector()&& SrcBitSize < DstBitSize;
  case Instruction::FPTrunc:
    return SrcTy->isFPOrFPVector() &&
           DstTy->isFPOrFPVector() && 
           SrcBitSize > DstBitSize;
  case Instruction::FPExt:
    return SrcTy->isFPOrFPVector() &&
           DstTy->isFPOrFPVector() && 
           SrcBitSize < DstBitSize;
  case Instruction::UIToFP:
  case Instruction::SIToFP:
    if (const VectorType *SVTy = dyn_cast<VectorType>(SrcTy)) {
      if (const VectorType *DVTy = dyn_cast<VectorType>(DstTy)) {
        return SVTy->getElementType()->isIntOrIntVector() &&
               DVTy->getElementType()->isFPOrFPVector() &&
               SVTy->getNumElements() == DVTy->getNumElements();
      }
    }
    return SrcTy->isIntOrIntVector() && DstTy->isFPOrFPVector();
  case Instruction::FPToUI:
  case Instruction::FPToSI:
    if (const VectorType *SVTy = dyn_cast<VectorType>(SrcTy)) {
      if (const VectorType *DVTy = dyn_cast<VectorType>(DstTy)) {
        return SVTy->getElementType()->isFPOrFPVector() &&
               DVTy->getElementType()->isIntOrIntVector() &&
               SVTy->getNumElements() == DVTy->getNumElements();
      }
    }
    return SrcTy->isFPOrFPVector() && DstTy->isIntOrIntVector();
  case Instruction::PtrToInt:
    return isa<PointerType>(SrcTy) && DstTy->isInteger();
  case Instruction::IntToPtr:
    return SrcTy->isInteger() && isa<PointerType>(DstTy);
  case Instruction::BitCast:
    // BitCast implies a no-op cast of type only. No bits change.
    // However, you can't cast pointers to anything but pointers.
    if (isa<PointerType>(SrcTy) != isa<PointerType>(DstTy))
      return false;

    // Now we know we're not dealing with a pointer/non-pointer mismatch. In all
    // these cases, the cast is okay if the source and destination bit widths
    // are identical.
    return SrcTy->getPrimitiveSizeInBits() == DstTy->getPrimitiveSizeInBits();
  }
}

TruncInst::TruncInst(
  Value *S, const Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, Trunc, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal Trunc");
}

TruncInst::TruncInst(
  Value *S, const Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, Trunc, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal Trunc");
}

ZExtInst::ZExtInst(
  Value *S, const Type *Ty, const Twine &Name, Instruction *InsertBefore
)  : CastInst(Ty, ZExt, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal ZExt");
}

ZExtInst::ZExtInst(
  Value *S, const Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
)  : CastInst(Ty, ZExt, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal ZExt");
}
SExtInst::SExtInst(
  Value *S, const Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, SExt, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal SExt");
}

SExtInst::SExtInst(
  Value *S, const Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
)  : CastInst(Ty, SExt, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal SExt");
}

FPTruncInst::FPTruncInst(
  Value *S, const Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, FPTrunc, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPTrunc");
}

FPTruncInst::FPTruncInst(
  Value *S, const Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, FPTrunc, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPTrunc");
}

FPExtInst::FPExtInst(
  Value *S, const Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, FPExt, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPExt");
}

FPExtInst::FPExtInst(
  Value *S, const Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, FPExt, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPExt");
}

UIToFPInst::UIToFPInst(
  Value *S, const Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, UIToFP, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal UIToFP");
}

UIToFPInst::UIToFPInst(
  Value *S, const Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, UIToFP, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal UIToFP");
}

SIToFPInst::SIToFPInst(
  Value *S, const Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, SIToFP, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal SIToFP");
}

SIToFPInst::SIToFPInst(
  Value *S, const Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, SIToFP, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal SIToFP");
}

FPToUIInst::FPToUIInst(
  Value *S, const Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, FPToUI, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPToUI");
}

FPToUIInst::FPToUIInst(
  Value *S, const Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, FPToUI, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPToUI");
}

FPToSIInst::FPToSIInst(
  Value *S, const Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, FPToSI, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPToSI");
}

FPToSIInst::FPToSIInst(
  Value *S, const Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, FPToSI, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPToSI");
}

PtrToIntInst::PtrToIntInst(
  Value *S, const Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, PtrToInt, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal PtrToInt");
}

PtrToIntInst::PtrToIntInst(
  Value *S, const Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, PtrToInt, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal PtrToInt");
}

IntToPtrInst::IntToPtrInst(
  Value *S, const Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, IntToPtr, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal IntToPtr");
}

IntToPtrInst::IntToPtrInst(
  Value *S, const Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, IntToPtr, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal IntToPtr");
}

BitCastInst::BitCastInst(
  Value *S, const Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, BitCast, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal BitCast");
}

BitCastInst::BitCastInst(
  Value *S, const Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, BitCast, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal BitCast");
}

//===----------------------------------------------------------------------===//
//                               CmpInst Classes
//===----------------------------------------------------------------------===//

CmpInst::CmpInst(const Type *ty, OtherOps op, unsigned short predicate,
                 Value *LHS, Value *RHS, const Twine &Name,
                 Instruction *InsertBefore)
  : Instruction(ty, op,
                OperandTraits<CmpInst>::op_begin(this),
                OperandTraits<CmpInst>::operands(this),
                InsertBefore) {
    Op<0>() = LHS;
    Op<1>() = RHS;
  SubclassData = predicate;
  setName(Name);
}

CmpInst::CmpInst(const Type *ty, OtherOps op, unsigned short predicate,
                 Value *LHS, Value *RHS, const Twine &Name,
                 BasicBlock *InsertAtEnd)
  : Instruction(ty, op,
                OperandTraits<CmpInst>::op_begin(this),
                OperandTraits<CmpInst>::operands(this),
                InsertAtEnd) {
  Op<0>() = LHS;
  Op<1>() = RHS;
  SubclassData = predicate;
  setName(Name);
}

CmpInst *
CmpInst::Create(LLVMContext &Context, OtherOps Op, unsigned short predicate,
                Value *S1, Value *S2, 
                const Twine &Name, Instruction *InsertBefore) {
  if (Op == Instruction::ICmp) {
    if (InsertBefore)
      return new ICmpInst(InsertBefore, CmpInst::Predicate(predicate),
                          S1, S2, Name);
    else
      return new ICmpInst(Context, CmpInst::Predicate(predicate),
                          S1, S2, Name);
  }
  
  if (InsertBefore)
    return new FCmpInst(InsertBefore, CmpInst::Predicate(predicate),
                        S1, S2, Name);
  else
    return new FCmpInst(Context, CmpInst::Predicate(predicate),
                        S1, S2, Name);
}

CmpInst *
CmpInst::Create(OtherOps Op, unsigned short predicate, Value *S1, Value *S2, 
                const Twine &Name, BasicBlock *InsertAtEnd) {
  if (Op == Instruction::ICmp) {
    return new ICmpInst(*InsertAtEnd, CmpInst::Predicate(predicate),
                        S1, S2, Name);
  }
  return new FCmpInst(*InsertAtEnd, CmpInst::Predicate(predicate),
                      S1, S2, Name);
}

void CmpInst::swapOperands() {
  if (ICmpInst *IC = dyn_cast<ICmpInst>(this))
    IC->swapOperands();
  else
    cast<FCmpInst>(this)->swapOperands();
}

bool CmpInst::isCommutative() {
  if (ICmpInst *IC = dyn_cast<ICmpInst>(this))
    return IC->isCommutative();
  return cast<FCmpInst>(this)->isCommutative();
}

bool CmpInst::isEquality() {
  if (ICmpInst *IC = dyn_cast<ICmpInst>(this))
    return IC->isEquality();
  return cast<FCmpInst>(this)->isEquality();
}


CmpInst::Predicate CmpInst::getInversePredicate(Predicate pred) {
  switch (pred) {
    default: assert(!"Unknown cmp predicate!");
    case ICMP_EQ: return ICMP_NE;
    case ICMP_NE: return ICMP_EQ;
    case ICMP_UGT: return ICMP_ULE;
    case ICMP_ULT: return ICMP_UGE;
    case ICMP_UGE: return ICMP_ULT;
    case ICMP_ULE: return ICMP_UGT;
    case ICMP_SGT: return ICMP_SLE;
    case ICMP_SLT: return ICMP_SGE;
    case ICMP_SGE: return ICMP_SLT;
    case ICMP_SLE: return ICMP_SGT;

    case FCMP_OEQ: return FCMP_UNE;
    case FCMP_ONE: return FCMP_UEQ;
    case FCMP_OGT: return FCMP_ULE;
    case FCMP_OLT: return FCMP_UGE;
    case FCMP_OGE: return FCMP_ULT;
    case FCMP_OLE: return FCMP_UGT;
    case FCMP_UEQ: return FCMP_ONE;
    case FCMP_UNE: return FCMP_OEQ;
    case FCMP_UGT: return FCMP_OLE;
    case FCMP_ULT: return FCMP_OGE;
    case FCMP_UGE: return FCMP_OLT;
    case FCMP_ULE: return FCMP_OGT;
    case FCMP_ORD: return FCMP_UNO;
    case FCMP_UNO: return FCMP_ORD;
    case FCMP_TRUE: return FCMP_FALSE;
    case FCMP_FALSE: return FCMP_TRUE;
  }
}

ICmpInst::Predicate ICmpInst::getSignedPredicate(Predicate pred) {
  switch (pred) {
    default: assert(! "Unknown icmp predicate!");
    case ICMP_EQ: case ICMP_NE: 
    case ICMP_SGT: case ICMP_SLT: case ICMP_SGE: case ICMP_SLE: 
       return pred;
    case ICMP_UGT: return ICMP_SGT;
    case ICMP_ULT: return ICMP_SLT;
    case ICMP_UGE: return ICMP_SGE;
    case ICMP_ULE: return ICMP_SLE;
  }
}

ICmpInst::Predicate ICmpInst::getUnsignedPredicate(Predicate pred) {
  switch (pred) {
    default: assert(! "Unknown icmp predicate!");
    case ICMP_EQ: case ICMP_NE: 
    case ICMP_UGT: case ICMP_ULT: case ICMP_UGE: case ICMP_ULE: 
       return pred;
    case ICMP_SGT: return ICMP_UGT;
    case ICMP_SLT: return ICMP_ULT;
    case ICMP_SGE: return ICMP_UGE;
    case ICMP_SLE: return ICMP_ULE;
  }
}

bool ICmpInst::isSignedPredicate(Predicate pred) {
  switch (pred) {
    default: assert(! "Unknown icmp predicate!");
    case ICMP_SGT: case ICMP_SLT: case ICMP_SGE: case ICMP_SLE: 
      return true;
    case ICMP_EQ:  case ICMP_NE: case ICMP_UGT: case ICMP_ULT: 
    case ICMP_UGE: case ICMP_ULE:
      return false;
  }
}

/// Initialize a set of values that all satisfy the condition with C.
///
ConstantRange 
ICmpInst::makeConstantRange(Predicate pred, const APInt &C) {
  APInt Lower(C);
  APInt Upper(C);
  uint32_t BitWidth = C.getBitWidth();
  switch (pred) {
  default: llvm_unreachable("Invalid ICmp opcode to ConstantRange ctor!");
  case ICmpInst::ICMP_EQ: Upper++; break;
  case ICmpInst::ICMP_NE: Lower++; break;
  case ICmpInst::ICMP_ULT: Lower = APInt::getMinValue(BitWidth); break;
  case ICmpInst::ICMP_SLT: Lower = APInt::getSignedMinValue(BitWidth); break;
  case ICmpInst::ICMP_UGT: 
    Lower++; Upper = APInt::getMinValue(BitWidth);        // Min = Next(Max)
    break;
  case ICmpInst::ICMP_SGT:
    Lower++; Upper = APInt::getSignedMinValue(BitWidth);  // Min = Next(Max)
    break;
  case ICmpInst::ICMP_ULE: 
    Lower = APInt::getMinValue(BitWidth); Upper++; 
    break;
  case ICmpInst::ICMP_SLE: 
    Lower = APInt::getSignedMinValue(BitWidth); Upper++; 
    break;
  case ICmpInst::ICMP_UGE:
    Upper = APInt::getMinValue(BitWidth);        // Min = Next(Max)
    break;
  case ICmpInst::ICMP_SGE:
    Upper = APInt::getSignedMinValue(BitWidth);  // Min = Next(Max)
    break;
  }
  return ConstantRange(Lower, Upper);
}

CmpInst::Predicate CmpInst::getSwappedPredicate(Predicate pred) {
  switch (pred) {
    default: assert(!"Unknown cmp predicate!");
    case ICMP_EQ: case ICMP_NE:
      return pred;
    case ICMP_SGT: return ICMP_SLT;
    case ICMP_SLT: return ICMP_SGT;
    case ICMP_SGE: return ICMP_SLE;
    case ICMP_SLE: return ICMP_SGE;
    case ICMP_UGT: return ICMP_ULT;
    case ICMP_ULT: return ICMP_UGT;
    case ICMP_UGE: return ICMP_ULE;
    case ICMP_ULE: return ICMP_UGE;
  
    case FCMP_FALSE: case FCMP_TRUE:
    case FCMP_OEQ: case FCMP_ONE:
    case FCMP_UEQ: case FCMP_UNE:
    case FCMP_ORD: case FCMP_UNO:
      return pred;
    case FCMP_OGT: return FCMP_OLT;
    case FCMP_OLT: return FCMP_OGT;
    case FCMP_OGE: return FCMP_OLE;
    case FCMP_OLE: return FCMP_OGE;
    case FCMP_UGT: return FCMP_ULT;
    case FCMP_ULT: return FCMP_UGT;
    case FCMP_UGE: return FCMP_ULE;
    case FCMP_ULE: return FCMP_UGE;
  }
}

bool CmpInst::isUnsigned(unsigned short predicate) {
  switch (predicate) {
    default: return false;
    case ICmpInst::ICMP_ULT: case ICmpInst::ICMP_ULE: case ICmpInst::ICMP_UGT: 
    case ICmpInst::ICMP_UGE: return true;
  }
}

bool CmpInst::isSigned(unsigned short predicate){
  switch (predicate) {
    default: return false;
    case ICmpInst::ICMP_SLT: case ICmpInst::ICMP_SLE: case ICmpInst::ICMP_SGT: 
    case ICmpInst::ICMP_SGE: return true;
  }
}

bool CmpInst::isOrdered(unsigned short predicate) {
  switch (predicate) {
    default: return false;
    case FCmpInst::FCMP_OEQ: case FCmpInst::FCMP_ONE: case FCmpInst::FCMP_OGT: 
    case FCmpInst::FCMP_OLT: case FCmpInst::FCMP_OGE: case FCmpInst::FCMP_OLE: 
    case FCmpInst::FCMP_ORD: return true;
  }
}
      
bool CmpInst::isUnordered(unsigned short predicate) {
  switch (predicate) {
    default: return false;
    case FCmpInst::FCMP_UEQ: case FCmpInst::FCMP_UNE: case FCmpInst::FCMP_UGT: 
    case FCmpInst::FCMP_ULT: case FCmpInst::FCMP_UGE: case FCmpInst::FCMP_ULE: 
    case FCmpInst::FCMP_UNO: return true;
  }
}

//===----------------------------------------------------------------------===//
//                        SwitchInst Implementation
//===----------------------------------------------------------------------===//

void SwitchInst::init(Value *Value, BasicBlock *Default, unsigned NumCases) {
  assert(Value && Default);
  ReservedSpace = 2+NumCases*2;
  NumOperands = 2;
  OperandList = allocHungoffUses(ReservedSpace);

  OperandList[0] = Value;
  OperandList[1] = Default;
}

/// SwitchInst ctor - Create a new switch instruction, specifying a value to
/// switch on and a default destination.  The number of additional cases can
/// be specified here to make memory allocation more efficient.  This
/// constructor can also autoinsert before another instruction.
SwitchInst::SwitchInst(Value *Value, BasicBlock *Default, unsigned NumCases,
                       Instruction *InsertBefore)
  : TerminatorInst(Type::getVoidTy(Value->getContext()), Instruction::Switch,
                   0, 0, InsertBefore) {
  init(Value, Default, NumCases);
}

/// SwitchInst ctor - Create a new switch instruction, specifying a value to
/// switch on and a default destination.  The number of additional cases can
/// be specified here to make memory allocation more efficient.  This
/// constructor also autoinserts at the end of the specified BasicBlock.
SwitchInst::SwitchInst(Value *Value, BasicBlock *Default, unsigned NumCases,
                       BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::getVoidTy(Value->getContext()), Instruction::Switch,
                   0, 0, InsertAtEnd) {
  init(Value, Default, NumCases);
}

SwitchInst::SwitchInst(const SwitchInst &SI)
  : TerminatorInst(Type::getVoidTy(SI.getContext()), Instruction::Switch,
                   allocHungoffUses(SI.getNumOperands()), SI.getNumOperands()) {
  Use *OL = OperandList, *InOL = SI.OperandList;
  for (unsigned i = 0, E = SI.getNumOperands(); i != E; i+=2) {
    OL[i] = InOL[i];
    OL[i+1] = InOL[i+1];
  }
  SubclassOptionalData = SI.SubclassOptionalData;
}

SwitchInst::~SwitchInst() {
  dropHungoffUses(OperandList);
}


/// addCase - Add an entry to the switch instruction...
///
void SwitchInst::addCase(ConstantInt *OnVal, BasicBlock *Dest) {
  unsigned OpNo = NumOperands;
  if (OpNo+2 > ReservedSpace)
    resizeOperands(0);  // Get more space!
  // Initialize some new operands.
  assert(OpNo+1 < ReservedSpace && "Growing didn't work!");
  NumOperands = OpNo+2;
  OperandList[OpNo] = OnVal;
  OperandList[OpNo+1] = Dest;
}

/// removeCase - This method removes the specified successor from the switch
/// instruction.  Note that this cannot be used to remove the default
/// destination (successor #0).
///
void SwitchInst::removeCase(unsigned idx) {
  assert(idx != 0 && "Cannot remove the default case!");
  assert(idx*2 < getNumOperands() && "Successor index out of range!!!");

  unsigned NumOps = getNumOperands();
  Use *OL = OperandList;

  // Move everything after this operand down.
  //
  // FIXME: we could just swap with the end of the list, then erase.  However,
  // client might not expect this to happen.  The code as it is thrashes the
  // use/def lists, which is kinda lame.
  for (unsigned i = (idx+1)*2; i != NumOps; i += 2) {
    OL[i-2] = OL[i];
    OL[i-2+1] = OL[i+1];
  }

  // Nuke the last value.
  OL[NumOps-2].set(0);
  OL[NumOps-2+1].set(0);
  NumOperands = NumOps-2;
}

/// resizeOperands - resize operands - This adjusts the length of the operands
/// list according to the following behavior:
///   1. If NumOps == 0, grow the operand list in response to a push_back style
///      of operation.  This grows the number of ops by 3 times.
///   2. If NumOps > NumOperands, reserve space for NumOps operands.
///   3. If NumOps == NumOperands, trim the reserved space.
///
void SwitchInst::resizeOperands(unsigned NumOps) {
  unsigned e = getNumOperands();
  if (NumOps == 0) {
    NumOps = e*3;
  } else if (NumOps*2 > NumOperands) {
    // No resize needed.
    if (ReservedSpace >= NumOps) return;
  } else if (NumOps == NumOperands) {
    if (ReservedSpace == NumOps) return;
  } else {
    return;
  }

  ReservedSpace = NumOps;
  Use *NewOps = allocHungoffUses(NumOps);
  Use *OldOps = OperandList;
  for (unsigned i = 0; i != e; ++i) {
      NewOps[i] = OldOps[i];
  }
  OperandList = NewOps;
  if (OldOps) Use::zap(OldOps, OldOps + e, true);
}


BasicBlock *SwitchInst::getSuccessorV(unsigned idx) const {
  return getSuccessor(idx);
}
unsigned SwitchInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}
void SwitchInst::setSuccessorV(unsigned idx, BasicBlock *B) {
  setSuccessor(idx, B);
}

// Define these methods here so vtables don't get emitted into every translation
// unit that uses these classes.

GetElementPtrInst *GetElementPtrInst::clone(LLVMContext&) const {
  GetElementPtrInst *New = new(getNumOperands()) GetElementPtrInst(*this);
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

BinaryOperator *BinaryOperator::clone(LLVMContext&) const {
  BinaryOperator *New = Create(getOpcode(), Op<0>(), Op<1>());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

FCmpInst* FCmpInst::clone(LLVMContext &Context) const {
  FCmpInst *New = new FCmpInst(Context, getPredicate(), Op<0>(), Op<1>());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}
ICmpInst* ICmpInst::clone(LLVMContext &Context) const {
  ICmpInst *New = new ICmpInst(Context, getPredicate(), Op<0>(), Op<1>());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

ExtractValueInst *ExtractValueInst::clone(LLVMContext&) const {
  ExtractValueInst *New = new ExtractValueInst(*this);
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}
InsertValueInst *InsertValueInst::clone(LLVMContext&) const {
  InsertValueInst *New = new InsertValueInst(*this);
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

MallocInst *MallocInst::clone(LLVMContext&) const {
  MallocInst *New = new MallocInst(getAllocatedType(),
                                   (Value*)getOperand(0),
                                   getAlignment());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

AllocaInst *AllocaInst::clone(LLVMContext&) const {
  AllocaInst *New = new AllocaInst(getAllocatedType(),
                                   (Value*)getOperand(0),
                                   getAlignment());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

FreeInst *FreeInst::clone(LLVMContext&) const {
  FreeInst *New = new FreeInst(getOperand(0));
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

LoadInst *LoadInst::clone(LLVMContext&) const {
  LoadInst *New = new LoadInst(getOperand(0),
                               Twine(), isVolatile(),
                               getAlignment());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

StoreInst *StoreInst::clone(LLVMContext&) const {
  StoreInst *New = new StoreInst(getOperand(0), getOperand(1),
                                 isVolatile(), getAlignment());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

CastInst *TruncInst::clone(LLVMContext&) const {
  TruncInst *New = new TruncInst(getOperand(0), getType());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

CastInst *ZExtInst::clone(LLVMContext&) const {
  ZExtInst *New = new ZExtInst(getOperand(0), getType());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

CastInst *SExtInst::clone(LLVMContext&) const {
  SExtInst *New = new SExtInst(getOperand(0), getType());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

CastInst *FPTruncInst::clone(LLVMContext&) const {
  FPTruncInst *New = new FPTruncInst(getOperand(0), getType());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

CastInst *FPExtInst::clone(LLVMContext&) const {
  FPExtInst *New = new FPExtInst(getOperand(0), getType());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

CastInst *UIToFPInst::clone(LLVMContext&) const {
  UIToFPInst *New = new UIToFPInst(getOperand(0), getType());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

CastInst *SIToFPInst::clone(LLVMContext&) const {
  SIToFPInst *New = new SIToFPInst(getOperand(0), getType());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

CastInst *FPToUIInst::clone(LLVMContext&) const {
  FPToUIInst *New = new FPToUIInst(getOperand(0), getType());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

CastInst *FPToSIInst::clone(LLVMContext&) const {
  FPToSIInst *New = new FPToSIInst(getOperand(0), getType());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

CastInst *PtrToIntInst::clone(LLVMContext&) const {
  PtrToIntInst *New = new PtrToIntInst(getOperand(0), getType());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

CastInst *IntToPtrInst::clone(LLVMContext&) const {
  IntToPtrInst *New = new IntToPtrInst(getOperand(0), getType());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

CastInst *BitCastInst::clone(LLVMContext&) const {
  BitCastInst *New = new BitCastInst(getOperand(0), getType());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

CallInst *CallInst::clone(LLVMContext&) const {
  CallInst *New = new(getNumOperands()) CallInst(*this);
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

SelectInst *SelectInst::clone(LLVMContext&) const {
  SelectInst *New = SelectInst::Create(getOperand(0),
                                       getOperand(1),
                                       getOperand(2));
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

VAArgInst *VAArgInst::clone(LLVMContext&) const {
  VAArgInst *New = new VAArgInst(getOperand(0), getType());
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

ExtractElementInst *ExtractElementInst::clone(LLVMContext&) const {
  ExtractElementInst *New = ExtractElementInst::Create(getOperand(0),
                                                       getOperand(1));
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

InsertElementInst *InsertElementInst::clone(LLVMContext&) const {
  InsertElementInst *New = InsertElementInst::Create(getOperand(0),
                                                     getOperand(1),
                                                     getOperand(2));
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

ShuffleVectorInst *ShuffleVectorInst::clone(LLVMContext&) const {
  ShuffleVectorInst *New = new ShuffleVectorInst(getOperand(0),
                                                 getOperand(1),
                                                 getOperand(2));
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

PHINode *PHINode::clone(LLVMContext&) const {
  PHINode *New = new PHINode(*this);
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

ReturnInst *ReturnInst::clone(LLVMContext&) const {
  ReturnInst *New = new(getNumOperands()) ReturnInst(*this);
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

BranchInst *BranchInst::clone(LLVMContext&) const {
  unsigned Ops(getNumOperands());
  BranchInst *New = new(Ops, Ops == 1) BranchInst(*this);
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

SwitchInst *SwitchInst::clone(LLVMContext&) const {
  SwitchInst *New = new SwitchInst(*this);
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

InvokeInst *InvokeInst::clone(LLVMContext&) const {
  InvokeInst *New = new(getNumOperands()) InvokeInst(*this);
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

UnwindInst *UnwindInst::clone(LLVMContext &C) const {
  UnwindInst *New = new UnwindInst(C);
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}

UnreachableInst *UnreachableInst::clone(LLVMContext &C) const {
  UnreachableInst *New = new UnreachableInst(C);
  New->SubclassOptionalData = SubclassOptionalData;
  return New;
}
