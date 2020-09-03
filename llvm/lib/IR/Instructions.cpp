//===- Instructions.cpp - Implement the LLVM instructions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements all of the non-inline methods for the LLVM instruction
// classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Instructions.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

using namespace llvm;

//===----------------------------------------------------------------------===//
//                            AllocaInst Class
//===----------------------------------------------------------------------===//

Optional<uint64_t>
AllocaInst::getAllocationSizeInBits(const DataLayout &DL) const {
  uint64_t Size = DL.getTypeAllocSizeInBits(getAllocatedType());
  if (isArrayAllocation()) {
    auto *C = dyn_cast<ConstantInt>(getArraySize());
    if (!C)
      return None;
    Size *= C->getZExtValue();
  }
  return Size;
}

//===----------------------------------------------------------------------===//
//                              SelectInst Class
//===----------------------------------------------------------------------===//

/// areInvalidOperands - Return a string if the specified operands are invalid
/// for a select operation, otherwise return null.
const char *SelectInst::areInvalidOperands(Value *Op0, Value *Op1, Value *Op2) {
  if (Op1->getType() != Op2->getType())
    return "both values to select must have same type";

  if (Op1->getType()->isTokenTy())
    return "select values cannot have token type";

  if (VectorType *VT = dyn_cast<VectorType>(Op0->getType())) {
    // Vector select.
    if (VT->getElementType() != Type::getInt1Ty(Op0->getContext()))
      return "vector select condition element type must be i1";
    VectorType *ET = dyn_cast<VectorType>(Op1->getType());
    if (!ET)
      return "selected values for vector select must be vectors";
    if (ET->getElementCount() != VT->getElementCount())
      return "vector select requires selected vectors to have "
                   "the same vector length as select condition";
  } else if (Op0->getType() != Type::getInt1Ty(Op0->getContext())) {
    return "select condition must be i1 or <n x i1>";
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
//                               PHINode Class
//===----------------------------------------------------------------------===//

PHINode::PHINode(const PHINode &PN)
    : Instruction(PN.getType(), Instruction::PHI, nullptr, PN.getNumOperands()),
      ReservedSpace(PN.getNumOperands()) {
  allocHungoffUses(PN.getNumOperands());
  std::copy(PN.op_begin(), PN.op_end(), op_begin());
  std::copy(PN.block_begin(), PN.block_end(), block_begin());
  SubclassOptionalData = PN.SubclassOptionalData;
}

// removeIncomingValue - Remove an incoming value.  This is useful if a
// predecessor basic block is deleted.
Value *PHINode::removeIncomingValue(unsigned Idx, bool DeletePHIIfEmpty) {
  Value *Removed = getIncomingValue(Idx);

  // Move everything after this operand down.
  //
  // FIXME: we could just swap with the end of the list, then erase.  However,
  // clients might not expect this to happen.  The code as it is thrashes the
  // use/def lists, which is kinda lame.
  std::copy(op_begin() + Idx + 1, op_end(), op_begin() + Idx);
  std::copy(block_begin() + Idx + 1, block_end(), block_begin() + Idx);

  // Nuke the last value.
  Op<-1>().set(nullptr);
  setNumHungOffUseOperands(getNumOperands() - 1);

  // If the PHI node is dead, because it has zero entries, nuke it now.
  if (getNumOperands() == 0 && DeletePHIIfEmpty) {
    // If anyone is using this PHI, make them use a dummy value instead...
    replaceAllUsesWith(UndefValue::get(getType()));
    eraseFromParent();
  }
  return Removed;
}

/// growOperands - grow operands - This grows the operand list in response
/// to a push_back style of operation.  This grows the number of ops by 1.5
/// times.
///
void PHINode::growOperands() {
  unsigned e = getNumOperands();
  unsigned NumOps = e + e / 2;
  if (NumOps < 2) NumOps = 2;      // 2 op PHI nodes are VERY common.

  ReservedSpace = NumOps;
  growHungoffUses(ReservedSpace, /* IsPhi */ true);
}

/// hasConstantValue - If the specified PHI node always merges together the same
/// value, return the value, otherwise return null.
Value *PHINode::hasConstantValue() const {
  // Exploit the fact that phi nodes always have at least one entry.
  Value *ConstantValue = getIncomingValue(0);
  for (unsigned i = 1, e = getNumIncomingValues(); i != e; ++i)
    if (getIncomingValue(i) != ConstantValue && getIncomingValue(i) != this) {
      if (ConstantValue != this)
        return nullptr; // Incoming values not all the same.
       // The case where the first value is this PHI.
      ConstantValue = getIncomingValue(i);
    }
  if (ConstantValue == this)
    return UndefValue::get(getType());
  return ConstantValue;
}

/// hasConstantOrUndefValue - Whether the specified PHI node always merges
/// together the same value, assuming that undefs result in the same value as
/// non-undefs.
/// Unlike \ref hasConstantValue, this does not return a value because the
/// unique non-undef incoming value need not dominate the PHI node.
bool PHINode::hasConstantOrUndefValue() const {
  Value *ConstantValue = nullptr;
  for (unsigned i = 0, e = getNumIncomingValues(); i != e; ++i) {
    Value *Incoming = getIncomingValue(i);
    if (Incoming != this && !isa<UndefValue>(Incoming)) {
      if (ConstantValue && ConstantValue != Incoming)
        return false;
      ConstantValue = Incoming;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
//                       LandingPadInst Implementation
//===----------------------------------------------------------------------===//

LandingPadInst::LandingPadInst(Type *RetTy, unsigned NumReservedValues,
                               const Twine &NameStr, Instruction *InsertBefore)
    : Instruction(RetTy, Instruction::LandingPad, nullptr, 0, InsertBefore) {
  init(NumReservedValues, NameStr);
}

LandingPadInst::LandingPadInst(Type *RetTy, unsigned NumReservedValues,
                               const Twine &NameStr, BasicBlock *InsertAtEnd)
    : Instruction(RetTy, Instruction::LandingPad, nullptr, 0, InsertAtEnd) {
  init(NumReservedValues, NameStr);
}

LandingPadInst::LandingPadInst(const LandingPadInst &LP)
    : Instruction(LP.getType(), Instruction::LandingPad, nullptr,
                  LP.getNumOperands()),
      ReservedSpace(LP.getNumOperands()) {
  allocHungoffUses(LP.getNumOperands());
  Use *OL = getOperandList();
  const Use *InOL = LP.getOperandList();
  for (unsigned I = 0, E = ReservedSpace; I != E; ++I)
    OL[I] = InOL[I];

  setCleanup(LP.isCleanup());
}

LandingPadInst *LandingPadInst::Create(Type *RetTy, unsigned NumReservedClauses,
                                       const Twine &NameStr,
                                       Instruction *InsertBefore) {
  return new LandingPadInst(RetTy, NumReservedClauses, NameStr, InsertBefore);
}

LandingPadInst *LandingPadInst::Create(Type *RetTy, unsigned NumReservedClauses,
                                       const Twine &NameStr,
                                       BasicBlock *InsertAtEnd) {
  return new LandingPadInst(RetTy, NumReservedClauses, NameStr, InsertAtEnd);
}

void LandingPadInst::init(unsigned NumReservedValues, const Twine &NameStr) {
  ReservedSpace = NumReservedValues;
  setNumHungOffUseOperands(0);
  allocHungoffUses(ReservedSpace);
  setName(NameStr);
  setCleanup(false);
}

/// growOperands - grow operands - This grows the operand list in response to a
/// push_back style of operation. This grows the number of ops by 2 times.
void LandingPadInst::growOperands(unsigned Size) {
  unsigned e = getNumOperands();
  if (ReservedSpace >= e + Size) return;
  ReservedSpace = (std::max(e, 1U) + Size / 2) * 2;
  growHungoffUses(ReservedSpace);
}

void LandingPadInst::addClause(Constant *Val) {
  unsigned OpNo = getNumOperands();
  growOperands(1);
  assert(OpNo < ReservedSpace && "Growing didn't work!");
  setNumHungOffUseOperands(getNumOperands() + 1);
  getOperandList()[OpNo] = Val;
}

//===----------------------------------------------------------------------===//
//                        CallBase Implementation
//===----------------------------------------------------------------------===//

CallBase *CallBase::Create(CallBase *CB, ArrayRef<OperandBundleDef> Bundles,
                           Instruction *InsertPt) {
  switch (CB->getOpcode()) {
  case Instruction::Call:
    return CallInst::Create(cast<CallInst>(CB), Bundles, InsertPt);
  case Instruction::Invoke:
    return InvokeInst::Create(cast<InvokeInst>(CB), Bundles, InsertPt);
  case Instruction::CallBr:
    return CallBrInst::Create(cast<CallBrInst>(CB), Bundles, InsertPt);
  default:
    llvm_unreachable("Unknown CallBase sub-class!");
  }
}

Function *CallBase::getCaller() { return getParent()->getParent(); }

unsigned CallBase::getNumSubclassExtraOperandsDynamic() const {
  assert(getOpcode() == Instruction::CallBr && "Unexpected opcode!");
  return cast<CallBrInst>(this)->getNumIndirectDests() + 1;
}

bool CallBase::isIndirectCall() const {
  const Value *V = getCalledOperand();
  if (isa<Function>(V) || isa<Constant>(V))
    return false;
  return !isInlineAsm();
}

/// Tests if this call site must be tail call optimized. Only a CallInst can
/// be tail call optimized.
bool CallBase::isMustTailCall() const {
  if (auto *CI = dyn_cast<CallInst>(this))
    return CI->isMustTailCall();
  return false;
}

/// Tests if this call site is marked as a tail call.
bool CallBase::isTailCall() const {
  if (auto *CI = dyn_cast<CallInst>(this))
    return CI->isTailCall();
  return false;
}

Intrinsic::ID CallBase::getIntrinsicID() const {
  if (auto *F = getCalledFunction())
    return F->getIntrinsicID();
  return Intrinsic::not_intrinsic;
}

bool CallBase::isReturnNonNull() const {
  if (hasRetAttr(Attribute::NonNull))
    return true;

  if (getDereferenceableBytes(AttributeList::ReturnIndex) > 0 &&
           !NullPointerIsDefined(getCaller(),
                                 getType()->getPointerAddressSpace()))
    return true;

  return false;
}

Value *CallBase::getReturnedArgOperand() const {
  unsigned Index;

  if (Attrs.hasAttrSomewhere(Attribute::Returned, &Index) && Index)
    return getArgOperand(Index - AttributeList::FirstArgIndex);
  if (const Function *F = getCalledFunction())
    if (F->getAttributes().hasAttrSomewhere(Attribute::Returned, &Index) &&
        Index)
      return getArgOperand(Index - AttributeList::FirstArgIndex);

  return nullptr;
}

bool CallBase::hasRetAttr(Attribute::AttrKind Kind) const {
  if (Attrs.hasAttribute(AttributeList::ReturnIndex, Kind))
    return true;

  // Look at the callee, if available.
  if (const Function *F = getCalledFunction())
    return F->getAttributes().hasAttribute(AttributeList::ReturnIndex, Kind);
  return false;
}

/// Determine whether the argument or parameter has the given attribute.
bool CallBase::paramHasAttr(unsigned ArgNo, Attribute::AttrKind Kind) const {
  assert(ArgNo < getNumArgOperands() && "Param index out of bounds!");

  if (Attrs.hasParamAttribute(ArgNo, Kind))
    return true;
  if (const Function *F = getCalledFunction())
    return F->getAttributes().hasParamAttribute(ArgNo, Kind);
  return false;
}

bool CallBase::hasFnAttrOnCalledFunction(Attribute::AttrKind Kind) const {
  if (const Function *F = getCalledFunction())
    return F->getAttributes().hasFnAttribute(Kind);
  return false;
}

bool CallBase::hasFnAttrOnCalledFunction(StringRef Kind) const {
  if (const Function *F = getCalledFunction())
    return F->getAttributes().hasFnAttribute(Kind);
  return false;
}

void CallBase::getOperandBundlesAsDefs(
    SmallVectorImpl<OperandBundleDef> &Defs) const {
  for (unsigned i = 0, e = getNumOperandBundles(); i != e; ++i)
    Defs.emplace_back(getOperandBundleAt(i));
}

CallBase::op_iterator
CallBase::populateBundleOperandInfos(ArrayRef<OperandBundleDef> Bundles,
                                     const unsigned BeginIndex) {
  auto It = op_begin() + BeginIndex;
  for (auto &B : Bundles)
    It = std::copy(B.input_begin(), B.input_end(), It);

  auto *ContextImpl = getContext().pImpl;
  auto BI = Bundles.begin();
  unsigned CurrentIndex = BeginIndex;

  for (auto &BOI : bundle_op_infos()) {
    assert(BI != Bundles.end() && "Incorrect allocation?");

    BOI.Tag = ContextImpl->getOrInsertBundleTag(BI->getTag());
    BOI.Begin = CurrentIndex;
    BOI.End = CurrentIndex + BI->input_size();
    CurrentIndex = BOI.End;
    BI++;
  }

  assert(BI == Bundles.end() && "Incorrect allocation?");

  return It;
}

CallBase::BundleOpInfo &CallBase::getBundleOpInfoForOperand(unsigned OpIdx) {
  /// When there isn't many bundles, we do a simple linear search.
  /// Else fallback to a binary-search that use the fact that bundles usually
  /// have similar number of argument to get faster convergence.
  if (bundle_op_info_end() - bundle_op_info_begin() < 8) {
    for (auto &BOI : bundle_op_infos())
      if (BOI.Begin <= OpIdx && OpIdx < BOI.End)
        return BOI;

    llvm_unreachable("Did not find operand bundle for operand!");
  }

  assert(OpIdx >= arg_size() && "the Idx is not in the operand bundles");
  assert(bundle_op_info_end() - bundle_op_info_begin() > 0 &&
         OpIdx < std::prev(bundle_op_info_end())->End &&
         "The Idx isn't in the operand bundle");

  /// We need a decimal number below and to prevent using floating point numbers
  /// we use an intergal value multiplied by this constant.
  constexpr unsigned NumberScaling = 1024;

  bundle_op_iterator Begin = bundle_op_info_begin();
  bundle_op_iterator End = bundle_op_info_end();
  bundle_op_iterator Current;

  while (Begin != End) {
    unsigned ScaledOperandPerBundle =
        NumberScaling * (std::prev(End)->End - Begin->Begin) / (End - Begin);
    Current = Begin + (((OpIdx - Begin->Begin) * NumberScaling) /
                       ScaledOperandPerBundle);
    if (Current >= End)
      Current = std::prev(End);
    assert(Current < End && Current >= Begin &&
           "the operand bundle doesn't cover every value in the range");
    if (OpIdx >= Current->Begin && OpIdx < Current->End)
      break;
    if (OpIdx >= Current->End)
      Begin = Current + 1;
    else
      End = Current;
  }

  assert(OpIdx >= Current->Begin && OpIdx < Current->End &&
         "the operand bundle doesn't cover every value in the range");
  return *Current;
}

//===----------------------------------------------------------------------===//
//                        CallInst Implementation
//===----------------------------------------------------------------------===//

void CallInst::init(FunctionType *FTy, Value *Func, ArrayRef<Value *> Args,
                    ArrayRef<OperandBundleDef> Bundles, const Twine &NameStr) {
  this->FTy = FTy;
  assert(getNumOperands() == Args.size() + CountBundleInputs(Bundles) + 1 &&
         "NumOperands not set up?");
  setCalledOperand(Func);

#ifndef NDEBUG
  assert((Args.size() == FTy->getNumParams() ||
          (FTy->isVarArg() && Args.size() > FTy->getNumParams())) &&
         "Calling a function with bad signature!");

  for (unsigned i = 0; i != Args.size(); ++i)
    assert((i >= FTy->getNumParams() ||
            FTy->getParamType(i) == Args[i]->getType()) &&
           "Calling a function with a bad signature!");
#endif

  llvm::copy(Args, op_begin());

  auto It = populateBundleOperandInfos(Bundles, Args.size());
  (void)It;
  assert(It + 1 == op_end() && "Should add up!");

  setName(NameStr);
}

void CallInst::init(FunctionType *FTy, Value *Func, const Twine &NameStr) {
  this->FTy = FTy;
  assert(getNumOperands() == 1 && "NumOperands not set up?");
  setCalledOperand(Func);

  assert(FTy->getNumParams() == 0 && "Calling a function with bad signature");

  setName(NameStr);
}

CallInst::CallInst(FunctionType *Ty, Value *Func, const Twine &Name,
                   Instruction *InsertBefore)
    : CallBase(Ty->getReturnType(), Instruction::Call,
               OperandTraits<CallBase>::op_end(this) - 1, 1, InsertBefore) {
  init(Ty, Func, Name);
}

CallInst::CallInst(FunctionType *Ty, Value *Func, const Twine &Name,
                   BasicBlock *InsertAtEnd)
    : CallBase(Ty->getReturnType(), Instruction::Call,
               OperandTraits<CallBase>::op_end(this) - 1, 1, InsertAtEnd) {
  init(Ty, Func, Name);
}

CallInst::CallInst(const CallInst &CI)
    : CallBase(CI.Attrs, CI.FTy, CI.getType(), Instruction::Call,
               OperandTraits<CallBase>::op_end(this) - CI.getNumOperands(),
               CI.getNumOperands()) {
  setTailCallKind(CI.getTailCallKind());
  setCallingConv(CI.getCallingConv());

  std::copy(CI.op_begin(), CI.op_end(), op_begin());
  std::copy(CI.bundle_op_info_begin(), CI.bundle_op_info_end(),
            bundle_op_info_begin());
  SubclassOptionalData = CI.SubclassOptionalData;
}

CallInst *CallInst::Create(CallInst *CI, ArrayRef<OperandBundleDef> OpB,
                           Instruction *InsertPt) {
  std::vector<Value *> Args(CI->arg_begin(), CI->arg_end());

  auto *NewCI = CallInst::Create(CI->getFunctionType(), CI->getCalledOperand(),
                                 Args, OpB, CI->getName(), InsertPt);
  NewCI->setTailCallKind(CI->getTailCallKind());
  NewCI->setCallingConv(CI->getCallingConv());
  NewCI->SubclassOptionalData = CI->SubclassOptionalData;
  NewCI->setAttributes(CI->getAttributes());
  NewCI->setDebugLoc(CI->getDebugLoc());
  return NewCI;
}

// Update profile weight for call instruction by scaling it using the ratio
// of S/T. The meaning of "branch_weights" meta data for call instruction is
// transfered to represent call count.
void CallInst::updateProfWeight(uint64_t S, uint64_t T) {
  auto *ProfileData = getMetadata(LLVMContext::MD_prof);
  if (ProfileData == nullptr)
    return;

  auto *ProfDataName = dyn_cast<MDString>(ProfileData->getOperand(0));
  if (!ProfDataName || (!ProfDataName->getString().equals("branch_weights") &&
                        !ProfDataName->getString().equals("VP")))
    return;

  if (T == 0) {
    LLVM_DEBUG(dbgs() << "Attempting to update profile weights will result in "
                         "div by 0. Ignoring. Likely the function "
                      << getParent()->getParent()->getName()
                      << " has 0 entry count, and contains call instructions "
                         "with non-zero prof info.");
    return;
  }

  MDBuilder MDB(getContext());
  SmallVector<Metadata *, 3> Vals;
  Vals.push_back(ProfileData->getOperand(0));
  APInt APS(128, S), APT(128, T);
  if (ProfDataName->getString().equals("branch_weights") &&
      ProfileData->getNumOperands() > 0) {
    // Using APInt::div may be expensive, but most cases should fit 64 bits.
    APInt Val(128, mdconst::dyn_extract<ConstantInt>(ProfileData->getOperand(1))
                       ->getValue()
                       .getZExtValue());
    Val *= APS;
    Vals.push_back(MDB.createConstant(ConstantInt::get(
        Type::getInt64Ty(getContext()), Val.udiv(APT).getLimitedValue())));
  } else if (ProfDataName->getString().equals("VP"))
    for (unsigned i = 1; i < ProfileData->getNumOperands(); i += 2) {
      // The first value is the key of the value profile, which will not change.
      Vals.push_back(ProfileData->getOperand(i));
      // Using APInt::div may be expensive, but most cases should fit 64 bits.
      APInt Val(128,
                mdconst::dyn_extract<ConstantInt>(ProfileData->getOperand(i + 1))
                    ->getValue()
                    .getZExtValue());
      Val *= APS;
      Vals.push_back(MDB.createConstant(
          ConstantInt::get(Type::getInt64Ty(getContext()),
                           Val.udiv(APT).getLimitedValue())));
    }
  setMetadata(LLVMContext::MD_prof, MDNode::get(getContext(), Vals));
}

/// IsConstantOne - Return true only if val is constant int 1
static bool IsConstantOne(Value *val) {
  assert(val && "IsConstantOne does not work with nullptr val");
  const ConstantInt *CVal = dyn_cast<ConstantInt>(val);
  return CVal && CVal->isOne();
}

static Instruction *createMalloc(Instruction *InsertBefore,
                                 BasicBlock *InsertAtEnd, Type *IntPtrTy,
                                 Type *AllocTy, Value *AllocSize,
                                 Value *ArraySize,
                                 ArrayRef<OperandBundleDef> OpB,
                                 Function *MallocF, const Twine &Name) {
  assert(((!InsertBefore && InsertAtEnd) || (InsertBefore && !InsertAtEnd)) &&
         "createMalloc needs either InsertBefore or InsertAtEnd");

  // malloc(type) becomes:
  //       bitcast (i8* malloc(typeSize)) to type*
  // malloc(type, arraySize) becomes:
  //       bitcast (i8* malloc(typeSize*arraySize)) to type*
  if (!ArraySize)
    ArraySize = ConstantInt::get(IntPtrTy, 1);
  else if (ArraySize->getType() != IntPtrTy) {
    if (InsertBefore)
      ArraySize = CastInst::CreateIntegerCast(ArraySize, IntPtrTy, false,
                                              "", InsertBefore);
    else
      ArraySize = CastInst::CreateIntegerCast(ArraySize, IntPtrTy, false,
                                              "", InsertAtEnd);
  }

  if (!IsConstantOne(ArraySize)) {
    if (IsConstantOne(AllocSize)) {
      AllocSize = ArraySize;         // Operand * 1 = Operand
    } else if (Constant *CO = dyn_cast<Constant>(ArraySize)) {
      Constant *Scale = ConstantExpr::getIntegerCast(CO, IntPtrTy,
                                                     false /*ZExt*/);
      // Malloc arg is constant product of type size and array size
      AllocSize = ConstantExpr::getMul(Scale, cast<Constant>(AllocSize));
    } else {
      // Multiply type size by the array size...
      if (InsertBefore)
        AllocSize = BinaryOperator::CreateMul(ArraySize, AllocSize,
                                              "mallocsize", InsertBefore);
      else
        AllocSize = BinaryOperator::CreateMul(ArraySize, AllocSize,
                                              "mallocsize", InsertAtEnd);
    }
  }

  assert(AllocSize->getType() == IntPtrTy && "malloc arg is wrong size");
  // Create the call to Malloc.
  BasicBlock *BB = InsertBefore ? InsertBefore->getParent() : InsertAtEnd;
  Module *M = BB->getParent()->getParent();
  Type *BPTy = Type::getInt8PtrTy(BB->getContext());
  FunctionCallee MallocFunc = MallocF;
  if (!MallocFunc)
    // prototype malloc as "void *malloc(size_t)"
    MallocFunc = M->getOrInsertFunction("malloc", BPTy, IntPtrTy);
  PointerType *AllocPtrType = PointerType::getUnqual(AllocTy);
  CallInst *MCall = nullptr;
  Instruction *Result = nullptr;
  if (InsertBefore) {
    MCall = CallInst::Create(MallocFunc, AllocSize, OpB, "malloccall",
                             InsertBefore);
    Result = MCall;
    if (Result->getType() != AllocPtrType)
      // Create a cast instruction to convert to the right type...
      Result = new BitCastInst(MCall, AllocPtrType, Name, InsertBefore);
  } else {
    MCall = CallInst::Create(MallocFunc, AllocSize, OpB, "malloccall");
    Result = MCall;
    if (Result->getType() != AllocPtrType) {
      InsertAtEnd->getInstList().push_back(MCall);
      // Create a cast instruction to convert to the right type...
      Result = new BitCastInst(MCall, AllocPtrType, Name);
    }
  }
  MCall->setTailCall();
  if (Function *F = dyn_cast<Function>(MallocFunc.getCallee())) {
    MCall->setCallingConv(F->getCallingConv());
    if (!F->returnDoesNotAlias())
      F->setReturnDoesNotAlias();
  }
  assert(!MCall->getType()->isVoidTy() && "Malloc has void return type");

  return Result;
}

/// CreateMalloc - Generate the IR for a call to malloc:
/// 1. Compute the malloc call's argument as the specified type's size,
///    possibly multiplied by the array size if the array size is not
///    constant 1.
/// 2. Call malloc with that argument.
/// 3. Bitcast the result of the malloc call to the specified type.
Instruction *CallInst::CreateMalloc(Instruction *InsertBefore,
                                    Type *IntPtrTy, Type *AllocTy,
                                    Value *AllocSize, Value *ArraySize,
                                    Function *MallocF,
                                    const Twine &Name) {
  return createMalloc(InsertBefore, nullptr, IntPtrTy, AllocTy, AllocSize,
                      ArraySize, None, MallocF, Name);
}
Instruction *CallInst::CreateMalloc(Instruction *InsertBefore,
                                    Type *IntPtrTy, Type *AllocTy,
                                    Value *AllocSize, Value *ArraySize,
                                    ArrayRef<OperandBundleDef> OpB,
                                    Function *MallocF,
                                    const Twine &Name) {
  return createMalloc(InsertBefore, nullptr, IntPtrTy, AllocTy, AllocSize,
                      ArraySize, OpB, MallocF, Name);
}

/// CreateMalloc - Generate the IR for a call to malloc:
/// 1. Compute the malloc call's argument as the specified type's size,
///    possibly multiplied by the array size if the array size is not
///    constant 1.
/// 2. Call malloc with that argument.
/// 3. Bitcast the result of the malloc call to the specified type.
/// Note: This function does not add the bitcast to the basic block, that is the
/// responsibility of the caller.
Instruction *CallInst::CreateMalloc(BasicBlock *InsertAtEnd,
                                    Type *IntPtrTy, Type *AllocTy,
                                    Value *AllocSize, Value *ArraySize,
                                    Function *MallocF, const Twine &Name) {
  return createMalloc(nullptr, InsertAtEnd, IntPtrTy, AllocTy, AllocSize,
                      ArraySize, None, MallocF, Name);
}
Instruction *CallInst::CreateMalloc(BasicBlock *InsertAtEnd,
                                    Type *IntPtrTy, Type *AllocTy,
                                    Value *AllocSize, Value *ArraySize,
                                    ArrayRef<OperandBundleDef> OpB,
                                    Function *MallocF, const Twine &Name) {
  return createMalloc(nullptr, InsertAtEnd, IntPtrTy, AllocTy, AllocSize,
                      ArraySize, OpB, MallocF, Name);
}

static Instruction *createFree(Value *Source,
                               ArrayRef<OperandBundleDef> Bundles,
                               Instruction *InsertBefore,
                               BasicBlock *InsertAtEnd) {
  assert(((!InsertBefore && InsertAtEnd) || (InsertBefore && !InsertAtEnd)) &&
         "createFree needs either InsertBefore or InsertAtEnd");
  assert(Source->getType()->isPointerTy() &&
         "Can not free something of nonpointer type!");

  BasicBlock *BB = InsertBefore ? InsertBefore->getParent() : InsertAtEnd;
  Module *M = BB->getParent()->getParent();

  Type *VoidTy = Type::getVoidTy(M->getContext());
  Type *IntPtrTy = Type::getInt8PtrTy(M->getContext());
  // prototype free as "void free(void*)"
  FunctionCallee FreeFunc = M->getOrInsertFunction("free", VoidTy, IntPtrTy);
  CallInst *Result = nullptr;
  Value *PtrCast = Source;
  if (InsertBefore) {
    if (Source->getType() != IntPtrTy)
      PtrCast = new BitCastInst(Source, IntPtrTy, "", InsertBefore);
    Result = CallInst::Create(FreeFunc, PtrCast, Bundles, "", InsertBefore);
  } else {
    if (Source->getType() != IntPtrTy)
      PtrCast = new BitCastInst(Source, IntPtrTy, "", InsertAtEnd);
    Result = CallInst::Create(FreeFunc, PtrCast, Bundles, "");
  }
  Result->setTailCall();
  if (Function *F = dyn_cast<Function>(FreeFunc.getCallee()))
    Result->setCallingConv(F->getCallingConv());

  return Result;
}

/// CreateFree - Generate the IR for a call to the builtin free function.
Instruction *CallInst::CreateFree(Value *Source, Instruction *InsertBefore) {
  return createFree(Source, None, InsertBefore, nullptr);
}
Instruction *CallInst::CreateFree(Value *Source,
                                  ArrayRef<OperandBundleDef> Bundles,
                                  Instruction *InsertBefore) {
  return createFree(Source, Bundles, InsertBefore, nullptr);
}

/// CreateFree - Generate the IR for a call to the builtin free function.
/// Note: This function does not add the call to the basic block, that is the
/// responsibility of the caller.
Instruction *CallInst::CreateFree(Value *Source, BasicBlock *InsertAtEnd) {
  Instruction *FreeCall = createFree(Source, None, nullptr, InsertAtEnd);
  assert(FreeCall && "CreateFree did not create a CallInst");
  return FreeCall;
}
Instruction *CallInst::CreateFree(Value *Source,
                                  ArrayRef<OperandBundleDef> Bundles,
                                  BasicBlock *InsertAtEnd) {
  Instruction *FreeCall = createFree(Source, Bundles, nullptr, InsertAtEnd);
  assert(FreeCall && "CreateFree did not create a CallInst");
  return FreeCall;
}

//===----------------------------------------------------------------------===//
//                        InvokeInst Implementation
//===----------------------------------------------------------------------===//

void InvokeInst::init(FunctionType *FTy, Value *Fn, BasicBlock *IfNormal,
                      BasicBlock *IfException, ArrayRef<Value *> Args,
                      ArrayRef<OperandBundleDef> Bundles,
                      const Twine &NameStr) {
  this->FTy = FTy;

  assert((int)getNumOperands() ==
             ComputeNumOperands(Args.size(), CountBundleInputs(Bundles)) &&
         "NumOperands not set up?");
  setNormalDest(IfNormal);
  setUnwindDest(IfException);
  setCalledOperand(Fn);

#ifndef NDEBUG
  assert(((Args.size() == FTy->getNumParams()) ||
          (FTy->isVarArg() && Args.size() > FTy->getNumParams())) &&
         "Invoking a function with bad signature");

  for (unsigned i = 0, e = Args.size(); i != e; i++)
    assert((i >= FTy->getNumParams() ||
            FTy->getParamType(i) == Args[i]->getType()) &&
           "Invoking a function with a bad signature!");
#endif

  llvm::copy(Args, op_begin());

  auto It = populateBundleOperandInfos(Bundles, Args.size());
  (void)It;
  assert(It + 3 == op_end() && "Should add up!");

  setName(NameStr);
}

InvokeInst::InvokeInst(const InvokeInst &II)
    : CallBase(II.Attrs, II.FTy, II.getType(), Instruction::Invoke,
               OperandTraits<CallBase>::op_end(this) - II.getNumOperands(),
               II.getNumOperands()) {
  setCallingConv(II.getCallingConv());
  std::copy(II.op_begin(), II.op_end(), op_begin());
  std::copy(II.bundle_op_info_begin(), II.bundle_op_info_end(),
            bundle_op_info_begin());
  SubclassOptionalData = II.SubclassOptionalData;
}

InvokeInst *InvokeInst::Create(InvokeInst *II, ArrayRef<OperandBundleDef> OpB,
                               Instruction *InsertPt) {
  std::vector<Value *> Args(II->arg_begin(), II->arg_end());

  auto *NewII = InvokeInst::Create(
      II->getFunctionType(), II->getCalledOperand(), II->getNormalDest(),
      II->getUnwindDest(), Args, OpB, II->getName(), InsertPt);
  NewII->setCallingConv(II->getCallingConv());
  NewII->SubclassOptionalData = II->SubclassOptionalData;
  NewII->setAttributes(II->getAttributes());
  NewII->setDebugLoc(II->getDebugLoc());
  return NewII;
}


LandingPadInst *InvokeInst::getLandingPadInst() const {
  return cast<LandingPadInst>(getUnwindDest()->getFirstNonPHI());
}

//===----------------------------------------------------------------------===//
//                        CallBrInst Implementation
//===----------------------------------------------------------------------===//

void CallBrInst::init(FunctionType *FTy, Value *Fn, BasicBlock *Fallthrough,
                      ArrayRef<BasicBlock *> IndirectDests,
                      ArrayRef<Value *> Args,
                      ArrayRef<OperandBundleDef> Bundles,
                      const Twine &NameStr) {
  this->FTy = FTy;

  assert((int)getNumOperands() ==
             ComputeNumOperands(Args.size(), IndirectDests.size(),
                                CountBundleInputs(Bundles)) &&
         "NumOperands not set up?");
  NumIndirectDests = IndirectDests.size();
  setDefaultDest(Fallthrough);
  for (unsigned i = 0; i != NumIndirectDests; ++i)
    setIndirectDest(i, IndirectDests[i]);
  setCalledOperand(Fn);

#ifndef NDEBUG
  assert(((Args.size() == FTy->getNumParams()) ||
          (FTy->isVarArg() && Args.size() > FTy->getNumParams())) &&
         "Calling a function with bad signature");

  for (unsigned i = 0, e = Args.size(); i != e; i++)
    assert((i >= FTy->getNumParams() ||
            FTy->getParamType(i) == Args[i]->getType()) &&
           "Calling a function with a bad signature!");
#endif

  std::copy(Args.begin(), Args.end(), op_begin());

  auto It = populateBundleOperandInfos(Bundles, Args.size());
  (void)It;
  assert(It + 2 + IndirectDests.size() == op_end() && "Should add up!");

  setName(NameStr);
}

void CallBrInst::updateArgBlockAddresses(unsigned i, BasicBlock *B) {
  assert(getNumIndirectDests() > i && "IndirectDest # out of range for callbr");
  if (BasicBlock *OldBB = getIndirectDest(i)) {
    BlockAddress *Old = BlockAddress::get(OldBB);
    BlockAddress *New = BlockAddress::get(B);
    for (unsigned ArgNo = 0, e = getNumArgOperands(); ArgNo != e; ++ArgNo)
      if (dyn_cast<BlockAddress>(getArgOperand(ArgNo)) == Old)
        setArgOperand(ArgNo, New);
  }
}

CallBrInst::CallBrInst(const CallBrInst &CBI)
    : CallBase(CBI.Attrs, CBI.FTy, CBI.getType(), Instruction::CallBr,
               OperandTraits<CallBase>::op_end(this) - CBI.getNumOperands(),
               CBI.getNumOperands()) {
  setCallingConv(CBI.getCallingConv());
  std::copy(CBI.op_begin(), CBI.op_end(), op_begin());
  std::copy(CBI.bundle_op_info_begin(), CBI.bundle_op_info_end(),
            bundle_op_info_begin());
  SubclassOptionalData = CBI.SubclassOptionalData;
  NumIndirectDests = CBI.NumIndirectDests;
}

CallBrInst *CallBrInst::Create(CallBrInst *CBI, ArrayRef<OperandBundleDef> OpB,
                               Instruction *InsertPt) {
  std::vector<Value *> Args(CBI->arg_begin(), CBI->arg_end());

  auto *NewCBI = CallBrInst::Create(
      CBI->getFunctionType(), CBI->getCalledOperand(), CBI->getDefaultDest(),
      CBI->getIndirectDests(), Args, OpB, CBI->getName(), InsertPt);
  NewCBI->setCallingConv(CBI->getCallingConv());
  NewCBI->SubclassOptionalData = CBI->SubclassOptionalData;
  NewCBI->setAttributes(CBI->getAttributes());
  NewCBI->setDebugLoc(CBI->getDebugLoc());
  NewCBI->NumIndirectDests = CBI->NumIndirectDests;
  return NewCBI;
}

//===----------------------------------------------------------------------===//
//                        ReturnInst Implementation
//===----------------------------------------------------------------------===//

ReturnInst::ReturnInst(const ReturnInst &RI)
    : Instruction(Type::getVoidTy(RI.getContext()), Instruction::Ret,
                  OperandTraits<ReturnInst>::op_end(this) - RI.getNumOperands(),
                  RI.getNumOperands()) {
  if (RI.getNumOperands())
    Op<0>() = RI.Op<0>();
  SubclassOptionalData = RI.SubclassOptionalData;
}

ReturnInst::ReturnInst(LLVMContext &C, Value *retVal, Instruction *InsertBefore)
    : Instruction(Type::getVoidTy(C), Instruction::Ret,
                  OperandTraits<ReturnInst>::op_end(this) - !!retVal, !!retVal,
                  InsertBefore) {
  if (retVal)
    Op<0>() = retVal;
}

ReturnInst::ReturnInst(LLVMContext &C, Value *retVal, BasicBlock *InsertAtEnd)
    : Instruction(Type::getVoidTy(C), Instruction::Ret,
                  OperandTraits<ReturnInst>::op_end(this) - !!retVal, !!retVal,
                  InsertAtEnd) {
  if (retVal)
    Op<0>() = retVal;
}

ReturnInst::ReturnInst(LLVMContext &Context, BasicBlock *InsertAtEnd)
    : Instruction(Type::getVoidTy(Context), Instruction::Ret,
                  OperandTraits<ReturnInst>::op_end(this), 0, InsertAtEnd) {}

//===----------------------------------------------------------------------===//
//                        ResumeInst Implementation
//===----------------------------------------------------------------------===//

ResumeInst::ResumeInst(const ResumeInst &RI)
    : Instruction(Type::getVoidTy(RI.getContext()), Instruction::Resume,
                  OperandTraits<ResumeInst>::op_begin(this), 1) {
  Op<0>() = RI.Op<0>();
}

ResumeInst::ResumeInst(Value *Exn, Instruction *InsertBefore)
    : Instruction(Type::getVoidTy(Exn->getContext()), Instruction::Resume,
                  OperandTraits<ResumeInst>::op_begin(this), 1, InsertBefore) {
  Op<0>() = Exn;
}

ResumeInst::ResumeInst(Value *Exn, BasicBlock *InsertAtEnd)
    : Instruction(Type::getVoidTy(Exn->getContext()), Instruction::Resume,
                  OperandTraits<ResumeInst>::op_begin(this), 1, InsertAtEnd) {
  Op<0>() = Exn;
}

//===----------------------------------------------------------------------===//
//                        CleanupReturnInst Implementation
//===----------------------------------------------------------------------===//

CleanupReturnInst::CleanupReturnInst(const CleanupReturnInst &CRI)
    : Instruction(CRI.getType(), Instruction::CleanupRet,
                  OperandTraits<CleanupReturnInst>::op_end(this) -
                      CRI.getNumOperands(),
                  CRI.getNumOperands()) {
  setSubclassData<Instruction::OpaqueField>(
      CRI.getSubclassData<Instruction::OpaqueField>());
  Op<0>() = CRI.Op<0>();
  if (CRI.hasUnwindDest())
    Op<1>() = CRI.Op<1>();
}

void CleanupReturnInst::init(Value *CleanupPad, BasicBlock *UnwindBB) {
  if (UnwindBB)
    setSubclassData<UnwindDestField>(true);

  Op<0>() = CleanupPad;
  if (UnwindBB)
    Op<1>() = UnwindBB;
}

CleanupReturnInst::CleanupReturnInst(Value *CleanupPad, BasicBlock *UnwindBB,
                                     unsigned Values, Instruction *InsertBefore)
    : Instruction(Type::getVoidTy(CleanupPad->getContext()),
                  Instruction::CleanupRet,
                  OperandTraits<CleanupReturnInst>::op_end(this) - Values,
                  Values, InsertBefore) {
  init(CleanupPad, UnwindBB);
}

CleanupReturnInst::CleanupReturnInst(Value *CleanupPad, BasicBlock *UnwindBB,
                                     unsigned Values, BasicBlock *InsertAtEnd)
    : Instruction(Type::getVoidTy(CleanupPad->getContext()),
                  Instruction::CleanupRet,
                  OperandTraits<CleanupReturnInst>::op_end(this) - Values,
                  Values, InsertAtEnd) {
  init(CleanupPad, UnwindBB);
}

//===----------------------------------------------------------------------===//
//                        CatchReturnInst Implementation
//===----------------------------------------------------------------------===//
void CatchReturnInst::init(Value *CatchPad, BasicBlock *BB) {
  Op<0>() = CatchPad;
  Op<1>() = BB;
}

CatchReturnInst::CatchReturnInst(const CatchReturnInst &CRI)
    : Instruction(Type::getVoidTy(CRI.getContext()), Instruction::CatchRet,
                  OperandTraits<CatchReturnInst>::op_begin(this), 2) {
  Op<0>() = CRI.Op<0>();
  Op<1>() = CRI.Op<1>();
}

CatchReturnInst::CatchReturnInst(Value *CatchPad, BasicBlock *BB,
                                 Instruction *InsertBefore)
    : Instruction(Type::getVoidTy(BB->getContext()), Instruction::CatchRet,
                  OperandTraits<CatchReturnInst>::op_begin(this), 2,
                  InsertBefore) {
  init(CatchPad, BB);
}

CatchReturnInst::CatchReturnInst(Value *CatchPad, BasicBlock *BB,
                                 BasicBlock *InsertAtEnd)
    : Instruction(Type::getVoidTy(BB->getContext()), Instruction::CatchRet,
                  OperandTraits<CatchReturnInst>::op_begin(this), 2,
                  InsertAtEnd) {
  init(CatchPad, BB);
}

//===----------------------------------------------------------------------===//
//                       CatchSwitchInst Implementation
//===----------------------------------------------------------------------===//

CatchSwitchInst::CatchSwitchInst(Value *ParentPad, BasicBlock *UnwindDest,
                                 unsigned NumReservedValues,
                                 const Twine &NameStr,
                                 Instruction *InsertBefore)
    : Instruction(ParentPad->getType(), Instruction::CatchSwitch, nullptr, 0,
                  InsertBefore) {
  if (UnwindDest)
    ++NumReservedValues;
  init(ParentPad, UnwindDest, NumReservedValues + 1);
  setName(NameStr);
}

CatchSwitchInst::CatchSwitchInst(Value *ParentPad, BasicBlock *UnwindDest,
                                 unsigned NumReservedValues,
                                 const Twine &NameStr, BasicBlock *InsertAtEnd)
    : Instruction(ParentPad->getType(), Instruction::CatchSwitch, nullptr, 0,
                  InsertAtEnd) {
  if (UnwindDest)
    ++NumReservedValues;
  init(ParentPad, UnwindDest, NumReservedValues + 1);
  setName(NameStr);
}

CatchSwitchInst::CatchSwitchInst(const CatchSwitchInst &CSI)
    : Instruction(CSI.getType(), Instruction::CatchSwitch, nullptr,
                  CSI.getNumOperands()) {
  init(CSI.getParentPad(), CSI.getUnwindDest(), CSI.getNumOperands());
  setNumHungOffUseOperands(ReservedSpace);
  Use *OL = getOperandList();
  const Use *InOL = CSI.getOperandList();
  for (unsigned I = 1, E = ReservedSpace; I != E; ++I)
    OL[I] = InOL[I];
}

void CatchSwitchInst::init(Value *ParentPad, BasicBlock *UnwindDest,
                           unsigned NumReservedValues) {
  assert(ParentPad && NumReservedValues);

  ReservedSpace = NumReservedValues;
  setNumHungOffUseOperands(UnwindDest ? 2 : 1);
  allocHungoffUses(ReservedSpace);

  Op<0>() = ParentPad;
  if (UnwindDest) {
    setSubclassData<UnwindDestField>(true);
    setUnwindDest(UnwindDest);
  }
}

/// growOperands - grow operands - This grows the operand list in response to a
/// push_back style of operation. This grows the number of ops by 2 times.
void CatchSwitchInst::growOperands(unsigned Size) {
  unsigned NumOperands = getNumOperands();
  assert(NumOperands >= 1);
  if (ReservedSpace >= NumOperands + Size)
    return;
  ReservedSpace = (NumOperands + Size / 2) * 2;
  growHungoffUses(ReservedSpace);
}

void CatchSwitchInst::addHandler(BasicBlock *Handler) {
  unsigned OpNo = getNumOperands();
  growOperands(1);
  assert(OpNo < ReservedSpace && "Growing didn't work!");
  setNumHungOffUseOperands(getNumOperands() + 1);
  getOperandList()[OpNo] = Handler;
}

void CatchSwitchInst::removeHandler(handler_iterator HI) {
  // Move all subsequent handlers up one.
  Use *EndDst = op_end() - 1;
  for (Use *CurDst = HI.getCurrent(); CurDst != EndDst; ++CurDst)
    *CurDst = *(CurDst + 1);
  // Null out the last handler use.
  *EndDst = nullptr;

  setNumHungOffUseOperands(getNumOperands() - 1);
}

//===----------------------------------------------------------------------===//
//                        FuncletPadInst Implementation
//===----------------------------------------------------------------------===//
void FuncletPadInst::init(Value *ParentPad, ArrayRef<Value *> Args,
                          const Twine &NameStr) {
  assert(getNumOperands() == 1 + Args.size() && "NumOperands not set up?");
  llvm::copy(Args, op_begin());
  setParentPad(ParentPad);
  setName(NameStr);
}

FuncletPadInst::FuncletPadInst(const FuncletPadInst &FPI)
    : Instruction(FPI.getType(), FPI.getOpcode(),
                  OperandTraits<FuncletPadInst>::op_end(this) -
                      FPI.getNumOperands(),
                  FPI.getNumOperands()) {
  std::copy(FPI.op_begin(), FPI.op_end(), op_begin());
  setParentPad(FPI.getParentPad());
}

FuncletPadInst::FuncletPadInst(Instruction::FuncletPadOps Op, Value *ParentPad,
                               ArrayRef<Value *> Args, unsigned Values,
                               const Twine &NameStr, Instruction *InsertBefore)
    : Instruction(ParentPad->getType(), Op,
                  OperandTraits<FuncletPadInst>::op_end(this) - Values, Values,
                  InsertBefore) {
  init(ParentPad, Args, NameStr);
}

FuncletPadInst::FuncletPadInst(Instruction::FuncletPadOps Op, Value *ParentPad,
                               ArrayRef<Value *> Args, unsigned Values,
                               const Twine &NameStr, BasicBlock *InsertAtEnd)
    : Instruction(ParentPad->getType(), Op,
                  OperandTraits<FuncletPadInst>::op_end(this) - Values, Values,
                  InsertAtEnd) {
  init(ParentPad, Args, NameStr);
}

//===----------------------------------------------------------------------===//
//                      UnreachableInst Implementation
//===----------------------------------------------------------------------===//

UnreachableInst::UnreachableInst(LLVMContext &Context,
                                 Instruction *InsertBefore)
    : Instruction(Type::getVoidTy(Context), Instruction::Unreachable, nullptr,
                  0, InsertBefore) {}
UnreachableInst::UnreachableInst(LLVMContext &Context, BasicBlock *InsertAtEnd)
    : Instruction(Type::getVoidTy(Context), Instruction::Unreachable, nullptr,
                  0, InsertAtEnd) {}

//===----------------------------------------------------------------------===//
//                        BranchInst Implementation
//===----------------------------------------------------------------------===//

void BranchInst::AssertOK() {
  if (isConditional())
    assert(getCondition()->getType()->isIntegerTy(1) &&
           "May only branch on boolean predicates!");
}

BranchInst::BranchInst(BasicBlock *IfTrue, Instruction *InsertBefore)
    : Instruction(Type::getVoidTy(IfTrue->getContext()), Instruction::Br,
                  OperandTraits<BranchInst>::op_end(this) - 1, 1,
                  InsertBefore) {
  assert(IfTrue && "Branch destination may not be null!");
  Op<-1>() = IfTrue;
}

BranchInst::BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
                       Instruction *InsertBefore)
    : Instruction(Type::getVoidTy(IfTrue->getContext()), Instruction::Br,
                  OperandTraits<BranchInst>::op_end(this) - 3, 3,
                  InsertBefore) {
  Op<-1>() = IfTrue;
  Op<-2>() = IfFalse;
  Op<-3>() = Cond;
#ifndef NDEBUG
  AssertOK();
#endif
}

BranchInst::BranchInst(BasicBlock *IfTrue, BasicBlock *InsertAtEnd)
    : Instruction(Type::getVoidTy(IfTrue->getContext()), Instruction::Br,
                  OperandTraits<BranchInst>::op_end(this) - 1, 1, InsertAtEnd) {
  assert(IfTrue && "Branch destination may not be null!");
  Op<-1>() = IfTrue;
}

BranchInst::BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
                       BasicBlock *InsertAtEnd)
    : Instruction(Type::getVoidTy(IfTrue->getContext()), Instruction::Br,
                  OperandTraits<BranchInst>::op_end(this) - 3, 3, InsertAtEnd) {
  Op<-1>() = IfTrue;
  Op<-2>() = IfFalse;
  Op<-3>() = Cond;
#ifndef NDEBUG
  AssertOK();
#endif
}

BranchInst::BranchInst(const BranchInst &BI)
    : Instruction(Type::getVoidTy(BI.getContext()), Instruction::Br,
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

void BranchInst::swapSuccessors() {
  assert(isConditional() &&
         "Cannot swap successors of an unconditional branch");
  Op<-1>().swap(Op<-2>());

  // Update profile metadata if present and it matches our structural
  // expectations.
  swapProfMetadata();
}

//===----------------------------------------------------------------------===//
//                        AllocaInst Implementation
//===----------------------------------------------------------------------===//

static Value *getAISize(LLVMContext &Context, Value *Amt) {
  if (!Amt)
    Amt = ConstantInt::get(Type::getInt32Ty(Context), 1);
  else {
    assert(!isa<BasicBlock>(Amt) &&
           "Passed basic block into allocation size parameter! Use other ctor");
    assert(Amt->getType()->isIntegerTy() &&
           "Allocation array size is not an integer!");
  }
  return Amt;
}

static Align computeAllocaDefaultAlign(Type *Ty, BasicBlock *BB) {
  assert(BB && "Insertion BB cannot be null when alignment not provided!");
  assert(BB->getParent() &&
         "BB must be in a Function when alignment not provided!");
  const DataLayout &DL = BB->getModule()->getDataLayout();
  return DL.getPrefTypeAlign(Ty);
}

static Align computeAllocaDefaultAlign(Type *Ty, Instruction *I) {
  assert(I && "Insertion position cannot be null when alignment not provided!");
  return computeAllocaDefaultAlign(Ty, I->getParent());
}

AllocaInst::AllocaInst(Type *Ty, unsigned AddrSpace, const Twine &Name,
                       Instruction *InsertBefore)
  : AllocaInst(Ty, AddrSpace, /*ArraySize=*/nullptr, Name, InsertBefore) {}

AllocaInst::AllocaInst(Type *Ty, unsigned AddrSpace, const Twine &Name,
                       BasicBlock *InsertAtEnd)
  : AllocaInst(Ty, AddrSpace, /*ArraySize=*/nullptr, Name, InsertAtEnd) {}

AllocaInst::AllocaInst(Type *Ty, unsigned AddrSpace, Value *ArraySize,
                       const Twine &Name, Instruction *InsertBefore)
    : AllocaInst(Ty, AddrSpace, ArraySize,
                 computeAllocaDefaultAlign(Ty, InsertBefore), Name,
                 InsertBefore) {}

AllocaInst::AllocaInst(Type *Ty, unsigned AddrSpace, Value *ArraySize,
                       const Twine &Name, BasicBlock *InsertAtEnd)
    : AllocaInst(Ty, AddrSpace, ArraySize,
                 computeAllocaDefaultAlign(Ty, InsertAtEnd), Name,
                 InsertAtEnd) {}

AllocaInst::AllocaInst(Type *Ty, unsigned AddrSpace, Value *ArraySize,
                       Align Align, const Twine &Name,
                       Instruction *InsertBefore)
    : UnaryInstruction(PointerType::get(Ty, AddrSpace), Alloca,
                       getAISize(Ty->getContext(), ArraySize), InsertBefore),
      AllocatedType(Ty) {
  setAlignment(Align);
  assert(!Ty->isVoidTy() && "Cannot allocate void!");
  setName(Name);
}

AllocaInst::AllocaInst(Type *Ty, unsigned AddrSpace, Value *ArraySize,
                       Align Align, const Twine &Name, BasicBlock *InsertAtEnd)
    : UnaryInstruction(PointerType::get(Ty, AddrSpace), Alloca,
                       getAISize(Ty->getContext(), ArraySize), InsertAtEnd),
      AllocatedType(Ty) {
  setAlignment(Align);
  assert(!Ty->isVoidTy() && "Cannot allocate void!");
  setName(Name);
}


bool AllocaInst::isArrayAllocation() const {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(getOperand(0)))
    return !CI->isOne();
  return true;
}

/// isStaticAlloca - Return true if this alloca is in the entry block of the
/// function and is a constant size.  If so, the code generator will fold it
/// into the prolog/epilog code, so it is basically free.
bool AllocaInst::isStaticAlloca() const {
  // Must be constant size.
  if (!isa<ConstantInt>(getArraySize())) return false;

  // Must be in the entry block.
  const BasicBlock *Parent = getParent();
  return Parent == &Parent->getParent()->front() && !isUsedWithInAlloca();
}

//===----------------------------------------------------------------------===//
//                           LoadInst Implementation
//===----------------------------------------------------------------------===//

void LoadInst::AssertOK() {
  assert(getOperand(0)->getType()->isPointerTy() &&
         "Ptr must have pointer type.");
  assert(!(isAtomic() && getAlignment() == 0) &&
         "Alignment required for atomic load");
}

static Align computeLoadStoreDefaultAlign(Type *Ty, BasicBlock *BB) {
  assert(BB && "Insertion BB cannot be null when alignment not provided!");
  assert(BB->getParent() &&
         "BB must be in a Function when alignment not provided!");
  const DataLayout &DL = BB->getModule()->getDataLayout();
  return DL.getABITypeAlign(Ty);
}

static Align computeLoadStoreDefaultAlign(Type *Ty, Instruction *I) {
  assert(I && "Insertion position cannot be null when alignment not provided!");
  return computeLoadStoreDefaultAlign(Ty, I->getParent());
}

LoadInst::LoadInst(Type *Ty, Value *Ptr, const Twine &Name,
                   Instruction *InsertBef)
    : LoadInst(Ty, Ptr, Name, /*isVolatile=*/false, InsertBef) {}

LoadInst::LoadInst(Type *Ty, Value *Ptr, const Twine &Name,
                   BasicBlock *InsertAE)
    : LoadInst(Ty, Ptr, Name, /*isVolatile=*/false, InsertAE) {}

LoadInst::LoadInst(Type *Ty, Value *Ptr, const Twine &Name, bool isVolatile,
                   Instruction *InsertBef)
    : LoadInst(Ty, Ptr, Name, isVolatile,
               computeLoadStoreDefaultAlign(Ty, InsertBef), InsertBef) {}

LoadInst::LoadInst(Type *Ty, Value *Ptr, const Twine &Name, bool isVolatile,
                   BasicBlock *InsertAE)
    : LoadInst(Ty, Ptr, Name, isVolatile,
               computeLoadStoreDefaultAlign(Ty, InsertAE), InsertAE) {}

LoadInst::LoadInst(Type *Ty, Value *Ptr, const Twine &Name, bool isVolatile,
                   Align Align, Instruction *InsertBef)
    : LoadInst(Ty, Ptr, Name, isVolatile, Align, AtomicOrdering::NotAtomic,
               SyncScope::System, InsertBef) {}

LoadInst::LoadInst(Type *Ty, Value *Ptr, const Twine &Name, bool isVolatile,
                   Align Align, BasicBlock *InsertAE)
    : LoadInst(Ty, Ptr, Name, isVolatile, Align, AtomicOrdering::NotAtomic,
               SyncScope::System, InsertAE) {}

LoadInst::LoadInst(Type *Ty, Value *Ptr, const Twine &Name, bool isVolatile,
                   Align Align, AtomicOrdering Order, SyncScope::ID SSID,
                   Instruction *InsertBef)
    : UnaryInstruction(Ty, Load, Ptr, InsertBef) {
  assert(Ty == cast<PointerType>(Ptr->getType())->getElementType());
  setVolatile(isVolatile);
  setAlignment(Align);
  setAtomic(Order, SSID);
  AssertOK();
  setName(Name);
}

LoadInst::LoadInst(Type *Ty, Value *Ptr, const Twine &Name, bool isVolatile,
                   Align Align, AtomicOrdering Order, SyncScope::ID SSID,
                   BasicBlock *InsertAE)
    : UnaryInstruction(Ty, Load, Ptr, InsertAE) {
  assert(Ty == cast<PointerType>(Ptr->getType())->getElementType());
  setVolatile(isVolatile);
  setAlignment(Align);
  setAtomic(Order, SSID);
  AssertOK();
  setName(Name);
}

//===----------------------------------------------------------------------===//
//                           StoreInst Implementation
//===----------------------------------------------------------------------===//

void StoreInst::AssertOK() {
  assert(getOperand(0) && getOperand(1) && "Both operands must be non-null!");
  assert(getOperand(1)->getType()->isPointerTy() &&
         "Ptr must have pointer type!");
  assert(getOperand(0)->getType() ==
                 cast<PointerType>(getOperand(1)->getType())->getElementType()
         && "Ptr must be a pointer to Val type!");
  assert(!(isAtomic() && getAlignment() == 0) &&
         "Alignment required for atomic store");
}

StoreInst::StoreInst(Value *val, Value *addr, Instruction *InsertBefore)
    : StoreInst(val, addr, /*isVolatile=*/false, InsertBefore) {}

StoreInst::StoreInst(Value *val, Value *addr, BasicBlock *InsertAtEnd)
    : StoreInst(val, addr, /*isVolatile=*/false, InsertAtEnd) {}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile,
                     Instruction *InsertBefore)
    : StoreInst(val, addr, isVolatile,
                computeLoadStoreDefaultAlign(val->getType(), InsertBefore),
                InsertBefore) {}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile,
                     BasicBlock *InsertAtEnd)
    : StoreInst(val, addr, isVolatile,
                computeLoadStoreDefaultAlign(val->getType(), InsertAtEnd),
                InsertAtEnd) {}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile, Align Align,
                     Instruction *InsertBefore)
    : StoreInst(val, addr, isVolatile, Align, AtomicOrdering::NotAtomic,
                SyncScope::System, InsertBefore) {}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile, Align Align,
                     BasicBlock *InsertAtEnd)
    : StoreInst(val, addr, isVolatile, Align, AtomicOrdering::NotAtomic,
                SyncScope::System, InsertAtEnd) {}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile, Align Align,
                     AtomicOrdering Order, SyncScope::ID SSID,
                     Instruction *InsertBefore)
    : Instruction(Type::getVoidTy(val->getContext()), Store,
                  OperandTraits<StoreInst>::op_begin(this),
                  OperandTraits<StoreInst>::operands(this), InsertBefore) {
  Op<0>() = val;
  Op<1>() = addr;
  setVolatile(isVolatile);
  setAlignment(Align);
  setAtomic(Order, SSID);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile, Align Align,
                     AtomicOrdering Order, SyncScope::ID SSID,
                     BasicBlock *InsertAtEnd)
    : Instruction(Type::getVoidTy(val->getContext()), Store,
                  OperandTraits<StoreInst>::op_begin(this),
                  OperandTraits<StoreInst>::operands(this), InsertAtEnd) {
  Op<0>() = val;
  Op<1>() = addr;
  setVolatile(isVolatile);
  setAlignment(Align);
  setAtomic(Order, SSID);
  AssertOK();
}


//===----------------------------------------------------------------------===//
//                       AtomicCmpXchgInst Implementation
//===----------------------------------------------------------------------===//

void AtomicCmpXchgInst::Init(Value *Ptr, Value *Cmp, Value *NewVal,
                             Align Alignment, AtomicOrdering SuccessOrdering,
                             AtomicOrdering FailureOrdering,
                             SyncScope::ID SSID) {
  Op<0>() = Ptr;
  Op<1>() = Cmp;
  Op<2>() = NewVal;
  setSuccessOrdering(SuccessOrdering);
  setFailureOrdering(FailureOrdering);
  setSyncScopeID(SSID);
  setAlignment(Alignment);

  assert(getOperand(0) && getOperand(1) && getOperand(2) &&
         "All operands must be non-null!");
  assert(getOperand(0)->getType()->isPointerTy() &&
         "Ptr must have pointer type!");
  assert(getOperand(1)->getType() ==
                 cast<PointerType>(getOperand(0)->getType())->getElementType()
         && "Ptr must be a pointer to Cmp type!");
  assert(getOperand(2)->getType() ==
                 cast<PointerType>(getOperand(0)->getType())->getElementType()
         && "Ptr must be a pointer to NewVal type!");
  assert(SuccessOrdering != AtomicOrdering::NotAtomic &&
         "AtomicCmpXchg instructions must be atomic!");
  assert(FailureOrdering != AtomicOrdering::NotAtomic &&
         "AtomicCmpXchg instructions must be atomic!");
  assert(!isStrongerThan(FailureOrdering, SuccessOrdering) &&
         "AtomicCmpXchg failure argument shall be no stronger than the success "
         "argument");
  assert(FailureOrdering != AtomicOrdering::Release &&
         FailureOrdering != AtomicOrdering::AcquireRelease &&
         "AtomicCmpXchg failure ordering cannot include release semantics");
}

AtomicCmpXchgInst::AtomicCmpXchgInst(Value *Ptr, Value *Cmp, Value *NewVal,
                                     Align Alignment,
                                     AtomicOrdering SuccessOrdering,
                                     AtomicOrdering FailureOrdering,
                                     SyncScope::ID SSID,
                                     Instruction *InsertBefore)
    : Instruction(
          StructType::get(Cmp->getType(), Type::getInt1Ty(Cmp->getContext())),
          AtomicCmpXchg, OperandTraits<AtomicCmpXchgInst>::op_begin(this),
          OperandTraits<AtomicCmpXchgInst>::operands(this), InsertBefore) {
  Init(Ptr, Cmp, NewVal, Alignment, SuccessOrdering, FailureOrdering, SSID);
}

AtomicCmpXchgInst::AtomicCmpXchgInst(Value *Ptr, Value *Cmp, Value *NewVal,
                                     Align Alignment,
                                     AtomicOrdering SuccessOrdering,
                                     AtomicOrdering FailureOrdering,
                                     SyncScope::ID SSID,
                                     BasicBlock *InsertAtEnd)
    : Instruction(
          StructType::get(Cmp->getType(), Type::getInt1Ty(Cmp->getContext())),
          AtomicCmpXchg, OperandTraits<AtomicCmpXchgInst>::op_begin(this),
          OperandTraits<AtomicCmpXchgInst>::operands(this), InsertAtEnd) {
  Init(Ptr, Cmp, NewVal, Alignment, SuccessOrdering, FailureOrdering, SSID);
}

//===----------------------------------------------------------------------===//
//                       AtomicRMWInst Implementation
//===----------------------------------------------------------------------===//

void AtomicRMWInst::Init(BinOp Operation, Value *Ptr, Value *Val,
                         Align Alignment, AtomicOrdering Ordering,
                         SyncScope::ID SSID) {
  Op<0>() = Ptr;
  Op<1>() = Val;
  setOperation(Operation);
  setOrdering(Ordering);
  setSyncScopeID(SSID);
  setAlignment(Alignment);

  assert(getOperand(0) && getOperand(1) &&
         "All operands must be non-null!");
  assert(getOperand(0)->getType()->isPointerTy() &&
         "Ptr must have pointer type!");
  assert(getOperand(1)->getType() ==
         cast<PointerType>(getOperand(0)->getType())->getElementType()
         && "Ptr must be a pointer to Val type!");
  assert(Ordering != AtomicOrdering::NotAtomic &&
         "AtomicRMW instructions must be atomic!");
}

AtomicRMWInst::AtomicRMWInst(BinOp Operation, Value *Ptr, Value *Val,
                             Align Alignment, AtomicOrdering Ordering,
                             SyncScope::ID SSID, Instruction *InsertBefore)
    : Instruction(Val->getType(), AtomicRMW,
                  OperandTraits<AtomicRMWInst>::op_begin(this),
                  OperandTraits<AtomicRMWInst>::operands(this), InsertBefore) {
  Init(Operation, Ptr, Val, Alignment, Ordering, SSID);
}

AtomicRMWInst::AtomicRMWInst(BinOp Operation, Value *Ptr, Value *Val,
                             Align Alignment, AtomicOrdering Ordering,
                             SyncScope::ID SSID, BasicBlock *InsertAtEnd)
    : Instruction(Val->getType(), AtomicRMW,
                  OperandTraits<AtomicRMWInst>::op_begin(this),
                  OperandTraits<AtomicRMWInst>::operands(this), InsertAtEnd) {
  Init(Operation, Ptr, Val, Alignment, Ordering, SSID);
}

StringRef AtomicRMWInst::getOperationName(BinOp Op) {
  switch (Op) {
  case AtomicRMWInst::Xchg:
    return "xchg";
  case AtomicRMWInst::Add:
    return "add";
  case AtomicRMWInst::Sub:
    return "sub";
  case AtomicRMWInst::And:
    return "and";
  case AtomicRMWInst::Nand:
    return "nand";
  case AtomicRMWInst::Or:
    return "or";
  case AtomicRMWInst::Xor:
    return "xor";
  case AtomicRMWInst::Max:
    return "max";
  case AtomicRMWInst::Min:
    return "min";
  case AtomicRMWInst::UMax:
    return "umax";
  case AtomicRMWInst::UMin:
    return "umin";
  case AtomicRMWInst::FAdd:
    return "fadd";
  case AtomicRMWInst::FSub:
    return "fsub";
  case AtomicRMWInst::BAD_BINOP:
    return "<invalid operation>";
  }

  llvm_unreachable("invalid atomicrmw operation");
}

//===----------------------------------------------------------------------===//
//                       FenceInst Implementation
//===----------------------------------------------------------------------===//

FenceInst::FenceInst(LLVMContext &C, AtomicOrdering Ordering,
                     SyncScope::ID SSID,
                     Instruction *InsertBefore)
  : Instruction(Type::getVoidTy(C), Fence, nullptr, 0, InsertBefore) {
  setOrdering(Ordering);
  setSyncScopeID(SSID);
}

FenceInst::FenceInst(LLVMContext &C, AtomicOrdering Ordering,
                     SyncScope::ID SSID,
                     BasicBlock *InsertAtEnd)
  : Instruction(Type::getVoidTy(C), Fence, nullptr, 0, InsertAtEnd) {
  setOrdering(Ordering);
  setSyncScopeID(SSID);
}

//===----------------------------------------------------------------------===//
//                       GetElementPtrInst Implementation
//===----------------------------------------------------------------------===//

void GetElementPtrInst::init(Value *Ptr, ArrayRef<Value *> IdxList,
                             const Twine &Name) {
  assert(getNumOperands() == 1 + IdxList.size() &&
         "NumOperands not initialized?");
  Op<0>() = Ptr;
  llvm::copy(IdxList, op_begin() + 1);
  setName(Name);
}

GetElementPtrInst::GetElementPtrInst(const GetElementPtrInst &GEPI)
    : Instruction(GEPI.getType(), GetElementPtr,
                  OperandTraits<GetElementPtrInst>::op_end(this) -
                      GEPI.getNumOperands(),
                  GEPI.getNumOperands()),
      SourceElementType(GEPI.SourceElementType),
      ResultElementType(GEPI.ResultElementType) {
  std::copy(GEPI.op_begin(), GEPI.op_end(), op_begin());
  SubclassOptionalData = GEPI.SubclassOptionalData;
}

Type *GetElementPtrInst::getTypeAtIndex(Type *Ty, Value *Idx) {
  if (auto *Struct = dyn_cast<StructType>(Ty)) {
    if (!Struct->indexValid(Idx))
      return nullptr;
    return Struct->getTypeAtIndex(Idx);
  }
  if (!Idx->getType()->isIntOrIntVectorTy())
    return nullptr;
  if (auto *Array = dyn_cast<ArrayType>(Ty))
    return Array->getElementType();
  if (auto *Vector = dyn_cast<VectorType>(Ty))
    return Vector->getElementType();
  return nullptr;
}

Type *GetElementPtrInst::getTypeAtIndex(Type *Ty, uint64_t Idx) {
  if (auto *Struct = dyn_cast<StructType>(Ty)) {
    if (Idx >= Struct->getNumElements())
      return nullptr;
    return Struct->getElementType(Idx);
  }
  if (auto *Array = dyn_cast<ArrayType>(Ty))
    return Array->getElementType();
  if (auto *Vector = dyn_cast<VectorType>(Ty))
    return Vector->getElementType();
  return nullptr;
}

template <typename IndexTy>
static Type *getIndexedTypeInternal(Type *Ty, ArrayRef<IndexTy> IdxList) {
  if (IdxList.empty())
    return Ty;
  for (IndexTy V : IdxList.slice(1)) {
    Ty = GetElementPtrInst::getTypeAtIndex(Ty, V);
    if (!Ty)
      return Ty;
  }
  return Ty;
}

Type *GetElementPtrInst::getIndexedType(Type *Ty, ArrayRef<Value *> IdxList) {
  return getIndexedTypeInternal(Ty, IdxList);
}

Type *GetElementPtrInst::getIndexedType(Type *Ty,
                                        ArrayRef<Constant *> IdxList) {
  return getIndexedTypeInternal(Ty, IdxList);
}

Type *GetElementPtrInst::getIndexedType(Type *Ty, ArrayRef<uint64_t> IdxList) {
  return getIndexedTypeInternal(Ty, IdxList);
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

void GetElementPtrInst::setIsInBounds(bool B) {
  cast<GEPOperator>(this)->setIsInBounds(B);
}

bool GetElementPtrInst::isInBounds() const {
  return cast<GEPOperator>(this)->isInBounds();
}

bool GetElementPtrInst::accumulateConstantOffset(const DataLayout &DL,
                                                 APInt &Offset) const {
  // Delegate to the generic GEPOperator implementation.
  return cast<GEPOperator>(this)->accumulateConstantOffset(DL, Offset);
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
  if (!Val->getType()->isVectorTy() || !Index->getType()->isIntegerTy())
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
  if (!Vec->getType()->isVectorTy())
    return false;   // First operand of insertelement must be vector type.

  if (Elt->getType() != cast<VectorType>(Vec->getType())->getElementType())
    return false;// Second operand of insertelement must be vector element type.

  if (!Index->getType()->isIntegerTy())
    return false;  // Third operand of insertelement must be i32.
  return true;
}

//===----------------------------------------------------------------------===//
//                      ShuffleVectorInst Implementation
//===----------------------------------------------------------------------===//

ShuffleVectorInst::ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                                     const Twine &Name,
                                     Instruction *InsertBefore)
    : Instruction(
          VectorType::get(cast<VectorType>(V1->getType())->getElementType(),
                          cast<VectorType>(Mask->getType())->getElementCount()),
          ShuffleVector, OperandTraits<ShuffleVectorInst>::op_begin(this),
          OperandTraits<ShuffleVectorInst>::operands(this), InsertBefore) {
  assert(isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector instruction operands!");

  Op<0>() = V1;
  Op<1>() = V2;
  SmallVector<int, 16> MaskArr;
  getShuffleMask(cast<Constant>(Mask), MaskArr);
  setShuffleMask(MaskArr);
  setName(Name);
}

ShuffleVectorInst::ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                                     const Twine &Name, BasicBlock *InsertAtEnd)
    : Instruction(
          VectorType::get(cast<VectorType>(V1->getType())->getElementType(),
                          cast<VectorType>(Mask->getType())->getElementCount()),
          ShuffleVector, OperandTraits<ShuffleVectorInst>::op_begin(this),
          OperandTraits<ShuffleVectorInst>::operands(this), InsertAtEnd) {
  assert(isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector instruction operands!");

  Op<0>() = V1;
  Op<1>() = V2;
  SmallVector<int, 16> MaskArr;
  getShuffleMask(cast<Constant>(Mask), MaskArr);
  setShuffleMask(MaskArr);
  setName(Name);
}

ShuffleVectorInst::ShuffleVectorInst(Value *V1, Value *V2, ArrayRef<int> Mask,
                                     const Twine &Name,
                                     Instruction *InsertBefore)
    : Instruction(
          VectorType::get(cast<VectorType>(V1->getType())->getElementType(),
                          Mask.size(), isa<ScalableVectorType>(V1->getType())),
          ShuffleVector, OperandTraits<ShuffleVectorInst>::op_begin(this),
          OperandTraits<ShuffleVectorInst>::operands(this), InsertBefore) {
  assert(isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector instruction operands!");
  Op<0>() = V1;
  Op<1>() = V2;
  setShuffleMask(Mask);
  setName(Name);
}

ShuffleVectorInst::ShuffleVectorInst(Value *V1, Value *V2, ArrayRef<int> Mask,
                                     const Twine &Name, BasicBlock *InsertAtEnd)
    : Instruction(
          VectorType::get(cast<VectorType>(V1->getType())->getElementType(),
                          Mask.size(), isa<ScalableVectorType>(V1->getType())),
          ShuffleVector, OperandTraits<ShuffleVectorInst>::op_begin(this),
          OperandTraits<ShuffleVectorInst>::operands(this), InsertAtEnd) {
  assert(isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector instruction operands!");

  Op<0>() = V1;
  Op<1>() = V2;
  setShuffleMask(Mask);
  setName(Name);
}

void ShuffleVectorInst::commute() {
  int NumOpElts = cast<VectorType>(Op<0>()->getType())->getNumElements();
  int NumMaskElts = ShuffleMask.size();
  SmallVector<int, 16> NewMask(NumMaskElts);
  for (int i = 0; i != NumMaskElts; ++i) {
    int MaskElt = getMaskValue(i);
    if (MaskElt == UndefMaskElem) {
      NewMask[i] = UndefMaskElem;
      continue;
    }
    assert(MaskElt >= 0 && MaskElt < 2 * NumOpElts && "Out-of-range mask");
    MaskElt = (MaskElt < NumOpElts) ? MaskElt + NumOpElts : MaskElt - NumOpElts;
    NewMask[i] = MaskElt;
  }
  setShuffleMask(NewMask);
  Op<0>().swap(Op<1>());
}

bool ShuffleVectorInst::isValidOperands(const Value *V1, const Value *V2,
                                        ArrayRef<int> Mask) {
  // V1 and V2 must be vectors of the same type.
  if (!isa<VectorType>(V1->getType()) || V1->getType() != V2->getType())
    return false;

  // Make sure the mask elements make sense.
  int V1Size = cast<VectorType>(V1->getType())->getElementCount().Min;
  for (int Elem : Mask)
    if (Elem != UndefMaskElem && Elem >= V1Size * 2)
      return false;

  if (isa<ScalableVectorType>(V1->getType()))
    if ((Mask[0] != 0 && Mask[0] != UndefMaskElem) || !is_splat(Mask))
      return false;

  return true;
}

bool ShuffleVectorInst::isValidOperands(const Value *V1, const Value *V2,
                                        const Value *Mask) {
  // V1 and V2 must be vectors of the same type.
  if (!V1->getType()->isVectorTy() || V1->getType() != V2->getType())
    return false;

  // Mask must be vector of i32, and must be the same kind of vector as the
  // input vectors
  auto *MaskTy = dyn_cast<VectorType>(Mask->getType());
  if (!MaskTy || !MaskTy->getElementType()->isIntegerTy(32) ||
      isa<ScalableVectorType>(MaskTy) != isa<ScalableVectorType>(V1->getType()))
    return false;

  // Check to see if Mask is valid.
  if (isa<UndefValue>(Mask) || isa<ConstantAggregateZero>(Mask))
    return true;

  if (const auto *MV = dyn_cast<ConstantVector>(Mask)) {
    unsigned V1Size = cast<VectorType>(V1->getType())->getNumElements();
    for (Value *Op : MV->operands()) {
      if (auto *CI = dyn_cast<ConstantInt>(Op)) {
        if (CI->uge(V1Size*2))
          return false;
      } else if (!isa<UndefValue>(Op)) {
        return false;
      }
    }
    return true;
  }

  if (const auto *CDS = dyn_cast<ConstantDataSequential>(Mask)) {
    unsigned V1Size = cast<VectorType>(V1->getType())->getNumElements();
    for (unsigned i = 0, e = MaskTy->getNumElements(); i != e; ++i)
      if (CDS->getElementAsInteger(i) >= V1Size*2)
        return false;
    return true;
  }

  return false;
}

void ShuffleVectorInst::getShuffleMask(const Constant *Mask,
                                       SmallVectorImpl<int> &Result) {
  unsigned NumElts = cast<VectorType>(Mask->getType())->getElementCount().Min;
  if (isa<ConstantAggregateZero>(Mask)) {
    Result.resize(NumElts, 0);
    return;
  }
  Result.reserve(NumElts);
  if (auto *CDS = dyn_cast<ConstantDataSequential>(Mask)) {
    for (unsigned i = 0; i != NumElts; ++i)
      Result.push_back(CDS->getElementAsInteger(i));
    return;
  }
  for (unsigned i = 0; i != NumElts; ++i) {
    Constant *C = Mask->getAggregateElement(i);
    Result.push_back(isa<UndefValue>(C) ? -1 :
                     cast<ConstantInt>(C)->getZExtValue());
  }
}

void ShuffleVectorInst::setShuffleMask(ArrayRef<int> Mask) {
  ShuffleMask.assign(Mask.begin(), Mask.end());
  ShuffleMaskForBitcode = convertShuffleMaskForBitcode(Mask, getType());
}
Constant *ShuffleVectorInst::convertShuffleMaskForBitcode(ArrayRef<int> Mask,
                                                          Type *ResultTy) {
  Type *Int32Ty = Type::getInt32Ty(ResultTy->getContext());
  if (isa<ScalableVectorType>(ResultTy)) {
    assert(is_splat(Mask) && "Unexpected shuffle");
    Type *VecTy = VectorType::get(Int32Ty, Mask.size(), true);
    if (Mask[0] == 0)
      return Constant::getNullValue(VecTy);
    return UndefValue::get(VecTy);
  }
  SmallVector<Constant *, 16> MaskConst;
  for (int Elem : Mask) {
    if (Elem == UndefMaskElem)
      MaskConst.push_back(UndefValue::get(Int32Ty));
    else
      MaskConst.push_back(ConstantInt::get(Int32Ty, Elem));
  }
  return ConstantVector::get(MaskConst);
}

static bool isSingleSourceMaskImpl(ArrayRef<int> Mask, int NumOpElts) {
  assert(!Mask.empty() && "Shuffle mask must contain elements");
  bool UsesLHS = false;
  bool UsesRHS = false;
  for (int i = 0, NumMaskElts = Mask.size(); i < NumMaskElts; ++i) {
    if (Mask[i] == -1)
      continue;
    assert(Mask[i] >= 0 && Mask[i] < (NumOpElts * 2) &&
           "Out-of-bounds shuffle mask element");
    UsesLHS |= (Mask[i] < NumOpElts);
    UsesRHS |= (Mask[i] >= NumOpElts);
    if (UsesLHS && UsesRHS)
      return false;
  }
  // Allow for degenerate case: completely undef mask means neither source is used.
  return UsesLHS || UsesRHS;
}

bool ShuffleVectorInst::isSingleSourceMask(ArrayRef<int> Mask) {
  // We don't have vector operand size information, so assume operands are the
  // same size as the mask.
  return isSingleSourceMaskImpl(Mask, Mask.size());
}

static bool isIdentityMaskImpl(ArrayRef<int> Mask, int NumOpElts) {
  if (!isSingleSourceMaskImpl(Mask, NumOpElts))
    return false;
  for (int i = 0, NumMaskElts = Mask.size(); i < NumMaskElts; ++i) {
    if (Mask[i] == -1)
      continue;
    if (Mask[i] != i && Mask[i] != (NumOpElts + i))
      return false;
  }
  return true;
}

bool ShuffleVectorInst::isIdentityMask(ArrayRef<int> Mask) {
  // We don't have vector operand size information, so assume operands are the
  // same size as the mask.
  return isIdentityMaskImpl(Mask, Mask.size());
}

bool ShuffleVectorInst::isReverseMask(ArrayRef<int> Mask) {
  if (!isSingleSourceMask(Mask))
    return false;
  for (int i = 0, NumElts = Mask.size(); i < NumElts; ++i) {
    if (Mask[i] == -1)
      continue;
    if (Mask[i] != (NumElts - 1 - i) && Mask[i] != (NumElts + NumElts - 1 - i))
      return false;
  }
  return true;
}

bool ShuffleVectorInst::isZeroEltSplatMask(ArrayRef<int> Mask) {
  if (!isSingleSourceMask(Mask))
    return false;
  for (int i = 0, NumElts = Mask.size(); i < NumElts; ++i) {
    if (Mask[i] == -1)
      continue;
    if (Mask[i] != 0 && Mask[i] != NumElts)
      return false;
  }
  return true;
}

bool ShuffleVectorInst::isSelectMask(ArrayRef<int> Mask) {
  // Select is differentiated from identity. It requires using both sources.
  if (isSingleSourceMask(Mask))
    return false;
  for (int i = 0, NumElts = Mask.size(); i < NumElts; ++i) {
    if (Mask[i] == -1)
      continue;
    if (Mask[i] != i && Mask[i] != (NumElts + i))
      return false;
  }
  return true;
}

bool ShuffleVectorInst::isTransposeMask(ArrayRef<int> Mask) {
  // Example masks that will return true:
  // v1 = <a, b, c, d>
  // v2 = <e, f, g, h>
  // trn1 = shufflevector v1, v2 <0, 4, 2, 6> = <a, e, c, g>
  // trn2 = shufflevector v1, v2 <1, 5, 3, 7> = <b, f, d, h>

  // 1. The number of elements in the mask must be a power-of-2 and at least 2.
  int NumElts = Mask.size();
  if (NumElts < 2 || !isPowerOf2_32(NumElts))
    return false;

  // 2. The first element of the mask must be either a 0 or a 1.
  if (Mask[0] != 0 && Mask[0] != 1)
    return false;

  // 3. The difference between the first 2 elements must be equal to the
  // number of elements in the mask.
  if ((Mask[1] - Mask[0]) != NumElts)
    return false;

  // 4. The difference between consecutive even-numbered and odd-numbered
  // elements must be equal to 2.
  for (int i = 2; i < NumElts; ++i) {
    int MaskEltVal = Mask[i];
    if (MaskEltVal == -1)
      return false;
    int MaskEltPrevVal = Mask[i - 2];
    if (MaskEltVal - MaskEltPrevVal != 2)
      return false;
  }
  return true;
}

bool ShuffleVectorInst::isExtractSubvectorMask(ArrayRef<int> Mask,
                                               int NumSrcElts, int &Index) {
  // Must extract from a single source.
  if (!isSingleSourceMaskImpl(Mask, NumSrcElts))
    return false;

  // Must be smaller (else this is an Identity shuffle).
  if (NumSrcElts <= (int)Mask.size())
    return false;

  // Find start of extraction, accounting that we may start with an UNDEF.
  int SubIndex = -1;
  for (int i = 0, e = Mask.size(); i != e; ++i) {
    int M = Mask[i];
    if (M < 0)
      continue;
    int Offset = (M % NumSrcElts) - i;
    if (0 <= SubIndex && SubIndex != Offset)
      return false;
    SubIndex = Offset;
  }

  if (0 <= SubIndex && SubIndex + (int)Mask.size() <= NumSrcElts) {
    Index = SubIndex;
    return true;
  }
  return false;
}

bool ShuffleVectorInst::isIdentityWithPadding() const {
  if (isa<UndefValue>(Op<2>()))
    return false;
  int NumOpElts = cast<VectorType>(Op<0>()->getType())->getNumElements();
  int NumMaskElts = cast<VectorType>(getType())->getNumElements();
  if (NumMaskElts <= NumOpElts)
    return false;

  // The first part of the mask must choose elements from exactly 1 source op.
  ArrayRef<int> Mask = getShuffleMask();
  if (!isIdentityMaskImpl(Mask, NumOpElts))
    return false;

  // All extending must be with undef elements.
  for (int i = NumOpElts; i < NumMaskElts; ++i)
    if (Mask[i] != -1)
      return false;

  return true;
}

bool ShuffleVectorInst::isIdentityWithExtract() const {
  if (isa<UndefValue>(Op<2>()))
    return false;

  // FIXME: Not currently possible to express a shuffle mask for a scalable
  // vector for this case
  if (isa<ScalableVectorType>(getType()))
    return false;

  int NumOpElts = cast<FixedVectorType>(Op<0>()->getType())->getNumElements();
  int NumMaskElts = cast<FixedVectorType>(getType())->getNumElements();
  if (NumMaskElts >= NumOpElts)
    return false;

  return isIdentityMaskImpl(getShuffleMask(), NumOpElts);
}

bool ShuffleVectorInst::isConcat() const {
  // Vector concatenation is differentiated from identity with padding.
  if (isa<UndefValue>(Op<0>()) || isa<UndefValue>(Op<1>()) ||
      isa<UndefValue>(Op<2>()))
    return false;

  int NumOpElts = cast<VectorType>(Op<0>()->getType())->getNumElements();
  int NumMaskElts = getType()->getNumElements();
  if (NumMaskElts != NumOpElts * 2)
    return false;

  // Use the mask length rather than the operands' vector lengths here. We
  // already know that the shuffle returns a vector twice as long as the inputs,
  // and neither of the inputs are undef vectors. If the mask picks consecutive
  // elements from both inputs, then this is a concatenation of the inputs.
  return isIdentityMaskImpl(getShuffleMask(), NumMaskElts);
}

//===----------------------------------------------------------------------===//
//                             InsertValueInst Class
//===----------------------------------------------------------------------===//

void InsertValueInst::init(Value *Agg, Value *Val, ArrayRef<unsigned> Idxs,
                           const Twine &Name) {
  assert(getNumOperands() == 2 && "NumOperands not initialized?");

  // There's no fundamental reason why we require at least one index
  // (other than weirdness with &*IdxBegin being invalid; see
  // getelementptr's init routine for example). But there's no
  // present need to support it.
  assert(!Idxs.empty() && "InsertValueInst must have at least one index");

  assert(ExtractValueInst::getIndexedType(Agg->getType(), Idxs) ==
         Val->getType() && "Inserted value must match indexed type!");
  Op<0>() = Agg;
  Op<1>() = Val;

  Indices.append(Idxs.begin(), Idxs.end());
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

//===----------------------------------------------------------------------===//
//                             ExtractValueInst Class
//===----------------------------------------------------------------------===//

void ExtractValueInst::init(ArrayRef<unsigned> Idxs, const Twine &Name) {
  assert(getNumOperands() == 1 && "NumOperands not initialized?");

  // There's no fundamental reason why we require at least one index.
  // But there's no present need to support it.
  assert(!Idxs.empty() && "ExtractValueInst must have at least one index");

  Indices.append(Idxs.begin(), Idxs.end());
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
Type *ExtractValueInst::getIndexedType(Type *Agg,
                                       ArrayRef<unsigned> Idxs) {
  for (unsigned Index : Idxs) {
    // We can't use CompositeType::indexValid(Index) here.
    // indexValid() always returns true for arrays because getelementptr allows
    // out-of-bounds indices. Since we don't allow those for extractvalue and
    // insertvalue we need to check array indexing manually.
    // Since the only other types we can index into are struct types it's just
    // as easy to check those manually as well.
    if (ArrayType *AT = dyn_cast<ArrayType>(Agg)) {
      if (Index >= AT->getNumElements())
        return nullptr;
      Agg = AT->getElementType();
    } else if (StructType *ST = dyn_cast<StructType>(Agg)) {
      if (Index >= ST->getNumElements())
        return nullptr;
      Agg = ST->getElementType(Index);
    } else {
      // Not a valid type to index into.
      return nullptr;
    }
  }
  return const_cast<Type*>(Agg);
}

//===----------------------------------------------------------------------===//
//                             UnaryOperator Class
//===----------------------------------------------------------------------===//

UnaryOperator::UnaryOperator(UnaryOps iType, Value *S,
                             Type *Ty, const Twine &Name,
                             Instruction *InsertBefore)
  : UnaryInstruction(Ty, iType, S, InsertBefore) {
  Op<0>() = S;
  setName(Name);
  AssertOK();
}

UnaryOperator::UnaryOperator(UnaryOps iType, Value *S,
                             Type *Ty, const Twine &Name,
                             BasicBlock *InsertAtEnd)
  : UnaryInstruction(Ty, iType, S, InsertAtEnd) {
  Op<0>() = S;
  setName(Name);
  AssertOK();
}

UnaryOperator *UnaryOperator::Create(UnaryOps Op, Value *S,
                                     const Twine &Name,
                                     Instruction *InsertBefore) {
  return new UnaryOperator(Op, S, S->getType(), Name, InsertBefore);
}

UnaryOperator *UnaryOperator::Create(UnaryOps Op, Value *S,
                                     const Twine &Name,
                                     BasicBlock *InsertAtEnd) {
  UnaryOperator *Res = Create(Op, S, Name);
  InsertAtEnd->getInstList().push_back(Res);
  return Res;
}

void UnaryOperator::AssertOK() {
  Value *LHS = getOperand(0);
  (void)LHS; // Silence warnings.
#ifndef NDEBUG
  switch (getOpcode()) {
  case FNeg:
    assert(getType() == LHS->getType() &&
           "Unary operation should return same type as operand!");
    assert(getType()->isFPOrFPVectorTy() &&
           "Tried to create a floating-point operation on a "
           "non-floating-point type!");
    break;
  default: llvm_unreachable("Invalid opcode provided");
  }
#endif
}

//===----------------------------------------------------------------------===//
//                             BinaryOperator Class
//===----------------------------------------------------------------------===//

BinaryOperator::BinaryOperator(BinaryOps iType, Value *S1, Value *S2,
                               Type *Ty, const Twine &Name,
                               Instruction *InsertBefore)
  : Instruction(Ty, iType,
                OperandTraits<BinaryOperator>::op_begin(this),
                OperandTraits<BinaryOperator>::operands(this),
                InsertBefore) {
  Op<0>() = S1;
  Op<1>() = S2;
  setName(Name);
  AssertOK();
}

BinaryOperator::BinaryOperator(BinaryOps iType, Value *S1, Value *S2,
                               Type *Ty, const Twine &Name,
                               BasicBlock *InsertAtEnd)
  : Instruction(Ty, iType,
                OperandTraits<BinaryOperator>::op_begin(this),
                OperandTraits<BinaryOperator>::operands(this),
                InsertAtEnd) {
  Op<0>() = S1;
  Op<1>() = S2;
  setName(Name);
  AssertOK();
}

void BinaryOperator::AssertOK() {
  Value *LHS = getOperand(0), *RHS = getOperand(1);
  (void)LHS; (void)RHS; // Silence warnings.
  assert(LHS->getType() == RHS->getType() &&
         "Binary operator operand types must match!");
#ifndef NDEBUG
  switch (getOpcode()) {
  case Add: case Sub:
  case Mul:
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert(getType()->isIntOrIntVectorTy() &&
           "Tried to create an integer operation on a non-integer type!");
    break;
  case FAdd: case FSub:
  case FMul:
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert(getType()->isFPOrFPVectorTy() &&
           "Tried to create a floating-point operation on a "
           "non-floating-point type!");
    break;
  case UDiv:
  case SDiv:
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert(getType()->isIntOrIntVectorTy() &&
           "Incorrect operand type (not integer) for S/UDIV");
    break;
  case FDiv:
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert(getType()->isFPOrFPVectorTy() &&
           "Incorrect operand type (not floating point) for FDIV");
    break;
  case URem:
  case SRem:
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert(getType()->isIntOrIntVectorTy() &&
           "Incorrect operand type (not integer) for S/UREM");
    break;
  case FRem:
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert(getType()->isFPOrFPVectorTy() &&
           "Incorrect operand type (not floating point) for FREM");
    break;
  case Shl:
  case LShr:
  case AShr:
    assert(getType() == LHS->getType() &&
           "Shift operation should return same type as operands!");
    assert(getType()->isIntOrIntVectorTy() &&
           "Tried to create a shift operation on a non-integral type!");
    break;
  case And: case Or:
  case Xor:
    assert(getType() == LHS->getType() &&
           "Logical operation should return same type as operands!");
    assert(getType()->isIntOrIntVectorTy() &&
           "Tried to create a logical operation on a non-integral type!");
    break;
  default: llvm_unreachable("Invalid opcode provided");
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

BinaryOperator *BinaryOperator::CreateNSWNeg(Value *Op, const Twine &Name,
                                             Instruction *InsertBefore) {
  Value *zero = ConstantFP::getZeroValueForNegation(Op->getType());
  return BinaryOperator::CreateNSWSub(zero, Op, Name, InsertBefore);
}

BinaryOperator *BinaryOperator::CreateNSWNeg(Value *Op, const Twine &Name,
                                             BasicBlock *InsertAtEnd) {
  Value *zero = ConstantFP::getZeroValueForNegation(Op->getType());
  return BinaryOperator::CreateNSWSub(zero, Op, Name, InsertAtEnd);
}

BinaryOperator *BinaryOperator::CreateNUWNeg(Value *Op, const Twine &Name,
                                             Instruction *InsertBefore) {
  Value *zero = ConstantFP::getZeroValueForNegation(Op->getType());
  return BinaryOperator::CreateNUWSub(zero, Op, Name, InsertBefore);
}

BinaryOperator *BinaryOperator::CreateNUWNeg(Value *Op, const Twine &Name,
                                             BasicBlock *InsertAtEnd) {
  Value *zero = ConstantFP::getZeroValueForNegation(Op->getType());
  return BinaryOperator::CreateNUWSub(zero, Op, Name, InsertAtEnd);
}

BinaryOperator *BinaryOperator::CreateNot(Value *Op, const Twine &Name,
                                          Instruction *InsertBefore) {
  Constant *C = Constant::getAllOnesValue(Op->getType());
  return new BinaryOperator(Instruction::Xor, Op, C,
                            Op->getType(), Name, InsertBefore);
}

BinaryOperator *BinaryOperator::CreateNot(Value *Op, const Twine &Name,
                                          BasicBlock *InsertAtEnd) {
  Constant *AllOnes = Constant::getAllOnesValue(Op->getType());
  return new BinaryOperator(Instruction::Xor, Op, AllOnes,
                            Op->getType(), Name, InsertAtEnd);
}

// Exchange the two operands to this instruction. This instruction is safe to
// use on any binary instruction and does not modify the semantics of the
// instruction. If the instruction is order-dependent (SetLT f.e.), the opcode
// is changed.
bool BinaryOperator::swapOperands() {
  if (!isCommutative())
    return true; // Can't commute operands
  Op<0>().swap(Op<1>());
  return false;
}

//===----------------------------------------------------------------------===//
//                             FPMathOperator Class
//===----------------------------------------------------------------------===//

float FPMathOperator::getFPAccuracy() const {
  const MDNode *MD =
      cast<Instruction>(this)->getMetadata(LLVMContext::MD_fpmath);
  if (!MD)
    return 0.0;
  ConstantFP *Accuracy = mdconst::extract<ConstantFP>(MD->getOperand(0));
  return Accuracy->getValueAPF().convertToFloat();
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
      return getOperand(0)->getType()->isIntegerTy() &&
        getType()->isIntegerTy();
  }
}

bool CastInst::isLosslessCast() const {
  // Only BitCast can be lossless, exit fast if we're not BitCast
  if (getOpcode() != Instruction::BitCast)
    return false;

  // Identity cast is always lossless
  Type *SrcTy = getOperand(0)->getType();
  Type *DstTy = getType();
  if (SrcTy == DstTy)
    return true;

  // Pointer to pointer is always lossless.
  if (SrcTy->isPointerTy())
    return DstTy->isPointerTy();
  return false;  // Other types have no identity values
}

/// This function determines if the CastInst does not require any bits to be
/// changed in order to effect the cast. Essentially, it identifies cases where
/// no code gen is necessary for the cast, hence the name no-op cast.  For
/// example, the following are all no-op casts:
/// # bitcast i32* %x to i8*
/// # bitcast <2 x i32> %x to <4 x i16>
/// # ptrtoint i32* %x to i32     ; on 32-bit plaforms only
/// Determine if the described cast is a no-op.
bool CastInst::isNoopCast(Instruction::CastOps Opcode,
                          Type *SrcTy,
                          Type *DestTy,
                          const DataLayout &DL) {
  switch (Opcode) {
    default: llvm_unreachable("Invalid CastOp");
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::AddrSpaceCast:
      // TODO: Target informations may give a more accurate answer here.
      return false;
    case Instruction::BitCast:
      return true;  // BitCast never modifies bits.
    case Instruction::PtrToInt:
      return DL.getIntPtrType(SrcTy)->getScalarSizeInBits() ==
             DestTy->getScalarSizeInBits();
    case Instruction::IntToPtr:
      return DL.getIntPtrType(DestTy)->getScalarSizeInBits() ==
             SrcTy->getScalarSizeInBits();
  }
}

bool CastInst::isNoopCast(const DataLayout &DL) const {
  return isNoopCast(getOpcode(), getOperand(0)->getType(), getType(), DL);
}

/// This function determines if a pair of casts can be eliminated and what
/// opcode should be used in the elimination. This assumes that there are two
/// instructions like this:
/// *  %F = firstOpcode SrcTy %x to MidTy
/// *  %S = secondOpcode MidTy %F to DstTy
/// The function returns a resultOpcode so these two casts can be replaced with:
/// *  %Replacement = resultOpcode %SrcTy %x to DstTy
/// If no such cast is permitted, the function returns 0.
unsigned CastInst::isEliminableCastPair(
  Instruction::CastOps firstOp, Instruction::CastOps secondOp,
  Type *SrcTy, Type *MidTy, Type *DstTy, Type *SrcIntPtrTy, Type *MidIntPtrTy,
  Type *DstIntPtrTy) {
  // Define the 144 possibilities for these two cast instructions. The values
  // in this matrix determine what to do in a given situation and select the
  // case in the switch below.  The rows correspond to firstOp, the columns
  // correspond to secondOp.  In looking at the table below, keep in mind
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
  // BITCAST       =       FirstClass   n/a       FirstClass    n/a
  // ADDRSPCST    n/a      Pointer      n/a        Pointer      n/a
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
    // T        F  F  U  S  F  F  P  I  B  A  -+
    // R  Z  S  P  P  I  I  T  P  2  N  T  S   |
    // U  E  E  2  2  2  2  R  E  I  T  C  C   +- secondOp
    // N  X  X  U  S  F  F  N  X  N  2  V  V   |
    // C  T  T  I  I  P  P  C  T  T  P  T  T  -+
    {  1, 0, 0,99,99, 0, 0,99,99,99, 0, 3, 0}, // Trunc         -+
    {  8, 1, 9,99,99, 2,17,99,99,99, 2, 3, 0}, // ZExt           |
    {  8, 0, 1,99,99, 0, 2,99,99,99, 0, 3, 0}, // SExt           |
    {  0, 0, 0,99,99, 0, 0,99,99,99, 0, 3, 0}, // FPToUI         |
    {  0, 0, 0,99,99, 0, 0,99,99,99, 0, 3, 0}, // FPToSI         |
    { 99,99,99, 0, 0,99,99, 0, 0,99,99, 4, 0}, // UIToFP         +- firstOp
    { 99,99,99, 0, 0,99,99, 0, 0,99,99, 4, 0}, // SIToFP         |
    { 99,99,99, 0, 0,99,99, 0, 0,99,99, 4, 0}, // FPTrunc        |
    { 99,99,99, 2, 2,99,99, 8, 2,99,99, 4, 0}, // FPExt          |
    {  1, 0, 0,99,99, 0, 0,99,99,99, 7, 3, 0}, // PtrToInt       |
    { 99,99,99,99,99,99,99,99,99,11,99,15, 0}, // IntToPtr       |
    {  5, 5, 5, 6, 6, 5, 5, 6, 6,16, 5, 1,14}, // BitCast        |
    {  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,13,12}, // AddrSpaceCast -+
  };

  // TODO: This logic could be encoded into the table above and handled in the
  // switch below.
  // If either of the casts are a bitcast from scalar to vector, disallow the
  // merging. However, any pair of bitcasts are allowed.
  bool IsFirstBitcast  = (firstOp == Instruction::BitCast);
  bool IsSecondBitcast = (secondOp == Instruction::BitCast);
  bool AreBothBitcasts = IsFirstBitcast && IsSecondBitcast;

  // Check if any of the casts convert scalars <-> vectors.
  if ((IsFirstBitcast  && isa<VectorType>(SrcTy) != isa<VectorType>(MidTy)) ||
      (IsSecondBitcast && isa<VectorType>(MidTy) != isa<VectorType>(DstTy)))
    if (!AreBothBitcasts)
      return 0;

  int ElimCase = CastResults[firstOp-Instruction::CastOpsBegin]
                            [secondOp-Instruction::CastOpsBegin];
  switch (ElimCase) {
    case 0:
      // Categorically disallowed.
      return 0;
    case 1:
      // Allowed, use first cast's opcode.
      return firstOp;
    case 2:
      // Allowed, use second cast's opcode.
      return secondOp;
    case 3:
      // No-op cast in second op implies firstOp as long as the DestTy
      // is integer and we are not converting between a vector and a
      // non-vector type.
      if (!SrcTy->isVectorTy() && DstTy->isIntegerTy())
        return firstOp;
      return 0;
    case 4:
      // No-op cast in second op implies firstOp as long as the DestTy
      // is floating point.
      if (DstTy->isFloatingPointTy())
        return firstOp;
      return 0;
    case 5:
      // No-op cast in first op implies secondOp as long as the SrcTy
      // is an integer.
      if (SrcTy->isIntegerTy())
        return secondOp;
      return 0;
    case 6:
      // No-op cast in first op implies secondOp as long as the SrcTy
      // is a floating point.
      if (SrcTy->isFloatingPointTy())
        return secondOp;
      return 0;
    case 7: {
      // Cannot simplify if address spaces are different!
      if (SrcTy->getPointerAddressSpace() != DstTy->getPointerAddressSpace())
        return 0;

      unsigned MidSize = MidTy->getScalarSizeInBits();
      // We can still fold this without knowing the actual sizes as long we
      // know that the intermediate pointer is the largest possible
      // pointer size.
      // FIXME: Is this always true?
      if (MidSize == 64)
        return Instruction::BitCast;

      // ptrtoint, inttoptr -> bitcast (ptr -> ptr) if int size is >= ptr size.
      if (!SrcIntPtrTy || DstIntPtrTy != SrcIntPtrTy)
        return 0;
      unsigned PtrSize = SrcIntPtrTy->getScalarSizeInBits();
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
    case 9:
      // zext, sext -> zext, because sext can't sign extend after zext
      return Instruction::ZExt;
    case 11: {
      // inttoptr, ptrtoint -> bitcast if SrcSize<=PtrSize and SrcSize==DstSize
      if (!MidIntPtrTy)
        return 0;
      unsigned PtrSize = MidIntPtrTy->getScalarSizeInBits();
      unsigned SrcSize = SrcTy->getScalarSizeInBits();
      unsigned DstSize = DstTy->getScalarSizeInBits();
      if (SrcSize <= PtrSize && SrcSize == DstSize)
        return Instruction::BitCast;
      return 0;
    }
    case 12:
      // addrspacecast, addrspacecast -> bitcast,       if SrcAS == DstAS
      // addrspacecast, addrspacecast -> addrspacecast, if SrcAS != DstAS
      if (SrcTy->getPointerAddressSpace() != DstTy->getPointerAddressSpace())
        return Instruction::AddrSpaceCast;
      return Instruction::BitCast;
    case 13:
      // FIXME: this state can be merged with (1), but the following assert
      // is useful to check the correcteness of the sequence due to semantic
      // change of bitcast.
      assert(
        SrcTy->isPtrOrPtrVectorTy() &&
        MidTy->isPtrOrPtrVectorTy() &&
        DstTy->isPtrOrPtrVectorTy() &&
        SrcTy->getPointerAddressSpace() != MidTy->getPointerAddressSpace() &&
        MidTy->getPointerAddressSpace() == DstTy->getPointerAddressSpace() &&
        "Illegal addrspacecast, bitcast sequence!");
      // Allowed, use first cast's opcode
      return firstOp;
    case 14:
      // bitcast, addrspacecast -> addrspacecast if the element type of
      // bitcast's source is the same as that of addrspacecast's destination.
      if (SrcTy->getScalarType()->getPointerElementType() ==
          DstTy->getScalarType()->getPointerElementType())
        return Instruction::AddrSpaceCast;
      return 0;
    case 15:
      // FIXME: this state can be merged with (1), but the following assert
      // is useful to check the correcteness of the sequence due to semantic
      // change of bitcast.
      assert(
        SrcTy->isIntOrIntVectorTy() &&
        MidTy->isPtrOrPtrVectorTy() &&
        DstTy->isPtrOrPtrVectorTy() &&
        MidTy->getPointerAddressSpace() == DstTy->getPointerAddressSpace() &&
        "Illegal inttoptr, bitcast sequence!");
      // Allowed, use first cast's opcode
      return firstOp;
    case 16:
      // FIXME: this state can be merged with (2), but the following assert
      // is useful to check the correcteness of the sequence due to semantic
      // change of bitcast.
      assert(
        SrcTy->isPtrOrPtrVectorTy() &&
        MidTy->isPtrOrPtrVectorTy() &&
        DstTy->isIntOrIntVectorTy() &&
        SrcTy->getPointerAddressSpace() == MidTy->getPointerAddressSpace() &&
        "Illegal bitcast, ptrtoint sequence!");
      // Allowed, use second cast's opcode
      return secondOp;
    case 17:
      // (sitofp (zext x)) -> (uitofp x)
      return Instruction::UIToFP;
    case 99:
      // Cast combination can't happen (error in input). This is for all cases
      // where the MidTy is not the same for the two cast instructions.
      llvm_unreachable("Invalid Cast Combination");
    default:
      llvm_unreachable("Error in CastResults table!!!");
  }
}

CastInst *CastInst::Create(Instruction::CastOps op, Value *S, Type *Ty,
  const Twine &Name, Instruction *InsertBefore) {
  assert(castIsValid(op, S, Ty) && "Invalid cast!");
  // Construct and return the appropriate CastInst subclass
  switch (op) {
  case Trunc:         return new TruncInst         (S, Ty, Name, InsertBefore);
  case ZExt:          return new ZExtInst          (S, Ty, Name, InsertBefore);
  case SExt:          return new SExtInst          (S, Ty, Name, InsertBefore);
  case FPTrunc:       return new FPTruncInst       (S, Ty, Name, InsertBefore);
  case FPExt:         return new FPExtInst         (S, Ty, Name, InsertBefore);
  case UIToFP:        return new UIToFPInst        (S, Ty, Name, InsertBefore);
  case SIToFP:        return new SIToFPInst        (S, Ty, Name, InsertBefore);
  case FPToUI:        return new FPToUIInst        (S, Ty, Name, InsertBefore);
  case FPToSI:        return new FPToSIInst        (S, Ty, Name, InsertBefore);
  case PtrToInt:      return new PtrToIntInst      (S, Ty, Name, InsertBefore);
  case IntToPtr:      return new IntToPtrInst      (S, Ty, Name, InsertBefore);
  case BitCast:       return new BitCastInst       (S, Ty, Name, InsertBefore);
  case AddrSpaceCast: return new AddrSpaceCastInst (S, Ty, Name, InsertBefore);
  default: llvm_unreachable("Invalid opcode provided");
  }
}

CastInst *CastInst::Create(Instruction::CastOps op, Value *S, Type *Ty,
  const Twine &Name, BasicBlock *InsertAtEnd) {
  assert(castIsValid(op, S, Ty) && "Invalid cast!");
  // Construct and return the appropriate CastInst subclass
  switch (op) {
  case Trunc:         return new TruncInst         (S, Ty, Name, InsertAtEnd);
  case ZExt:          return new ZExtInst          (S, Ty, Name, InsertAtEnd);
  case SExt:          return new SExtInst          (S, Ty, Name, InsertAtEnd);
  case FPTrunc:       return new FPTruncInst       (S, Ty, Name, InsertAtEnd);
  case FPExt:         return new FPExtInst         (S, Ty, Name, InsertAtEnd);
  case UIToFP:        return new UIToFPInst        (S, Ty, Name, InsertAtEnd);
  case SIToFP:        return new SIToFPInst        (S, Ty, Name, InsertAtEnd);
  case FPToUI:        return new FPToUIInst        (S, Ty, Name, InsertAtEnd);
  case FPToSI:        return new FPToSIInst        (S, Ty, Name, InsertAtEnd);
  case PtrToInt:      return new PtrToIntInst      (S, Ty, Name, InsertAtEnd);
  case IntToPtr:      return new IntToPtrInst      (S, Ty, Name, InsertAtEnd);
  case BitCast:       return new BitCastInst       (S, Ty, Name, InsertAtEnd);
  case AddrSpaceCast: return new AddrSpaceCastInst (S, Ty, Name, InsertAtEnd);
  default: llvm_unreachable("Invalid opcode provided");
  }
}

CastInst *CastInst::CreateZExtOrBitCast(Value *S, Type *Ty,
                                        const Twine &Name,
                                        Instruction *InsertBefore) {
  if (S->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return Create(Instruction::BitCast, S, Ty, Name, InsertBefore);
  return Create(Instruction::ZExt, S, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreateZExtOrBitCast(Value *S, Type *Ty,
                                        const Twine &Name,
                                        BasicBlock *InsertAtEnd) {
  if (S->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return Create(Instruction::BitCast, S, Ty, Name, InsertAtEnd);
  return Create(Instruction::ZExt, S, Ty, Name, InsertAtEnd);
}

CastInst *CastInst::CreateSExtOrBitCast(Value *S, Type *Ty,
                                        const Twine &Name,
                                        Instruction *InsertBefore) {
  if (S->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return Create(Instruction::BitCast, S, Ty, Name, InsertBefore);
  return Create(Instruction::SExt, S, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreateSExtOrBitCast(Value *S, Type *Ty,
                                        const Twine &Name,
                                        BasicBlock *InsertAtEnd) {
  if (S->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return Create(Instruction::BitCast, S, Ty, Name, InsertAtEnd);
  return Create(Instruction::SExt, S, Ty, Name, InsertAtEnd);
}

CastInst *CastInst::CreateTruncOrBitCast(Value *S, Type *Ty,
                                         const Twine &Name,
                                         Instruction *InsertBefore) {
  if (S->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return Create(Instruction::BitCast, S, Ty, Name, InsertBefore);
  return Create(Instruction::Trunc, S, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreateTruncOrBitCast(Value *S, Type *Ty,
                                         const Twine &Name,
                                         BasicBlock *InsertAtEnd) {
  if (S->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return Create(Instruction::BitCast, S, Ty, Name, InsertAtEnd);
  return Create(Instruction::Trunc, S, Ty, Name, InsertAtEnd);
}

CastInst *CastInst::CreatePointerCast(Value *S, Type *Ty,
                                      const Twine &Name,
                                      BasicBlock *InsertAtEnd) {
  assert(S->getType()->isPtrOrPtrVectorTy() && "Invalid cast");
  assert((Ty->isIntOrIntVectorTy() || Ty->isPtrOrPtrVectorTy()) &&
         "Invalid cast");
  assert(Ty->isVectorTy() == S->getType()->isVectorTy() && "Invalid cast");
  assert((!Ty->isVectorTy() ||
          cast<VectorType>(Ty)->getNumElements() ==
              cast<VectorType>(S->getType())->getNumElements()) &&
         "Invalid cast");

  if (Ty->isIntOrIntVectorTy())
    return Create(Instruction::PtrToInt, S, Ty, Name, InsertAtEnd);

  return CreatePointerBitCastOrAddrSpaceCast(S, Ty, Name, InsertAtEnd);
}

/// Create a BitCast or a PtrToInt cast instruction
CastInst *CastInst::CreatePointerCast(Value *S, Type *Ty,
                                      const Twine &Name,
                                      Instruction *InsertBefore) {
  assert(S->getType()->isPtrOrPtrVectorTy() && "Invalid cast");
  assert((Ty->isIntOrIntVectorTy() || Ty->isPtrOrPtrVectorTy()) &&
         "Invalid cast");
  assert(Ty->isVectorTy() == S->getType()->isVectorTy() && "Invalid cast");
  assert((!Ty->isVectorTy() ||
          cast<VectorType>(Ty)->getNumElements() ==
              cast<VectorType>(S->getType())->getNumElements()) &&
         "Invalid cast");

  if (Ty->isIntOrIntVectorTy())
    return Create(Instruction::PtrToInt, S, Ty, Name, InsertBefore);

  return CreatePointerBitCastOrAddrSpaceCast(S, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreatePointerBitCastOrAddrSpaceCast(
  Value *S, Type *Ty,
  const Twine &Name,
  BasicBlock *InsertAtEnd) {
  assert(S->getType()->isPtrOrPtrVectorTy() && "Invalid cast");
  assert(Ty->isPtrOrPtrVectorTy() && "Invalid cast");

  if (S->getType()->getPointerAddressSpace() != Ty->getPointerAddressSpace())
    return Create(Instruction::AddrSpaceCast, S, Ty, Name, InsertAtEnd);

  return Create(Instruction::BitCast, S, Ty, Name, InsertAtEnd);
}

CastInst *CastInst::CreatePointerBitCastOrAddrSpaceCast(
  Value *S, Type *Ty,
  const Twine &Name,
  Instruction *InsertBefore) {
  assert(S->getType()->isPtrOrPtrVectorTy() && "Invalid cast");
  assert(Ty->isPtrOrPtrVectorTy() && "Invalid cast");

  if (S->getType()->getPointerAddressSpace() != Ty->getPointerAddressSpace())
    return Create(Instruction::AddrSpaceCast, S, Ty, Name, InsertBefore);

  return Create(Instruction::BitCast, S, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreateBitOrPointerCast(Value *S, Type *Ty,
                                           const Twine &Name,
                                           Instruction *InsertBefore) {
  if (S->getType()->isPointerTy() && Ty->isIntegerTy())
    return Create(Instruction::PtrToInt, S, Ty, Name, InsertBefore);
  if (S->getType()->isIntegerTy() && Ty->isPointerTy())
    return Create(Instruction::IntToPtr, S, Ty, Name, InsertBefore);

  return Create(Instruction::BitCast, S, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreateIntegerCast(Value *C, Type *Ty,
                                      bool isSigned, const Twine &Name,
                                      Instruction *InsertBefore) {
  assert(C->getType()->isIntOrIntVectorTy() && Ty->isIntOrIntVectorTy() &&
         "Invalid integer cast");
  unsigned SrcBits = C->getType()->getScalarSizeInBits();
  unsigned DstBits = Ty->getScalarSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::Trunc :
      (isSigned ? Instruction::SExt : Instruction::ZExt)));
  return Create(opcode, C, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreateIntegerCast(Value *C, Type *Ty,
                                      bool isSigned, const Twine &Name,
                                      BasicBlock *InsertAtEnd) {
  assert(C->getType()->isIntOrIntVectorTy() && Ty->isIntOrIntVectorTy() &&
         "Invalid cast");
  unsigned SrcBits = C->getType()->getScalarSizeInBits();
  unsigned DstBits = Ty->getScalarSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::Trunc :
      (isSigned ? Instruction::SExt : Instruction::ZExt)));
  return Create(opcode, C, Ty, Name, InsertAtEnd);
}

CastInst *CastInst::CreateFPCast(Value *C, Type *Ty,
                                 const Twine &Name,
                                 Instruction *InsertBefore) {
  assert(C->getType()->isFPOrFPVectorTy() && Ty->isFPOrFPVectorTy() &&
         "Invalid cast");
  unsigned SrcBits = C->getType()->getScalarSizeInBits();
  unsigned DstBits = Ty->getScalarSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::FPTrunc : Instruction::FPExt));
  return Create(opcode, C, Ty, Name, InsertBefore);
}

CastInst *CastInst::CreateFPCast(Value *C, Type *Ty,
                                 const Twine &Name,
                                 BasicBlock *InsertAtEnd) {
  assert(C->getType()->isFPOrFPVectorTy() && Ty->isFPOrFPVectorTy() &&
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
bool CastInst::isCastable(Type *SrcTy, Type *DestTy) {
  if (!SrcTy->isFirstClassType() || !DestTy->isFirstClassType())
    return false;

  if (SrcTy == DestTy)
    return true;

  if (VectorType *SrcVecTy = dyn_cast<VectorType>(SrcTy))
    if (VectorType *DestVecTy = dyn_cast<VectorType>(DestTy))
      if (SrcVecTy->getNumElements() == DestVecTy->getNumElements()) {
        // An element by element cast.  Valid if casting the elements is valid.
        SrcTy = SrcVecTy->getElementType();
        DestTy = DestVecTy->getElementType();
      }

  // Get the bit sizes, we'll need these
  TypeSize SrcBits = SrcTy->getPrimitiveSizeInBits();   // 0 for ptr
  TypeSize DestBits = DestTy->getPrimitiveSizeInBits(); // 0 for ptr

  // Run through the possibilities ...
  if (DestTy->isIntegerTy()) {               // Casting to integral
    if (SrcTy->isIntegerTy())                // Casting from integral
        return true;
    if (SrcTy->isFloatingPointTy())   // Casting from floating pt
      return true;
    if (SrcTy->isVectorTy())          // Casting from vector
      return DestBits == SrcBits;
                                      // Casting from something else
    return SrcTy->isPointerTy();
  }
  if (DestTy->isFloatingPointTy()) {  // Casting to floating pt
    if (SrcTy->isIntegerTy())                // Casting from integral
      return true;
    if (SrcTy->isFloatingPointTy())   // Casting from floating pt
      return true;
    if (SrcTy->isVectorTy())          // Casting from vector
      return DestBits == SrcBits;
                                    // Casting from something else
    return false;
  }
  if (DestTy->isVectorTy())         // Casting to vector
    return DestBits == SrcBits;
  if (DestTy->isPointerTy()) {        // Casting to pointer
    if (SrcTy->isPointerTy())                // Casting from pointer
      return true;
    return SrcTy->isIntegerTy();             // Casting from integral
  }
  if (DestTy->isX86_MMXTy()) {
    if (SrcTy->isVectorTy())
      return DestBits == SrcBits;       // 64-bit vector to MMX
    return false;
  }                                    // Casting to something else
  return false;
}

bool CastInst::isBitCastable(Type *SrcTy, Type *DestTy) {
  if (!SrcTy->isFirstClassType() || !DestTy->isFirstClassType())
    return false;

  if (SrcTy == DestTy)
    return true;

  if (VectorType *SrcVecTy = dyn_cast<VectorType>(SrcTy)) {
    if (VectorType *DestVecTy = dyn_cast<VectorType>(DestTy)) {
      if (SrcVecTy->getElementCount() == DestVecTy->getElementCount()) {
        // An element by element cast. Valid if casting the elements is valid.
        SrcTy = SrcVecTy->getElementType();
        DestTy = DestVecTy->getElementType();
      }
    }
  }

  if (PointerType *DestPtrTy = dyn_cast<PointerType>(DestTy)) {
    if (PointerType *SrcPtrTy = dyn_cast<PointerType>(SrcTy)) {
      return SrcPtrTy->getAddressSpace() == DestPtrTy->getAddressSpace();
    }
  }

  TypeSize SrcBits = SrcTy->getPrimitiveSizeInBits();   // 0 for ptr
  TypeSize DestBits = DestTy->getPrimitiveSizeInBits(); // 0 for ptr

  // Could still have vectors of pointers if the number of elements doesn't
  // match
  if (SrcBits.getKnownMinSize() == 0 || DestBits.getKnownMinSize() == 0)
    return false;

  if (SrcBits != DestBits)
    return false;

  if (DestTy->isX86_MMXTy() || SrcTy->isX86_MMXTy())
    return false;

  return true;
}

bool CastInst::isBitOrNoopPointerCastable(Type *SrcTy, Type *DestTy,
                                          const DataLayout &DL) {
  // ptrtoint and inttoptr are not allowed on non-integral pointers
  if (auto *PtrTy = dyn_cast<PointerType>(SrcTy))
    if (auto *IntTy = dyn_cast<IntegerType>(DestTy))
      return (IntTy->getBitWidth() == DL.getPointerTypeSizeInBits(PtrTy) &&
              !DL.isNonIntegralPointerType(PtrTy));
  if (auto *PtrTy = dyn_cast<PointerType>(DestTy))
    if (auto *IntTy = dyn_cast<IntegerType>(SrcTy))
      return (IntTy->getBitWidth() == DL.getPointerTypeSizeInBits(PtrTy) &&
              !DL.isNonIntegralPointerType(PtrTy));

  return isBitCastable(SrcTy, DestTy);
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
  const Value *Src, bool SrcIsSigned, Type *DestTy, bool DestIsSigned) {
  Type *SrcTy = Src->getType();

  assert(SrcTy->isFirstClassType() && DestTy->isFirstClassType() &&
         "Only first class types are castable!");

  if (SrcTy == DestTy)
    return BitCast;

  // FIXME: Check address space sizes here
  if (VectorType *SrcVecTy = dyn_cast<VectorType>(SrcTy))
    if (VectorType *DestVecTy = dyn_cast<VectorType>(DestTy))
      if (SrcVecTy->getNumElements() == DestVecTy->getNumElements()) {
        // An element by element cast.  Find the appropriate opcode based on the
        // element types.
        SrcTy = SrcVecTy->getElementType();
        DestTy = DestVecTy->getElementType();
      }

  // Get the bit sizes, we'll need these
  unsigned SrcBits = SrcTy->getPrimitiveSizeInBits();   // 0 for ptr
  unsigned DestBits = DestTy->getPrimitiveSizeInBits(); // 0 for ptr

  // Run through the possibilities ...
  if (DestTy->isIntegerTy()) {                      // Casting to integral
    if (SrcTy->isIntegerTy()) {                     // Casting from integral
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
    } else if (SrcTy->isFloatingPointTy()) {        // Casting from floating pt
      if (DestIsSigned)
        return FPToSI;                              // FP -> sint
      else
        return FPToUI;                              // FP -> uint
    } else if (SrcTy->isVectorTy()) {
      assert(DestBits == SrcBits &&
             "Casting vector to integer of different width");
      return BitCast;                             // Same size, no-op cast
    } else {
      assert(SrcTy->isPointerTy() &&
             "Casting from a value that is not first-class type");
      return PtrToInt;                              // ptr -> int
    }
  } else if (DestTy->isFloatingPointTy()) {         // Casting to floating pt
    if (SrcTy->isIntegerTy()) {                     // Casting from integral
      if (SrcIsSigned)
        return SIToFP;                              // sint -> FP
      else
        return UIToFP;                              // uint -> FP
    } else if (SrcTy->isFloatingPointTy()) {        // Casting from floating pt
      if (DestBits < SrcBits) {
        return FPTrunc;                             // FP -> smaller FP
      } else if (DestBits > SrcBits) {
        return FPExt;                               // FP -> larger FP
      } else  {
        return BitCast;                             // same size, no-op cast
      }
    } else if (SrcTy->isVectorTy()) {
      assert(DestBits == SrcBits &&
             "Casting vector to floating point of different width");
      return BitCast;                             // same size, no-op cast
    }
    llvm_unreachable("Casting pointer or non-first class to float");
  } else if (DestTy->isVectorTy()) {
    assert(DestBits == SrcBits &&
           "Illegal cast to vector (wrong type or size)");
    return BitCast;
  } else if (DestTy->isPointerTy()) {
    if (SrcTy->isPointerTy()) {
      if (DestTy->getPointerAddressSpace() != SrcTy->getPointerAddressSpace())
        return AddrSpaceCast;
      return BitCast;                               // ptr -> ptr
    } else if (SrcTy->isIntegerTy()) {
      return IntToPtr;                              // int -> ptr
    }
    llvm_unreachable("Casting pointer to other than pointer or int");
  } else if (DestTy->isX86_MMXTy()) {
    if (SrcTy->isVectorTy()) {
      assert(DestBits == SrcBits && "Casting vector of wrong width to X86_MMX");
      return BitCast;                               // 64-bit vector to MMX
    }
    llvm_unreachable("Illegal cast to X86_MMX");
  }
  llvm_unreachable("Casting to type that is not first-class");
}

//===----------------------------------------------------------------------===//
//                    CastInst SubClass Constructors
//===----------------------------------------------------------------------===//

/// Check that the construction parameters for a CastInst are correct. This
/// could be broken out into the separate constructors but it is useful to have
/// it in one place and to eliminate the redundant code for getting the sizes
/// of the types involved.
bool
CastInst::castIsValid(Instruction::CastOps op, Value *S, Type *DstTy) {
  // Check for type sanity on the arguments
  Type *SrcTy = S->getType();

  if (!SrcTy->isFirstClassType() || !DstTy->isFirstClassType() ||
      SrcTy->isAggregateType() || DstTy->isAggregateType())
    return false;

  // Get the size of the types in bits, and whether we are dealing
  // with vector types, we'll need this later.
  bool SrcIsVec = isa<VectorType>(SrcTy);
  bool DstIsVec = isa<VectorType>(DstTy);
  unsigned SrcScalarBitSize = SrcTy->getScalarSizeInBits();
  unsigned DstScalarBitSize = DstTy->getScalarSizeInBits();

  // If these are vector types, get the lengths of the vectors (using zero for
  // scalar types means that checking that vector lengths match also checks that
  // scalars are not being converted to vectors or vectors to scalars).
  ElementCount SrcEC = SrcIsVec ? cast<VectorType>(SrcTy)->getElementCount()
                                : ElementCount(0, false);
  ElementCount DstEC = DstIsVec ? cast<VectorType>(DstTy)->getElementCount()
                                : ElementCount(0, false);

  // Switch on the opcode provided
  switch (op) {
  default: return false; // This is an input error
  case Instruction::Trunc:
    return SrcTy->isIntOrIntVectorTy() && DstTy->isIntOrIntVectorTy() &&
           SrcEC == DstEC && SrcScalarBitSize > DstScalarBitSize;
  case Instruction::ZExt:
    return SrcTy->isIntOrIntVectorTy() && DstTy->isIntOrIntVectorTy() &&
           SrcEC == DstEC && SrcScalarBitSize < DstScalarBitSize;
  case Instruction::SExt:
    return SrcTy->isIntOrIntVectorTy() && DstTy->isIntOrIntVectorTy() &&
           SrcEC == DstEC && SrcScalarBitSize < DstScalarBitSize;
  case Instruction::FPTrunc:
    return SrcTy->isFPOrFPVectorTy() && DstTy->isFPOrFPVectorTy() &&
           SrcEC == DstEC && SrcScalarBitSize > DstScalarBitSize;
  case Instruction::FPExt:
    return SrcTy->isFPOrFPVectorTy() && DstTy->isFPOrFPVectorTy() &&
           SrcEC == DstEC && SrcScalarBitSize < DstScalarBitSize;
  case Instruction::UIToFP:
  case Instruction::SIToFP:
    return SrcTy->isIntOrIntVectorTy() && DstTy->isFPOrFPVectorTy() &&
           SrcEC == DstEC;
  case Instruction::FPToUI:
  case Instruction::FPToSI:
    return SrcTy->isFPOrFPVectorTy() && DstTy->isIntOrIntVectorTy() &&
           SrcEC == DstEC;
  case Instruction::PtrToInt:
    if (SrcEC != DstEC)
      return false;
    return SrcTy->isPtrOrPtrVectorTy() && DstTy->isIntOrIntVectorTy();
  case Instruction::IntToPtr:
    if (SrcEC != DstEC)
      return false;
    return SrcTy->isIntOrIntVectorTy() && DstTy->isPtrOrPtrVectorTy();
  case Instruction::BitCast: {
    PointerType *SrcPtrTy = dyn_cast<PointerType>(SrcTy->getScalarType());
    PointerType *DstPtrTy = dyn_cast<PointerType>(DstTy->getScalarType());

    // BitCast implies a no-op cast of type only. No bits change.
    // However, you can't cast pointers to anything but pointers.
    if (!SrcPtrTy != !DstPtrTy)
      return false;

    // For non-pointer cases, the cast is okay if the source and destination bit
    // widths are identical.
    if (!SrcPtrTy)
      return SrcTy->getPrimitiveSizeInBits() == DstTy->getPrimitiveSizeInBits();

    // If both are pointers then the address spaces must match.
    if (SrcPtrTy->getAddressSpace() != DstPtrTy->getAddressSpace())
      return false;

    // A vector of pointers must have the same number of elements.
    if (SrcIsVec && DstIsVec)
      return SrcEC == DstEC;
    if (SrcIsVec)
      return SrcEC == ElementCount(1, false);
    if (DstIsVec)
      return DstEC == ElementCount(1, false);

    return true;
  }
  case Instruction::AddrSpaceCast: {
    PointerType *SrcPtrTy = dyn_cast<PointerType>(SrcTy->getScalarType());
    if (!SrcPtrTy)
      return false;

    PointerType *DstPtrTy = dyn_cast<PointerType>(DstTy->getScalarType());
    if (!DstPtrTy)
      return false;

    if (SrcPtrTy->getAddressSpace() == DstPtrTy->getAddressSpace())
      return false;

    return SrcEC == DstEC;
  }
  }
}

TruncInst::TruncInst(
  Value *S, Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, Trunc, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal Trunc");
}

TruncInst::TruncInst(
  Value *S, Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, Trunc, S, Name, InsertAtEnd) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal Trunc");
}

ZExtInst::ZExtInst(
  Value *S, Type *Ty, const Twine &Name, Instruction *InsertBefore
)  : CastInst(Ty, ZExt, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal ZExt");
}

ZExtInst::ZExtInst(
  Value *S, Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
)  : CastInst(Ty, ZExt, S, Name, InsertAtEnd) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal ZExt");
}
SExtInst::SExtInst(
  Value *S, Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, SExt, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal SExt");
}

SExtInst::SExtInst(
  Value *S, Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
)  : CastInst(Ty, SExt, S, Name, InsertAtEnd) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal SExt");
}

FPTruncInst::FPTruncInst(
  Value *S, Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, FPTrunc, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPTrunc");
}

FPTruncInst::FPTruncInst(
  Value *S, Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, FPTrunc, S, Name, InsertAtEnd) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPTrunc");
}

FPExtInst::FPExtInst(
  Value *S, Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, FPExt, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPExt");
}

FPExtInst::FPExtInst(
  Value *S, Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, FPExt, S, Name, InsertAtEnd) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPExt");
}

UIToFPInst::UIToFPInst(
  Value *S, Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, UIToFP, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal UIToFP");
}

UIToFPInst::UIToFPInst(
  Value *S, Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, UIToFP, S, Name, InsertAtEnd) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal UIToFP");
}

SIToFPInst::SIToFPInst(
  Value *S, Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, SIToFP, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal SIToFP");
}

SIToFPInst::SIToFPInst(
  Value *S, Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, SIToFP, S, Name, InsertAtEnd) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal SIToFP");
}

FPToUIInst::FPToUIInst(
  Value *S, Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, FPToUI, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPToUI");
}

FPToUIInst::FPToUIInst(
  Value *S, Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, FPToUI, S, Name, InsertAtEnd) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPToUI");
}

FPToSIInst::FPToSIInst(
  Value *S, Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, FPToSI, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPToSI");
}

FPToSIInst::FPToSIInst(
  Value *S, Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, FPToSI, S, Name, InsertAtEnd) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPToSI");
}

PtrToIntInst::PtrToIntInst(
  Value *S, Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, PtrToInt, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal PtrToInt");
}

PtrToIntInst::PtrToIntInst(
  Value *S, Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, PtrToInt, S, Name, InsertAtEnd) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal PtrToInt");
}

IntToPtrInst::IntToPtrInst(
  Value *S, Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, IntToPtr, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal IntToPtr");
}

IntToPtrInst::IntToPtrInst(
  Value *S, Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, IntToPtr, S, Name, InsertAtEnd) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal IntToPtr");
}

BitCastInst::BitCastInst(
  Value *S, Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, BitCast, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal BitCast");
}

BitCastInst::BitCastInst(
  Value *S, Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, BitCast, S, Name, InsertAtEnd) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal BitCast");
}

AddrSpaceCastInst::AddrSpaceCastInst(
  Value *S, Type *Ty, const Twine &Name, Instruction *InsertBefore
) : CastInst(Ty, AddrSpaceCast, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal AddrSpaceCast");
}

AddrSpaceCastInst::AddrSpaceCastInst(
  Value *S, Type *Ty, const Twine &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, AddrSpaceCast, S, Name, InsertAtEnd) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal AddrSpaceCast");
}

//===----------------------------------------------------------------------===//
//                               CmpInst Classes
//===----------------------------------------------------------------------===//

CmpInst::CmpInst(Type *ty, OtherOps op, Predicate predicate, Value *LHS,
                 Value *RHS, const Twine &Name, Instruction *InsertBefore,
                 Instruction *FlagsSource)
  : Instruction(ty, op,
                OperandTraits<CmpInst>::op_begin(this),
                OperandTraits<CmpInst>::operands(this),
                InsertBefore) {
  Op<0>() = LHS;
  Op<1>() = RHS;
  setPredicate((Predicate)predicate);
  setName(Name);
  if (FlagsSource)
    copyIRFlags(FlagsSource);
}

CmpInst::CmpInst(Type *ty, OtherOps op, Predicate predicate, Value *LHS,
                 Value *RHS, const Twine &Name, BasicBlock *InsertAtEnd)
  : Instruction(ty, op,
                OperandTraits<CmpInst>::op_begin(this),
                OperandTraits<CmpInst>::operands(this),
                InsertAtEnd) {
  Op<0>() = LHS;
  Op<1>() = RHS;
  setPredicate((Predicate)predicate);
  setName(Name);
}

CmpInst *
CmpInst::Create(OtherOps Op, Predicate predicate, Value *S1, Value *S2,
                const Twine &Name, Instruction *InsertBefore) {
  if (Op == Instruction::ICmp) {
    if (InsertBefore)
      return new ICmpInst(InsertBefore, CmpInst::Predicate(predicate),
                          S1, S2, Name);
    else
      return new ICmpInst(CmpInst::Predicate(predicate),
                          S1, S2, Name);
  }

  if (InsertBefore)
    return new FCmpInst(InsertBefore, CmpInst::Predicate(predicate),
                        S1, S2, Name);
  else
    return new FCmpInst(CmpInst::Predicate(predicate),
                        S1, S2, Name);
}

CmpInst *
CmpInst::Create(OtherOps Op, Predicate predicate, Value *S1, Value *S2,
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

bool CmpInst::isCommutative() const {
  if (const ICmpInst *IC = dyn_cast<ICmpInst>(this))
    return IC->isCommutative();
  return cast<FCmpInst>(this)->isCommutative();
}

bool CmpInst::isEquality() const {
  if (const ICmpInst *IC = dyn_cast<ICmpInst>(this))
    return IC->isEquality();
  return cast<FCmpInst>(this)->isEquality();
}

CmpInst::Predicate CmpInst::getInversePredicate(Predicate pred) {
  switch (pred) {
    default: llvm_unreachable("Unknown cmp predicate!");
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

StringRef CmpInst::getPredicateName(Predicate Pred) {
  switch (Pred) {
  default:                   return "unknown";
  case FCmpInst::FCMP_FALSE: return "false";
  case FCmpInst::FCMP_OEQ:   return "oeq";
  case FCmpInst::FCMP_OGT:   return "ogt";
  case FCmpInst::FCMP_OGE:   return "oge";
  case FCmpInst::FCMP_OLT:   return "olt";
  case FCmpInst::FCMP_OLE:   return "ole";
  case FCmpInst::FCMP_ONE:   return "one";
  case FCmpInst::FCMP_ORD:   return "ord";
  case FCmpInst::FCMP_UNO:   return "uno";
  case FCmpInst::FCMP_UEQ:   return "ueq";
  case FCmpInst::FCMP_UGT:   return "ugt";
  case FCmpInst::FCMP_UGE:   return "uge";
  case FCmpInst::FCMP_ULT:   return "ult";
  case FCmpInst::FCMP_ULE:   return "ule";
  case FCmpInst::FCMP_UNE:   return "une";
  case FCmpInst::FCMP_TRUE:  return "true";
  case ICmpInst::ICMP_EQ:    return "eq";
  case ICmpInst::ICMP_NE:    return "ne";
  case ICmpInst::ICMP_SGT:   return "sgt";
  case ICmpInst::ICMP_SGE:   return "sge";
  case ICmpInst::ICMP_SLT:   return "slt";
  case ICmpInst::ICMP_SLE:   return "sle";
  case ICmpInst::ICMP_UGT:   return "ugt";
  case ICmpInst::ICMP_UGE:   return "uge";
  case ICmpInst::ICMP_ULT:   return "ult";
  case ICmpInst::ICMP_ULE:   return "ule";
  }
}

ICmpInst::Predicate ICmpInst::getSignedPredicate(Predicate pred) {
  switch (pred) {
    default: llvm_unreachable("Unknown icmp predicate!");
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
    default: llvm_unreachable("Unknown icmp predicate!");
    case ICMP_EQ: case ICMP_NE:
    case ICMP_UGT: case ICMP_ULT: case ICMP_UGE: case ICMP_ULE:
       return pred;
    case ICMP_SGT: return ICMP_UGT;
    case ICMP_SLT: return ICMP_ULT;
    case ICMP_SGE: return ICMP_UGE;
    case ICMP_SLE: return ICMP_ULE;
  }
}

CmpInst::Predicate CmpInst::getFlippedStrictnessPredicate(Predicate pred) {
  switch (pred) {
    default: llvm_unreachable("Unknown or unsupported cmp predicate!");
    case ICMP_SGT: return ICMP_SGE;
    case ICMP_SLT: return ICMP_SLE;
    case ICMP_SGE: return ICMP_SGT;
    case ICMP_SLE: return ICMP_SLT;
    case ICMP_UGT: return ICMP_UGE;
    case ICMP_ULT: return ICMP_ULE;
    case ICMP_UGE: return ICMP_UGT;
    case ICMP_ULE: return ICMP_ULT;

    case FCMP_OGT: return FCMP_OGE;
    case FCMP_OLT: return FCMP_OLE;
    case FCMP_OGE: return FCMP_OGT;
    case FCMP_OLE: return FCMP_OLT;
    case FCMP_UGT: return FCMP_UGE;
    case FCMP_ULT: return FCMP_ULE;
    case FCMP_UGE: return FCMP_UGT;
    case FCMP_ULE: return FCMP_ULT;
  }
}

CmpInst::Predicate CmpInst::getSwappedPredicate(Predicate pred) {
  switch (pred) {
    default: llvm_unreachable("Unknown cmp predicate!");
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

CmpInst::Predicate CmpInst::getNonStrictPredicate(Predicate pred) {
  switch (pred) {
  case ICMP_SGT: return ICMP_SGE;
  case ICMP_SLT: return ICMP_SLE;
  case ICMP_UGT: return ICMP_UGE;
  case ICMP_ULT: return ICMP_ULE;
  case FCMP_OGT: return FCMP_OGE;
  case FCMP_OLT: return FCMP_OLE;
  case FCMP_UGT: return FCMP_UGE;
  case FCMP_ULT: return FCMP_ULE;
  default: return pred;
  }
}

CmpInst::Predicate CmpInst::getSignedPredicate(Predicate pred) {
  assert(CmpInst::isUnsigned(pred) && "Call only with signed predicates!");

  switch (pred) {
  default:
    llvm_unreachable("Unknown predicate!");
  case CmpInst::ICMP_ULT:
    return CmpInst::ICMP_SLT;
  case CmpInst::ICMP_ULE:
    return CmpInst::ICMP_SLE;
  case CmpInst::ICMP_UGT:
    return CmpInst::ICMP_SGT;
  case CmpInst::ICMP_UGE:
    return CmpInst::ICMP_SGE;
  }
}

bool CmpInst::isUnsigned(Predicate predicate) {
  switch (predicate) {
    default: return false;
    case ICmpInst::ICMP_ULT: case ICmpInst::ICMP_ULE: case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_UGE: return true;
  }
}

bool CmpInst::isSigned(Predicate predicate) {
  switch (predicate) {
    default: return false;
    case ICmpInst::ICMP_SLT: case ICmpInst::ICMP_SLE: case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_SGE: return true;
  }
}

bool CmpInst::isOrdered(Predicate predicate) {
  switch (predicate) {
    default: return false;
    case FCmpInst::FCMP_OEQ: case FCmpInst::FCMP_ONE: case FCmpInst::FCMP_OGT:
    case FCmpInst::FCMP_OLT: case FCmpInst::FCMP_OGE: case FCmpInst::FCMP_OLE:
    case FCmpInst::FCMP_ORD: return true;
  }
}

bool CmpInst::isUnordered(Predicate predicate) {
  switch (predicate) {
    default: return false;
    case FCmpInst::FCMP_UEQ: case FCmpInst::FCMP_UNE: case FCmpInst::FCMP_UGT:
    case FCmpInst::FCMP_ULT: case FCmpInst::FCMP_UGE: case FCmpInst::FCMP_ULE:
    case FCmpInst::FCMP_UNO: return true;
  }
}

bool CmpInst::isTrueWhenEqual(Predicate predicate) {
  switch(predicate) {
    default: return false;
    case ICMP_EQ:   case ICMP_UGE: case ICMP_ULE: case ICMP_SGE: case ICMP_SLE:
    case FCMP_TRUE: case FCMP_UEQ: case FCMP_UGE: case FCMP_ULE: return true;
  }
}

bool CmpInst::isFalseWhenEqual(Predicate predicate) {
  switch(predicate) {
  case ICMP_NE:    case ICMP_UGT: case ICMP_ULT: case ICMP_SGT: case ICMP_SLT:
  case FCMP_FALSE: case FCMP_ONE: case FCMP_OGT: case FCMP_OLT: return true;
  default: return false;
  }
}

bool CmpInst::isImpliedTrueByMatchingCmp(Predicate Pred1, Predicate Pred2) {
  // If the predicates match, then we know the first condition implies the
  // second is true.
  if (Pred1 == Pred2)
    return true;

  switch (Pred1) {
  default:
    break;
  case ICMP_EQ:
    // A == B implies A >=u B, A <=u B, A >=s B, and A <=s B are true.
    return Pred2 == ICMP_UGE || Pred2 == ICMP_ULE || Pred2 == ICMP_SGE ||
           Pred2 == ICMP_SLE;
  case ICMP_UGT: // A >u B implies A != B and A >=u B are true.
    return Pred2 == ICMP_NE || Pred2 == ICMP_UGE;
  case ICMP_ULT: // A <u B implies A != B and A <=u B are true.
    return Pred2 == ICMP_NE || Pred2 == ICMP_ULE;
  case ICMP_SGT: // A >s B implies A != B and A >=s B are true.
    return Pred2 == ICMP_NE || Pred2 == ICMP_SGE;
  case ICMP_SLT: // A <s B implies A != B and A <=s B are true.
    return Pred2 == ICMP_NE || Pred2 == ICMP_SLE;
  }
  return false;
}

bool CmpInst::isImpliedFalseByMatchingCmp(Predicate Pred1, Predicate Pred2) {
  return isImpliedTrueByMatchingCmp(Pred1, getInversePredicate(Pred2));
}

//===----------------------------------------------------------------------===//
//                        SwitchInst Implementation
//===----------------------------------------------------------------------===//

void SwitchInst::init(Value *Value, BasicBlock *Default, unsigned NumReserved) {
  assert(Value && Default && NumReserved);
  ReservedSpace = NumReserved;
  setNumHungOffUseOperands(2);
  allocHungoffUses(ReservedSpace);

  Op<0>() = Value;
  Op<1>() = Default;
}

/// SwitchInst ctor - Create a new switch instruction, specifying a value to
/// switch on and a default destination.  The number of additional cases can
/// be specified here to make memory allocation more efficient.  This
/// constructor can also autoinsert before another instruction.
SwitchInst::SwitchInst(Value *Value, BasicBlock *Default, unsigned NumCases,
                       Instruction *InsertBefore)
    : Instruction(Type::getVoidTy(Value->getContext()), Instruction::Switch,
                  nullptr, 0, InsertBefore) {
  init(Value, Default, 2+NumCases*2);
}

/// SwitchInst ctor - Create a new switch instruction, specifying a value to
/// switch on and a default destination.  The number of additional cases can
/// be specified here to make memory allocation more efficient.  This
/// constructor also autoinserts at the end of the specified BasicBlock.
SwitchInst::SwitchInst(Value *Value, BasicBlock *Default, unsigned NumCases,
                       BasicBlock *InsertAtEnd)
    : Instruction(Type::getVoidTy(Value->getContext()), Instruction::Switch,
                  nullptr, 0, InsertAtEnd) {
  init(Value, Default, 2+NumCases*2);
}

SwitchInst::SwitchInst(const SwitchInst &SI)
    : Instruction(SI.getType(), Instruction::Switch, nullptr, 0) {
  init(SI.getCondition(), SI.getDefaultDest(), SI.getNumOperands());
  setNumHungOffUseOperands(SI.getNumOperands());
  Use *OL = getOperandList();
  const Use *InOL = SI.getOperandList();
  for (unsigned i = 2, E = SI.getNumOperands(); i != E; i += 2) {
    OL[i] = InOL[i];
    OL[i+1] = InOL[i+1];
  }
  SubclassOptionalData = SI.SubclassOptionalData;
}

/// addCase - Add an entry to the switch instruction...
///
void SwitchInst::addCase(ConstantInt *OnVal, BasicBlock *Dest) {
  unsigned NewCaseIdx = getNumCases();
  unsigned OpNo = getNumOperands();
  if (OpNo+2 > ReservedSpace)
    growOperands();  // Get more space!
  // Initialize some new operands.
  assert(OpNo+1 < ReservedSpace && "Growing didn't work!");
  setNumHungOffUseOperands(OpNo+2);
  CaseHandle Case(this, NewCaseIdx);
  Case.setValue(OnVal);
  Case.setSuccessor(Dest);
}

/// removeCase - This method removes the specified case and its successor
/// from the switch instruction.
SwitchInst::CaseIt SwitchInst::removeCase(CaseIt I) {
  unsigned idx = I->getCaseIndex();

  assert(2 + idx*2 < getNumOperands() && "Case index out of range!!!");

  unsigned NumOps = getNumOperands();
  Use *OL = getOperandList();

  // Overwrite this case with the end of the list.
  if (2 + (idx + 1) * 2 != NumOps) {
    OL[2 + idx * 2] = OL[NumOps - 2];
    OL[2 + idx * 2 + 1] = OL[NumOps - 1];
  }

  // Nuke the last value.
  OL[NumOps-2].set(nullptr);
  OL[NumOps-2+1].set(nullptr);
  setNumHungOffUseOperands(NumOps-2);

  return CaseIt(this, idx);
}

/// growOperands - grow operands - This grows the operand list in response
/// to a push_back style of operation.  This grows the number of ops by 3 times.
///
void SwitchInst::growOperands() {
  unsigned e = getNumOperands();
  unsigned NumOps = e*3;

  ReservedSpace = NumOps;
  growHungoffUses(ReservedSpace);
}

MDNode *
SwitchInstProfUpdateWrapper::getProfBranchWeightsMD(const SwitchInst &SI) {
  if (MDNode *ProfileData = SI.getMetadata(LLVMContext::MD_prof))
    if (auto *MDName = dyn_cast<MDString>(ProfileData->getOperand(0)))
      if (MDName->getString() == "branch_weights")
        return ProfileData;
  return nullptr;
}

MDNode *SwitchInstProfUpdateWrapper::buildProfBranchWeightsMD() {
  assert(Changed && "called only if metadata has changed");

  if (!Weights)
    return nullptr;

  assert(SI.getNumSuccessors() == Weights->size() &&
         "num of prof branch_weights must accord with num of successors");

  bool AllZeroes =
      all_of(Weights.getValue(), [](uint32_t W) { return W == 0; });

  if (AllZeroes || Weights.getValue().size() < 2)
    return nullptr;

  return MDBuilder(SI.getParent()->getContext()).createBranchWeights(*Weights);
}

void SwitchInstProfUpdateWrapper::init() {
  MDNode *ProfileData = getProfBranchWeightsMD(SI);
  if (!ProfileData)
    return;

  if (ProfileData->getNumOperands() != SI.getNumSuccessors() + 1) {
    llvm_unreachable("number of prof branch_weights metadata operands does "
                     "not correspond to number of succesors");
  }

  SmallVector<uint32_t, 8> Weights;
  for (unsigned CI = 1, CE = SI.getNumSuccessors(); CI <= CE; ++CI) {
    ConstantInt *C = mdconst::extract<ConstantInt>(ProfileData->getOperand(CI));
    uint32_t CW = C->getValue().getZExtValue();
    Weights.push_back(CW);
  }
  this->Weights = std::move(Weights);
}

SwitchInst::CaseIt
SwitchInstProfUpdateWrapper::removeCase(SwitchInst::CaseIt I) {
  if (Weights) {
    assert(SI.getNumSuccessors() == Weights->size() &&
           "num of prof branch_weights must accord with num of successors");
    Changed = true;
    // Copy the last case to the place of the removed one and shrink.
    // This is tightly coupled with the way SwitchInst::removeCase() removes
    // the cases in SwitchInst::removeCase(CaseIt).
    Weights.getValue()[I->getCaseIndex() + 1] = Weights.getValue().back();
    Weights.getValue().pop_back();
  }
  return SI.removeCase(I);
}

void SwitchInstProfUpdateWrapper::addCase(
    ConstantInt *OnVal, BasicBlock *Dest,
    SwitchInstProfUpdateWrapper::CaseWeightOpt W) {
  SI.addCase(OnVal, Dest);

  if (!Weights && W && *W) {
    Changed = true;
    Weights = SmallVector<uint32_t, 8>(SI.getNumSuccessors(), 0);
    Weights.getValue()[SI.getNumSuccessors() - 1] = *W;
  } else if (Weights) {
    Changed = true;
    Weights.getValue().push_back(W ? *W : 0);
  }
  if (Weights)
    assert(SI.getNumSuccessors() == Weights->size() &&
           "num of prof branch_weights must accord with num of successors");
}

SymbolTableList<Instruction>::iterator
SwitchInstProfUpdateWrapper::eraseFromParent() {
  // Instruction is erased. Mark as unchanged to not touch it in the destructor.
  Changed = false;
  if (Weights)
    Weights->resize(0);
  return SI.eraseFromParent();
}

SwitchInstProfUpdateWrapper::CaseWeightOpt
SwitchInstProfUpdateWrapper::getSuccessorWeight(unsigned idx) {
  if (!Weights)
    return None;
  return Weights.getValue()[idx];
}

void SwitchInstProfUpdateWrapper::setSuccessorWeight(
    unsigned idx, SwitchInstProfUpdateWrapper::CaseWeightOpt W) {
  if (!W)
    return;

  if (!Weights && *W)
    Weights = SmallVector<uint32_t, 8>(SI.getNumSuccessors(), 0);

  if (Weights) {
    auto &OldW = Weights.getValue()[idx];
    if (*W != OldW) {
      Changed = true;
      OldW = *W;
    }
  }
}

SwitchInstProfUpdateWrapper::CaseWeightOpt
SwitchInstProfUpdateWrapper::getSuccessorWeight(const SwitchInst &SI,
                                                unsigned idx) {
  if (MDNode *ProfileData = getProfBranchWeightsMD(SI))
    if (ProfileData->getNumOperands() == SI.getNumSuccessors() + 1)
      return mdconst::extract<ConstantInt>(ProfileData->getOperand(idx + 1))
          ->getValue()
          .getZExtValue();

  return None;
}

//===----------------------------------------------------------------------===//
//                        IndirectBrInst Implementation
//===----------------------------------------------------------------------===//

void IndirectBrInst::init(Value *Address, unsigned NumDests) {
  assert(Address && Address->getType()->isPointerTy() &&
         "Address of indirectbr must be a pointer");
  ReservedSpace = 1+NumDests;
  setNumHungOffUseOperands(1);
  allocHungoffUses(ReservedSpace);

  Op<0>() = Address;
}


/// growOperands - grow operands - This grows the operand list in response
/// to a push_back style of operation.  This grows the number of ops by 2 times.
///
void IndirectBrInst::growOperands() {
  unsigned e = getNumOperands();
  unsigned NumOps = e*2;

  ReservedSpace = NumOps;
  growHungoffUses(ReservedSpace);
}

IndirectBrInst::IndirectBrInst(Value *Address, unsigned NumCases,
                               Instruction *InsertBefore)
    : Instruction(Type::getVoidTy(Address->getContext()),
                  Instruction::IndirectBr, nullptr, 0, InsertBefore) {
  init(Address, NumCases);
}

IndirectBrInst::IndirectBrInst(Value *Address, unsigned NumCases,
                               BasicBlock *InsertAtEnd)
    : Instruction(Type::getVoidTy(Address->getContext()),
                  Instruction::IndirectBr, nullptr, 0, InsertAtEnd) {
  init(Address, NumCases);
}

IndirectBrInst::IndirectBrInst(const IndirectBrInst &IBI)
    : Instruction(Type::getVoidTy(IBI.getContext()), Instruction::IndirectBr,
                  nullptr, IBI.getNumOperands()) {
  allocHungoffUses(IBI.getNumOperands());
  Use *OL = getOperandList();
  const Use *InOL = IBI.getOperandList();
  for (unsigned i = 0, E = IBI.getNumOperands(); i != E; ++i)
    OL[i] = InOL[i];
  SubclassOptionalData = IBI.SubclassOptionalData;
}

/// addDestination - Add a destination.
///
void IndirectBrInst::addDestination(BasicBlock *DestBB) {
  unsigned OpNo = getNumOperands();
  if (OpNo+1 > ReservedSpace)
    growOperands();  // Get more space!
  // Initialize some new operands.
  assert(OpNo < ReservedSpace && "Growing didn't work!");
  setNumHungOffUseOperands(OpNo+1);
  getOperandList()[OpNo] = DestBB;
}

/// removeDestination - This method removes the specified successor from the
/// indirectbr instruction.
void IndirectBrInst::removeDestination(unsigned idx) {
  assert(idx < getNumOperands()-1 && "Successor index out of range!");

  unsigned NumOps = getNumOperands();
  Use *OL = getOperandList();

  // Replace this value with the last one.
  OL[idx+1] = OL[NumOps-1];

  // Nuke the last value.
  OL[NumOps-1].set(nullptr);
  setNumHungOffUseOperands(NumOps-1);
}

//===----------------------------------------------------------------------===//
//                            FreezeInst Implementation
//===----------------------------------------------------------------------===//

FreezeInst::FreezeInst(Value *S,
                       const Twine &Name, Instruction *InsertBefore)
    : UnaryInstruction(S->getType(), Freeze, S, InsertBefore) {
  setName(Name);
}

FreezeInst::FreezeInst(Value *S,
                       const Twine &Name, BasicBlock *InsertAtEnd)
    : UnaryInstruction(S->getType(), Freeze, S, InsertAtEnd) {
  setName(Name);
}

//===----------------------------------------------------------------------===//
//                           cloneImpl() implementations
//===----------------------------------------------------------------------===//

// Define these methods here so vtables don't get emitted into every translation
// unit that uses these classes.

GetElementPtrInst *GetElementPtrInst::cloneImpl() const {
  return new (getNumOperands()) GetElementPtrInst(*this);
}

UnaryOperator *UnaryOperator::cloneImpl() const {
  return Create(getOpcode(), Op<0>());
}

BinaryOperator *BinaryOperator::cloneImpl() const {
  return Create(getOpcode(), Op<0>(), Op<1>());
}

FCmpInst *FCmpInst::cloneImpl() const {
  return new FCmpInst(getPredicate(), Op<0>(), Op<1>());
}

ICmpInst *ICmpInst::cloneImpl() const {
  return new ICmpInst(getPredicate(), Op<0>(), Op<1>());
}

ExtractValueInst *ExtractValueInst::cloneImpl() const {
  return new ExtractValueInst(*this);
}

InsertValueInst *InsertValueInst::cloneImpl() const {
  return new InsertValueInst(*this);
}

AllocaInst *AllocaInst::cloneImpl() const {
  AllocaInst *Result =
      new AllocaInst(getAllocatedType(), getType()->getAddressSpace(),
                     getOperand(0), getAlign());
  Result->setUsedWithInAlloca(isUsedWithInAlloca());
  Result->setSwiftError(isSwiftError());
  return Result;
}

LoadInst *LoadInst::cloneImpl() const {
  return new LoadInst(getType(), getOperand(0), Twine(), isVolatile(),
                      getAlign(), getOrdering(), getSyncScopeID());
}

StoreInst *StoreInst::cloneImpl() const {
  return new StoreInst(getOperand(0), getOperand(1), isVolatile(), getAlign(),
                       getOrdering(), getSyncScopeID());
}

AtomicCmpXchgInst *AtomicCmpXchgInst::cloneImpl() const {
  AtomicCmpXchgInst *Result = new AtomicCmpXchgInst(
      getOperand(0), getOperand(1), getOperand(2), getAlign(),
      getSuccessOrdering(), getFailureOrdering(), getSyncScopeID());
  Result->setVolatile(isVolatile());
  Result->setWeak(isWeak());
  return Result;
}

AtomicRMWInst *AtomicRMWInst::cloneImpl() const {
  AtomicRMWInst *Result =
      new AtomicRMWInst(getOperation(), getOperand(0), getOperand(1),
                        getAlign(), getOrdering(), getSyncScopeID());
  Result->setVolatile(isVolatile());
  return Result;
}

FenceInst *FenceInst::cloneImpl() const {
  return new FenceInst(getContext(), getOrdering(), getSyncScopeID());
}

TruncInst *TruncInst::cloneImpl() const {
  return new TruncInst(getOperand(0), getType());
}

ZExtInst *ZExtInst::cloneImpl() const {
  return new ZExtInst(getOperand(0), getType());
}

SExtInst *SExtInst::cloneImpl() const {
  return new SExtInst(getOperand(0), getType());
}

FPTruncInst *FPTruncInst::cloneImpl() const {
  return new FPTruncInst(getOperand(0), getType());
}

FPExtInst *FPExtInst::cloneImpl() const {
  return new FPExtInst(getOperand(0), getType());
}

UIToFPInst *UIToFPInst::cloneImpl() const {
  return new UIToFPInst(getOperand(0), getType());
}

SIToFPInst *SIToFPInst::cloneImpl() const {
  return new SIToFPInst(getOperand(0), getType());
}

FPToUIInst *FPToUIInst::cloneImpl() const {
  return new FPToUIInst(getOperand(0), getType());
}

FPToSIInst *FPToSIInst::cloneImpl() const {
  return new FPToSIInst(getOperand(0), getType());
}

PtrToIntInst *PtrToIntInst::cloneImpl() const {
  return new PtrToIntInst(getOperand(0), getType());
}

IntToPtrInst *IntToPtrInst::cloneImpl() const {
  return new IntToPtrInst(getOperand(0), getType());
}

BitCastInst *BitCastInst::cloneImpl() const {
  return new BitCastInst(getOperand(0), getType());
}

AddrSpaceCastInst *AddrSpaceCastInst::cloneImpl() const {
  return new AddrSpaceCastInst(getOperand(0), getType());
}

CallInst *CallInst::cloneImpl() const {
  if (hasOperandBundles()) {
    unsigned DescriptorBytes = getNumOperandBundles() * sizeof(BundleOpInfo);
    return new(getNumOperands(), DescriptorBytes) CallInst(*this);
  }
  return  new(getNumOperands()) CallInst(*this);
}

SelectInst *SelectInst::cloneImpl() const {
  return SelectInst::Create(getOperand(0), getOperand(1), getOperand(2));
}

VAArgInst *VAArgInst::cloneImpl() const {
  return new VAArgInst(getOperand(0), getType());
}

ExtractElementInst *ExtractElementInst::cloneImpl() const {
  return ExtractElementInst::Create(getOperand(0), getOperand(1));
}

InsertElementInst *InsertElementInst::cloneImpl() const {
  return InsertElementInst::Create(getOperand(0), getOperand(1), getOperand(2));
}

ShuffleVectorInst *ShuffleVectorInst::cloneImpl() const {
  return new ShuffleVectorInst(getOperand(0), getOperand(1), getShuffleMask());
}

PHINode *PHINode::cloneImpl() const { return new PHINode(*this); }

LandingPadInst *LandingPadInst::cloneImpl() const {
  return new LandingPadInst(*this);
}

ReturnInst *ReturnInst::cloneImpl() const {
  return new(getNumOperands()) ReturnInst(*this);
}

BranchInst *BranchInst::cloneImpl() const {
  return new(getNumOperands()) BranchInst(*this);
}

SwitchInst *SwitchInst::cloneImpl() const { return new SwitchInst(*this); }

IndirectBrInst *IndirectBrInst::cloneImpl() const {
  return new IndirectBrInst(*this);
}

InvokeInst *InvokeInst::cloneImpl() const {
  if (hasOperandBundles()) {
    unsigned DescriptorBytes = getNumOperandBundles() * sizeof(BundleOpInfo);
    return new(getNumOperands(), DescriptorBytes) InvokeInst(*this);
  }
  return new(getNumOperands()) InvokeInst(*this);
}

CallBrInst *CallBrInst::cloneImpl() const {
  if (hasOperandBundles()) {
    unsigned DescriptorBytes = getNumOperandBundles() * sizeof(BundleOpInfo);
    return new (getNumOperands(), DescriptorBytes) CallBrInst(*this);
  }
  return new (getNumOperands()) CallBrInst(*this);
}

ResumeInst *ResumeInst::cloneImpl() const { return new (1) ResumeInst(*this); }

CleanupReturnInst *CleanupReturnInst::cloneImpl() const {
  return new (getNumOperands()) CleanupReturnInst(*this);
}

CatchReturnInst *CatchReturnInst::cloneImpl() const {
  return new (getNumOperands()) CatchReturnInst(*this);
}

CatchSwitchInst *CatchSwitchInst::cloneImpl() const {
  return new CatchSwitchInst(*this);
}

FuncletPadInst *FuncletPadInst::cloneImpl() const {
  return new (getNumOperands()) FuncletPadInst(*this);
}

UnreachableInst *UnreachableInst::cloneImpl() const {
  LLVMContext &Context = getContext();
  return new UnreachableInst(Context);
}

FreezeInst *FreezeInst::cloneImpl() const {
  return new FreezeInst(getOperand(0));
}
