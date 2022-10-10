//===-- Instruction.cpp - Implement the Instruction class -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Instruction class for the IR library.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Instruction.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
using namespace llvm;

Instruction::Instruction(Type *ty, unsigned it, Use *Ops, unsigned NumOps,
                         Instruction *InsertBefore)
  : User(ty, Value::InstructionVal + it, Ops, NumOps), Parent(nullptr) {

  // If requested, insert this instruction into a basic block...
  if (InsertBefore) {
    BasicBlock *BB = InsertBefore->getParent();
    assert(BB && "Instruction to insert before is not in a basic block!");
    BB->getInstList().insert(InsertBefore->getIterator(), this);
  }
}

Instruction::Instruction(Type *ty, unsigned it, Use *Ops, unsigned NumOps,
                         BasicBlock *InsertAtEnd)
  : User(ty, Value::InstructionVal + it, Ops, NumOps), Parent(nullptr) {

  // append this instruction into the basic block
  assert(InsertAtEnd && "Basic block to append to may not be NULL!");
  InsertAtEnd->getInstList().push_back(this);
}

Instruction::~Instruction() {
  assert(!Parent && "Instruction still linked in the program!");

  // Replace any extant metadata uses of this instruction with undef to
  // preserve debug info accuracy. Some alternatives include:
  // - Treat Instruction like any other Value, and point its extant metadata
  //   uses to an empty ValueAsMetadata node. This makes extant dbg.value uses
  //   trivially dead (i.e. fair game for deletion in many passes), leading to
  //   stale dbg.values being in effect for too long.
  // - Call salvageDebugInfoOrMarkUndef. Not needed to make instruction removal
  //   correct. OTOH results in wasted work in some common cases (e.g. when all
  //   instructions in a BasicBlock are deleted).
  if (isUsedByMetadata())
    ValueAsMetadata::handleRAUW(this, UndefValue::get(getType()));
}


void Instruction::setParent(BasicBlock *P) {
  Parent = P;
}

const Module *Instruction::getModule() const {
  return getParent()->getModule();
}

const Function *Instruction::getFunction() const {
  return getParent()->getParent();
}

void Instruction::removeFromParent() {
  getParent()->getInstList().remove(getIterator());
}

iplist<Instruction>::iterator Instruction::eraseFromParent() {
  return getParent()->getInstList().erase(getIterator());
}

/// Insert an unlinked instruction into a basic block immediately before the
/// specified instruction.
void Instruction::insertBefore(Instruction *InsertPos) {
  InsertPos->getParent()->getInstList().insert(InsertPos->getIterator(), this);
}

/// Insert an unlinked instruction into a basic block immediately after the
/// specified instruction.
void Instruction::insertAfter(Instruction *InsertPos) {
  InsertPos->getParent()->getInstList().insertAfter(InsertPos->getIterator(),
                                                    this);
}

/// Unlink this instruction from its current basic block and insert it into the
/// basic block that MovePos lives in, right before MovePos.
void Instruction::moveBefore(Instruction *MovePos) {
  moveBefore(*MovePos->getParent(), MovePos->getIterator());
}

void Instruction::moveAfter(Instruction *MovePos) {
  moveBefore(*MovePos->getParent(), ++MovePos->getIterator());
}

void Instruction::moveBefore(BasicBlock &BB,
                             SymbolTableList<Instruction>::iterator I) {
  assert(I == BB.end() || I->getParent() == &BB);
  BB.getInstList().splice(I, getParent()->getInstList(), getIterator());
}

bool Instruction::comesBefore(const Instruction *Other) const {
  assert(Parent && Other->Parent &&
         "instructions without BB parents have no order");
  assert(Parent == Other->Parent && "cross-BB instruction order comparison");
  if (!Parent->isInstrOrderValid())
    Parent->renumberInstructions();
  return Order < Other->Order;
}

Instruction *Instruction::getInsertionPointAfterDef() {
  assert(!getType()->isVoidTy() && "Instruction must define result");
  BasicBlock *InsertBB;
  BasicBlock::iterator InsertPt;
  if (auto *PN = dyn_cast<PHINode>(this)) {
    InsertBB = PN->getParent();
    InsertPt = InsertBB->getFirstInsertionPt();
  } else if (auto *II = dyn_cast<InvokeInst>(this)) {
    InsertBB = II->getNormalDest();
    InsertPt = InsertBB->getFirstInsertionPt();
  } else if (auto *CB = dyn_cast<CallBrInst>(this)) {
    InsertBB = CB->getDefaultDest();
    InsertPt = InsertBB->getFirstInsertionPt();
  } else {
    assert(!isTerminator() && "Only invoke/callbr terminators return value");
    InsertBB = getParent();
    InsertPt = std::next(getIterator());
  }

  // catchswitch blocks don't have any legal insertion point (because they
  // are both an exception pad and a terminator).
  if (InsertPt == InsertBB->end())
    return nullptr;
  return &*InsertPt;
}

bool Instruction::isOnlyUserOfAnyOperand() {
  return any_of(operands(), [](Value *V) { return V->hasOneUser(); });
}

void Instruction::setHasNoUnsignedWrap(bool b) {
  cast<OverflowingBinaryOperator>(this)->setHasNoUnsignedWrap(b);
}

void Instruction::setHasNoSignedWrap(bool b) {
  cast<OverflowingBinaryOperator>(this)->setHasNoSignedWrap(b);
}

void Instruction::setIsExact(bool b) {
  cast<PossiblyExactOperator>(this)->setIsExact(b);
}

bool Instruction::hasNoUnsignedWrap() const {
  return cast<OverflowingBinaryOperator>(this)->hasNoUnsignedWrap();
}

bool Instruction::hasNoSignedWrap() const {
  return cast<OverflowingBinaryOperator>(this)->hasNoSignedWrap();
}

bool Instruction::hasPoisonGeneratingFlags() const {
  return cast<Operator>(this)->hasPoisonGeneratingFlags();
}

void Instruction::dropPoisonGeneratingFlags() {
  switch (getOpcode()) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::Shl:
    cast<OverflowingBinaryOperator>(this)->setHasNoUnsignedWrap(false);
    cast<OverflowingBinaryOperator>(this)->setHasNoSignedWrap(false);
    break;

  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::AShr:
  case Instruction::LShr:
    cast<PossiblyExactOperator>(this)->setIsExact(false);
    break;

  case Instruction::GetElementPtr:
    cast<GetElementPtrInst>(this)->setIsInBounds(false);
    break;
  }
  if (isa<FPMathOperator>(this)) {
    setHasNoNaNs(false);
    setHasNoInfs(false);
  }

  assert(!hasPoisonGeneratingFlags() && "must be kept in sync");
}

void Instruction::dropUndefImplyingAttrsAndUnknownMetadata(
    ArrayRef<unsigned> KnownIDs) {
  dropUnknownNonDebugMetadata(KnownIDs);
  auto *CB = dyn_cast<CallBase>(this);
  if (!CB)
    return;
  // For call instructions, we also need to drop parameter and return attributes
  // that are can cause UB if the call is moved to a location where the
  // attribute is not valid.
  AttributeList AL = CB->getAttributes();
  if (AL.isEmpty())
    return;
  AttributeMask UBImplyingAttributes =
      AttributeFuncs::getUBImplyingAttributes();
  for (unsigned ArgNo = 0; ArgNo < CB->arg_size(); ArgNo++)
    CB->removeParamAttrs(ArgNo, UBImplyingAttributes);
  CB->removeRetAttrs(UBImplyingAttributes);
}

bool Instruction::isExact() const {
  return cast<PossiblyExactOperator>(this)->isExact();
}

void Instruction::setFast(bool B) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setFast(B);
}

void Instruction::setHasAllowReassoc(bool B) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setHasAllowReassoc(B);
}

void Instruction::setHasNoNaNs(bool B) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setHasNoNaNs(B);
}

void Instruction::setHasNoInfs(bool B) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setHasNoInfs(B);
}

void Instruction::setHasNoSignedZeros(bool B) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setHasNoSignedZeros(B);
}

void Instruction::setHasAllowReciprocal(bool B) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setHasAllowReciprocal(B);
}

void Instruction::setHasAllowContract(bool B) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setHasAllowContract(B);
}

void Instruction::setHasApproxFunc(bool B) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setHasApproxFunc(B);
}

void Instruction::setFastMathFlags(FastMathFlags FMF) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setFastMathFlags(FMF);
}

void Instruction::copyFastMathFlags(FastMathFlags FMF) {
  assert(isa<FPMathOperator>(this) && "copying fast-math flag on invalid op");
  cast<FPMathOperator>(this)->copyFastMathFlags(FMF);
}

bool Instruction::isFast() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->isFast();
}

bool Instruction::hasAllowReassoc() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->hasAllowReassoc();
}

bool Instruction::hasNoNaNs() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->hasNoNaNs();
}

bool Instruction::hasNoInfs() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->hasNoInfs();
}

bool Instruction::hasNoSignedZeros() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->hasNoSignedZeros();
}

bool Instruction::hasAllowReciprocal() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->hasAllowReciprocal();
}

bool Instruction::hasAllowContract() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->hasAllowContract();
}

bool Instruction::hasApproxFunc() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->hasApproxFunc();
}

FastMathFlags Instruction::getFastMathFlags() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->getFastMathFlags();
}

void Instruction::copyFastMathFlags(const Instruction *I) {
  copyFastMathFlags(I->getFastMathFlags());
}

void Instruction::copyIRFlags(const Value *V, bool IncludeWrapFlags) {
  // Copy the wrapping flags.
  if (IncludeWrapFlags && isa<OverflowingBinaryOperator>(this)) {
    if (auto *OB = dyn_cast<OverflowingBinaryOperator>(V)) {
      setHasNoSignedWrap(OB->hasNoSignedWrap());
      setHasNoUnsignedWrap(OB->hasNoUnsignedWrap());
    }
  }

  // Copy the exact flag.
  if (auto *PE = dyn_cast<PossiblyExactOperator>(V))
    if (isa<PossiblyExactOperator>(this))
      setIsExact(PE->isExact());

  // Copy the fast-math flags.
  if (auto *FP = dyn_cast<FPMathOperator>(V))
    if (isa<FPMathOperator>(this))
      copyFastMathFlags(FP->getFastMathFlags());

  if (auto *SrcGEP = dyn_cast<GetElementPtrInst>(V))
    if (auto *DestGEP = dyn_cast<GetElementPtrInst>(this))
      DestGEP->setIsInBounds(SrcGEP->isInBounds() || DestGEP->isInBounds());
}

void Instruction::andIRFlags(const Value *V) {
  if (auto *OB = dyn_cast<OverflowingBinaryOperator>(V)) {
    if (isa<OverflowingBinaryOperator>(this)) {
      setHasNoSignedWrap(hasNoSignedWrap() && OB->hasNoSignedWrap());
      setHasNoUnsignedWrap(hasNoUnsignedWrap() && OB->hasNoUnsignedWrap());
    }
  }

  if (auto *PE = dyn_cast<PossiblyExactOperator>(V))
    if (isa<PossiblyExactOperator>(this))
      setIsExact(isExact() && PE->isExact());

  if (auto *FP = dyn_cast<FPMathOperator>(V)) {
    if (isa<FPMathOperator>(this)) {
      FastMathFlags FM = getFastMathFlags();
      FM &= FP->getFastMathFlags();
      copyFastMathFlags(FM);
    }
  }

  if (auto *SrcGEP = dyn_cast<GetElementPtrInst>(V))
    if (auto *DestGEP = dyn_cast<GetElementPtrInst>(this))
      DestGEP->setIsInBounds(SrcGEP->isInBounds() && DestGEP->isInBounds());
}

const char *Instruction::getOpcodeName(unsigned OpCode) {
  switch (OpCode) {
  // Terminators
  case Ret:    return "ret";
  case Br:     return "br";
  case Switch: return "switch";
  case IndirectBr: return "indirectbr";
  case Invoke: return "invoke";
  case Resume: return "resume";
  case Unreachable: return "unreachable";
  case CleanupRet: return "cleanupret";
  case CatchRet: return "catchret";
  case CatchPad: return "catchpad";
  case CatchSwitch: return "catchswitch";
  case CallBr: return "callbr";

  // Standard unary operators...
  case FNeg: return "fneg";

  // Standard binary operators...
  case Add: return "add";
  case FAdd: return "fadd";
  case Sub: return "sub";
  case FSub: return "fsub";
  case Mul: return "mul";
  case FMul: return "fmul";
  case UDiv: return "udiv";
  case SDiv: return "sdiv";
  case FDiv: return "fdiv";
  case URem: return "urem";
  case SRem: return "srem";
  case FRem: return "frem";

  // Logical operators...
  case And: return "and";
  case Or : return "or";
  case Xor: return "xor";

  // Memory instructions...
  case Alloca:        return "alloca";
  case Load:          return "load";
  case Store:         return "store";
  case AtomicCmpXchg: return "cmpxchg";
  case AtomicRMW:     return "atomicrmw";
  case Fence:         return "fence";
  case GetElementPtr: return "getelementptr";

  // Convert instructions...
  case Trunc:         return "trunc";
  case ZExt:          return "zext";
  case SExt:          return "sext";
  case FPTrunc:       return "fptrunc";
  case FPExt:         return "fpext";
  case FPToUI:        return "fptoui";
  case FPToSI:        return "fptosi";
  case UIToFP:        return "uitofp";
  case SIToFP:        return "sitofp";
  case IntToPtr:      return "inttoptr";
  case PtrToInt:      return "ptrtoint";
  case BitCast:       return "bitcast";
  case AddrSpaceCast: return "addrspacecast";

  // Other instructions...
  case ICmp:           return "icmp";
  case FCmp:           return "fcmp";
  case PHI:            return "phi";
  case Select:         return "select";
  case Call:           return "call";
  case Shl:            return "shl";
  case LShr:           return "lshr";
  case AShr:           return "ashr";
  case VAArg:          return "va_arg";
  case ExtractElement: return "extractelement";
  case InsertElement:  return "insertelement";
  case ShuffleVector:  return "shufflevector";
  case ExtractValue:   return "extractvalue";
  case InsertValue:    return "insertvalue";
  case LandingPad:     return "landingpad";
  case CleanupPad:     return "cleanuppad";
  case Freeze:         return "freeze";

  default: return "<Invalid operator> ";
  }
}

/// Return true if both instructions have the same special state. This must be
/// kept in sync with FunctionComparator::cmpOperations in
/// lib/Transforms/IPO/MergeFunctions.cpp.
static bool haveSameSpecialState(const Instruction *I1, const Instruction *I2,
                                 bool IgnoreAlignment = false) {
  assert(I1->getOpcode() == I2->getOpcode() &&
         "Can not compare special state of different instructions");

  if (const AllocaInst *AI = dyn_cast<AllocaInst>(I1))
    return AI->getAllocatedType() == cast<AllocaInst>(I2)->getAllocatedType() &&
           (AI->getAlign() == cast<AllocaInst>(I2)->getAlign() ||
            IgnoreAlignment);
  if (const LoadInst *LI = dyn_cast<LoadInst>(I1))
    return LI->isVolatile() == cast<LoadInst>(I2)->isVolatile() &&
           (LI->getAlign() == cast<LoadInst>(I2)->getAlign() ||
            IgnoreAlignment) &&
           LI->getOrdering() == cast<LoadInst>(I2)->getOrdering() &&
           LI->getSyncScopeID() == cast<LoadInst>(I2)->getSyncScopeID();
  if (const StoreInst *SI = dyn_cast<StoreInst>(I1))
    return SI->isVolatile() == cast<StoreInst>(I2)->isVolatile() &&
           (SI->getAlign() == cast<StoreInst>(I2)->getAlign() ||
            IgnoreAlignment) &&
           SI->getOrdering() == cast<StoreInst>(I2)->getOrdering() &&
           SI->getSyncScopeID() == cast<StoreInst>(I2)->getSyncScopeID();
  if (const CmpInst *CI = dyn_cast<CmpInst>(I1))
    return CI->getPredicate() == cast<CmpInst>(I2)->getPredicate();
  if (const CallInst *CI = dyn_cast<CallInst>(I1))
    return CI->isTailCall() == cast<CallInst>(I2)->isTailCall() &&
           CI->getCallingConv() == cast<CallInst>(I2)->getCallingConv() &&
           CI->getAttributes() == cast<CallInst>(I2)->getAttributes() &&
           CI->hasIdenticalOperandBundleSchema(*cast<CallInst>(I2));
  if (const InvokeInst *CI = dyn_cast<InvokeInst>(I1))
    return CI->getCallingConv() == cast<InvokeInst>(I2)->getCallingConv() &&
           CI->getAttributes() == cast<InvokeInst>(I2)->getAttributes() &&
           CI->hasIdenticalOperandBundleSchema(*cast<InvokeInst>(I2));
  if (const CallBrInst *CI = dyn_cast<CallBrInst>(I1))
    return CI->getCallingConv() == cast<CallBrInst>(I2)->getCallingConv() &&
           CI->getAttributes() == cast<CallBrInst>(I2)->getAttributes() &&
           CI->hasIdenticalOperandBundleSchema(*cast<CallBrInst>(I2));
  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(I1))
    return IVI->getIndices() == cast<InsertValueInst>(I2)->getIndices();
  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(I1))
    return EVI->getIndices() == cast<ExtractValueInst>(I2)->getIndices();
  if (const FenceInst *FI = dyn_cast<FenceInst>(I1))
    return FI->getOrdering() == cast<FenceInst>(I2)->getOrdering() &&
           FI->getSyncScopeID() == cast<FenceInst>(I2)->getSyncScopeID();
  if (const AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(I1))
    return CXI->isVolatile() == cast<AtomicCmpXchgInst>(I2)->isVolatile() &&
           CXI->isWeak() == cast<AtomicCmpXchgInst>(I2)->isWeak() &&
           CXI->getSuccessOrdering() ==
               cast<AtomicCmpXchgInst>(I2)->getSuccessOrdering() &&
           CXI->getFailureOrdering() ==
               cast<AtomicCmpXchgInst>(I2)->getFailureOrdering() &&
           CXI->getSyncScopeID() ==
               cast<AtomicCmpXchgInst>(I2)->getSyncScopeID();
  if (const AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I1))
    return RMWI->getOperation() == cast<AtomicRMWInst>(I2)->getOperation() &&
           RMWI->isVolatile() == cast<AtomicRMWInst>(I2)->isVolatile() &&
           RMWI->getOrdering() == cast<AtomicRMWInst>(I2)->getOrdering() &&
           RMWI->getSyncScopeID() == cast<AtomicRMWInst>(I2)->getSyncScopeID();
  if (const ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(I1))
    return SVI->getShuffleMask() ==
           cast<ShuffleVectorInst>(I2)->getShuffleMask();
  if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I1))
    return GEP->getSourceElementType() ==
           cast<GetElementPtrInst>(I2)->getSourceElementType();

  return true;
}

bool Instruction::isIdenticalTo(const Instruction *I) const {
  return isIdenticalToWhenDefined(I) &&
         SubclassOptionalData == I->SubclassOptionalData;
}

bool Instruction::isIdenticalToWhenDefined(const Instruction *I) const {
  if (getOpcode() != I->getOpcode() ||
      getNumOperands() != I->getNumOperands() ||
      getType() != I->getType())
    return false;

  // If both instructions have no operands, they are identical.
  if (getNumOperands() == 0 && I->getNumOperands() == 0)
    return haveSameSpecialState(this, I);

  // We have two instructions of identical opcode and #operands.  Check to see
  // if all operands are the same.
  if (!std::equal(op_begin(), op_end(), I->op_begin()))
    return false;

  // WARNING: this logic must be kept in sync with EliminateDuplicatePHINodes()!
  if (const PHINode *thisPHI = dyn_cast<PHINode>(this)) {
    const PHINode *otherPHI = cast<PHINode>(I);
    return std::equal(thisPHI->block_begin(), thisPHI->block_end(),
                      otherPHI->block_begin());
  }

  return haveSameSpecialState(this, I);
}

// Keep this in sync with FunctionComparator::cmpOperations in
// lib/Transforms/IPO/MergeFunctions.cpp.
bool Instruction::isSameOperationAs(const Instruction *I,
                                    unsigned flags) const {
  bool IgnoreAlignment = flags & CompareIgnoringAlignment;
  bool UseScalarTypes  = flags & CompareUsingScalarTypes;

  if (getOpcode() != I->getOpcode() ||
      getNumOperands() != I->getNumOperands() ||
      (UseScalarTypes ?
       getType()->getScalarType() != I->getType()->getScalarType() :
       getType() != I->getType()))
    return false;

  // We have two instructions of identical opcode and #operands.  Check to see
  // if all operands are the same type
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (UseScalarTypes ?
        getOperand(i)->getType()->getScalarType() !=
          I->getOperand(i)->getType()->getScalarType() :
        getOperand(i)->getType() != I->getOperand(i)->getType())
      return false;

  return haveSameSpecialState(this, I, IgnoreAlignment);
}

bool Instruction::isUsedOutsideOfBlock(const BasicBlock *BB) const {
  for (const Use &U : uses()) {
    // PHI nodes uses values in the corresponding predecessor block.  For other
    // instructions, just check to see whether the parent of the use matches up.
    const Instruction *I = cast<Instruction>(U.getUser());
    const PHINode *PN = dyn_cast<PHINode>(I);
    if (!PN) {
      if (I->getParent() != BB)
        return true;
      continue;
    }

    if (PN->getIncomingBlock(U) != BB)
      return true;
  }
  return false;
}

bool Instruction::mayReadFromMemory() const {
  switch (getOpcode()) {
  default: return false;
  case Instruction::VAArg:
  case Instruction::Load:
  case Instruction::Fence: // FIXME: refine definition of mayReadFromMemory
  case Instruction::AtomicCmpXchg:
  case Instruction::AtomicRMW:
  case Instruction::CatchPad:
  case Instruction::CatchRet:
    return true;
  case Instruction::Call:
  case Instruction::Invoke:
  case Instruction::CallBr:
    return !cast<CallBase>(this)->onlyWritesMemory();
  case Instruction::Store:
    return !cast<StoreInst>(this)->isUnordered();
  }
}

bool Instruction::mayWriteToMemory() const {
  switch (getOpcode()) {
  default: return false;
  case Instruction::Fence: // FIXME: refine definition of mayWriteToMemory
  case Instruction::Store:
  case Instruction::VAArg:
  case Instruction::AtomicCmpXchg:
  case Instruction::AtomicRMW:
  case Instruction::CatchPad:
  case Instruction::CatchRet:
    return true;
  case Instruction::Call:
  case Instruction::Invoke:
  case Instruction::CallBr:
    return !cast<CallBase>(this)->onlyReadsMemory();
  case Instruction::Load:
    return !cast<LoadInst>(this)->isUnordered();
  }
}

bool Instruction::isAtomic() const {
  switch (getOpcode()) {
  default:
    return false;
  case Instruction::AtomicCmpXchg:
  case Instruction::AtomicRMW:
  case Instruction::Fence:
    return true;
  case Instruction::Load:
    return cast<LoadInst>(this)->getOrdering() != AtomicOrdering::NotAtomic;
  case Instruction::Store:
    return cast<StoreInst>(this)->getOrdering() != AtomicOrdering::NotAtomic;
  }
}

bool Instruction::hasAtomicLoad() const {
  assert(isAtomic());
  switch (getOpcode()) {
  default:
    return false;
  case Instruction::AtomicCmpXchg:
  case Instruction::AtomicRMW:
  case Instruction::Load:
    return true;
  }
}

bool Instruction::hasAtomicStore() const {
  assert(isAtomic());
  switch (getOpcode()) {
  default:
    return false;
  case Instruction::AtomicCmpXchg:
  case Instruction::AtomicRMW:
  case Instruction::Store:
    return true;
  }
}

bool Instruction::isVolatile() const {
  switch (getOpcode()) {
  default:
    return false;
  case Instruction::AtomicRMW:
    return cast<AtomicRMWInst>(this)->isVolatile();
  case Instruction::Store:
    return cast<StoreInst>(this)->isVolatile();
  case Instruction::Load:
    return cast<LoadInst>(this)->isVolatile();
  case Instruction::AtomicCmpXchg:
    return cast<AtomicCmpXchgInst>(this)->isVolatile();
  case Instruction::Call:
  case Instruction::Invoke:
    // There are a very limited number of intrinsics with volatile flags.
    if (auto *II = dyn_cast<IntrinsicInst>(this)) {
      if (auto *MI = dyn_cast<MemIntrinsic>(II))
        return MI->isVolatile();
      switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::matrix_column_major_load:
        return cast<ConstantInt>(II->getArgOperand(2))->isOne();
      case Intrinsic::matrix_column_major_store:
        return cast<ConstantInt>(II->getArgOperand(3))->isOne();
      }
    }
    return false;
  }
}

bool Instruction::mayThrow() const {
  if (const CallInst *CI = dyn_cast<CallInst>(this))
    return !CI->doesNotThrow();
  if (const auto *CRI = dyn_cast<CleanupReturnInst>(this))
    return CRI->unwindsToCaller();
  if (const auto *CatchSwitch = dyn_cast<CatchSwitchInst>(this))
    return CatchSwitch->unwindsToCaller();
  return isa<ResumeInst>(this);
}

bool Instruction::mayHaveSideEffects() const {
  return mayWriteToMemory() || mayThrow() || !willReturn();
}

bool Instruction::isSafeToRemove() const {
  return (!isa<CallInst>(this) || !this->mayHaveSideEffects()) &&
         !this->isTerminator() && !this->isEHPad();
}

bool Instruction::willReturn() const {
  // Volatile store isn't guaranteed to return; see LangRef.
  if (auto *SI = dyn_cast<StoreInst>(this))
    return !SI->isVolatile();

  if (const auto *CB = dyn_cast<CallBase>(this))
    // FIXME: Temporarily assume that all side-effect free intrinsics will
    // return. Remove this workaround once all intrinsics are appropriately
    // annotated.
    return CB->hasFnAttr(Attribute::WillReturn) ||
           (isa<IntrinsicInst>(CB) && CB->onlyReadsMemory());
  return true;
}

bool Instruction::isLifetimeStartOrEnd() const {
  auto *II = dyn_cast<IntrinsicInst>(this);
  if (!II)
    return false;
  Intrinsic::ID ID = II->getIntrinsicID();
  return ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end;
}

bool Instruction::isLaunderOrStripInvariantGroup() const {
  auto *II = dyn_cast<IntrinsicInst>(this);
  if (!II)
    return false;
  Intrinsic::ID ID = II->getIntrinsicID();
  return ID == Intrinsic::launder_invariant_group ||
         ID == Intrinsic::strip_invariant_group;
}

bool Instruction::isDebugOrPseudoInst() const {
  return isa<DbgInfoIntrinsic>(this) || isa<PseudoProbeInst>(this);
}

const Instruction *
Instruction::getNextNonDebugInstruction(bool SkipPseudoOp) const {
  for (const Instruction *I = getNextNode(); I; I = I->getNextNode())
    if (!isa<DbgInfoIntrinsic>(I) && !(SkipPseudoOp && isa<PseudoProbeInst>(I)))
      return I;
  return nullptr;
}

const Instruction *
Instruction::getPrevNonDebugInstruction(bool SkipPseudoOp) const {
  for (const Instruction *I = getPrevNode(); I; I = I->getPrevNode())
    if (!isa<DbgInfoIntrinsic>(I) && !(SkipPseudoOp && isa<PseudoProbeInst>(I)))
      return I;
  return nullptr;
}

bool Instruction::isAssociative() const {
  unsigned Opcode = getOpcode();
  if (isAssociative(Opcode))
    return true;

  switch (Opcode) {
  case FMul:
  case FAdd:
    return cast<FPMathOperator>(this)->hasAllowReassoc() &&
           cast<FPMathOperator>(this)->hasNoSignedZeros();
  default:
    return false;
  }
}

bool Instruction::isCommutative() const {
  if (auto *II = dyn_cast<IntrinsicInst>(this))
    return II->isCommutative();
  // TODO: Should allow icmp/fcmp?
  return isCommutative(getOpcode());
}

unsigned Instruction::getNumSuccessors() const {
  switch (getOpcode()) {
#define HANDLE_TERM_INST(N, OPC, CLASS)                                        \
  case Instruction::OPC:                                                       \
    return static_cast<const CLASS *>(this)->getNumSuccessors();
#include "llvm/IR/Instruction.def"
  default:
    break;
  }
  llvm_unreachable("not a terminator");
}

BasicBlock *Instruction::getSuccessor(unsigned idx) const {
  switch (getOpcode()) {
#define HANDLE_TERM_INST(N, OPC, CLASS)                                        \
  case Instruction::OPC:                                                       \
    return static_cast<const CLASS *>(this)->getSuccessor(idx);
#include "llvm/IR/Instruction.def"
  default:
    break;
  }
  llvm_unreachable("not a terminator");
}

void Instruction::setSuccessor(unsigned idx, BasicBlock *B) {
  switch (getOpcode()) {
#define HANDLE_TERM_INST(N, OPC, CLASS)                                        \
  case Instruction::OPC:                                                       \
    return static_cast<CLASS *>(this)->setSuccessor(idx, B);
#include "llvm/IR/Instruction.def"
  default:
    break;
  }
  llvm_unreachable("not a terminator");
}

void Instruction::replaceSuccessorWith(BasicBlock *OldBB, BasicBlock *NewBB) {
  for (unsigned Idx = 0, NumSuccessors = Instruction::getNumSuccessors();
       Idx != NumSuccessors; ++Idx)
    if (getSuccessor(Idx) == OldBB)
      setSuccessor(Idx, NewBB);
}

Instruction *Instruction::cloneImpl() const {
  llvm_unreachable("Subclass of Instruction failed to implement cloneImpl");
}

void Instruction::swapProfMetadata() {
  MDNode *ProfileData = getMetadata(LLVMContext::MD_prof);
  if (!ProfileData || ProfileData->getNumOperands() != 3 ||
      !isa<MDString>(ProfileData->getOperand(0)))
    return;

  MDString *MDName = cast<MDString>(ProfileData->getOperand(0));
  if (MDName->getString() != "branch_weights")
    return;

  // The first operand is the name. Fetch them backwards and build a new one.
  Metadata *Ops[] = {ProfileData->getOperand(0), ProfileData->getOperand(2),
                     ProfileData->getOperand(1)};
  setMetadata(LLVMContext::MD_prof,
              MDNode::get(ProfileData->getContext(), Ops));
}

void Instruction::copyMetadata(const Instruction &SrcInst,
                               ArrayRef<unsigned> WL) {
  if (!SrcInst.hasMetadata())
    return;

  DenseSet<unsigned> WLS;
  for (unsigned M : WL)
    WLS.insert(M);

  // Otherwise, enumerate and copy over metadata from the old instruction to the
  // new one.
  SmallVector<std::pair<unsigned, MDNode *>, 4> TheMDs;
  SrcInst.getAllMetadataOtherThanDebugLoc(TheMDs);
  for (const auto &MD : TheMDs) {
    if (WL.empty() || WLS.count(MD.first))
      setMetadata(MD.first, MD.second);
  }
  if (WL.empty() || WLS.count(LLVMContext::MD_dbg))
    setDebugLoc(SrcInst.getDebugLoc());
}

Instruction *Instruction::clone() const {
  Instruction *New = nullptr;
  switch (getOpcode()) {
  default:
    llvm_unreachable("Unhandled Opcode.");
#define HANDLE_INST(num, opc, clas)                                            \
  case Instruction::opc:                                                       \
    New = cast<clas>(this)->cloneImpl();                                       \
    break;
#include "llvm/IR/Instruction.def"
#undef HANDLE_INST
  }

  New->SubclassOptionalData = SubclassOptionalData;
  New->copyMetadata(*this);
  return New;
}
