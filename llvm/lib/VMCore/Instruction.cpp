//===-- Instruction.cpp - Implement the Instruction class -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Instruction class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Type.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/LeakDetector.h"
using namespace llvm;

Instruction::Instruction(const Type *ty, unsigned it, Use *Ops, unsigned NumOps,
                         Instruction *InsertBefore)
  : User(ty, Value::InstructionVal + it, Ops, NumOps), Parent(0) {
  // Make sure that we get added to a basicblock
  LeakDetector::addGarbageObject(this);

  // If requested, insert this instruction into a basic block...
  if (InsertBefore) {
    assert(InsertBefore->getParent() &&
           "Instruction to insert before is not in a basic block!");
    InsertBefore->getParent()->getInstList().insert(InsertBefore, this);
  }
}

Instruction::Instruction(const Type *ty, unsigned it, Use *Ops, unsigned NumOps,
                         BasicBlock *InsertAtEnd)
  : User(ty, Value::InstructionVal + it, Ops, NumOps), Parent(0) {
  // Make sure that we get added to a basicblock
  LeakDetector::addGarbageObject(this);

  // append this instruction into the basic block
  assert(InsertAtEnd && "Basic block to append to may not be NULL!");
  InsertAtEnd->getInstList().push_back(this);
}


// Out of line virtual method, so the vtable, etc has a home.
Instruction::~Instruction() {
  assert(Parent == 0 && "Instruction still linked in the program!");
}


void Instruction::setParent(BasicBlock *P) {
  if (getParent()) {
    if (!P) LeakDetector::addGarbageObject(this);
  } else {
    if (P) LeakDetector::removeGarbageObject(this);
  }

  Parent = P;
}

void Instruction::removeFromParent() {
  getParent()->getInstList().remove(this);
}

void Instruction::eraseFromParent() {
  getParent()->getInstList().erase(this);
}

/// insertBefore - Insert an unlinked instructions into a basic block
/// immediately before the specified instruction.
void Instruction::insertBefore(Instruction *InsertPos) {
  InsertPos->getParent()->getInstList().insert(InsertPos, this);
}

/// insertAfter - Insert an unlinked instructions into a basic block
/// immediately after the specified instruction.
void Instruction::insertAfter(Instruction *InsertPos) {
  InsertPos->getParent()->getInstList().insertAfter(InsertPos, this);
}

/// moveBefore - Unlink this instruction from its current basic block and
/// insert it into the basic block that MovePos lives in, right before
/// MovePos.
void Instruction::moveBefore(Instruction *MovePos) {
  MovePos->getParent()->getInstList().splice(MovePos,getParent()->getInstList(),
                                             this);
}


const char *Instruction::getOpcodeName(unsigned OpCode) {
  switch (OpCode) {
  // Terminators
  case Ret:    return "ret";
  case Br:     return "br";
  case Switch: return "switch";
  case Invoke: return "invoke";
  case Unwind: return "unwind";
  case Unreachable: return "unreachable";

  // Standard binary operators...
  case Add: return "add";
  case Sub: return "sub";
  case Mul: return "mul";
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
  case Malloc:        return "malloc";
  case Free:          return "free";
  case Alloca:        return "alloca";
  case Load:          return "load";
  case Store:         return "store";
  case GetElementPtr: return "getelementptr";

  // Convert instructions...
  case Trunc:     return "trunc";
  case ZExt:      return "zext";
  case SExt:      return "sext";
  case FPTrunc:   return "fptrunc";
  case FPExt:     return "fpext";
  case FPToUI:    return "fptoui";
  case FPToSI:    return "fptosi";
  case UIToFP:    return "uitofp";
  case SIToFP:    return "sitofp";
  case IntToPtr:  return "inttoptr";
  case PtrToInt:  return "ptrtoint";
  case BitCast:   return "bitcast";

  // Other instructions...
  case ICmp:           return "icmp";
  case FCmp:           return "fcmp";
  case VICmp:          return "vicmp";
  case VFCmp:          return "vfcmp";
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

  default: return "<Invalid operator> ";
  }

  return 0;
}

/// isIdenticalTo - Return true if the specified instruction is exactly
/// identical to the current one.  This means that all operands match and any
/// extra information (e.g. load is volatile) agree.
bool Instruction::isIdenticalTo(const Instruction *I) const {
  if (getOpcode() != I->getOpcode() ||
      getNumOperands() != I->getNumOperands() ||
      getType() != I->getType())
    return false;

  // We have two instructions of identical opcode and #operands.  Check to see
  // if all operands are the same.
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (getOperand(i) != I->getOperand(i))
      return false;

  // Check special state that is a part of some instructions.
  if (const LoadInst *LI = dyn_cast<LoadInst>(this))
    return LI->isVolatile() == cast<LoadInst>(I)->isVolatile() &&
           LI->getAlignment() == cast<LoadInst>(I)->getAlignment();
  if (const StoreInst *SI = dyn_cast<StoreInst>(this))
    return SI->isVolatile() == cast<StoreInst>(I)->isVolatile() &&
           SI->getAlignment() == cast<StoreInst>(I)->getAlignment();
  if (const CmpInst *CI = dyn_cast<CmpInst>(this))
    return CI->getPredicate() == cast<CmpInst>(I)->getPredicate();
  if (const CallInst *CI = dyn_cast<CallInst>(this))
    return CI->isTailCall() == cast<CallInst>(I)->isTailCall() &&
           CI->getCallingConv() == cast<CallInst>(I)->getCallingConv() &&
           CI->getAttributes().getRawPointer() ==
             cast<CallInst>(I)->getAttributes().getRawPointer();
  if (const InvokeInst *CI = dyn_cast<InvokeInst>(this))
    return CI->getCallingConv() == cast<InvokeInst>(I)->getCallingConv() &&
           CI->getAttributes().getRawPointer() ==
             cast<InvokeInst>(I)->getAttributes().getRawPointer();
  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(this)) {
    if (IVI->getNumIndices() != cast<InsertValueInst>(I)->getNumIndices())
      return false;
    for (unsigned i = 0, e = IVI->getNumIndices(); i != e; ++i)
      if (IVI->idx_begin()[i] != cast<InsertValueInst>(I)->idx_begin()[i])
        return false;
    return true;
  }
  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(this)) {
    if (EVI->getNumIndices() != cast<ExtractValueInst>(I)->getNumIndices())
      return false;
    for (unsigned i = 0, e = EVI->getNumIndices(); i != e; ++i)
      if (EVI->idx_begin()[i] != cast<ExtractValueInst>(I)->idx_begin()[i])
        return false;
    return true;
  }

  return true;
}

// isSameOperationAs
bool Instruction::isSameOperationAs(const Instruction *I) const {
  if (getOpcode() != I->getOpcode() || getType() != I->getType() ||
      getNumOperands() != I->getNumOperands())
    return false;

  // We have two instructions of identical opcode and #operands.  Check to see
  // if all operands are the same type
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (getOperand(i)->getType() != I->getOperand(i)->getType())
      return false;

  // Check special state that is a part of some instructions.
  if (const LoadInst *LI = dyn_cast<LoadInst>(this))
    return LI->isVolatile() == cast<LoadInst>(I)->isVolatile() &&
           LI->getAlignment() == cast<LoadInst>(I)->getAlignment();
  if (const StoreInst *SI = dyn_cast<StoreInst>(this))
    return SI->isVolatile() == cast<StoreInst>(I)->isVolatile() &&
           SI->getAlignment() == cast<StoreInst>(I)->getAlignment();
  if (const CmpInst *CI = dyn_cast<CmpInst>(this))
    return CI->getPredicate() == cast<CmpInst>(I)->getPredicate();
  if (const CallInst *CI = dyn_cast<CallInst>(this))
    return CI->isTailCall() == cast<CallInst>(I)->isTailCall() &&
           CI->getCallingConv() == cast<CallInst>(I)->getCallingConv() &&
           CI->getAttributes().getRawPointer() ==
             cast<CallInst>(I)->getAttributes().getRawPointer();
  if (const InvokeInst *CI = dyn_cast<InvokeInst>(this))
    return CI->getCallingConv() == cast<InvokeInst>(I)->getCallingConv() &&
           CI->getAttributes().getRawPointer() ==
             cast<InvokeInst>(I)->getAttributes().getRawPointer();
  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(this)) {
    if (IVI->getNumIndices() != cast<InsertValueInst>(I)->getNumIndices())
      return false;
    for (unsigned i = 0, e = IVI->getNumIndices(); i != e; ++i)
      if (IVI->idx_begin()[i] != cast<InsertValueInst>(I)->idx_begin()[i])
        return false;
    return true;
  }
  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(this)) {
    if (EVI->getNumIndices() != cast<ExtractValueInst>(I)->getNumIndices())
      return false;
    for (unsigned i = 0, e = EVI->getNumIndices(); i != e; ++i)
      if (EVI->idx_begin()[i] != cast<ExtractValueInst>(I)->idx_begin()[i])
        return false;
    return true;
  }

  return true;
}

/// isUsedOutsideOfBlock - Return true if there are any uses of I outside of the
/// specified block.  Note that PHI nodes are considered to evaluate their
/// operands in the corresponding predecessor block.
bool Instruction::isUsedOutsideOfBlock(const BasicBlock *BB) const {
  for (use_const_iterator UI = use_begin(), E = use_end(); UI != E; ++UI) {
    // PHI nodes uses values in the corresponding predecessor block.  For other
    // instructions, just check to see whether the parent of the use matches up.
    const PHINode *PN = dyn_cast<PHINode>(*UI);
    if (PN == 0) {
      if (cast<Instruction>(*UI)->getParent() != BB)
        return true;
      continue;
    }
    
    if (PN->getIncomingBlock(UI) != BB)
      return true;
  }
  return false;    
}

/// mayReadFromMemory - Return true if this instruction may read memory.
///
bool Instruction::mayReadFromMemory() const {
  switch (getOpcode()) {
  default: return false;
  case Instruction::Free:
  case Instruction::VAArg:
  case Instruction::Load:
    return true;
  case Instruction::Call:
    return !cast<CallInst>(this)->doesNotAccessMemory();
  case Instruction::Invoke:
    return !cast<InvokeInst>(this)->doesNotAccessMemory();
  case Instruction::Store:
    return cast<StoreInst>(this)->isVolatile();
  }
}

/// mayWriteToMemory - Return true if this instruction may modify memory.
///
bool Instruction::mayWriteToMemory() const {
  switch (getOpcode()) {
  default: return false;
  case Instruction::Free:
  case Instruction::Store:
  case Instruction::VAArg:
    return true;
  case Instruction::Call:
    return !cast<CallInst>(this)->onlyReadsMemory();
  case Instruction::Invoke:
    return !cast<InvokeInst>(this)->onlyReadsMemory();
  case Instruction::Load:
    return cast<LoadInst>(this)->isVolatile();
  }
}

/// isAssociative - Return true if the instruction is associative:
///
///   Associative operators satisfy:  x op (y op z) === (x op y) op z)
///
/// In LLVM, the Add, Mul, And, Or, and Xor operators are associative, when not
/// applied to floating point types.
///
bool Instruction::isAssociative(unsigned Opcode, const Type *Ty) {
  if (Opcode == And || Opcode == Or || Opcode == Xor)
    return true;

  // Add/Mul reassociate unless they are FP or FP vectors.
  if (Opcode == Add || Opcode == Mul)
    return !Ty->isFPOrFPVector();
  return 0;
}

/// isCommutative - Return true if the instruction is commutative:
///
///   Commutative operators satisfy: (x op y) === (y op x)
///
/// In LLVM, these are the associative operators, plus SetEQ and SetNE, when
/// applied to any type.
///
bool Instruction::isCommutative(unsigned op) {
  switch (op) {
  case Add:
  case Mul:
  case And:
  case Or:
  case Xor:
    return true;
  default:
    return false;
  }
}

/// isTrapping - Return true if the instruction may trap.
///
bool Instruction::isTrapping(unsigned op) {
  switch(op) {
  case UDiv:
  case SDiv:
  case FDiv:
  case URem:
  case SRem:
  case FRem:
  case Load:
  case Store:
  case Call:
  case Invoke:
  case VAArg:
    return true;
  default:
    return false;
  }
}
