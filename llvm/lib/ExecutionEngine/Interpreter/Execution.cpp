//===-- Execution.cpp - Implement code to simulate the program ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains the actual instruction interpreter.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "interpreter"
#include "Interpreter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cmath>
using namespace llvm;

STATISTIC(NumDynamicInsts, "Number of dynamic instructions executed");

static cl::opt<bool> PrintVolatile("interpreter-print-volatile", cl::Hidden,
          cl::desc("make the interpreter print every volatile load and store"));

//===----------------------------------------------------------------------===//
//                     Various Helper Functions
//===----------------------------------------------------------------------===//

static void SetValue(Value *V, GenericValue Val, ExecutionContext &SF) {
  SF.Values[V] = Val;
}

//===----------------------------------------------------------------------===//
//                    Binary Instruction Implementations
//===----------------------------------------------------------------------===//

#define IMPLEMENT_BINARY_OPERATOR(OP, TY) \
   case Type::TY##TyID: \
     Dest.TY##Val = Src1.TY##Val OP Src2.TY##Val; \
     break

static void executeFAddInst(GenericValue &Dest, GenericValue Src1,
                            GenericValue Src2, Type *Ty) {
  switch (Ty->getTypeID()) {
    IMPLEMENT_BINARY_OPERATOR(+, Float);
    IMPLEMENT_BINARY_OPERATOR(+, Double);
  default:
    dbgs() << "Unhandled type for FAdd instruction: " << *Ty << "\n";
    llvm_unreachable(0);
  }
}

static void executeFSubInst(GenericValue &Dest, GenericValue Src1,
                            GenericValue Src2, Type *Ty) {
  switch (Ty->getTypeID()) {
    IMPLEMENT_BINARY_OPERATOR(-, Float);
    IMPLEMENT_BINARY_OPERATOR(-, Double);
  default:
    dbgs() << "Unhandled type for FSub instruction: " << *Ty << "\n";
    llvm_unreachable(0);
  }
}

static void executeFMulInst(GenericValue &Dest, GenericValue Src1,
                            GenericValue Src2, Type *Ty) {
  switch (Ty->getTypeID()) {
    IMPLEMENT_BINARY_OPERATOR(*, Float);
    IMPLEMENT_BINARY_OPERATOR(*, Double);
  default:
    dbgs() << "Unhandled type for FMul instruction: " << *Ty << "\n";
    llvm_unreachable(0);
  }
}

static void executeFDivInst(GenericValue &Dest, GenericValue Src1, 
                            GenericValue Src2, Type *Ty) {
  switch (Ty->getTypeID()) {
    IMPLEMENT_BINARY_OPERATOR(/, Float);
    IMPLEMENT_BINARY_OPERATOR(/, Double);
  default:
    dbgs() << "Unhandled type for FDiv instruction: " << *Ty << "\n";
    llvm_unreachable(0);
  }
}

static void executeFRemInst(GenericValue &Dest, GenericValue Src1, 
                            GenericValue Src2, Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::FloatTyID:
    Dest.FloatVal = fmod(Src1.FloatVal, Src2.FloatVal);
    break;
  case Type::DoubleTyID:
    Dest.DoubleVal = fmod(Src1.DoubleVal, Src2.DoubleVal);
    break;
  default:
    dbgs() << "Unhandled type for Rem instruction: " << *Ty << "\n";
    llvm_unreachable(0);
  }
}

#define IMPLEMENT_INTEGER_ICMP(OP, TY) \
   case Type::IntegerTyID:  \
      Dest.IntVal = APInt(1,Src1.IntVal.OP(Src2.IntVal)); \
      break;

// Handle pointers specially because they must be compared with only as much
// width as the host has.  We _do not_ want to be comparing 64 bit values when
// running on a 32-bit target, otherwise the upper 32 bits might mess up
// comparisons if they contain garbage.
#define IMPLEMENT_POINTER_ICMP(OP) \
   case Type::PointerTyID: \
      Dest.IntVal = APInt(1,(void*)(intptr_t)Src1.PointerVal OP \
                            (void*)(intptr_t)Src2.PointerVal); \
      break;

static GenericValue executeICMP_EQ(GenericValue Src1, GenericValue Src2,
                                   Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_INTEGER_ICMP(eq,Ty);
    IMPLEMENT_POINTER_ICMP(==);
  default:
    dbgs() << "Unhandled type for ICMP_EQ predicate: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeICMP_NE(GenericValue Src1, GenericValue Src2,
                                   Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_INTEGER_ICMP(ne,Ty);
    IMPLEMENT_POINTER_ICMP(!=);
  default:
    dbgs() << "Unhandled type for ICMP_NE predicate: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeICMP_ULT(GenericValue Src1, GenericValue Src2,
                                    Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_INTEGER_ICMP(ult,Ty);
    IMPLEMENT_POINTER_ICMP(<);
  default:
    dbgs() << "Unhandled type for ICMP_ULT predicate: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeICMP_SLT(GenericValue Src1, GenericValue Src2,
                                    Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_INTEGER_ICMP(slt,Ty);
    IMPLEMENT_POINTER_ICMP(<);
  default:
    dbgs() << "Unhandled type for ICMP_SLT predicate: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeICMP_UGT(GenericValue Src1, GenericValue Src2,
                                    Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_INTEGER_ICMP(ugt,Ty);
    IMPLEMENT_POINTER_ICMP(>);
  default:
    dbgs() << "Unhandled type for ICMP_UGT predicate: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeICMP_SGT(GenericValue Src1, GenericValue Src2,
                                    Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_INTEGER_ICMP(sgt,Ty);
    IMPLEMENT_POINTER_ICMP(>);
  default:
    dbgs() << "Unhandled type for ICMP_SGT predicate: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeICMP_ULE(GenericValue Src1, GenericValue Src2,
                                    Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_INTEGER_ICMP(ule,Ty);
    IMPLEMENT_POINTER_ICMP(<=);
  default:
    dbgs() << "Unhandled type for ICMP_ULE predicate: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeICMP_SLE(GenericValue Src1, GenericValue Src2,
                                    Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_INTEGER_ICMP(sle,Ty);
    IMPLEMENT_POINTER_ICMP(<=);
  default:
    dbgs() << "Unhandled type for ICMP_SLE predicate: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeICMP_UGE(GenericValue Src1, GenericValue Src2,
                                    Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_INTEGER_ICMP(uge,Ty);
    IMPLEMENT_POINTER_ICMP(>=);
  default:
    dbgs() << "Unhandled type for ICMP_UGE predicate: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeICMP_SGE(GenericValue Src1, GenericValue Src2,
                                    Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_INTEGER_ICMP(sge,Ty);
    IMPLEMENT_POINTER_ICMP(>=);
  default:
    dbgs() << "Unhandled type for ICMP_SGE predicate: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

void Interpreter::visitICmpInst(ICmpInst &I) {
  ExecutionContext &SF = ECStack.back();
  Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue R;   // Result
  
  switch (I.getPredicate()) {
  case ICmpInst::ICMP_EQ:  R = executeICMP_EQ(Src1,  Src2, Ty); break;
  case ICmpInst::ICMP_NE:  R = executeICMP_NE(Src1,  Src2, Ty); break;
  case ICmpInst::ICMP_ULT: R = executeICMP_ULT(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_SLT: R = executeICMP_SLT(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_UGT: R = executeICMP_UGT(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_SGT: R = executeICMP_SGT(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_ULE: R = executeICMP_ULE(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_SLE: R = executeICMP_SLE(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_UGE: R = executeICMP_UGE(Src1, Src2, Ty); break;
  case ICmpInst::ICMP_SGE: R = executeICMP_SGE(Src1, Src2, Ty); break;
  default:
    dbgs() << "Don't know how to handle this ICmp predicate!\n-->" << I;
    llvm_unreachable(0);
  }
 
  SetValue(&I, R, SF);
}

#define IMPLEMENT_FCMP(OP, TY) \
   case Type::TY##TyID: \
     Dest.IntVal = APInt(1,Src1.TY##Val OP Src2.TY##Val); \
     break

static GenericValue executeFCMP_OEQ(GenericValue Src1, GenericValue Src2,
                                   Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_FCMP(==, Float);
    IMPLEMENT_FCMP(==, Double);
  default:
    dbgs() << "Unhandled type for FCmp EQ instruction: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeFCMP_ONE(GenericValue Src1, GenericValue Src2,
                                   Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_FCMP(!=, Float);
    IMPLEMENT_FCMP(!=, Double);

  default:
    dbgs() << "Unhandled type for FCmp NE instruction: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeFCMP_OLE(GenericValue Src1, GenericValue Src2,
                                   Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_FCMP(<=, Float);
    IMPLEMENT_FCMP(<=, Double);
  default:
    dbgs() << "Unhandled type for FCmp LE instruction: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeFCMP_OGE(GenericValue Src1, GenericValue Src2,
                                   Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_FCMP(>=, Float);
    IMPLEMENT_FCMP(>=, Double);
  default:
    dbgs() << "Unhandled type for FCmp GE instruction: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeFCMP_OLT(GenericValue Src1, GenericValue Src2,
                                   Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_FCMP(<, Float);
    IMPLEMENT_FCMP(<, Double);
  default:
    dbgs() << "Unhandled type for FCmp LT instruction: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

static GenericValue executeFCMP_OGT(GenericValue Src1, GenericValue Src2,
                                     Type *Ty) {
  GenericValue Dest;
  switch (Ty->getTypeID()) {
    IMPLEMENT_FCMP(>, Float);
    IMPLEMENT_FCMP(>, Double);
  default:
    dbgs() << "Unhandled type for FCmp GT instruction: " << *Ty << "\n";
    llvm_unreachable(0);
  }
  return Dest;
}

#define IMPLEMENT_UNORDERED(TY, X,Y)                                     \
  if (TY->isFloatTy()) {                                                 \
    if (X.FloatVal != X.FloatVal || Y.FloatVal != Y.FloatVal) {          \
      Dest.IntVal = APInt(1,true);                                       \
      return Dest;                                                       \
    }                                                                    \
  } else if (X.DoubleVal != X.DoubleVal || Y.DoubleVal != Y.DoubleVal) { \
    Dest.IntVal = APInt(1,true);                                         \
    return Dest;                                                         \
  }


static GenericValue executeFCMP_UEQ(GenericValue Src1, GenericValue Src2,
                                   Type *Ty) {
  GenericValue Dest;
  IMPLEMENT_UNORDERED(Ty, Src1, Src2)
  return executeFCMP_OEQ(Src1, Src2, Ty);
}

static GenericValue executeFCMP_UNE(GenericValue Src1, GenericValue Src2,
                                   Type *Ty) {
  GenericValue Dest;
  IMPLEMENT_UNORDERED(Ty, Src1, Src2)
  return executeFCMP_ONE(Src1, Src2, Ty);
}

static GenericValue executeFCMP_ULE(GenericValue Src1, GenericValue Src2,
                                   Type *Ty) {
  GenericValue Dest;
  IMPLEMENT_UNORDERED(Ty, Src1, Src2)
  return executeFCMP_OLE(Src1, Src2, Ty);
}

static GenericValue executeFCMP_UGE(GenericValue Src1, GenericValue Src2,
                                   Type *Ty) {
  GenericValue Dest;
  IMPLEMENT_UNORDERED(Ty, Src1, Src2)
  return executeFCMP_OGE(Src1, Src2, Ty);
}

static GenericValue executeFCMP_ULT(GenericValue Src1, GenericValue Src2,
                                   Type *Ty) {
  GenericValue Dest;
  IMPLEMENT_UNORDERED(Ty, Src1, Src2)
  return executeFCMP_OLT(Src1, Src2, Ty);
}

static GenericValue executeFCMP_UGT(GenericValue Src1, GenericValue Src2,
                                     Type *Ty) {
  GenericValue Dest;
  IMPLEMENT_UNORDERED(Ty, Src1, Src2)
  return executeFCMP_OGT(Src1, Src2, Ty);
}

static GenericValue executeFCMP_ORD(GenericValue Src1, GenericValue Src2,
                                     Type *Ty) {
  GenericValue Dest;
  if (Ty->isFloatTy())
    Dest.IntVal = APInt(1,(Src1.FloatVal == Src1.FloatVal && 
                           Src2.FloatVal == Src2.FloatVal));
  else
    Dest.IntVal = APInt(1,(Src1.DoubleVal == Src1.DoubleVal && 
                           Src2.DoubleVal == Src2.DoubleVal));
  return Dest;
}

static GenericValue executeFCMP_UNO(GenericValue Src1, GenericValue Src2,
                                     Type *Ty) {
  GenericValue Dest;
  if (Ty->isFloatTy())
    Dest.IntVal = APInt(1,(Src1.FloatVal != Src1.FloatVal || 
                           Src2.FloatVal != Src2.FloatVal));
  else
    Dest.IntVal = APInt(1,(Src1.DoubleVal != Src1.DoubleVal || 
                           Src2.DoubleVal != Src2.DoubleVal));
  return Dest;
}

void Interpreter::visitFCmpInst(FCmpInst &I) {
  ExecutionContext &SF = ECStack.back();
  Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue R;   // Result
  
  switch (I.getPredicate()) {
  case FCmpInst::FCMP_FALSE: R.IntVal = APInt(1,false); break;
  case FCmpInst::FCMP_TRUE:  R.IntVal = APInt(1,true); break;
  case FCmpInst::FCMP_ORD:   R = executeFCMP_ORD(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_UNO:   R = executeFCMP_UNO(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_UEQ:   R = executeFCMP_UEQ(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_OEQ:   R = executeFCMP_OEQ(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_UNE:   R = executeFCMP_UNE(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_ONE:   R = executeFCMP_ONE(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_ULT:   R = executeFCMP_ULT(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_OLT:   R = executeFCMP_OLT(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_UGT:   R = executeFCMP_UGT(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_OGT:   R = executeFCMP_OGT(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_ULE:   R = executeFCMP_ULE(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_OLE:   R = executeFCMP_OLE(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_UGE:   R = executeFCMP_UGE(Src1, Src2, Ty); break;
  case FCmpInst::FCMP_OGE:   R = executeFCMP_OGE(Src1, Src2, Ty); break;
  default:
    dbgs() << "Don't know how to handle this FCmp predicate!\n-->" << I;
    llvm_unreachable(0);
  }
 
  SetValue(&I, R, SF);
}

static GenericValue executeCmpInst(unsigned predicate, GenericValue Src1, 
                                   GenericValue Src2, Type *Ty) {
  GenericValue Result;
  switch (predicate) {
  case ICmpInst::ICMP_EQ:    return executeICMP_EQ(Src1, Src2, Ty);
  case ICmpInst::ICMP_NE:    return executeICMP_NE(Src1, Src2, Ty);
  case ICmpInst::ICMP_UGT:   return executeICMP_UGT(Src1, Src2, Ty);
  case ICmpInst::ICMP_SGT:   return executeICMP_SGT(Src1, Src2, Ty);
  case ICmpInst::ICMP_ULT:   return executeICMP_ULT(Src1, Src2, Ty);
  case ICmpInst::ICMP_SLT:   return executeICMP_SLT(Src1, Src2, Ty);
  case ICmpInst::ICMP_UGE:   return executeICMP_UGE(Src1, Src2, Ty);
  case ICmpInst::ICMP_SGE:   return executeICMP_SGE(Src1, Src2, Ty);
  case ICmpInst::ICMP_ULE:   return executeICMP_ULE(Src1, Src2, Ty);
  case ICmpInst::ICMP_SLE:   return executeICMP_SLE(Src1, Src2, Ty);
  case FCmpInst::FCMP_ORD:   return executeFCMP_ORD(Src1, Src2, Ty);
  case FCmpInst::FCMP_UNO:   return executeFCMP_UNO(Src1, Src2, Ty);
  case FCmpInst::FCMP_OEQ:   return executeFCMP_OEQ(Src1, Src2, Ty);
  case FCmpInst::FCMP_UEQ:   return executeFCMP_UEQ(Src1, Src2, Ty);
  case FCmpInst::FCMP_ONE:   return executeFCMP_ONE(Src1, Src2, Ty);
  case FCmpInst::FCMP_UNE:   return executeFCMP_UNE(Src1, Src2, Ty);
  case FCmpInst::FCMP_OLT:   return executeFCMP_OLT(Src1, Src2, Ty);
  case FCmpInst::FCMP_ULT:   return executeFCMP_ULT(Src1, Src2, Ty);
  case FCmpInst::FCMP_OGT:   return executeFCMP_OGT(Src1, Src2, Ty);
  case FCmpInst::FCMP_UGT:   return executeFCMP_UGT(Src1, Src2, Ty);
  case FCmpInst::FCMP_OLE:   return executeFCMP_OLE(Src1, Src2, Ty);
  case FCmpInst::FCMP_ULE:   return executeFCMP_ULE(Src1, Src2, Ty);
  case FCmpInst::FCMP_OGE:   return executeFCMP_OGE(Src1, Src2, Ty);
  case FCmpInst::FCMP_UGE:   return executeFCMP_UGE(Src1, Src2, Ty);
  case FCmpInst::FCMP_FALSE: { 
    GenericValue Result;
    Result.IntVal = APInt(1, false);
    return Result;
  }
  case FCmpInst::FCMP_TRUE: {
    GenericValue Result;
    Result.IntVal = APInt(1, true);
    return Result;
  }
  default:
    dbgs() << "Unhandled Cmp predicate\n";
    llvm_unreachable(0);
  }
}

void Interpreter::visitBinaryOperator(BinaryOperator &I) {
  ExecutionContext &SF = ECStack.back();
  Type *Ty    = I.getOperand(0)->getType();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue R;   // Result

  switch (I.getOpcode()) {
  case Instruction::Add:   R.IntVal = Src1.IntVal + Src2.IntVal; break;
  case Instruction::Sub:   R.IntVal = Src1.IntVal - Src2.IntVal; break;
  case Instruction::Mul:   R.IntVal = Src1.IntVal * Src2.IntVal; break;
  case Instruction::FAdd:  executeFAddInst(R, Src1, Src2, Ty); break;
  case Instruction::FSub:  executeFSubInst(R, Src1, Src2, Ty); break;
  case Instruction::FMul:  executeFMulInst(R, Src1, Src2, Ty); break;
  case Instruction::FDiv:  executeFDivInst(R, Src1, Src2, Ty); break;
  case Instruction::FRem:  executeFRemInst(R, Src1, Src2, Ty); break;
  case Instruction::UDiv:  R.IntVal = Src1.IntVal.udiv(Src2.IntVal); break;
  case Instruction::SDiv:  R.IntVal = Src1.IntVal.sdiv(Src2.IntVal); break;
  case Instruction::URem:  R.IntVal = Src1.IntVal.urem(Src2.IntVal); break;
  case Instruction::SRem:  R.IntVal = Src1.IntVal.srem(Src2.IntVal); break;
  case Instruction::And:   R.IntVal = Src1.IntVal & Src2.IntVal; break;
  case Instruction::Or:    R.IntVal = Src1.IntVal | Src2.IntVal; break;
  case Instruction::Xor:   R.IntVal = Src1.IntVal ^ Src2.IntVal; break;
  default:
    dbgs() << "Don't know how to handle this binary operator!\n-->" << I;
    llvm_unreachable(0);
  }

  SetValue(&I, R, SF);
}

static GenericValue executeSelectInst(GenericValue Src1, GenericValue Src2,
                                      GenericValue Src3) {
  return Src1.IntVal == 0 ? Src3 : Src2;
}

void Interpreter::visitSelectInst(SelectInst &I) {
  ExecutionContext &SF = ECStack.back();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue Src3 = getOperandValue(I.getOperand(2), SF);
  GenericValue R = executeSelectInst(Src1, Src2, Src3);
  SetValue(&I, R, SF);
}


//===----------------------------------------------------------------------===//
//                     Terminator Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::exitCalled(GenericValue GV) {
  // runAtExitHandlers() assumes there are no stack frames, but
  // if exit() was called, then it had a stack frame. Blow away
  // the stack before interpreting atexit handlers.
  ECStack.clear();
  runAtExitHandlers();
  exit(GV.IntVal.zextOrTrunc(32).getZExtValue());
}

/// Pop the last stack frame off of ECStack and then copy the result
/// back into the result variable if we are not returning void. The
/// result variable may be the ExitValue, or the Value of the calling
/// CallInst if there was a previous stack frame. This method may
/// invalidate any ECStack iterators you have. This method also takes
/// care of switching to the normal destination BB, if we are returning
/// from an invoke.
///
void Interpreter::popStackAndReturnValueToCaller(Type *RetTy,
                                                 GenericValue Result) {
  // Pop the current stack frame.
  ECStack.pop_back();

  if (ECStack.empty()) {  // Finished main.  Put result into exit code...
    if (RetTy && !RetTy->isVoidTy()) {          // Nonvoid return type?
      ExitValue = Result;   // Capture the exit value of the program
    } else {
      memset(&ExitValue.Untyped, 0, sizeof(ExitValue.Untyped));
    }
  } else {
    // If we have a previous stack frame, and we have a previous call,
    // fill in the return value...
    ExecutionContext &CallingSF = ECStack.back();
    if (Instruction *I = CallingSF.Caller.getInstruction()) {
      // Save result...
      if (!CallingSF.Caller.getType()->isVoidTy())
        SetValue(I, Result, CallingSF);
      if (InvokeInst *II = dyn_cast<InvokeInst> (I))
        SwitchToNewBasicBlock (II->getNormalDest (), CallingSF);
      CallingSF.Caller = CallSite();          // We returned from the call...
    }
  }
}

void Interpreter::visitReturnInst(ReturnInst &I) {
  ExecutionContext &SF = ECStack.back();
  Type *RetTy = Type::getVoidTy(I.getContext());
  GenericValue Result;

  // Save away the return value... (if we are not 'ret void')
  if (I.getNumOperands()) {
    RetTy  = I.getReturnValue()->getType();
    Result = getOperandValue(I.getReturnValue(), SF);
  }

  popStackAndReturnValueToCaller(RetTy, Result);
}

void Interpreter::visitUnreachableInst(UnreachableInst &I) {
  report_fatal_error("Program executed an 'unreachable' instruction!");
}

void Interpreter::visitBranchInst(BranchInst &I) {
  ExecutionContext &SF = ECStack.back();
  BasicBlock *Dest;

  Dest = I.getSuccessor(0);          // Uncond branches have a fixed dest...
  if (!I.isUnconditional()) {
    Value *Cond = I.getCondition();
    if (getOperandValue(Cond, SF).IntVal == 0) // If false cond...
      Dest = I.getSuccessor(1);
  }
  SwitchToNewBasicBlock(Dest, SF);
}

void Interpreter::visitSwitchInst(SwitchInst &I) {
  ExecutionContext &SF = ECStack.back();
  Value* Cond = I.getCondition();
  Type *ElTy = Cond->getType();
  GenericValue CondVal = getOperandValue(Cond, SF);

  // Check to see if any of the cases match...
  BasicBlock *Dest = 0;
  for (SwitchInst::CaseIt i = I.case_begin(), e = I.case_end(); i != e; ++i) {
    IntegersSubset& Case = i.getCaseValueEx();
    if (Case.isSingleNumber()) {
      // FIXME: Currently work with ConstantInt based numbers.
      const ConstantInt *CI = Case.getSingleNumber(0).toConstantInt();
      GenericValue Val = getOperandValue(const_cast<ConstantInt*>(CI), SF);
      if (executeICMP_EQ(Val, CondVal, ElTy).IntVal != 0) {
        Dest = cast<BasicBlock>(i.getCaseSuccessor());
        break;        
      }
    }
    if (Case.isSingleNumbersOnly()) {
      for (unsigned n = 0, en = Case.getNumItems(); n != en; ++n) {
        // FIXME: Currently work with ConstantInt based numbers.
        const ConstantInt *CI = Case.getSingleNumber(n).toConstantInt();
        GenericValue Val = getOperandValue(const_cast<ConstantInt*>(CI), SF);
        if (executeICMP_EQ(Val, CondVal, ElTy).IntVal != 0) {
          Dest = cast<BasicBlock>(i.getCaseSuccessor());
          break;        
        }
      }      
    } else
      for (unsigned n = 0, en = Case.getNumItems(); n != en; ++n) {
        IntegersSubset::Range r = Case.getItem(n);
        // FIXME: Currently work with ConstantInt based numbers.
        const ConstantInt *LowCI = r.getLow().toConstantInt();
        const ConstantInt *HighCI = r.getHigh().toConstantInt();
        GenericValue Low = getOperandValue(const_cast<ConstantInt*>(LowCI), SF);
        GenericValue High = getOperandValue(const_cast<ConstantInt*>(HighCI), SF);
        if (executeICMP_ULE(Low, CondVal, ElTy).IntVal != 0 &&
            executeICMP_ULE(CondVal, High, ElTy).IntVal != 0) {
          Dest = cast<BasicBlock>(i.getCaseSuccessor());
          break;        
        }
      }
  }
  if (!Dest) Dest = I.getDefaultDest();   // No cases matched: use default
  SwitchToNewBasicBlock(Dest, SF);
}

void Interpreter::visitIndirectBrInst(IndirectBrInst &I) {
  ExecutionContext &SF = ECStack.back();
  void *Dest = GVTOP(getOperandValue(I.getAddress(), SF));
  SwitchToNewBasicBlock((BasicBlock*)Dest, SF);
}


// SwitchToNewBasicBlock - This method is used to jump to a new basic block.
// This function handles the actual updating of block and instruction iterators
// as well as execution of all of the PHI nodes in the destination block.
//
// This method does this because all of the PHI nodes must be executed
// atomically, reading their inputs before any of the results are updated.  Not
// doing this can cause problems if the PHI nodes depend on other PHI nodes for
// their inputs.  If the input PHI node is updated before it is read, incorrect
// results can happen.  Thus we use a two phase approach.
//
void Interpreter::SwitchToNewBasicBlock(BasicBlock *Dest, ExecutionContext &SF){
  BasicBlock *PrevBB = SF.CurBB;      // Remember where we came from...
  SF.CurBB   = Dest;                  // Update CurBB to branch destination
  SF.CurInst = SF.CurBB->begin();     // Update new instruction ptr...

  if (!isa<PHINode>(SF.CurInst)) return;  // Nothing fancy to do

  // Loop over all of the PHI nodes in the current block, reading their inputs.
  std::vector<GenericValue> ResultValues;

  for (; PHINode *PN = dyn_cast<PHINode>(SF.CurInst); ++SF.CurInst) {
    // Search for the value corresponding to this previous bb...
    int i = PN->getBasicBlockIndex(PrevBB);
    assert(i != -1 && "PHINode doesn't contain entry for predecessor??");
    Value *IncomingValue = PN->getIncomingValue(i);

    // Save the incoming value for this PHI node...
    ResultValues.push_back(getOperandValue(IncomingValue, SF));
  }

  // Now loop over all of the PHI nodes setting their values...
  SF.CurInst = SF.CurBB->begin();
  for (unsigned i = 0; isa<PHINode>(SF.CurInst); ++SF.CurInst, ++i) {
    PHINode *PN = cast<PHINode>(SF.CurInst);
    SetValue(PN, ResultValues[i], SF);
  }
}

//===----------------------------------------------------------------------===//
//                     Memory Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::visitAllocaInst(AllocaInst &I) {
  ExecutionContext &SF = ECStack.back();

  Type *Ty = I.getType()->getElementType();  // Type to be allocated

  // Get the number of elements being allocated by the array...
  unsigned NumElements = 
    getOperandValue(I.getOperand(0), SF).IntVal.getZExtValue();

  unsigned TypeSize = (size_t)TD.getTypeAllocSize(Ty);

  // Avoid malloc-ing zero bytes, use max()...
  unsigned MemToAlloc = std::max(1U, NumElements * TypeSize);

  // Allocate enough memory to hold the type...
  void *Memory = malloc(MemToAlloc);

  DEBUG(dbgs() << "Allocated Type: " << *Ty << " (" << TypeSize << " bytes) x " 
               << NumElements << " (Total: " << MemToAlloc << ") at "
               << uintptr_t(Memory) << '\n');

  GenericValue Result = PTOGV(Memory);
  assert(Result.PointerVal != 0 && "Null pointer returned by malloc!");
  SetValue(&I, Result, SF);

  if (I.getOpcode() == Instruction::Alloca)
    ECStack.back().Allocas.add(Memory);
}

// getElementOffset - The workhorse for getelementptr.
//
GenericValue Interpreter::executeGEPOperation(Value *Ptr, gep_type_iterator I,
                                              gep_type_iterator E,
                                              ExecutionContext &SF) {
  assert(Ptr->getType()->isPointerTy() &&
         "Cannot getElementOffset of a nonpointer type!");

  uint64_t Total = 0;

  for (; I != E; ++I) {
    if (StructType *STy = dyn_cast<StructType>(*I)) {
      const StructLayout *SLO = TD.getStructLayout(STy);

      const ConstantInt *CPU = cast<ConstantInt>(I.getOperand());
      unsigned Index = unsigned(CPU->getZExtValue());

      Total += SLO->getElementOffset(Index);
    } else {
      SequentialType *ST = cast<SequentialType>(*I);
      // Get the index number for the array... which must be long type...
      GenericValue IdxGV = getOperandValue(I.getOperand(), SF);

      int64_t Idx;
      unsigned BitWidth = 
        cast<IntegerType>(I.getOperand()->getType())->getBitWidth();
      if (BitWidth == 32)
        Idx = (int64_t)(int32_t)IdxGV.IntVal.getZExtValue();
      else {
        assert(BitWidth == 64 && "Invalid index type for getelementptr");
        Idx = (int64_t)IdxGV.IntVal.getZExtValue();
      }
      Total += TD.getTypeAllocSize(ST->getElementType())*Idx;
    }
  }

  GenericValue Result;
  Result.PointerVal = ((char*)getOperandValue(Ptr, SF).PointerVal) + Total;
  DEBUG(dbgs() << "GEP Index " << Total << " bytes.\n");
  return Result;
}

void Interpreter::visitGetElementPtrInst(GetElementPtrInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeGEPOperation(I.getPointerOperand(),
                                   gep_type_begin(I), gep_type_end(I), SF), SF);
}

void Interpreter::visitLoadInst(LoadInst &I) {
  ExecutionContext &SF = ECStack.back();
  GenericValue SRC = getOperandValue(I.getPointerOperand(), SF);
  GenericValue *Ptr = (GenericValue*)GVTOP(SRC);
  GenericValue Result;
  LoadValueFromMemory(Result, Ptr, I.getType());
  SetValue(&I, Result, SF);
  if (I.isVolatile() && PrintVolatile)
    dbgs() << "Volatile load " << I;
}

void Interpreter::visitStoreInst(StoreInst &I) {
  ExecutionContext &SF = ECStack.back();
  GenericValue Val = getOperandValue(I.getOperand(0), SF);
  GenericValue SRC = getOperandValue(I.getPointerOperand(), SF);
  StoreValueToMemory(Val, (GenericValue *)GVTOP(SRC),
                     I.getOperand(0)->getType());
  if (I.isVolatile() && PrintVolatile)
    dbgs() << "Volatile store: " << I;
}

//===----------------------------------------------------------------------===//
//                 Miscellaneous Instruction Implementations
//===----------------------------------------------------------------------===//

void Interpreter::visitCallSite(CallSite CS) {
  ExecutionContext &SF = ECStack.back();

  // Check to see if this is an intrinsic function call...
  Function *F = CS.getCalledFunction();
  if (F && F->isDeclaration())
    switch (F->getIntrinsicID()) {
    case Intrinsic::not_intrinsic:
      break;
    case Intrinsic::vastart: { // va_start
      GenericValue ArgIndex;
      ArgIndex.UIntPairVal.first = ECStack.size() - 1;
      ArgIndex.UIntPairVal.second = 0;
      SetValue(CS.getInstruction(), ArgIndex, SF);
      return;
    }
    case Intrinsic::vaend:    // va_end is a noop for the interpreter
      return;
    case Intrinsic::vacopy:   // va_copy: dest = src
      SetValue(CS.getInstruction(), getOperandValue(*CS.arg_begin(), SF), SF);
      return;
    default:
      // If it is an unknown intrinsic function, use the intrinsic lowering
      // class to transform it into hopefully tasty LLVM code.
      //
      BasicBlock::iterator me(CS.getInstruction());
      BasicBlock *Parent = CS.getInstruction()->getParent();
      bool atBegin(Parent->begin() == me);
      if (!atBegin)
        --me;
      IL->LowerIntrinsicCall(cast<CallInst>(CS.getInstruction()));

      // Restore the CurInst pointer to the first instruction newly inserted, if
      // any.
      if (atBegin) {
        SF.CurInst = Parent->begin();
      } else {
        SF.CurInst = me;
        ++SF.CurInst;
      }
      return;
    }


  SF.Caller = CS;
  std::vector<GenericValue> ArgVals;
  const unsigned NumArgs = SF.Caller.arg_size();
  ArgVals.reserve(NumArgs);
  uint16_t pNum = 1;
  for (CallSite::arg_iterator i = SF.Caller.arg_begin(),
         e = SF.Caller.arg_end(); i != e; ++i, ++pNum) {
    Value *V = *i;
    ArgVals.push_back(getOperandValue(V, SF));
  }

  // To handle indirect calls, we must get the pointer value from the argument
  // and treat it as a function pointer.
  GenericValue SRC = getOperandValue(SF.Caller.getCalledValue(), SF);
  callFunction((Function*)GVTOP(SRC), ArgVals);
}

void Interpreter::visitShl(BinaryOperator &I) {
  ExecutionContext &SF = ECStack.back();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue Dest;
  if (Src2.IntVal.getZExtValue() < Src1.IntVal.getBitWidth())
    Dest.IntVal = Src1.IntVal.shl(Src2.IntVal.getZExtValue());
  else
    Dest.IntVal = Src1.IntVal;
  
  SetValue(&I, Dest, SF);
}

void Interpreter::visitLShr(BinaryOperator &I) {
  ExecutionContext &SF = ECStack.back();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue Dest;
  if (Src2.IntVal.getZExtValue() < Src1.IntVal.getBitWidth())
    Dest.IntVal = Src1.IntVal.lshr(Src2.IntVal.getZExtValue());
  else
    Dest.IntVal = Src1.IntVal;
  
  SetValue(&I, Dest, SF);
}

void Interpreter::visitAShr(BinaryOperator &I) {
  ExecutionContext &SF = ECStack.back();
  GenericValue Src1 = getOperandValue(I.getOperand(0), SF);
  GenericValue Src2 = getOperandValue(I.getOperand(1), SF);
  GenericValue Dest;
  if (Src2.IntVal.getZExtValue() < Src1.IntVal.getBitWidth())
    Dest.IntVal = Src1.IntVal.ashr(Src2.IntVal.getZExtValue());
  else
    Dest.IntVal = Src1.IntVal;
  
  SetValue(&I, Dest, SF);
}

GenericValue Interpreter::executeTruncInst(Value *SrcVal, Type *DstTy,
                                           ExecutionContext &SF) {
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);
  IntegerType *DITy = cast<IntegerType>(DstTy);
  unsigned DBitWidth = DITy->getBitWidth();
  Dest.IntVal = Src.IntVal.trunc(DBitWidth);
  return Dest;
}

GenericValue Interpreter::executeSExtInst(Value *SrcVal, Type *DstTy,
                                          ExecutionContext &SF) {
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);
  IntegerType *DITy = cast<IntegerType>(DstTy);
  unsigned DBitWidth = DITy->getBitWidth();
  Dest.IntVal = Src.IntVal.sext(DBitWidth);
  return Dest;
}

GenericValue Interpreter::executeZExtInst(Value *SrcVal, Type *DstTy,
                                          ExecutionContext &SF) {
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);
  IntegerType *DITy = cast<IntegerType>(DstTy);
  unsigned DBitWidth = DITy->getBitWidth();
  Dest.IntVal = Src.IntVal.zext(DBitWidth);
  return Dest;
}

GenericValue Interpreter::executeFPTruncInst(Value *SrcVal, Type *DstTy,
                                             ExecutionContext &SF) {
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);
  assert(SrcVal->getType()->isDoubleTy() && DstTy->isFloatTy() &&
         "Invalid FPTrunc instruction");
  Dest.FloatVal = (float) Src.DoubleVal;
  return Dest;
}

GenericValue Interpreter::executeFPExtInst(Value *SrcVal, Type *DstTy,
                                           ExecutionContext &SF) {
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);
  assert(SrcVal->getType()->isFloatTy() && DstTy->isDoubleTy() &&
         "Invalid FPTrunc instruction");
  Dest.DoubleVal = (double) Src.FloatVal;
  return Dest;
}

GenericValue Interpreter::executeFPToUIInst(Value *SrcVal, Type *DstTy,
                                            ExecutionContext &SF) {
  Type *SrcTy = SrcVal->getType();
  uint32_t DBitWidth = cast<IntegerType>(DstTy)->getBitWidth();
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);
  assert(SrcTy->isFloatingPointTy() && "Invalid FPToUI instruction");

  if (SrcTy->getTypeID() == Type::FloatTyID)
    Dest.IntVal = APIntOps::RoundFloatToAPInt(Src.FloatVal, DBitWidth);
  else
    Dest.IntVal = APIntOps::RoundDoubleToAPInt(Src.DoubleVal, DBitWidth);
  return Dest;
}

GenericValue Interpreter::executeFPToSIInst(Value *SrcVal, Type *DstTy,
                                            ExecutionContext &SF) {
  Type *SrcTy = SrcVal->getType();
  uint32_t DBitWidth = cast<IntegerType>(DstTy)->getBitWidth();
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);
  assert(SrcTy->isFloatingPointTy() && "Invalid FPToSI instruction");

  if (SrcTy->getTypeID() == Type::FloatTyID)
    Dest.IntVal = APIntOps::RoundFloatToAPInt(Src.FloatVal, DBitWidth);
  else
    Dest.IntVal = APIntOps::RoundDoubleToAPInt(Src.DoubleVal, DBitWidth);
  return Dest;
}

GenericValue Interpreter::executeUIToFPInst(Value *SrcVal, Type *DstTy,
                                            ExecutionContext &SF) {
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);
  assert(DstTy->isFloatingPointTy() && "Invalid UIToFP instruction");

  if (DstTy->getTypeID() == Type::FloatTyID)
    Dest.FloatVal = APIntOps::RoundAPIntToFloat(Src.IntVal);
  else
    Dest.DoubleVal = APIntOps::RoundAPIntToDouble(Src.IntVal);
  return Dest;
}

GenericValue Interpreter::executeSIToFPInst(Value *SrcVal, Type *DstTy,
                                            ExecutionContext &SF) {
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);
  assert(DstTy->isFloatingPointTy() && "Invalid SIToFP instruction");

  if (DstTy->getTypeID() == Type::FloatTyID)
    Dest.FloatVal = APIntOps::RoundSignedAPIntToFloat(Src.IntVal);
  else
    Dest.DoubleVal = APIntOps::RoundSignedAPIntToDouble(Src.IntVal);
  return Dest;

}

GenericValue Interpreter::executePtrToIntInst(Value *SrcVal, Type *DstTy,
                                              ExecutionContext &SF) {
  uint32_t DBitWidth = cast<IntegerType>(DstTy)->getBitWidth();
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);
  assert(SrcVal->getType()->isPointerTy() && "Invalid PtrToInt instruction");

  Dest.IntVal = APInt(DBitWidth, (intptr_t) Src.PointerVal);
  return Dest;
}

GenericValue Interpreter::executeIntToPtrInst(Value *SrcVal, Type *DstTy,
                                              ExecutionContext &SF) {
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);
  assert(DstTy->isPointerTy() && "Invalid PtrToInt instruction");

  uint32_t PtrSize = TD.getPointerSizeInBits();
  if (PtrSize != Src.IntVal.getBitWidth())
    Src.IntVal = Src.IntVal.zextOrTrunc(PtrSize);

  Dest.PointerVal = PointerTy(intptr_t(Src.IntVal.getZExtValue()));
  return Dest;
}

GenericValue Interpreter::executeBitCastInst(Value *SrcVal, Type *DstTy,
                                             ExecutionContext &SF) {
  
  Type *SrcTy = SrcVal->getType();
  GenericValue Dest, Src = getOperandValue(SrcVal, SF);
  if (DstTy->isPointerTy()) {
    assert(SrcTy->isPointerTy() && "Invalid BitCast");
    Dest.PointerVal = Src.PointerVal;
  } else if (DstTy->isIntegerTy()) {
    if (SrcTy->isFloatTy()) {
      Dest.IntVal = APInt::floatToBits(Src.FloatVal);
    } else if (SrcTy->isDoubleTy()) {
      Dest.IntVal = APInt::doubleToBits(Src.DoubleVal);
    } else if (SrcTy->isIntegerTy()) {
      Dest.IntVal = Src.IntVal;
    } else 
      llvm_unreachable("Invalid BitCast");
  } else if (DstTy->isFloatTy()) {
    if (SrcTy->isIntegerTy())
      Dest.FloatVal = Src.IntVal.bitsToFloat();
    else
      Dest.FloatVal = Src.FloatVal;
  } else if (DstTy->isDoubleTy()) {
    if (SrcTy->isIntegerTy())
      Dest.DoubleVal = Src.IntVal.bitsToDouble();
    else
      Dest.DoubleVal = Src.DoubleVal;
  } else
    llvm_unreachable("Invalid Bitcast");

  return Dest;
}

void Interpreter::visitTruncInst(TruncInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeTruncInst(I.getOperand(0), I.getType(), SF), SF);
}

void Interpreter::visitSExtInst(SExtInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeSExtInst(I.getOperand(0), I.getType(), SF), SF);
}

void Interpreter::visitZExtInst(ZExtInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeZExtInst(I.getOperand(0), I.getType(), SF), SF);
}

void Interpreter::visitFPTruncInst(FPTruncInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeFPTruncInst(I.getOperand(0), I.getType(), SF), SF);
}

void Interpreter::visitFPExtInst(FPExtInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeFPExtInst(I.getOperand(0), I.getType(), SF), SF);
}

void Interpreter::visitUIToFPInst(UIToFPInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeUIToFPInst(I.getOperand(0), I.getType(), SF), SF);
}

void Interpreter::visitSIToFPInst(SIToFPInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeSIToFPInst(I.getOperand(0), I.getType(), SF), SF);
}

void Interpreter::visitFPToUIInst(FPToUIInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeFPToUIInst(I.getOperand(0), I.getType(), SF), SF);
}

void Interpreter::visitFPToSIInst(FPToSIInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeFPToSIInst(I.getOperand(0), I.getType(), SF), SF);
}

void Interpreter::visitPtrToIntInst(PtrToIntInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executePtrToIntInst(I.getOperand(0), I.getType(), SF), SF);
}

void Interpreter::visitIntToPtrInst(IntToPtrInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeIntToPtrInst(I.getOperand(0), I.getType(), SF), SF);
}

void Interpreter::visitBitCastInst(BitCastInst &I) {
  ExecutionContext &SF = ECStack.back();
  SetValue(&I, executeBitCastInst(I.getOperand(0), I.getType(), SF), SF);
}

#define IMPLEMENT_VAARG(TY) \
   case Type::TY##TyID: Dest.TY##Val = Src.TY##Val; break

void Interpreter::visitVAArgInst(VAArgInst &I) {
  ExecutionContext &SF = ECStack.back();

  // Get the incoming valist parameter.  LLI treats the valist as a
  // (ec-stack-depth var-arg-index) pair.
  GenericValue VAList = getOperandValue(I.getOperand(0), SF);
  GenericValue Dest;
  GenericValue Src = ECStack[VAList.UIntPairVal.first]
                      .VarArgs[VAList.UIntPairVal.second];
  Type *Ty = I.getType();
  switch (Ty->getTypeID()) {
    case Type::IntegerTyID: Dest.IntVal = Src.IntVal;
    IMPLEMENT_VAARG(Pointer);
    IMPLEMENT_VAARG(Float);
    IMPLEMENT_VAARG(Double);
  default:
    dbgs() << "Unhandled dest type for vaarg instruction: " << *Ty << "\n";
    llvm_unreachable(0);
  }

  // Set the Value of this Instruction.
  SetValue(&I, Dest, SF);

  // Move the pointer to the next vararg.
  ++VAList.UIntPairVal.second;
}

GenericValue Interpreter::getConstantExprValue (ConstantExpr *CE,
                                                ExecutionContext &SF) {
  switch (CE->getOpcode()) {
  case Instruction::Trunc:   
      return executeTruncInst(CE->getOperand(0), CE->getType(), SF);
  case Instruction::ZExt:
      return executeZExtInst(CE->getOperand(0), CE->getType(), SF);
  case Instruction::SExt:
      return executeSExtInst(CE->getOperand(0), CE->getType(), SF);
  case Instruction::FPTrunc:
      return executeFPTruncInst(CE->getOperand(0), CE->getType(), SF);
  case Instruction::FPExt:
      return executeFPExtInst(CE->getOperand(0), CE->getType(), SF);
  case Instruction::UIToFP:
      return executeUIToFPInst(CE->getOperand(0), CE->getType(), SF);
  case Instruction::SIToFP:
      return executeSIToFPInst(CE->getOperand(0), CE->getType(), SF);
  case Instruction::FPToUI:
      return executeFPToUIInst(CE->getOperand(0), CE->getType(), SF);
  case Instruction::FPToSI:
      return executeFPToSIInst(CE->getOperand(0), CE->getType(), SF);
  case Instruction::PtrToInt:
      return executePtrToIntInst(CE->getOperand(0), CE->getType(), SF);
  case Instruction::IntToPtr:
      return executeIntToPtrInst(CE->getOperand(0), CE->getType(), SF);
  case Instruction::BitCast:
      return executeBitCastInst(CE->getOperand(0), CE->getType(), SF);
  case Instruction::GetElementPtr:
    return executeGEPOperation(CE->getOperand(0), gep_type_begin(CE),
                               gep_type_end(CE), SF);
  case Instruction::FCmp:
  case Instruction::ICmp:
    return executeCmpInst(CE->getPredicate(),
                          getOperandValue(CE->getOperand(0), SF),
                          getOperandValue(CE->getOperand(1), SF),
                          CE->getOperand(0)->getType());
  case Instruction::Select:
    return executeSelectInst(getOperandValue(CE->getOperand(0), SF),
                             getOperandValue(CE->getOperand(1), SF),
                             getOperandValue(CE->getOperand(2), SF));
  default :
    break;
  }

  // The cases below here require a GenericValue parameter for the result
  // so we initialize one, compute it and then return it.
  GenericValue Op0 = getOperandValue(CE->getOperand(0), SF);
  GenericValue Op1 = getOperandValue(CE->getOperand(1), SF);
  GenericValue Dest;
  Type * Ty = CE->getOperand(0)->getType();
  switch (CE->getOpcode()) {
  case Instruction::Add:  Dest.IntVal = Op0.IntVal + Op1.IntVal; break;
  case Instruction::Sub:  Dest.IntVal = Op0.IntVal - Op1.IntVal; break;
  case Instruction::Mul:  Dest.IntVal = Op0.IntVal * Op1.IntVal; break;
  case Instruction::FAdd: executeFAddInst(Dest, Op0, Op1, Ty); break;
  case Instruction::FSub: executeFSubInst(Dest, Op0, Op1, Ty); break;
  case Instruction::FMul: executeFMulInst(Dest, Op0, Op1, Ty); break;
  case Instruction::FDiv: executeFDivInst(Dest, Op0, Op1, Ty); break;
  case Instruction::FRem: executeFRemInst(Dest, Op0, Op1, Ty); break;
  case Instruction::SDiv: Dest.IntVal = Op0.IntVal.sdiv(Op1.IntVal); break;
  case Instruction::UDiv: Dest.IntVal = Op0.IntVal.udiv(Op1.IntVal); break;
  case Instruction::URem: Dest.IntVal = Op0.IntVal.urem(Op1.IntVal); break;
  case Instruction::SRem: Dest.IntVal = Op0.IntVal.srem(Op1.IntVal); break;
  case Instruction::And:  Dest.IntVal = Op0.IntVal & Op1.IntVal; break;
  case Instruction::Or:   Dest.IntVal = Op0.IntVal | Op1.IntVal; break;
  case Instruction::Xor:  Dest.IntVal = Op0.IntVal ^ Op1.IntVal; break;
  case Instruction::Shl:  
    Dest.IntVal = Op0.IntVal.shl(Op1.IntVal.getZExtValue());
    break;
  case Instruction::LShr: 
    Dest.IntVal = Op0.IntVal.lshr(Op1.IntVal.getZExtValue());
    break;
  case Instruction::AShr: 
    Dest.IntVal = Op0.IntVal.ashr(Op1.IntVal.getZExtValue());
    break;
  default:
    dbgs() << "Unhandled ConstantExpr: " << *CE << "\n";
    llvm_unreachable("Unhandled ConstantExpr");
  }
  return Dest;
}

GenericValue Interpreter::getOperandValue(Value *V, ExecutionContext &SF) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    return getConstantExprValue(CE, SF);
  } else if (Constant *CPV = dyn_cast<Constant>(V)) {
    return getConstantValue(CPV);
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    return PTOGV(getPointerToGlobal(GV));
  } else {
    return SF.Values[V];
  }
}

//===----------------------------------------------------------------------===//
//                        Dispatch and Execution Code
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// callFunction - Execute the specified function...
//
void Interpreter::callFunction(Function *F,
                               const std::vector<GenericValue> &ArgVals) {
  assert((ECStack.empty() || ECStack.back().Caller.getInstruction() == 0 ||
          ECStack.back().Caller.arg_size() == ArgVals.size()) &&
         "Incorrect number of arguments passed into function call!");
  // Make a new stack frame... and fill it in.
  ECStack.push_back(ExecutionContext());
  ExecutionContext &StackFrame = ECStack.back();
  StackFrame.CurFunction = F;

  // Special handling for external functions.
  if (F->isDeclaration()) {
    GenericValue Result = callExternalFunction (F, ArgVals);
    // Simulate a 'ret' instruction of the appropriate type.
    popStackAndReturnValueToCaller (F->getReturnType (), Result);
    return;
  }

  // Get pointers to first LLVM BB & Instruction in function.
  StackFrame.CurBB     = F->begin();
  StackFrame.CurInst   = StackFrame.CurBB->begin();

  // Run through the function arguments and initialize their values...
  assert((ArgVals.size() == F->arg_size() ||
         (ArgVals.size() > F->arg_size() && F->getFunctionType()->isVarArg()))&&
         "Invalid number of values passed to function invocation!");

  // Handle non-varargs arguments...
  unsigned i = 0;
  for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end(); 
       AI != E; ++AI, ++i)
    SetValue(AI, ArgVals[i], StackFrame);

  // Handle varargs arguments...
  StackFrame.VarArgs.assign(ArgVals.begin()+i, ArgVals.end());
}


void Interpreter::run() {
  while (!ECStack.empty()) {
    // Interpret a single instruction & increment the "PC".
    ExecutionContext &SF = ECStack.back();  // Current stack frame
    Instruction &I = *SF.CurInst++;         // Increment before execute

    // Track the number of dynamic instructions executed.
    ++NumDynamicInsts;

    DEBUG(dbgs() << "About to interpret: " << I);
    visit(I);   // Dispatch to one of the visit* methods...
#if 0
    // This is not safe, as visiting the instruction could lower it and free I.
DEBUG(
    if (!isa<CallInst>(I) && !isa<InvokeInst>(I) && 
        I.getType() != Type::VoidTy) {
      dbgs() << "  --> ";
      const GenericValue &Val = SF.Values[&I];
      switch (I.getType()->getTypeID()) {
      default: llvm_unreachable("Invalid GenericValue Type");
      case Type::VoidTyID:    dbgs() << "void"; break;
      case Type::FloatTyID:   dbgs() << "float " << Val.FloatVal; break;
      case Type::DoubleTyID:  dbgs() << "double " << Val.DoubleVal; break;
      case Type::PointerTyID: dbgs() << "void* " << intptr_t(Val.PointerVal);
        break;
      case Type::IntegerTyID: 
        dbgs() << "i" << Val.IntVal.getBitWidth() << " "
               << Val.IntVal.toStringUnsigned(10)
               << " (0x" << Val.IntVal.toStringUnsigned(16) << ")\n";
        break;
      }
    });
#endif
  }
}
