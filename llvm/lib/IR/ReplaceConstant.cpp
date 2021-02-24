//===- ReplaceConstant.cpp - Replace LLVM constant expression--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a utility function for replacing LLVM constant
// expressions by instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ReplaceConstant.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/NoFolder.h"

namespace llvm {
// Replace a constant expression by instructions with equivalent operations at
// a specified location.
Instruction *createReplacementInstr(ConstantExpr *CE, Instruction *Instr) {
  IRBuilder<NoFolder> Builder(Instr);
  unsigned OpCode = CE->getOpcode();
  switch (OpCode) {
  case Instruction::GetElementPtr: {
    SmallVector<Value *, 4> CEOpVec(CE->operands());
    ArrayRef<Value *> CEOps(CEOpVec);
    return dyn_cast<Instruction>(
        Builder.CreateInBoundsGEP(cast<GEPOperator>(CE)->getSourceElementType(),
                                  CEOps[0], CEOps.slice(1)));
  }
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    return dyn_cast<Instruction>(
        Builder.CreateBinOp((Instruction::BinaryOps)OpCode, CE->getOperand(0),
                            CE->getOperand(1), CE->getName()));
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
  case Instruction::AddrSpaceCast:
    return dyn_cast<Instruction>(
        Builder.CreateCast((Instruction::CastOps)OpCode, CE->getOperand(0),
                           CE->getType(), CE->getName()));
  default:
    llvm_unreachable("Unhandled constant expression!\n");
  }
}
} // namespace llvm
