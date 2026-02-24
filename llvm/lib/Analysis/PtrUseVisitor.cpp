//===- PtrUseVisitor.cpp - InstVisitors over a pointers uses --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implementation of the pointer use visitors.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

void detail::PtrUseVisitorBase::enqueueUsers(Value &I) {
  for (Use &U : I.uses()) {
    if (VisitedUses.insert(&U).second) {
      UseToVisit NewU = {
        UseToVisit::UseAndIsOffsetKnownPair(&U, IsOffsetKnown),
        Offset
      };
      Worklist.push_back(std::move(NewU));
    }
  }
}

bool detail::PtrUseVisitorBase::adjustOffsetForGEP(GetElementPtrInst &GEPI) {
  if (!IsOffsetKnown)
    return false;

  APInt TmpOffset(DL.getIndexTypeSizeInBits(GEPI.getType()), 0);
  if (GEPI.accumulateConstantOffset(DL, TmpOffset)) {
    Offset += TmpOffset.sextOrTrunc(Offset.getBitWidth());
    return true;
  }

  return false;
}

bool detail::PtrUseVisitorBase::adjustOffsetForSGEP(StructuredGEPInst &SGEP) {
  if (!IsOffsetKnown)
    return false;

  Type *CurrentType = SGEP.getBaseType();
  unsigned int OffsetBitWidth = DL.getIndexTypeSizeInBits(SGEP.getType());
  APInt TmpOffset(OffsetBitWidth, 0);

  for (unsigned I = 0; I < SGEP.getNumIndices(); I++) {
    Value *V = SGEP.getIndexOperand(I);
    ConstantInt *CI = dyn_cast<ConstantInt>(V);
    if (!CI) {
      IsOffsetKnown = false;
      return false;
    }

    if (ArrayType *AT = dyn_cast<ArrayType>(CurrentType)) {
      uint32_t EltTypeSize = DL.getTypeSizeInBits(AT->getElementType()) / 8;
      TmpOffset += CI->getZExtValue() * EltTypeSize;
      CurrentType = AT->getElementType();
    } else if (StructType *ST = dyn_cast<StructType>(CurrentType)) {
      const auto &STL = DL.getStructLayout(ST);
      TmpOffset += STL->getElementOffset(CI->getZExtValue());
      CurrentType = ST->getElementType(CI->getZExtValue());
    } else {
      llvm_unreachable("unimplemented");
    }
  }

  Offset += TmpOffset;
  return true;
}
