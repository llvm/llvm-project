//===-- Operator.cpp - Implement the LLVM operators -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the non-inline methods for the LLVM Operator classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Operator.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"

#include "ConstantsContext.h"

namespace llvm {
Type *GEPOperator::getSourceElementType() const {
  if (auto *I = dyn_cast<GetElementPtrInst>(this))
    return I->getSourceElementType();
  return cast<GetElementPtrConstantExpr>(this)->getSourceElementType();
}

Type *GEPOperator::getResultElementType() const {
  if (auto *I = dyn_cast<GetElementPtrInst>(this))
    return I->getResultElementType();
  return cast<GetElementPtrConstantExpr>(this)->getResultElementType();
}

bool GEPOperator::accumulateConstantOffset(const DataLayout &DL,
                                           APInt &Offset) const {
  assert(Offset.getBitWidth() ==
             DL.getIndexSizeInBits(getPointerAddressSpace()) &&
         "The offset bit width does not match DL specification.");

  for (gep_type_iterator GTI = gep_type_begin(this), GTE = gep_type_end(this);
       GTI != GTE; ++GTI) {
    ConstantInt *OpC = dyn_cast<ConstantInt>(GTI.getOperand());
    if (!OpC)
      return false;
    if (OpC->isZero())
      continue;

    // Handle a struct index, which adds its field offset to the pointer.
    if (StructType *STy = GTI.getStructTypeOrNull()) {
      unsigned ElementIdx = OpC->getZExtValue();
      const StructLayout *SL = DL.getStructLayout(STy);
      Offset += APInt(Offset.getBitWidth(), SL->getElementOffset(ElementIdx));
      continue;
    }

    // For array or vector indices, scale the index by the size of the type.
    APInt Index = OpC->getValue().sextOrTrunc(Offset.getBitWidth());
    Offset += Index * APInt(Offset.getBitWidth(),
                            DL.getTypeAllocSize(GTI.getIndexedType()));
  }
  return true;
}
}
