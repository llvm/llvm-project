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

bool GEPOperator::accumulateConstantOffset(
    const DataLayout &DL, APInt &Offset,
    function_ref<bool(Value &, APInt &)> ExternalAnalysis) const {
   assert(Offset.getBitWidth() ==
              DL.getIndexSizeInBits(getPointerAddressSpace()) &&
          "The offset bit width does not match DL specification.");

  bool UsedExternalAnalysis = false;
  auto AccumulateOffset = [&](APInt Index, uint64_t Size) -> bool {
    Index = Index.sextOrTrunc(Offset.getBitWidth());
    APInt IndexedSize = APInt(Offset.getBitWidth(), Size);
    // For array or vector indices, scale the index by the size of the type.
    if (!UsedExternalAnalysis) {
      Offset += Index * IndexedSize;
    } else {
      // External Analysis can return a result higher/lower than the value
      // represents. We need to detect overflow/underflow.
      bool Overflow = false;
      APInt OffsetPlus = Index.smul_ov(IndexedSize, Overflow);
      if (Overflow)
        return false;
      Offset = Offset.sadd_ov(OffsetPlus, Overflow);
      if (Overflow)
        return false;
    }
    return true;
  };

  for (gep_type_iterator GTI = gep_type_begin(this), GTE = gep_type_end(this);
       GTI != GTE; ++GTI) {
    // Scalable vectors are multiplied by a runtime constant.
    bool ScalableType = false;
    if (isa<ScalableVectorType>(GTI.getIndexedType()))
      ScalableType = true;

    Value *V = GTI.getOperand();
    StructType *STy = GTI.getStructTypeOrNull();
    // Handle ConstantInt if possible.
    if (auto ConstOffset = dyn_cast<ConstantInt>(V)) {
      if (ConstOffset->isZero())
        continue;
      // if the type is scalable and the constant is not zero (vscale * n * 0 =
      // 0) bailout.
      if (ScalableType)
        return false;
      // Handle a struct index, which adds its field offset to the pointer.
      if (STy) {
        unsigned ElementIdx = ConstOffset->getZExtValue();
        const StructLayout *SL = DL.getStructLayout(STy);
        // Element offset is in bytes.
        if (!AccumulateOffset(
                APInt(Offset.getBitWidth(), SL->getElementOffset(ElementIdx)),
                1))
          return false;
        continue;
      }
      if (!AccumulateOffset(ConstOffset->getValue(),
                            DL.getTypeAllocSize(GTI.getIndexedType())))
        return false;
      continue;
    }

    // The operand is not constant, check if an external analysis was provided.
    // External analsis is not applicable to a struct type.
    if (!ExternalAnalysis || STy || ScalableType)
      return false;
    APInt AnalysisIndex;
    if (!ExternalAnalysis(*V, AnalysisIndex))
      return false;
    UsedExternalAnalysis = true;
    if (!AccumulateOffset(AnalysisIndex,
                          DL.getTypeAllocSize(GTI.getIndexedType())))
      return false;
  }
  return true;
}
}
