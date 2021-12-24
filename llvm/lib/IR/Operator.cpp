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
bool Operator::hasPoisonGeneratingFlags() const {
  switch (getOpcode()) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::Shl: {
    auto *OBO = cast<OverflowingBinaryOperator>(this);
    return OBO->hasNoUnsignedWrap() || OBO->hasNoSignedWrap();
  }
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::AShr:
  case Instruction::LShr:
    return cast<PossiblyExactOperator>(this)->isExact();
  case Instruction::GetElementPtr: {
    auto *GEP = cast<GEPOperator>(this);
    // Note: inrange exists on constexpr only
    return GEP->isInBounds() || GEP->getInRangeIndex() != None;
  }
  default:
    if (const auto *FP = dyn_cast<FPMathOperator>(this))
      return FP->hasNoNaNs() || FP->hasNoInfs();
    return false;
  }
}

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

Align GEPOperator::getMaxPreservedAlignment(const DataLayout &DL) const {
  /// compute the worse possible offset for every level of the GEP et accumulate
  /// the minimum alignment into Result.

  Align Result = Align(llvm::Value::MaximumAlignment);
  for (gep_type_iterator GTI = gep_type_begin(this), GTE = gep_type_end(this);
       GTI != GTE; ++GTI) {
    int64_t Offset = 1;
    ConstantInt *OpC = dyn_cast<ConstantInt>(GTI.getOperand());

    if (StructType *STy = GTI.getStructTypeOrNull()) {
      const StructLayout *SL = DL.getStructLayout(STy);
      Offset = SL->getElementOffset(OpC->getZExtValue());
    } else {
      assert(GTI.isSequential() && "should be sequencial");
      /// If the index isn't know we take 1 because it is the index that will
      /// give the worse alignment of the offset.
      int64_t ElemCount = 1;
      if (OpC)
        ElemCount = OpC->getZExtValue();
      Offset = DL.getTypeAllocSize(GTI.getIndexedType()) * ElemCount;
    }
    Result = Align(MinAlign(Offset, Result.value()));
  }
  return Result;
}

bool GEPOperator::accumulateConstantOffset(
    const DataLayout &DL, APInt &Offset,
    function_ref<bool(Value &, APInt &)> ExternalAnalysis) const {
  assert(Offset.getBitWidth() ==
             DL.getIndexSizeInBits(getPointerAddressSpace()) &&
         "The offset bit width does not match DL specification.");
  SmallVector<const Value *> Index(llvm::drop_begin(operand_values()));
  return GEPOperator::accumulateConstantOffset(getSourceElementType(), Index,
                                               DL, Offset, ExternalAnalysis);
}

bool GEPOperator::accumulateConstantOffset(
    Type *SourceType, ArrayRef<const Value *> Index, const DataLayout &DL,
    APInt &Offset, function_ref<bool(Value &, APInt &)> ExternalAnalysis) {
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
  auto begin = generic_gep_type_iterator<decltype(Index.begin())>::begin(
      SourceType, Index.begin());
  auto end = generic_gep_type_iterator<decltype(Index.end())>::end(Index.end());
  for (auto GTI = begin, GTE = end; GTI != GTE; ++GTI) {
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

bool GEPOperator::collectOffset(
    const DataLayout &DL, unsigned BitWidth,
    MapVector<Value *, APInt> &VariableOffsets,
    APInt &ConstantOffset) const {
  assert(BitWidth == DL.getIndexSizeInBits(getPointerAddressSpace()) &&
         "The offset bit width does not match DL specification.");

  auto CollectConstantOffset = [&](APInt Index, uint64_t Size) {
    Index = Index.sextOrTrunc(BitWidth);
    APInt IndexedSize = APInt(BitWidth, Size);
    ConstantOffset += Index * IndexedSize;
  };

  for (gep_type_iterator GTI = gep_type_begin(this), GTE = gep_type_end(this);
       GTI != GTE; ++GTI) {
    // Scalable vectors are multiplied by a runtime constant.
    bool ScalableType = isa<ScalableVectorType>(GTI.getIndexedType());

    Value *V = GTI.getOperand();
    StructType *STy = GTI.getStructTypeOrNull();
    // Handle ConstantInt if possible.
    if (auto ConstOffset = dyn_cast<ConstantInt>(V)) {
      if (ConstOffset->isZero())
        continue;
      // If the type is scalable and the constant is not zero (vscale * n * 0 =
      // 0) bailout.
      // TODO: If the runtime value is accessible at any point before DWARF
      // emission, then we could potentially keep a forward reference to it
      // in the debug value to be filled in later.
      if (ScalableType)
        return false;
      // Handle a struct index, which adds its field offset to the pointer.
      if (STy) {
        unsigned ElementIdx = ConstOffset->getZExtValue();
        const StructLayout *SL = DL.getStructLayout(STy);
        // Element offset is in bytes.
        CollectConstantOffset(APInt(BitWidth, SL->getElementOffset(ElementIdx)),
                              1);
        continue;
      }
      CollectConstantOffset(ConstOffset->getValue(),
                            DL.getTypeAllocSize(GTI.getIndexedType()));
      continue;
    }

    if (STy || ScalableType)
      return false;
    APInt IndexedSize =
        APInt(BitWidth, DL.getTypeAllocSize(GTI.getIndexedType()));
    // Insert an initial offset of 0 for V iff none exists already, then
    // increment the offset by IndexedSize.
    if (!IndexedSize.isZero()) {
      VariableOffsets.insert({V, APInt(BitWidth, 0)});
      VariableOffsets[V] += IndexedSize;
    }
  }
  return true;
}

void FastMathFlags::print(raw_ostream &O) const {
  if (all())
    O << " fast";
  else {
    if (allowReassoc())
      O << " reassoc";
    if (noNaNs())
      O << " nnan";
    if (noInfs())
      O << " ninf";
    if (noSignedZeros())
      O << " nsz";
    if (allowReciprocal())
      O << " arcp";
    if (allowContract())
      O << " contract";
    if (approxFunc())
      O << " afn";
  }
}
} // namespace llvm
