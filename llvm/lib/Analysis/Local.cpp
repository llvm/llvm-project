//===- Local.cpp - Functions to perform local transformations -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform various local transformations to the
// program.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Utils/Local.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

Value *llvm::emitGEPOffset(IRBuilderBase *Builder, const DataLayout &DL,
                           User *GEP, bool NoAssumptions) {
  GEPOperator *GEPOp = cast<GEPOperator>(GEP);
  Type *IntIdxTy = DL.getIndexType(GEP->getType());
  Value *Result = nullptr;

  // If the GEP is inbounds, we know that none of the addressing operations will
  // overflow in a signed sense.
  bool isInBounds = GEPOp->isInBounds() && !NoAssumptions;
  auto AddOffset = [&](Value *Offset) {
    if (Result)
      Result = Builder->CreateAdd(Result, Offset, GEP->getName() + ".offs",
                                  false /*NUW*/, isInBounds /*NSW*/);
    else
      Result = Offset;
  };

  // Build a mask for high order bits.
  unsigned IntPtrWidth = IntIdxTy->getScalarType()->getIntegerBitWidth();
  uint64_t PtrSizeMask =
      std::numeric_limits<uint64_t>::max() >> (64 - IntPtrWidth);

  gep_type_iterator GTI = gep_type_begin(GEP);
  for (User::op_iterator i = GEP->op_begin() + 1, e = GEP->op_end(); i != e;
       ++i, ++GTI) {
    Value *Op = *i;
    uint64_t Size = DL.getTypeAllocSize(GTI.getIndexedType()) & PtrSizeMask;
    if (Constant *OpC = dyn_cast<Constant>(Op)) {
      if (OpC->isZeroValue())
        continue;

      // Handle a struct index, which adds its field offset to the pointer.
      if (StructType *STy = GTI.getStructTypeOrNull()) {
        uint64_t OpValue = OpC->getUniqueInteger().getZExtValue();
        Size = DL.getStructLayout(STy)->getElementOffset(OpValue);
        if (!Size)
          continue;

        AddOffset(ConstantInt::get(IntIdxTy, Size));
        continue;
      }
    }

    // Splat the index if needed.
    if (IntIdxTy->isVectorTy() && !Op->getType()->isVectorTy())
      Op = Builder->CreateVectorSplat(
          cast<FixedVectorType>(IntIdxTy)->getNumElements(), Op);

    // Convert to correct type.
    if (Op->getType() != IntIdxTy)
      Op = Builder->CreateIntCast(Op, IntIdxTy, true, Op->getName() + ".c");
    if (Size != 1) {
      // We'll let instcombine(mul) convert this to a shl if possible.
      Op = Builder->CreateMul(Op, ConstantInt::get(IntIdxTy, Size),
                              GEP->getName() + ".idx", false /*NUW*/,
                              isInBounds /*NSW*/);
    }
    AddOffset(Op);
  }
  return Result ? Result : Constant::getNullValue(IntIdxTy);
}
