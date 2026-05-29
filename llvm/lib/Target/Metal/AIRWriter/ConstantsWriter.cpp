//===- ConstantsWriter.cpp - AIR constants-block writer ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ValueEnumerator.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

namespace llvm {
namespace metal {

void emitConstantsBlock(BitstreamWriter &W, ValueEnumerator &E,
                        ArrayRef<const Constant *> Constants,
                        unsigned CodeSize) {
  if (Constants.empty())
    return;
  W.EnterSubblock(bitc::CONSTANTS_BLOCK_ID, CodeSize);

  Type *LastType = nullptr;
  for (auto *C : Constants) {
    SmallVector<uint64_t, 8> V;
    if (C->getType() != LastType) {
      LastType = C->getType();
      V.push_back(E.typeIdx(LastType));
      W.EmitRecord(bitc::CST_CODE_SETTYPE, V);
      V.clear();
    }
    if (isa<UndefValue>(C)) {
      W.EmitRecord(bitc::CST_CODE_UNDEF, V);
    } else if (auto *CI = dyn_cast<ConstantInt>(C)) {
      // Always use INTEGER for ints (even zero) - Metal doesn't
      // accept NULL for integer types in some contexts.
      // Use ZExtValue for i1 (Metal convention), SExtValue for wider types.
      int64_t Val = CI->getType()->isIntegerTy(1)
                        ? (int64_t)CI->getZExtValue()
                        : CI->getSExtValue();
      V.push_back(Val >= 0 ? uint64_t(Val) << 1
                           : (uint64_t(-Val) << 1) | 1);
      W.EmitRecord(bitc::CST_CODE_INTEGER, V);
    } else if (auto *CF = dyn_cast<ConstantFP>(C)) {
      V.push_back(CF->getValueAPF().bitcastToAPInt().getZExtValue());
      W.EmitRecord(bitc::CST_CODE_FLOAT, V);
    } else if (auto *CDV = dyn_cast<ConstantDataVector>(C)) {
      for (unsigned I = 0; I < CDV->getNumElements(); I++) {
        if (CDV->getElementType()->isIntegerTy())
          V.push_back(CDV->getElementAsInteger(I));
        else
          V.push_back(
              CDV->getElementAsAPFloat(I).bitcastToAPInt().getZExtValue());
      }
      W.EmitRecord(bitc::CST_CODE_DATA, V);
    } else if (auto *CDA = dyn_cast<ConstantDataArray>(C)) {
      // Metal v1: emit as AGGREGATE referencing individual element constants
      for (unsigned I = 0; I < CDA->getNumElements(); I++)
        V.push_back(E.moduleConstIdx(CDA->getElementAsConstant(I)));
      W.EmitRecord(bitc::CST_CODE_AGGREGATE, V);
    } else if (isa<ConstantArray>(C) || isa<ConstantStruct>(C) ||
               isa<ConstantVector>(C)) {
      // Aggregate: list element value IDs (elements already in table)
      for (unsigned I = 0; I < C->getNumOperands(); I++)
        V.push_back(E.moduleConstIdx(cast<Constant>(C->getOperand(I))));
      W.EmitRecord(bitc::CST_CODE_AGGREGATE, V);
    } else if (C->isNullValue()) {
      // NULL for aggregate zero, pointer null, vector zero
      W.EmitRecord(bitc::CST_CODE_NULL, V);
    } else {
      W.EmitRecord(bitc::CST_CODE_NULL, V);
    }
  }
  W.ExitBlock();
}

} // namespace metal
} // namespace llvm
