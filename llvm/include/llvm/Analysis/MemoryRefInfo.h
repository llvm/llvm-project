//===--------- Definition of the MemoryRefInfo class -*- C++ -*------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MemoryRefInfo class that is used when getting
// the information of a memory reference instruction.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ANALYSIS_MEMORYREFINFO_H
#define LLVM_ANALYSIS_MEMORYREFINFO_H

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/TypeSize.h"

namespace llvm {
class MemoryRefInfo {
public:
  Use *PtrUse = nullptr;
  bool IsWrite;
  Type *OpType;
  TypeSize TypeStoreSize = TypeSize::getFixed(0);
  MaybeAlign Alignment;
  // The mask Value, if we're looking at a masked load/store.
  Value *MaybeMask;
  // The EVL Value, if we're looking at a vp intrinsic.
  Value *MaybeEVL;
  // The Stride Value, if we're looking at a strided load/store.
  Value *MaybeStride;

  MemoryRefInfo() = default;
  MemoryRefInfo(Instruction *I, unsigned OperandNo, bool IsWrite,
                class Type *OpType, MaybeAlign Alignment,
                Value *MaybeMask = nullptr, Value *MaybeEVL = nullptr,
                Value *MaybeStride = nullptr)
      : IsWrite(IsWrite), OpType(OpType), Alignment(Alignment),
        MaybeMask(MaybeMask), MaybeEVL(MaybeEVL), MaybeStride(MaybeStride) {
    const DataLayout &DL = I->getDataLayout();
    TypeStoreSize = DL.getTypeStoreSizeInBits(OpType);
    PtrUse = &I->getOperandUse(OperandNo);
  }

  Instruction *getInsn() { return cast<Instruction>(PtrUse->getUser()); }
  Value *getPtr() { return PtrUse->get(); }
  operator bool() { return PtrUse != nullptr; }
};

} // namespace llvm
#endif
