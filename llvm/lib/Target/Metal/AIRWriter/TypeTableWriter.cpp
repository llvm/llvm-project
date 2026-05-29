//===- TypeTableWriter.cpp - AIR TYPE_BLOCK writer --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ValueEnumerator.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/DerivedTypes.h"

using namespace llvm;

namespace llvm {
namespace metal {

void emitTypeBlock(BitstreamWriter &W, ValueEnumerator &E) {
  W.EnterSubblock(bitc::TYPE_BLOCK_ID_NEW, 4);

  SmallVector<uint64_t, 8> V;
  V.push_back(E.types.size());
  W.EmitRecord(bitc::TYPE_CODE_NUMENTRY, V);

  for (auto &Entry : E.types) {
    Type *T = Entry.type;
    V.clear();

    if (T->isVoidTy()) {
      W.EmitRecord(bitc::TYPE_CODE_VOID, V);
    } else if (T->isFloatTy()) {
      W.EmitRecord(bitc::TYPE_CODE_FLOAT, V);
    } else if (T->isDoubleTy()) {
      W.EmitRecord(bitc::TYPE_CODE_DOUBLE, V);
    } else if (T->isHalfTy()) {
      W.EmitRecord(bitc::TYPE_CODE_HALF, V);
    } else if (T->isBFloatTy()) {
      W.EmitRecord(bitc::TYPE_CODE_BFLOAT, V);
    } else if (T->isIntegerTy()) {
      V.push_back(cast<IntegerType>(T)->getBitWidth());
      W.EmitRecord(bitc::TYPE_CODE_INTEGER, V);
    } else if (T->isMetadataTy()) {
      W.EmitRecord(bitc::TYPE_CODE_METADATA, V);
    } else if (T->isLabelTy()) {
      W.EmitRecord(bitc::TYPE_CODE_LABEL, V);
    } else if (isa<PointerType>(T)) {
      // Entry.pointee is the typed pointer's pointee
      V.push_back(E.typeIdx(Entry.pointee));
      V.push_back(cast<PointerType>(T)->getAddressSpace());
      W.EmitRecord(bitc::TYPE_CODE_POINTER, V);
    } else if (auto *VT = dyn_cast<FixedVectorType>(T)) {
      V.push_back(VT->getNumElements());
      V.push_back(E.typeIdx(VT->getElementType()));
      W.EmitRecord(bitc::TYPE_CODE_VECTOR, V);
    } else if (auto *AT = dyn_cast<ArrayType>(T)) {
      V.push_back(AT->getNumElements());
      V.push_back(E.typeIdx(AT->getElementType()));
      W.EmitRecord(bitc::TYPE_CODE_ARRAY, V);
    } else if (auto *FT = dyn_cast<FunctionType>(T)) {
      V.push_back(FT->isVarArg() ? 1 : 0);
      // Use stored return type index for pointer returns with custom pointee
      auto RIt = E.funcTypeReturnIndex.find(FT);
      if (RIt != E.funcTypeReturnIndex.end())
        V.push_back(RIt->second);
      else
        V.push_back(E.typeIdx(FT->getReturnType()));
      // Use stored per-param indices for correct multi-pointee support
      auto PIt = E.funcTypeParamIndices.find(FT);
      if (PIt != E.funcTypeParamIndices.end()) {
        for (unsigned Idx : PIt->second)
          V.push_back(Idx);
      } else {
        for (auto *PT : FT->params())
          V.push_back(E.typeIdx(PT));
      }
      W.EmitRecord(bitc::TYPE_CODE_FUNCTION, V);
    } else if (auto *ST = dyn_cast<StructType>(T)) {
      if (ST->isOpaque()) {
        // Opaque struct (e.g., %event_t = type opaque).
        // Metal/LLVM v1 bitcode: STRUCT_NAME + TYPE_CODE_OPAQUE [isPacked=0]
        if (ST->hasName()) {
          SmallVector<uint64_t, 32> NV;
          for (char C : ST->getName())
            NV.push_back((uint64_t)(unsigned char)C);
          W.EmitRecord(bitc::TYPE_CODE_STRUCT_NAME, NV);
        }
        V.push_back(0); // isPacked
        W.EmitRecord(bitc::TYPE_CODE_OPAQUE, V);
      } else if (ST->hasName()) {
        SmallVector<uint64_t, 32> NV;
        for (char C : ST->getName())
          NV.push_back((uint64_t)(unsigned char)C);
        W.EmitRecord(bitc::TYPE_CODE_STRUCT_NAME, NV);
        V.push_back(ST->isPacked() ? 1 : 0);
        for (auto *ET : ST->elements())
          V.push_back(E.typeIdx(ET));
        W.EmitRecord(bitc::TYPE_CODE_STRUCT_NAMED, V);
      } else {
        V.push_back(ST->isPacked() ? 1 : 0);
        for (auto *ET : ST->elements())
          V.push_back(E.typeIdx(ET));
        W.EmitRecord(bitc::TYPE_CODE_STRUCT_ANON, V);
      }
    }
  }
  W.ExitBlock();
}

} // namespace metal
} // namespace llvm
