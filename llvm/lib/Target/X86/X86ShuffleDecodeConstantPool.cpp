//===-- X86ShuffleDecodeConstantPool.cpp - X86 shuffle decode -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Define several functions to decode x86 specific shuffle semantics using
// constants from the constant pool.
//
//===----------------------------------------------------------------------===//

#include "X86ShuffleDecodeConstantPool.h"
#include "Utils/X86ShuffleDecode.h"
#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/IR/Constants.h"

//===----------------------------------------------------------------------===//
//  Vector Mask Decoding
//===----------------------------------------------------------------------===//

namespace llvm {

void DecodePSHUFBMask(const Constant *C, SmallVectorImpl<int> &ShuffleMask) {
  Type *MaskTy = C->getType();
  // It is not an error for the PSHUFB mask to not be a vector of i8 because the
  // constant pool uniques constants by their bit representation.
  // e.g. the following take up the same space in the constant pool:
  //   i128 -170141183420855150465331762880109871104
  //
  //   <2 x i64> <i64 -9223372034707292160, i64 -9223372034707292160>
  //
  //   <4 x i32> <i32 -2147483648, i32 -2147483648,
  //              i32 -2147483648, i32 -2147483648>

#ifndef NDEBUG
  unsigned MaskTySize = MaskTy->getPrimitiveSizeInBits();
  assert(MaskTySize == 128 || MaskTySize == 256 || MaskTySize == 512);
#endif

  // This is a straightforward byte vector.
  if (MaskTy->isVectorTy() && MaskTy->getVectorElementType()->isIntegerTy(8)) {
    int NumElements = MaskTy->getVectorNumElements();
    ShuffleMask.reserve(NumElements);

    for (int i = 0; i < NumElements; ++i) {
      // For AVX vectors with 32 bytes the base of the shuffle is the 16-byte
      // lane of the vector we're inside.
      int Base = i & ~0xf;
      Constant *COp = C->getAggregateElement(i);
      if (!COp) {
        ShuffleMask.clear();
        return;
      } else if (isa<UndefValue>(COp)) {
        ShuffleMask.push_back(SM_SentinelUndef);
        continue;
      }
      uint64_t Element = cast<ConstantInt>(COp)->getZExtValue();
      // If the high bit (7) of the byte is set, the element is zeroed.
      if (Element & (1 << 7))
        ShuffleMask.push_back(SM_SentinelZero);
      else {
        // Only the least significant 4 bits of the byte are used.
        int Index = Base + (Element & 0xf);
        ShuffleMask.push_back(Index);
      }
    }
  }
  // TODO: Handle funny-looking vectors too.
}

void DecodeVPERMILPMask(const Constant *C, unsigned ElSize,
                        SmallVectorImpl<int> &ShuffleMask) {
  Type *MaskTy = C->getType();
  // It is not an error for the PSHUFB mask to not be a vector of i8 because the
  // constant pool uniques constants by their bit representation.
  // e.g. the following take up the same space in the constant pool:
  //   i128 -170141183420855150465331762880109871104
  //
  //   <2 x i64> <i64 -9223372034707292160, i64 -9223372034707292160>
  //
  //   <4 x i32> <i32 -2147483648, i32 -2147483648,
  //              i32 -2147483648, i32 -2147483648>

  unsigned MaskTySize = MaskTy->getPrimitiveSizeInBits();

  if (MaskTySize != 128 && MaskTySize != 256) // FIXME: Add support for AVX-512.
    return;

  // Only support vector types.
  if (!MaskTy->isVectorTy())
    return;

  // Make sure its an integer type.
  Type *VecEltTy = MaskTy->getVectorElementType();
  if (!VecEltTy->isIntegerTy())
    return;

  // Support any element type from byte up to element size.
  // This is necesary primarily because 64-bit elements get split to 32-bit
  // in the constant pool on 32-bit target.
  unsigned EltTySize = VecEltTy->getIntegerBitWidth();
  if (EltTySize < 8 || EltTySize > ElSize)
    return;

  unsigned NumElements = MaskTySize / ElSize;
  assert((NumElements == 2 || NumElements == 4 || NumElements == 8) &&
         "Unexpected number of vector elements.");
  ShuffleMask.reserve(NumElements);
  unsigned NumElementsPerLane = 128 / ElSize;
  unsigned Factor = ElSize / EltTySize;

  for (unsigned i = 0; i < NumElements; ++i) {
    Constant *COp = C->getAggregateElement(i * Factor);
    if (!COp) {
      ShuffleMask.clear();
      return;
    } else if (isa<UndefValue>(COp)) {
      ShuffleMask.push_back(SM_SentinelUndef);
      continue;
    }
    int Index = i & ~(NumElementsPerLane - 1);
    uint64_t Element = cast<ConstantInt>(COp)->getZExtValue();
    if (ElSize == 64)
      Index += (Element >> 1) & 0x1;
    else
      Index += Element & 0x3;
    ShuffleMask.push_back(Index);
  }

  // TODO: Handle funny-looking vectors too.
}

void DecodeVPERMVMask(const Constant *C, MVT VT,
                      SmallVectorImpl<int> &ShuffleMask) {
  Type *MaskTy = C->getType();
  if (MaskTy->isVectorTy()) {
    unsigned NumElements = MaskTy->getVectorNumElements();
    if (NumElements == VT.getVectorNumElements()) {
      for (unsigned i = 0; i < NumElements; ++i) {
        Constant *COp = C->getAggregateElement(i);
        if (!COp || (!isa<UndefValue>(COp) && !isa<ConstantInt>(COp))) {
          ShuffleMask.clear();
          return;
        }
        if (isa<UndefValue>(COp))
          ShuffleMask.push_back(SM_SentinelUndef);
        else {
          uint64_t Element = cast<ConstantInt>(COp)->getZExtValue();
          Element &= (1 << NumElements) - 1;
          ShuffleMask.push_back(Element);
        }
      }
    }
    return;
  }
  // Scalar value; just broadcast it
  if (!isa<ConstantInt>(C))
    return;
  uint64_t Element = cast<ConstantInt>(C)->getZExtValue();
  int NumElements = VT.getVectorNumElements();
  Element &= (1 << NumElements) - 1;
  for (int i = 0; i < NumElements; ++i)
    ShuffleMask.push_back(Element);
}

void DecodeVPERMV3Mask(const Constant *C, MVT VT,
                       SmallVectorImpl<int> &ShuffleMask) {
  Type *MaskTy = C->getType();
  unsigned NumElements = MaskTy->getVectorNumElements();
  if (NumElements == VT.getVectorNumElements()) {
    for (unsigned i = 0; i < NumElements; ++i) {
      Constant *COp = C->getAggregateElement(i);
      if (!COp) {
        ShuffleMask.clear();
        return;
      }
      if (isa<UndefValue>(COp))
        ShuffleMask.push_back(SM_SentinelUndef);
      else {
        uint64_t Element = cast<ConstantInt>(COp)->getZExtValue();
        Element &= (1 << NumElements*2) - 1;
        ShuffleMask.push_back(Element);
      }
    }
  }
}
} // llvm namespace
