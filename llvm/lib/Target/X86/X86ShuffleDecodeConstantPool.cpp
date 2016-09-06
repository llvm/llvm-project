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

  if (!MaskTy->isVectorTy())
    return;
  int NumElts = MaskTy->getVectorNumElements();

  Type *EltTy = MaskTy->getVectorElementType();
  if (!EltTy->isIntegerTy())
    return;

  // The shuffle mask requires a byte vector - decode cases with
  // wider elements as well.
  unsigned BitWidth = cast<IntegerType>(EltTy)->getBitWidth();
  if ((BitWidth % 8) != 0)
    return;

  int Scale = BitWidth / 8;
  int NumBytes = NumElts * Scale;
  ShuffleMask.reserve(NumBytes);

  for (int i = 0; i != NumElts; ++i) {
    Constant *COp = C->getAggregateElement(i);
    if (!COp) {
      ShuffleMask.clear();
      return;
    } else if (isa<UndefValue>(COp)) {
      ShuffleMask.append(Scale, SM_SentinelUndef);
      continue;
    }

    APInt APElt = cast<ConstantInt>(COp)->getValue();
    for (int j = 0; j != Scale; ++j) {
      // For AVX vectors with 32 bytes the base of the shuffle is the 16-byte
      // lane of the vector we're inside.
      int Base = ((i * Scale) + j) & ~0xf;

      uint64_t Element = APElt.getLoBits(8).getZExtValue();
      APElt = APElt.lshr(8);

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

  assert(NumBytes == (int)ShuffleMask.size() && "Unexpected shuffle mask size");
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

  if (ElSize != 32 && ElSize != 64)
    return;

  unsigned MaskTySize = MaskTy->getPrimitiveSizeInBits();
  if (MaskTySize != 128 && MaskTySize != 256 && MaskTySize != 512)
    return;

  // Only support vector types.
  if (!MaskTy->isVectorTy())
    return;

  // Make sure its an integer type.
  Type *VecEltTy = MaskTy->getVectorElementType();
  if (!VecEltTy->isIntegerTy())
    return;

  // Support any element type from byte up to element size.
  // This is necessary primarily because 64-bit elements get split to 32-bit
  // in the constant pool on 32-bit target.
  unsigned EltTySize = VecEltTy->getIntegerBitWidth();
  if (EltTySize < 8 || EltTySize > ElSize)
    return;

  unsigned NumElements = MaskTySize / ElSize;
  assert((NumElements == 2 || NumElements == 4 || NumElements == 8 ||
          NumElements == 16) &&
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

void DecodeVPERMIL2PMask(const Constant *C, unsigned M2Z, unsigned ElSize,
                         SmallVectorImpl<int> &ShuffleMask) {
  Type *MaskTy = C->getType();

  unsigned MaskTySize = MaskTy->getPrimitiveSizeInBits();
  if (MaskTySize != 128 && MaskTySize != 256)
    return;

  // Only support vector types.
  if (!MaskTy->isVectorTy())
    return;

  // Make sure its an integer type.
  Type *VecEltTy = MaskTy->getVectorElementType();
  if (!VecEltTy->isIntegerTy())
    return;

  // Support any element type from byte up to element size.
  // This is necessary primarily because 64-bit elements get split to 32-bit
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

    // VPERMIL2 Operation.
    // Bits[3] - Match Bit.
    // Bits[2:1] - (Per Lane) PD Shuffle Mask.
    // Bits[2:0] - (Per Lane) PS Shuffle Mask.
    uint64_t Selector = cast<ConstantInt>(COp)->getZExtValue();
    unsigned MatchBit = (Selector >> 3) & 0x1;

    // M2Z[0:1]     MatchBit
    //   0Xb           X        Source selected by Selector index.
    //   10b           0        Source selected by Selector index.
    //   10b           1        Zero.
    //   11b           0        Zero.
    //   11b           1        Source selected by Selector index.
    if ((M2Z & 0x2) != 0u && MatchBit != (M2Z & 0x1)) {
      ShuffleMask.push_back(SM_SentinelZero);
      continue;
    }

    int Index = i & ~(NumElementsPerLane - 1);
    if (ElSize == 64)
      Index += (Selector >> 1) & 0x1;
    else
      Index += Selector & 0x3;

    int Src = (Selector >> 2) & 0x1;
    Index += Src * NumElements;
    ShuffleMask.push_back(Index);
  }

  // TODO: Handle funny-looking vectors too.
}

void DecodeVPPERMMask(const Constant *C, SmallVectorImpl<int> &ShuffleMask) {
  Type *MaskTy = C->getType();
  assert(MaskTy->getPrimitiveSizeInBits() == 128);

  // Only support vector types.
  if (!MaskTy->isVectorTy())
    return;

  // Make sure its an integer type.
  Type *VecEltTy = MaskTy->getVectorElementType();
  if (!VecEltTy->isIntegerTy())
    return;

  // The shuffle mask requires a byte vector - decode cases with
  // wider elements as well.
  unsigned BitWidth = cast<IntegerType>(VecEltTy)->getBitWidth();
  if ((BitWidth % 8) != 0)
    return;

  int NumElts = MaskTy->getVectorNumElements();
  int Scale = BitWidth / 8;
  int NumBytes = NumElts * Scale;
  ShuffleMask.reserve(NumBytes);

  for (int i = 0; i != NumElts; ++i) {
    Constant *COp = C->getAggregateElement(i);
    if (!COp) {
      ShuffleMask.clear();
      return;
    } else if (isa<UndefValue>(COp)) {
      ShuffleMask.append(Scale, SM_SentinelUndef);
      continue;
    }

    // VPPERM Operation
    // Bits[4:0] - Byte Index (0 - 31)
    // Bits[7:5] - Permute Operation
    //
    // Permute Operation:
    // 0 - Source byte (no logical operation).
    // 1 - Invert source byte.
    // 2 - Bit reverse of source byte.
    // 3 - Bit reverse of inverted source byte.
    // 4 - 00h (zero - fill).
    // 5 - FFh (ones - fill).
    // 6 - Most significant bit of source byte replicated in all bit positions.
    // 7 - Invert most significant bit of source byte and replicate in all bit positions.
    APInt MaskElt = cast<ConstantInt>(COp)->getValue();
    for (int j = 0; j != Scale; ++j) {
      APInt Index = MaskElt.getLoBits(5);
      APInt PermuteOp = MaskElt.lshr(5).getLoBits(3);
      MaskElt = MaskElt.lshr(8);

      if (PermuteOp == 4) {
        ShuffleMask.push_back(SM_SentinelZero);
        continue;
      }
      if (PermuteOp != 0) {
        ShuffleMask.clear();
        return;
      }
      ShuffleMask.push_back((int)Index.getZExtValue());
    }
  }

  assert(NumBytes == (int)ShuffleMask.size() && "Unexpected shuffle mask size");
}

void DecodeVPERMVMask(const Constant *C, MVT VT,
                      SmallVectorImpl<int> &ShuffleMask) {
  Type *MaskTy = C->getType();
  if (MaskTy->isVectorTy()) {
    unsigned NumElements = MaskTy->getVectorNumElements();
    if (NumElements == VT.getVectorNumElements()) {
      unsigned EltMaskSize = Log2_64(NumElements);
      for (unsigned i = 0; i < NumElements; ++i) {
        Constant *COp = C->getAggregateElement(i);
        if (!COp || (!isa<UndefValue>(COp) && !isa<ConstantInt>(COp))) {
          ShuffleMask.clear();
          return;
        }
        if (isa<UndefValue>(COp))
          ShuffleMask.push_back(SM_SentinelUndef);
        else {
          APInt Element = cast<ConstantInt>(COp)->getValue();
          Element = Element.getLoBits(EltMaskSize);
          ShuffleMask.push_back(Element.getZExtValue());
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
    unsigned EltMaskSize = Log2_64(NumElements * 2);
    for (unsigned i = 0; i < NumElements; ++i) {
      Constant *COp = C->getAggregateElement(i);
      if (!COp) {
        ShuffleMask.clear();
        return;
      }
      if (isa<UndefValue>(COp))
        ShuffleMask.push_back(SM_SentinelUndef);
      else {
        APInt Element = cast<ConstantInt>(COp)->getValue();
        Element = Element.getLoBits(EltMaskSize);
        ShuffleMask.push_back(Element.getZExtValue());
      }
    }
  }
}
} // llvm namespace
