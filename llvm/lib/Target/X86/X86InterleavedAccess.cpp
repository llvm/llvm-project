//===------- X86InterleavedAccess.cpp --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the interleaved accesses
// optimization generating X86-specific instructions/intrinsics for interleaved
// access groups.
//
//===----------------------------------------------------------------------===//

#include "X86ISelLowering.h"
#include "X86TargetMachine.h"

using namespace llvm;

/// Returns true if the interleaved access group represented by the shuffles
/// is supported for the subtarget. Returns false otherwise.
static bool isSupported(const X86Subtarget &SubTarget,
                        const LoadInst *LI,
                        const ArrayRef<ShuffleVectorInst *> &Shuffles,
                        unsigned Factor) {

  const DataLayout &DL = Shuffles[0]->getModule()->getDataLayout();
  VectorType *ShuffleVecTy = Shuffles[0]->getType();
  unsigned ShuffleVecSize = DL.getTypeSizeInBits(ShuffleVecTy);
  Type *ShuffleEltTy = ShuffleVecTy->getVectorElementType();

  if (DL.getTypeSizeInBits(LI->getType()) < Factor * ShuffleVecSize)
    return false;

  // Currently, lowering is supported for 64 bits on AVX.
  if (!SubTarget.hasAVX() || ShuffleVecSize != 256 ||
      DL.getTypeSizeInBits(ShuffleEltTy) != 64 ||
      Factor != 4)
    return false;

  return true;
}

/// \brief Lower interleaved load(s) into target specific instructions/
/// intrinsics. Lowering sequence varies depending on the vector-types, factor,
/// number of shuffles and ISA.
/// Currently, lowering is supported for 4x64 bits with Factor = 4 on AVX.
bool X86TargetLowering::lowerInterleavedLoad(
    LoadInst *LI, ArrayRef<ShuffleVectorInst *> Shuffles,
    ArrayRef<unsigned> Indices, unsigned Factor) const {
  assert(Factor >= 2 && Factor <= getMaxSupportedInterleaveFactor() &&
         "Invalid interleave factor");
  assert(!Shuffles.empty() && "Empty shufflevector input");
  assert(Shuffles.size() == Indices.size() &&
         "Unmatched number of shufflevectors and indices");

  if (!isSupported(Subtarget, LI, Shuffles, Factor))
    return false;

  VectorType *ShuffleVecTy = Shuffles[0]->getType();

  Type *VecBasePtrTy = ShuffleVecTy->getPointerTo(LI->getPointerAddressSpace());

  IRBuilder<> Builder(LI);
  SmallVector<Instruction *, 4> NewLoads;
  SmallVector<Value *, 4> NewShuffles;
  NewShuffles.resize(Factor);

  Value *VecBasePtr =
      Builder.CreateBitCast(LI->getPointerOperand(), VecBasePtrTy);

  // Generate 4 loads of type v4xT64
  for (unsigned Part = 0; Part < Factor; Part++) {
    // TODO: Support inbounds GEP
    Value *NewBasePtr =
        Builder.CreateGEP(VecBasePtr, Builder.getInt32(Part));
    Instruction *NewLoad =
        Builder.CreateAlignedLoad(NewBasePtr, LI->getAlignment());
    NewLoads.push_back(NewLoad);
  }

  // dst = src1[0,1],src2[0,1]
  uint32_t IntMask1[] = {0, 1, 4, 5};
  ArrayRef<unsigned int> ShuffleMask = makeArrayRef(IntMask1, 4);
  Value *IntrVec1 =
      Builder.CreateShuffleVector(NewLoads[0], NewLoads[2], ShuffleMask);
  Value *IntrVec2 =
      Builder.CreateShuffleVector(NewLoads[1], NewLoads[3], ShuffleMask);

  // dst = src1[2,3],src2[2,3]
  uint32_t IntMask2[] = {2, 3, 6, 7};
  ShuffleMask = makeArrayRef(IntMask2, 4);
  Value *IntrVec3 =
      Builder.CreateShuffleVector(NewLoads[0], NewLoads[2], ShuffleMask);
  Value *IntrVec4 =
      Builder.CreateShuffleVector(NewLoads[1], NewLoads[3], ShuffleMask);

  // dst = src1[0],src2[0],src1[2],src2[2]
  uint32_t IntMask3[] = {0, 4, 2, 6};
  ShuffleMask = makeArrayRef(IntMask3, 4);
  NewShuffles[0] = Builder.CreateShuffleVector(IntrVec1, IntrVec2, ShuffleMask);
  NewShuffles[2] = Builder.CreateShuffleVector(IntrVec3, IntrVec4, ShuffleMask);

  // dst = src1[1],src2[1],src1[3],src2[3]
  uint32_t IntMask4[] = {1, 5, 3, 7};
  ShuffleMask = makeArrayRef(IntMask4, 4);
  NewShuffles[1] = Builder.CreateShuffleVector(IntrVec1, IntrVec2, ShuffleMask);
  NewShuffles[3] = Builder.CreateShuffleVector(IntrVec3, IntrVec4, ShuffleMask);

  for (unsigned i = 0; i < Shuffles.size(); i++) {
    unsigned Index = Indices[i];
    Shuffles[i]->replaceAllUsesWith(NewShuffles[Index]);
  }

  return true;
}
