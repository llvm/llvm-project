//===-- AGXTargetTransformInfo.h - AGX specific TTI -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGXTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_AGX_AGXTARGETTRANSFORMINFO_H

#include "AGXTargetMachine.h"
#include "llvm/CodeGen/BasicTTIImpl.h"

namespace llvm {

class AGXTTIImpl final : public BasicTTIImplBase<AGXTTIImpl> {
  using BaseT = BasicTTIImplBase<AGXTTIImpl>;
  friend BaseT;

  const AGXSubtarget *ST;
  const AGXTargetLowering *TLI;

  const AGXSubtarget *getST() const { return ST; }
  const AGXTargetLowering *getTLI() const { return TLI; }

public:
  explicit AGXTTIImpl(const AGXTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  bool hasBranchDivergence(const Function *F = nullptr) const override {
    return true;
  }

  ValueUniformity getValueUniformity(const Value *V) const override;

  std::optional<Instruction *>
  instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const override;

  unsigned getFlatAddressSpace() const override {
    return AGX::GENERIC_ADDRESS;
  }

  unsigned getNumberOfRegisters(unsigned ClassID) const override {
    return ST->getMaxGPRs();
  }

  TypeSize
  getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const override {
    return TypeSize::getFixed(32);
  }

  unsigned getInliningThresholdMultiplier() const override { return 11; }

  // Device and threadgroup accesses of up to four 32-bit values are one
  // instruction (xy / xyz / xyzw) at any 4-byte alignment (ptx-re-blockers
  // B33); the load/store vectorizer merges neighbouring ones.
  unsigned getLoadStoreVecRegBitWidth(unsigned AS) const override {
    return 128;
  }
  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes, Align Alignment,
                                   unsigned AS) const override {
    return vectorizableChain(ChainSizeInBytes, Alignment, AS);
  }
  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes, Align Alignment,
                                    unsigned AS) const override {
    return vectorizableChain(ChainSizeInBytes, Alignment, AS);
  }
  static bool vectorizableChain(unsigned Bytes, Align Alignment, unsigned AS) {
    return Alignment >= Align(4) && Bytes % 4 == 0 && Bytes <= 16 &&
           (AS == AGX::DEVICE_ADDRESS || AS == AGX::THREADGROUP_ADDRESS);
  }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGXTARGETTRANSFORMINFO_H
