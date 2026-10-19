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
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGXTARGETTRANSFORMINFO_H
