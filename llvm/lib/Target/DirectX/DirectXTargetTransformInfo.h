//===- DirectXTargetTransformInfo.h - DirectX TTI ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DIRECTX_DIRECTXTARGETTRANSFORMINFO_H
#define LLVM_DIRECTX_DIRECTXTARGETTRANSFORMINFO_H

#include "DirectXSubtarget.h"
#include "DirectXTargetMachine.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

namespace llvm {
class DirectXTTIImpl final : public BasicTTIImplBase<DirectXTTIImpl> {
  using BaseT = BasicTTIImplBase<DirectXTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const DirectXSubtarget *ST;
  const DirectXTargetLowering *TLI;
  // True when native 16-bit types are enabled (i.e. -enable-16bit-types was
  // passed), indicated by the dx.nativelowprec module flag.
  const bool HasNativeLowPrecision;

  const DirectXSubtarget *getST() const { return ST; }
  const DirectXTargetLowering *getTLI() const { return TLI; }

  static bool readNativeLowPrecisionFlag(const Function &F) {
    if (auto *Flag = mdconst::extract_or_null<ConstantInt>(
            F.getParent()->getModuleFlag("dx.nativelowprec")))
      return Flag->getValue().getBoolValue();
    return false;
  }

public:
  explicit DirectXTTIImpl(const DirectXTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()),
        HasNativeLowPrecision(readNativeLowPrecisionFlag(F)) {}
  unsigned getMinVectorRegisterBitWidth() const override { return 32; }
  bool isTargetIntrinsicWithScalarOpAtArg(Intrinsic::ID ID,
                                          unsigned ScalarOpdIdx) const override;
  bool isTargetIntrinsicWithOverloadTypeAtArg(Intrinsic::ID ID,
                                              int OpdIdx) const override;
  unsigned getMinimumLookupTableEntryBitWidth() const override;

  InstructionCost getPartialReductionCost(
      unsigned Opcode, Type *InputTypeA, Type *InputTypeB, Type *AccumType,
      ElementCount VF, TTI::PartialReductionExtendKind OpAExtend,
      TTI::PartialReductionExtendKind OpBExtend, std::optional<unsigned> BinOp,
      TTI::TargetCostKind CostKind,
      std::optional<FastMathFlags> FMF) const override {
    return InstructionCost::getInvalid();
  }
};
} // namespace llvm

#endif // LLVM_DIRECTX_DIRECTXTARGETTRANSFORMINFO_H
