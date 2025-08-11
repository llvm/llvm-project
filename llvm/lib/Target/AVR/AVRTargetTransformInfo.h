//===- AVRTargetTransformInfo.h - AVR specific TTI --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines a TargetTransformInfoImplBase conforming object specific
/// to the AVR target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AVR_AVRTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_AVR_AVRTARGETTRANSFORMINFO_H

#include "AVRSubtarget.h"
#include "AVRTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Function.h"
#include <optional>

namespace llvm {

class AVRTTIImpl final : public BasicTTIImplBase<AVRTTIImpl> {
  using BaseT = BasicTTIImplBase<AVRTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const AVRSubtarget *ST;
  const AVRTargetLowering *TLI;

  const AVRSubtarget *getST() const { return ST; }
  const AVRTargetLowering *getTLI() const { return TLI; }

public:
  explicit AVRTTIImpl(const AVRTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  bool isLSRCostLess(const TargetTransformInfo::LSRCost &C1,
                     const TargetTransformInfo::LSRCost &C2) const override {
    if (C2.Insns == ~0u)
      return true;
    return 2 * C1.Insns + C1.AddRecCost + C1.SetupCost + C1.NumRegs <
           2 * C2.Insns + C2.AddRecCost + C2.SetupCost + C2.NumRegs;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AVR_AVRTARGETTRANSFORMINFO_H
