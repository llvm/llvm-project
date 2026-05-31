//===- AIRTargetTransformInfo.h - AIR TTI -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_AIR_AIRTARGETTRANSFORMINFO_H

#include "AIRSubtarget.h"
#include "AIRTargetMachine.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Function.h"

namespace llvm {
class AIRTTIImpl final : public BasicTTIImplBase<AIRTTIImpl> {
  using BaseT = BasicTTIImplBase<AIRTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const AIRSubtarget *ST;
  const AIRTargetLowering *TLI;

  const AIRSubtarget *getST() const { return ST; }
  const AIRTargetLowering *getTLI() const { return TLI; }

public:
  explicit AIRTTIImpl(const AIRTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRTARGETTRANSFORMINFO_H
