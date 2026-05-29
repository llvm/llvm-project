//===- MetalTargetTransformInfo.h - Metal TTI -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_METAL_METALTARGETTRANSFORMINFO_H

#include "MetalSubtarget.h"
#include "MetalTargetMachine.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Function.h"

namespace llvm {
class MetalTTIImpl final : public BasicTTIImplBase<MetalTTIImpl> {
  using BaseT = BasicTTIImplBase<MetalTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const MetalSubtarget *ST;
  const MetalTargetLowering *TLI;

  const MetalSubtarget *getST() const { return ST; }
  const MetalTargetLowering *getTLI() const { return TLI; }

public:
  explicit MetalTTIImpl(const MetalTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALTARGETTRANSFORMINFO_H
