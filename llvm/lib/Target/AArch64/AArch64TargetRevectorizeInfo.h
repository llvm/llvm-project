//===- AArch64TargetRevectorizeInfo.h - AArch64 TRVI -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64TARGETREVECTORIZEINFO_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64TARGETREVECTORIZEINFO_H

#include "llvm/Analysis/TargetRevectorizeInfo.h"

namespace llvm {

class AArch64Subtarget;

class AArch64RevectorizeInfoImpl final : public TargetRevectorizeInfoImplBase {
  // The target machine owns its subtargets and outlives function analyses.
  const AArch64Subtarget &ST;

public:
  AArch64RevectorizeInfoImpl(const TargetTransformInfo &TTI,
                             const AArch64Subtarget &ST)
      : TargetRevectorizeInfoImplBase(TTI), ST(ST) {}

  bool isTargetIntrinsicVectorizable(Intrinsic::ID ID) const override;

  Instruction *vectorizeTargetIntrinsic(Intrinsic::ID VectorIID,
                                        ArrayRef<Type *> TysForDecl,
                                        ArrayRef<Value *> WideArgs,
                                        ElementCount VF,
                                        IRBuilderBase &Builder) const override;

  InstructionCost
  getTargetIntrinsicVectorizationCost(Intrinsic::ID FromID, Type *WideRetTy,
                                      ArrayRef<Type *> WideArgTys,
                                      ElementCount VF) const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AARCH64_AARCH64TARGETREVECTORIZEINFO_H
