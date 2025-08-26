//===- LoongArchTargetTransformInfo.h - LoongArch specific TTI --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfoImplBase conforming object specific to the
/// LoongArch target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LOONGARCH_LOONGARCHTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_LOONGARCH_LOONGARCHTARGETTRANSFORMINFO_H

#include "LoongArchSubtarget.h"
#include "LoongArchTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"

namespace llvm {

class LoongArchTTIImpl : public BasicTTIImplBase<LoongArchTTIImpl> {
  typedef BasicTTIImplBase<LoongArchTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  enum LoongArchRegisterClass { GPRRC, FPRRC, VRRC };
  const LoongArchSubtarget *ST;
  const LoongArchTargetLowering *TLI;

  const LoongArchSubtarget *getST() const { return ST; }
  const LoongArchTargetLowering *getTLI() const { return TLI; }

public:
  explicit LoongArchTTIImpl(const LoongArchTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  TypeSize
  getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const override;
  unsigned getNumberOfRegisters(unsigned ClassID) const override;
  unsigned getRegisterClassForType(bool Vector,
                                   Type *Ty = nullptr) const override;
  unsigned getMaxInterleaveFactor(ElementCount VF) const override;
  const char *getRegisterClassName(unsigned ClassID) const override;
  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth) const override;

  unsigned getCacheLineSize() const override;
  unsigned getPrefetchDistance() const override;
  bool enableWritePrefetching() const override;

  bool shouldExpandReduction(const IntrinsicInst *II) const override;

  // TODO: Implement more hooks to provide TTI machinery for LoongArch.
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_LOONGARCH_LOONGARCHTARGETTRANSFORMINFO_H
