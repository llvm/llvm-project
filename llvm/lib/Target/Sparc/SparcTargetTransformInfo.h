//===-- SparcTargetTransformInfo.cpp - SPARC specific TTI--------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfoImplBase conforming object specific to the
/// SPARC target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPARC_SPARCTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_SPARC_SPARCTARGETTRANSFORMINFO_H

#include "SparcTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"

namespace llvm {

class SparcTTIImpl final : public BasicTTIImplBase<SparcTTIImpl> {
  typedef BasicTTIImplBase<SparcTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const SparcSubtarget *ST;
  const SparcTargetLowering *TLI;

  const SparcSubtarget *getST() const { return ST; }
  const SparcTargetLowering *getTLI() const { return TLI; }

public:
  explicit SparcTTIImpl(const SparcTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  /// \name Scalar TTI Implementations
  /// @{

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth) const override;
  /// @}
};

} // end namespace llvm

#endif
