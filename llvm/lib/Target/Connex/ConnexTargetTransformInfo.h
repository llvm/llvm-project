//===-- ConnexTargetTransformInfo.h - Connex specific TTI -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains a TargetTransformInfo::Concept conforming object specific
/// to the Connex target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

// Inspired from XCore/XCoreTargetTransformInfo.h

#ifndef LLVM_LIB_TARGET_CONNEX_CONNEXTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_CONNEX_CONNEXTARGETTRANSFORMINFO_H

#include "Connex.h"
#include "ConnexTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class ConnexTTIImpl : public BasicTTIImplBase<ConnexTTIImpl> {
  typedef BasicTTIImplBase<ConnexTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const ConnexSubtarget *ST;
  const ConnexTargetLowering *TLI;

  const ConnexSubtarget *getST() const {
    LLVM_DEBUG(dbgs() << "Entered getST()\n");
    return ST;
  }

  const ConnexTargetLowering *getTLI() const {
    LLVM_DEBUG(dbgs() << "Entered getTLI()\n");
    return TLI;
  }

public:
  // Inspired a bit from AArch64/AArch64TargetTransformInfo.h (from Oct 2025)
  bool isLegalMaskedGather(Type *DataTy, Align Alignment) const override {
    // Inspired from X86TargetTransformInfo.cpp
    LLVM_DEBUG(dbgs() << "Entered isLegalMaskedGather()\n");

    /*
    // Some CPUs have better gather performance than others.
    // TODO: Remove the explicit ST->hasAVX512()?, That would mean we would only
    // enable gather with a -march.
    if (!(ST->hasAVX512() || (ST->hasFastGather() && ST->hasAVX2())))
      return false;

    // This function is called now in two cases: from the Loop Vectorizer
    // and from the Scalarizer.
    // When the Loop Vectorizer asks about legality of the feature,
    // the vectorization factor is not calculated yet. The Loop Vectorizer
    // sends a scalar type and the decision is based on the width of the
    // scalar element.
    // Later on, the cost model will estimate usage this intrinsic based on
    // the vector type.
    // The Scalarizer asks again about legality. It sends a vector type.
    // In this case we can reject non-power-of-2 vectors.
    // We also reject single element vectors as the type legalizer can't
    // scalarize it.
    if (isa<VectorType>(DataTy)) {
      unsigned NumElts = DataTy->getVectorNumElements();
      if (NumElts == 1 || !isPowerOf2_32(NumElts))
        return false;
    }
    Type *ScalarTy = DataTy->getScalarType();
    if (ScalarTy->isPointerTy())
      return true;

    if (ScalarTy->isFloatTy() || ScalarTy->isDoubleTy())
      return true;

    if (!ScalarTy->isIntegerTy())
      return false;

    unsigned IntWidth = ScalarTy->getIntegerBitWidth();
    return IntWidth == 32 || IntWidth == 64;
    */

    Type *ScalarTy = DataTy->getScalarType();

    if (ScalarTy->isHalfTy())
      return true;

    if (ScalarTy->isIntegerTy()) {
      unsigned IntWidth = ScalarTy->getIntegerBitWidth();
      LLVM_DEBUG(dbgs() << "isLegalMaskedGather(): IntWidth = "
                        << IntWidth << "\n");
      return (IntWidth == 16) || (IntWidth == 32);
    }

    return false;
  }

  // Inspired a bit from AArch64/AArch64TargetTransformInfo.h (from Oct 2025)
  bool isLegalMaskedScatter(Type *DataType, Align Alignment) const override {
    LLVM_DEBUG(dbgs() << "Entered isLegalMaskedScatter()\n");

    // Inspired from X86TargetTransformInfo.cpp
    return isLegalMaskedGather(DataType, Alignment);
  }

public:
  explicit ConnexTTIImpl(const ConnexTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl()),
        TLI(ST->getTargetLowering()) {
    LLVM_DEBUG(dbgs() << "Entered constructor ConnexTTIImpl()\n");
  }

  /*
  unsigned getNumberOfRegisters(bool Vector) {
    if (Vector) {
      return 0;
    }
    return 12;
  }
  */
};

} // end namespace llvm

#endif
