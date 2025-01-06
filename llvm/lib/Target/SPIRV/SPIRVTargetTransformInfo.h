//===- SPIRVTargetTransformInfo.h - SPIR-V specific TTI ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// \file
// This file contains a TargetTransformInfo::Concept conforming object specific
// to the SPIRV target machine. It uses the target's detailed information to
// provide more precise answers to certain TTI queries, while letting the
// target independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVTARGETTRANSFORMINFO_H

#include "SPIRV.h"
#include "SPIRVTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"

namespace llvm {
class SPIRVTTIImpl : public BasicTTIImplBase<SPIRVTTIImpl> {
  using BaseT = BasicTTIImplBase<SPIRVTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const SPIRVSubtarget *ST;
  const SPIRVTargetLowering *TLI;

  const TargetSubtargetInfo *getST() const { return ST; }
  const SPIRVTargetLowering *getTLI() const { return TLI; }

public:
  explicit SPIRVTTIImpl(const SPIRVTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth) {
    // SPIR-V natively supports OpBitcount, per 3.53.14 in the spec, as such it
    // is reasonable to assume the Op is fast / preferable to the expanded loop.
    // Furthermore, this prevents information being lost if transforms are
    // applied to SPIR-V before lowering to a concrete target.
    if (!isPowerOf2_32(TyWidth) || TyWidth > 64)
      return TTI::PSK_Software; // Arbitrary bit-width INT is not core SPIR-V.
    return TTI::PSK_FastHardware;
  }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVTARGETTRANSFORMINFO_H
