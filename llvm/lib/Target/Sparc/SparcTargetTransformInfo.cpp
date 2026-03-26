//===-- SparcTargetTransformInfo.cpp - SPARC specific TTI -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SparcTargetTransformInfo.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "sparctti"

TargetTransformInfo::PopcntSupportKind
SparcTTIImpl::getPopcntSupport(unsigned TyWidth) const {
  assert(isPowerOf2_32(TyWidth) && "Type width must be power of 2");
  if (ST->usePopc())
    return TTI::PSK_FastHardware;
  return TTI::PSK_Software;
}

unsigned SparcTTIImpl::getRegisterClassForType(bool Vector, Type *Ty) const {
  if (Vector)
    return VRRC;
  if (Ty &&
      (Ty->getScalarType()->isFloatTy() || Ty->getScalarType()->isDoubleTy()))
    return FPRRC;
  if (Ty && (Ty->getScalarType()->isFP128Ty()))
    return FP128RRC;
  return GPRRC;
}

unsigned SparcTTIImpl::getNumberOfRegisters(unsigned ClassID) const {
  switch (ClassID) {
  case GPRRC:
    // %g0, %g6, %g7, %o6, %i6, and %i7 are used for special purposes so we
    // discount them here.
    return 26;
  case FPRRC:
    return 32;
  case FP128RRC:
    return 16;
  case VRRC:
    // TODO We have vector capabilities as part of the VIS extensions, but the
    // codegen doesn't currently use it. Revisit this when vector codegen is
    // ready.
    return 0;
  }

  llvm_unreachable("Unsupported register class");
}

TypeSize
SparcTTIImpl::getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const {
  switch (K) {
  case TargetTransformInfo::RGK_Scalar:
    // TODO When targeting V8+ ABI, G and O registers are 64-bit.
    return TypeSize::getFixed(ST->is64Bit() ? 64 : 32);
  case TargetTransformInfo::RGK_FixedWidthVector:
    // TODO We have vector capabilities as part of the VIS extensions, but the
    // codegen doesn't currently use it. Revisit this when vector codegen is
    // ready.
    return TypeSize::getFixed(0);
  case TargetTransformInfo::RGK_ScalableVector:
    return TypeSize::getScalable(0);
  }

  llvm_unreachable("Unsupported register kind");
}
