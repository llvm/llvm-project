//===-- llvm/CodeGenTypes/LowLevelType.cpp
//---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file implements the more header-heavy bits of the LLT class to
/// avoid polluting users' namespaces.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

bool LLT::ExtendedLLT = false;

static LLT::FpSemantics getFpSemanticsForMVT(MVT VT) {
  switch (VT.getScalarType().SimpleTy) {
  default:
    llvm_unreachable("Unknown FP format");
  case MVT::f16:
    return LLT::FpSemantics::S_IEEEhalf;
  case MVT::bf16:
    return LLT::FpSemantics::S_BFloat;
  case MVT::f32:
    return LLT::FpSemantics::S_IEEEsingle;
  case MVT::f64:
    return LLT::FpSemantics::S_IEEEdouble;
  case MVT::f80:
    return LLT::FpSemantics::S_x87DoubleExtended;
  case MVT::f128:
    return LLT::FpSemantics::S_IEEEquad;
  case MVT::ppcf128:
    return LLT::FpSemantics::S_PPCDoubleDouble;
  }
}

LLT::LLT(MVT VT) {
  if (!ExtendedLLT) {
    if (VT.isVector()) {
      bool AsVector = VT.getVectorMinNumElements() > 1 || VT.isScalableVector();
      Kind Info = AsVector ? Kind::VECTOR_ANY : Kind::ANY_SCALAR;
      init(Info, VT.getVectorElementCount(),
           VT.getVectorElementType().getSizeInBits());
    } else if (VT.isValid() && !VT.isScalableTargetExtVT()) {
      init(Kind::ANY_SCALAR, ElementCount::getFixed(0), VT.getSizeInBits());
    } else {
      this->Info = Kind::INVALID;
      this->RawData = 0;
    }
    return;
  }

  bool IsFloatingPoint = VT.isFloatingPoint();
  bool AsVector = VT.isVector() &&
                  (VT.getVectorMinNumElements() > 1 || VT.isScalableVector());

  if (AsVector) {
    if (IsFloatingPoint)
      init(LLT::Kind::VECTOR_FLOAT, VT.getVectorElementCount(),
           VT.getVectorElementType().getSizeInBits(), getFpSemanticsForMVT(VT));
    else
      init(LLT::Kind::VECTOR_INTEGER, VT.getVectorElementCount(),
           VT.getVectorElementType().getSizeInBits());
  } else if (VT.isValid() && !VT.isScalableTargetExtVT()) {
    // Aggregates are no different from real scalars as far as GlobalISel is
    // concerned.
    if (IsFloatingPoint)
      init(LLT::Kind::FLOAT, ElementCount::getFixed(0), VT.getSizeInBits(),
           getFpSemanticsForMVT(VT));
    else
      init(LLT::Kind::INTEGER, ElementCount::getFixed(0), VT.getSizeInBits());
  } else {
    this->Info = Kind::INVALID;
    this->RawData = 0;
  }
  return;
}

void LLT::print(raw_ostream &OS) const {
  if (isVector()) {
    OS << "<";
    OS << getElementCount() << " x " << getElementType() << ">";
  } else if (isPointer()) {
    OS << "p" << getAddressSpace();
  } else if (isBFloat16()) {
    OS << "bf16";
  } else if (isPPCF128()) {
    OS << "ppcf128";
  } else if (isFloatIEEE()) {
    OS << "f" << getScalarSizeInBits();
  } else if (isInteger()) {
    OS << "i" << getScalarSizeInBits();
  } else if (isValid()) {
    assert(isScalar() && "unexpected type");
    OS << "s" << getScalarSizeInBits();
  } else {
    OS << "LLT_invalid";
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void LLT::dump() const {
  print(dbgs());
  dbgs() << '\n';
}
#endif
