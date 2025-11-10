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

static std::optional<LLT::FPVariant> deriveFPInfo(MVT VT) {
  if (!VT.isFloatingPoint())
    return std::nullopt;

  switch (VT.getScalarType().SimpleTy) {
  case MVT::bf16:
    return LLT::FPVariant::BF16;
  case MVT::f80:
    return LLT::FPVariant::EXTENDED_FP80;
  case MVT::ppcf128:
    return LLT::FPVariant::PPC128_FLOAT;
  default:
    return LLT::FPVariant::IEEE_FLOAT;
  }
}

LLT::LLT(MVT VT, bool AllowExtendedLLT) {
  if (!AllowExtendedLLT) {
    if (VT.isVector()) {
      bool AsVector = VT.getVectorMinNumElements() > 1 || VT.isScalableVector();
      Kind Info = AsVector ? Kind::VECTOR_ANY : Kind::ANY_SCALAR;
      init(Info, VT.getVectorElementCount(),
           VT.getVectorElementType().getSizeInBits(), 0, FPVariant::IEEE_FLOAT);
    } else if (VT.isValid() && !VT.isScalableTargetExtVT()) {
      init(Kind::ANY_SCALAR, ElementCount::getFixed(0), VT.getSizeInBits(), 0,
           FPVariant::IEEE_FLOAT);
    } else {
      this->Info = Kind::INVALID;
      this->RawData = 0;
    }
    return;
  }

  auto FP = deriveFPInfo(VT);
  bool AsVector = VT.isVector() &&
                  (VT.getVectorMinNumElements() > 1 || VT.isScalableVector());

  LLT::Kind Info;
  if (FP.has_value())
    Info = AsVector ? LLT::Kind::VECTOR_FLOAT : LLT::Kind::FLOAT;
  else
    Info = AsVector ? LLT::Kind::VECTOR_INTEGER : LLT::Kind::INTEGER;

  if (VT.isVector()) {
    init(Info, VT.getVectorElementCount(),
         VT.getVectorElementType().getSizeInBits(), 0,
         FP.value_or(LLT::FPVariant::IEEE_FLOAT));
  } else if (VT.isValid() && !VT.isScalableTargetExtVT()) {
    // Aggregates are no different from real scalars as far as GlobalISel is
    // concerned.
    init(Info, ElementCount::getFixed(0), VT.getSizeInBits(), 0,
         FP.value_or(LLT::FPVariant::IEEE_FLOAT));
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
  } else if (isBFloat(16)) {
    OS << "bf16";
  } else if (isPPCF128()) {
    OS << "ppcf128";
  } else if (isFloat()) {
    assert(!isVariantFloat() && "unknown float variant");
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
