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

cl::opt<bool> llvm::EnableFPInfo(
    "enable-fpinfo",
    cl::desc("Enable low level types to carry floating point information"),
    cl::Optional, cl::Hidden, cl::init(true));

static std::optional<LLT::FPInfo> deriveFPInfo(MVT VT) {
  if (!VT.isFloatingPoint())
    return std::nullopt;

  switch (VT.getScalarType().SimpleTy) {
  case MVT::bf16:
  case MVT::ppcf128:
    return LLT::FPInfo::VARIANT_FLOAT_1;
  default:
    return LLT::FPInfo::IEEE_FLOAT;
  }
}

LLT::LLT(MVT VT, bool EnableFPInfo) {
    auto FP = EnableFPInfo ? deriveFPInfo(VT) : std::nullopt;
    bool AsVector = VT.isVector() && (VT.getVectorMinNumElements() > 1 || VT.isScalableVector());
      
    Kind Info;
    if (EnableFPInfo && FP.has_value())
        Info = AsVector ? Kind::VECTOR_FLOAT : Kind::FLOAT;
    else
        Info = AsVector ? Kind::VECTOR_INTEGER : Kind::INTEGER;

    if (VT.isVector()) {
      init(Info,
           VT.getVectorElementCount(),
           VT.getVectorElementType().getSizeInBits(),
           /*AddressSpace=*/0, FP.value_or(FPInfo::IEEE_FLOAT));
    } else if (VT.isValid() && !VT.isScalableTargetExtVT()) {
      // Aggregates are no different from real scalars as far as GlobalISel is
      // concerned.
      init(Info, ElementCount::getFixed(0), VT.getSizeInBits(),
           /*AddressSpace=*/0, FP.value_or(FPInfo::IEEE_FLOAT));
    } else {
      this->Info = static_cast<Kind>(0);
      this->RawData = 0;
    }
  }

void LLT::print(raw_ostream &OS) const {
  if (isVector()) {
    OS << "<";
    OS << getElementCount() << " x " << getElementType() << ">";
  } else if (isPointer()) {
    OS << "p" << getAddressSpace();
  } else if (EnableFPInfo && isBFloat()) {
    OS << "bf16";
  } else if (EnableFPInfo && isPPCF128()) {
    OS << "ppcf128";
  } else if (EnableFPInfo && isFloat()) {
    assert(!isVariantFloat() && "unknown float variant");
    OS << "f" << getScalarSizeInBits();
  } else if (EnableFPInfo && isInteger()) {
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

const constexpr LLT::BitFieldInfo LLT::ScalarSizeFieldInfo;
const constexpr LLT::BitFieldInfo LLT::ScalarFPFieldInfo;
const constexpr LLT::BitFieldInfo LLT::PointerSizeFieldInfo;
const constexpr LLT::BitFieldInfo LLT::PointerAddressSpaceFieldInfo;
const constexpr LLT::BitFieldInfo LLT::VectorElementsFieldInfo;
const constexpr LLT::BitFieldInfo LLT::VectorScalableFieldInfo;
