//===-- llvm/Support/LowLevelType.cpp -------------------------------------===//
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

#include "llvm/Support/LowLevelTypeImpl.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

LLT::LLT(MVT VT) {
  if (VT.isVector()) {
    init(/*IsPointer=*/false, VT.getVectorNumElements() > 1,
         VT.getVectorNumElements(), VT.getVectorElementType().getSizeInBits(),
         /*AddressSpace=*/0);
  } else if (VT.isValid()) {
    // Aggregates are no different from real scalars as far as GlobalISel is
    // concerned.
    assert(VT.getSizeInBits().isNonZero() && "invalid zero-sized type");
    init(/*IsPointer=*/false, /*IsVector=*/false, /*NumElements=*/0,
         VT.getSizeInBits(), /*AddressSpace=*/0);
  } else {
    IsPointer = false;
    IsVector = false;
    RawData = 0;
  }
}

void LLT::print(raw_ostream &OS) const {
  if (isVector())
    OS << "<" << getNumElements() << " x " << getElementType() << ">";
  else if (isPointer())
    OS << "p" << getAddressSpace();
  else if (isValid()) {
    assert(isScalar() && "unexpected type");
    OS << "s" << getScalarSizeInBits();
  } else
    OS << "LLT_invalid";
}

const constexpr LLT::BitFieldInfo LLT::ScalarSizeFieldInfo;
const constexpr LLT::BitFieldInfo LLT::PointerSizeFieldInfo;
const constexpr LLT::BitFieldInfo LLT::PointerAddressSpaceFieldInfo;
const constexpr LLT::BitFieldInfo LLT::VectorElementsFieldInfo;
const constexpr LLT::BitFieldInfo LLT::VectorSizeFieldInfo;
const constexpr LLT::BitFieldInfo LLT::PointerVectorElementsFieldInfo;
const constexpr LLT::BitFieldInfo LLT::PointerVectorSizeFieldInfo;
const constexpr LLT::BitFieldInfo LLT::PointerVectorAddressSpaceFieldInfo;
