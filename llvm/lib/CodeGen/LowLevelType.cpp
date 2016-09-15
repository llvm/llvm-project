//===-- llvm/CodeGen/GlobalISel/LowLevelType.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file implements the more header-heavy bits of the LLT class to
/// avoid polluting users' namespaces.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

LLT::LLT(Type &Ty, const DataLayout &DL) {
  if (auto VTy = dyn_cast<VectorType>(&Ty)) {
    SizeInBits = VTy->getElementType()->getPrimitiveSizeInBits();
    ElementsOrAddrSpace = VTy->getNumElements();
    Kind = ElementsOrAddrSpace == 1 ? Scalar : Vector;
  } else if (auto PTy = dyn_cast<PointerType>(&Ty)) {
    Kind = Pointer;
    SizeInBits = DL.getTypeSizeInBits(&Ty);
    ElementsOrAddrSpace = PTy->getAddressSpace();
  } else if (Ty.isSized()) {
    // Aggregates are no different from real scalars as far as GlobalISel is
    // concerned.
    Kind = Scalar;
    SizeInBits = DL.getTypeSizeInBits(&Ty);
    ElementsOrAddrSpace = 1;
    assert(SizeInBits != 0 && "invalid zero-sized type");
  } else {
    Kind = Invalid;
    SizeInBits = ElementsOrAddrSpace = 0;
  }
}

void LLT::print(raw_ostream &OS) const {
  if (isVector())
    OS << "<" << ElementsOrAddrSpace << " x s" << SizeInBits << ">";
  else if (isPointer())
    OS << "p" << getAddressSpace();
  else if (isValid()) {
    assert(isScalar() && "unexpected type");
    OS << "s" << getScalarSizeInBits();
  } else
    llvm_unreachable("trying to print an invalid type");
}
