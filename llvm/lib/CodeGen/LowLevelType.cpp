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

LLT::LLT(Type &Ty, const DataLayout *DL) {
  if (auto VTy = dyn_cast<VectorType>(&Ty)) {
    SizeOrAddrSpace = VTy->getElementType()->getPrimitiveSizeInBits();
    NumElements = VTy->getNumElements();
    Kind = NumElements == 1 ? Scalar : Vector;
  } else if (auto PTy = dyn_cast<PointerType>(&Ty)) {
    Kind = Pointer;
    SizeOrAddrSpace = PTy->getAddressSpace();
    NumElements = 1;
  } else if (Ty.isSized()) {
    // Aggregates are no different from real scalars as far as GlobalISel is
    // concerned.
    Kind = Scalar;
    SizeOrAddrSpace =
        DL ? DL->getTypeSizeInBits(&Ty) : Ty.getPrimitiveSizeInBits();
    NumElements = 1;
    assert(SizeOrAddrSpace != 0 && "invalid zero-sized type");
  } else {
    Kind = Unsized;
    SizeOrAddrSpace = NumElements = 0;
  }
}

void LLT::print(raw_ostream &OS) const {
  if (isVector())
    OS << "<" << NumElements << " x s" << SizeOrAddrSpace << ">";
  else if (isPointer())
    OS << "p" << getAddressSpace();
  else if (isSized())
    OS << "s" << getScalarSizeInBits();
  else if (isValid())
    OS << "unsized";
  else
    llvm_unreachable("trying to print an invalid type");
}
