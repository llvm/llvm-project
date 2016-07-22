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
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

LLT::LLT(const Type &Ty) {
  if (auto VTy = dyn_cast<VectorType>(&Ty)) {
    ScalarSize = VTy->getElementType()->getPrimitiveSizeInBits();
    NumElements = VTy->getNumElements();
    Kind = NumElements == 1 ? Scalar : Vector;
  } else if (Ty.isSized()) {
    // Aggregates are no different from real scalars as far as GlobalISel is
    // concerned.
    Kind = Scalar;
    ScalarSize = Ty.getPrimitiveSizeInBits();
    NumElements = 1;
  } else {
    Kind = Unsized;
    ScalarSize = NumElements = 0;
  }
}

void LLT::print(raw_ostream &OS) const {
  if (isVector())
    OS << "<" << NumElements << " x s" << ScalarSize << ">";
  else if (isSized())
    OS << "s" << ScalarSize;
  else if (isValid())
    OS << "unsized";
  else
    llvm_unreachable("trying to print an invalid type");
}
