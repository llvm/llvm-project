//===- lib/CodeGen/GlobalISel/LegalizerMutations.cpp - Mutations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A library of mutation factories to use for LegalityMutation.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"

using namespace llvm;

LegalizeMutation LegalizeMutations::changeTo(unsigned TypeIdx, LLT Ty) {
  return
      [=](const LegalityQuery &Query) { return std::make_pair(TypeIdx, Ty); };
}

LegalizeMutation LegalizeMutations::changeTo(unsigned TypeIdx,
                                             unsigned FromTypeIdx) {
  return [=](const LegalityQuery &Query) {
    return std::make_pair(TypeIdx, Query.Types[FromTypeIdx]);
  };
}

LegalizeMutation LegalizeMutations::widenScalarToNextPow2(unsigned TypeIdx,
                                                          unsigned Min) {
  return [=](const LegalityQuery &Query) {
    unsigned NewSizeInBits =
        1 << Log2_32_Ceil(Query.Types[TypeIdx].getSizeInBits());
    if (NewSizeInBits < Min)
      NewSizeInBits = Min;
    return std::make_pair(TypeIdx, LLT::scalar(NewSizeInBits));
  };
}

LegalizeMutation LegalizeMutations::moreElementsToNextPow2(unsigned TypeIdx,
                                                           unsigned Min) {
  return [=](const LegalityQuery &Query) {
    const LLT VecTy = Query.Types[TypeIdx];
    unsigned NewNumElements =
        std::max(1u << Log2_32_Ceil(VecTy.getNumElements()), Min);
    return std::make_pair(TypeIdx,
                          LLT::vector(NewNumElements, VecTy.getElementType()));
  };
}

LegalizeMutation LegalizeMutations::scalarize(unsigned TypeIdx) {
  return [=](const LegalityQuery &Query) {
    return std::make_pair(TypeIdx, Query.Types[TypeIdx].getElementType());
  };
}
