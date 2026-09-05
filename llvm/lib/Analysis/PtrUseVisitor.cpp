//===- PtrUseVisitor.cpp - InstVisitors over a pointers uses --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implementation of the pointer use visitors.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

void detail::PtrUseVisitorBase::enqueueUsers(Value &I) {
  for (Use &U : I.uses()) {
    if (VisitedUses.insert(&U).second) {
      UseToVisit NewU = {UseToVisit::UseAndIsOffsetKnownPair(&U, IsOffsetKnown),
                         Offset, HighOffset};
      Worklist.push_back(std::move(NewU));
    }
  }
}

bool detail::PtrUseVisitorBase::adjustOffsetForGEP(
    GetElementPtrInst &GEPI,
    function_ref<bool(const Value &, ConstantRange &)> RangeAnalysis) {
  if (!IsOffsetKnown)
    return false;

  if (!RangeAnalysis) {
    APInt TmpOffset(DL.getIndexTypeSizeInBits(GEPI.getType()), 0);
    if (!GEPI.accumulateConstantOffset(DL, TmpOffset))
      return false;
    Offset += TmpOffset.sextOrTrunc(Offset.getBitWidth());
    HighOffset += TmpOffset.sextOrTrunc(HighOffset.getBitWidth());
    return true;
  }

  auto AccumulateBound = [&](APInt &Accum, bool IsUpperBound) {
    auto ExternalAnalysis = [&](Value &V, APInt &Index) {
      ConstantRange CR(Index.getBitWidth(), /*isFullSet=*/false);
      if (!RangeAnalysis(V, CR))
        return false;
      Index = IsUpperBound ? CR.getSignedMax() : CR.getSignedMin();
      return true;
    };
    APInt Tmp(DL.getIndexTypeSizeInBits(GEPI.getType()), 0);
    if (!GEPI.accumulateConstantOffset(DL, Tmp, ExternalAnalysis))
      return false;
    Accum += Tmp.sextOrTrunc(Accum.getBitWidth());
    return true;
  };

  return AccumulateBound(Offset, /*IsUpperBound=*/false) &&
         AccumulateBound(HighOffset, /*IsUpperBound=*/true);
}
