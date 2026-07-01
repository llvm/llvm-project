//===------ FPTransformChecker.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file implements a service that helps checking conditions for
/// floating-point related IR transformations.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/FPTransformChecker.h"
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/IR/IntrinsicInst.h"

using namespace llvm;

FPTransformChecker::FPTransformChecker(const SimplifyQuery &Q)
    : FPTransformChecker(Q.CxtI) {}

void FPTransformChecker::init(const Function *Func) {
  if (Func) {
    if (Func->getAttributes().hasFnAttr(Attribute::StrictFP)) {
      U.F.StrictFP = true;
      U.F.Rounding = static_cast<unsigned>(RoundingMode::Dynamic);
      U.F.Exceptions = fp::ebStrict;
    } else {
      U.F.StrictFP = false;
      U.F.Rounding = static_cast<unsigned>(RoundingMode::NearestTiesToEven);
      U.F.Exceptions = fp::ebIgnore;
    }
  } else {
    // Conservatively assume the most restrictive case.
    U.F.StrictFP = true;
    U.F.Rounding = static_cast<unsigned>(RoundingMode::Dynamic);
    U.F.Exceptions = fp::ebStrict;
  }
}

void FPTransformChecker::with(const Instruction *I) {
  if (I) {
    if (auto *MathOp = dyn_cast<FPMathOperator>(I))
      U.F.FastMath = MathOp->getFastMathFlags().getAsOpaqueInt();
    if (const auto *CI = dyn_cast<ConstrainedFPIntrinsic>(I)) {
      U.F.StrictFP = true;
      U.F.Rounding = static_cast<unsigned>(
          CI->getRoundingMode().value_or(RoundingMode::Dynamic));
      U.F.Exceptions = CI->getExceptionBehavior().value_or(fp::ebStrict);
    }
  }
}

FPTransformChecker::FPTransformChecker(const Instruction *I) {
  U.Flags = 0;
  const Function *Func = nullptr;
  if (I) {
    if (const BasicBlock *BB = I->getParent())
      if (const Function *F = BB->getParent())
        Func = F;
  }
  init(Func);
  with(I);
}
