//===- Assumptions.cpp ------ Collection of helpers for assumptions -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Assumptions.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"

using namespace llvm;

bool llvm::hasAssumption(Function &F,
                         const KnownAssumptionString &AssumptionStr) {
  const Attribute &A = F.getFnAttribute(AssumptionAttrKey);
  if (!A.isValid())
    return false;
  assert(A.isStringAttribute() && "Expected a string attribute!");

  SmallVector<StringRef, 8> Strings;
  A.getValueAsString().split(Strings, ",");

  return llvm::any_of(Strings, [=](StringRef Assumption) {
    return Assumption == AssumptionStr;
  });
}

StringSet<> llvm::KnownAssumptionStrings({
    "omp_no_openmp",          // OpenMP 5.1
    "omp_no_openmp_routines", // OpenMP 5.1
    "omp_no_parallelism",     // OpenMP 5.1
    "ompx_spmd_amenable",     // OpenMPOpt extension
});
