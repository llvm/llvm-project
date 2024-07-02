//===- Flattener.cpp - Presburger ParseStruct Flattener ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Flattener class for flattening the parse tree
// produced by the parser for the Presburger library.
//
//===----------------------------------------------------------------------===//

#include "Flattener.h"
#include "ParseStructs.h"

using namespace mlir;
using namespace presburger;

void Flattener::visitDiv(const PureAffineExprImpl &div) {
  int64_t divisor = div.getDivisor();

  // First construct the linear part of the divisor.
  auto dividend = div.collectLinearTerms().getPadded(
      info.getLocalVarStartIdx() + localExprs.size() + 1);

  // Next, insert the non-linear coefficients.
  for (const auto &[hash, adjustedMulFactor, adjustedLinearTerm] :
       div.getNonLinearCoeffs()) {

    // adjustedMulFactor will be mulFactor * -divisor in case of mod, and
    // mulFactor in case of floordiv.
    dividend[lookupLocal(hash)] = adjustedMulFactor;

    // The expansion is either a modExpansion which we previously stored, or the
    // adjustedLinearTerm, which is correct in the case when we're encountering
    // the innermost mod for the first time.
    CoefficientVector expansion =
        lookupModExpansion(hash)
            .value_or(adjustedLinearTerm)
            .getPadded(info.getLocalVarStartIdx() + localExprs.size() + 1);
    dividend += expansion;

    // If this is a mod, insert the new computed expansion, which is the
    // dividend * mulFactor.
    if (div.isMod())
      localModExpansion.insert({div.hash(), dividend * div.getMulFactor()});
  }

  cst.addLocalFloorDiv(dividend, divisor);
  localExprs.insert(div.hash());
}

void Flattener::flatten(unsigned row, PureAffineExprImpl &div) {
  // Visit divs inner to outer.
  for (auto &nestedDiv : div.getNestedDivTerms())
    flatten(row, *nestedDiv);

  if (div.hasDivisor()) {
    visitDiv(div);
    return;
  }

  // Hit multiple times every time we have a linear sub-expression, but the
  // row is overwritten to consider only the outermost div, which is hit
  // last.

  // Set the linear part of the row.
  setRow(row, div.getLinearDividend());

  // Set the non-linear coefficients.
  for (const auto &[hash, adjustedMulFactor, adjustedLinearTerm] :
       div.getNonLinearCoeffs()) {
    flatMatrix(row, lookupLocal(hash)) = adjustedMulFactor;
    CoefficientVector expansion =
        lookupModExpansion(hash).value_or(adjustedLinearTerm);
    addToRow(row, expansion);
  }
}

std::pair<IntMatrix, IntegerPolyhedron> Flattener::flatten() {
  // Use the same flattener to simplify each expression successively. This way
  // local variables / expressions are shared.
  for (const auto &[row, expr] : enumerate(exprs))
    flatten(row, *expr);

  return {flatMatrix, cst};
}
