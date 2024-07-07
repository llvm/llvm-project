//===- Flattener.h - Presburger ParseStruct Flattener -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PARSER_FLATTENER_H
#define MLIR_ANALYSIS_PRESBURGER_PARSER_FLATTENER_H

#include "ParseStructs.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::presburger {
using llvm::SmallDenseMap;
using llvm::SmallSetVector;

class Flattener : public FinalParseResult {
public:
  // The final flattened result is stored here.
  IntMatrix flatMatrix;

  // We maintain a set of divs that we have seen while flattening. The size of
  // this set is at most info.numDivs, hitting info.numDivs at the end of the
  // flattening, if that expression contains all the possible divs.
  SmallSetVector<size_t, 4> localExprs;

  // We maintain a mapping between local mods and their expansions. The vector
  // is the dividend.
  SmallDenseMap<size_t, CoefficientVector, 2> localModExpansion;

  Flattener(FinalParseResult &&parseResult)
      : FinalParseResult(std::move(parseResult)),
        flatMatrix(exprs.size(), space.getNumCols()) {}

  std::pair<IntMatrix, IntegerPolyhedron> flatten();

private:
  void flatten(unsigned row, PureAffineExprImpl &div);
  void visitDiv(const PureAffineExprImpl &div);

  void addToRow(unsigned row, const CoefficientVector &l) {
    flatMatrix.addToRow(row,
                        getDynamicAPIntVec(l.getPadded(space.getNumCols())));
  }
  void setRow(unsigned row, const CoefficientVector &l) {
    flatMatrix.setRow(row, getDynamicAPIntVec(l.getPadded(space.getNumCols())));
  }

  unsigned lookupLocal(size_t hash) {
    const auto *it = find(localExprs, hash);
    assert(it != localExprs.end() &&
           "Local expression not found; walking from inner to outer?");
    return space.getLocalVarStartIdx() + it - localExprs.begin();
  }
  std::optional<CoefficientVector> lookupModExpansion(size_t hash) {
    return localModExpansion.contains(hash)
               ? std::make_optional(localModExpansion.at(hash))
               : std::nullopt;
  }
};
} // namespace mlir::presburger

#endif // MLIR_ANALYSIS_PRESBURGER_PARSER_FLATTENER_H
