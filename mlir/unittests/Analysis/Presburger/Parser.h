//===- Parser.h - Parser for Presburger library -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines functions to parse strings into Presburger library
// constructs.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_UNITTESTS_ANALYSIS_PRESBURGER_PARSER_H
#define MLIR_UNITTESTS_ANALYSIS_PRESBURGER_PARSER_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"

namespace mlir {
namespace presburger {

/// Parses an IntegerPolyhedron from a StringRef. It is expected that the string
/// represents a valid IntegerSet.
inline IntegerPolyhedron parseIntegerPolyhedron(StringRef str) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  return affine::FlatAffineValueConstraints(parseIntegerSet(str, &context));
}

/// Parse a list of StringRefs to IntegerRelation and combine them into a
/// PresburgerSet by using the union operation. It is expected that the strings
/// are all valid IntegerSet representation and that all of them have compatible
/// spaces.
inline PresburgerSet parsePresburgerSet(ArrayRef<StringRef> strs) {
  assert(!strs.empty() && "strs should not be empty");

  IntegerPolyhedron initPoly = parseIntegerPolyhedron(strs[0]);
  PresburgerSet result(initPoly);
  for (unsigned i = 1, e = strs.size(); i < e; ++i)
    result.unionInPlace(parseIntegerPolyhedron(strs[i]));
  return result;
}

inline MultiAffineFunction parseMultiAffineFunction(StringRef str) {
  MLIRContext context(MLIRContext::Threading::DISABLED);

  // TODO: Add default constructor for MultiAffineFunction.
  MultiAffineFunction multiAff(PresburgerSpace::getRelationSpace(),
                               Matrix(0, 1));
  if (getMultiAffineFunctionFromMap(parseAffineMap(str, &context), multiAff)
          .failed())
    llvm_unreachable(
        "Failed to parse MultiAffineFunction because of semi-affinity");
  return multiAff;
}

inline PWMAFunction
parsePWMAF(ArrayRef<std::pair<StringRef, StringRef>> pieces) {
  assert(!pieces.empty() && "At least one piece should be present.");

  MLIRContext context(MLIRContext::Threading::DISABLED);

  IntegerPolyhedron initDomain = parseIntegerPolyhedron(pieces[0].first);
  MultiAffineFunction initMultiAff = parseMultiAffineFunction(pieces[0].second);

  PWMAFunction func(PresburgerSpace::getRelationSpace(
      initMultiAff.getNumDomainVars(), initMultiAff.getNumOutputs(),
      initMultiAff.getNumSymbolVars()));

  func.addPiece({PresburgerSet(initDomain), initMultiAff});
  for (unsigned i = 1, e = pieces.size(); i < e; ++i)
    func.addPiece({PresburgerSet(parseIntegerPolyhedron(pieces[i].first)),
                   parseMultiAffineFunction(pieces[i].second)});
  return func;
}

inline IntegerRelation parseRelationFromSet(StringRef set, unsigned numDomain) {
  IntegerRelation rel = parseIntegerPolyhedron(set);

  rel.convertVarKind(VarKind::SetDim, 0, numDomain, VarKind::Domain);

  return rel;
}

inline PresburgerRelation
parsePresburgerRelationFromPresburgerSet(ArrayRef<StringRef> strs,
                                         unsigned numDomain) {
  assert(!strs.empty() && "strs should not be empty");

  IntegerRelation rel = parseIntegerPolyhedron(strs[0]);
  PresburgerRelation result(rel);
  for (unsigned i = 1, e = strs.size(); i < e; ++i)
    result.unionInPlace(parseIntegerPolyhedron(strs[i]));
  result.convertVarKind(VarKind::SetDim, 0, numDomain, VarKind::Domain, 0);
  return result;
}

} // namespace presburger
} // namespace mlir

#endif // MLIR_UNITTESTS_ANALYSIS_PRESBURGER_PARSER_H
