//===- lib/CodeGen/GlobalISel/LegalizerPredicates.cpp - Predicates --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A library of predicate factories to use for LegalityPredicate.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"

using namespace llvm;

LegalityPredicate
LegalityPredicates::all(LegalityPredicate P0, LegalityPredicate P1) {
  return [=](const LegalityQuery &Query) {
    return P0(Query) && P1(Query);
  };
}

LegalityPredicate
LegalityPredicates::typeInSet(unsigned TypeIdx,
                              std::initializer_list<LLT> TypesInit) {
  SmallVector<LLT, 4> Types = TypesInit;
  return [=](const LegalityQuery &Query) {
    return std::find(Types.begin(), Types.end(), Query.Types[TypeIdx]) != Types.end();
  };
}

LegalityPredicate LegalityPredicates::typePairInSet(
    unsigned TypeIdx0, unsigned TypeIdx1,
    std::initializer_list<std::pair<LLT, LLT>> TypesInit) {
  SmallVector<std::pair<LLT, LLT>, 4> Types = TypesInit;
  return [=](const LegalityQuery &Query) {
    std::pair<LLT, LLT> Match = {Query.Types[TypeIdx0], Query.Types[TypeIdx1]};
    return std::find(Types.begin(), Types.end(), Match) != Types.end();
  };
}

LegalityPredicate LegalityPredicates::isScalar(unsigned TypeIdx) {
  return [=](const LegalityQuery &Query) {
    return Query.Types[TypeIdx].isScalar();
  };
}

LegalityPredicate LegalityPredicates::narrowerThan(unsigned TypeIdx,
                                                   unsigned Size) {
  return [=](const LegalityQuery &Query) {
    const LLT &QueryTy = Query.Types[TypeIdx];
    return QueryTy.isScalar() && QueryTy.getSizeInBits() < Size;
  };
}

LegalityPredicate LegalityPredicates::widerThan(unsigned TypeIdx,
                                                unsigned Size) {
  return [=](const LegalityQuery &Query) {
    const LLT &QueryTy = Query.Types[TypeIdx];
    return QueryTy.isScalar() && QueryTy.getSizeInBits() > Size;
  };
}

LegalityPredicate LegalityPredicates::sizeNotPow2(unsigned TypeIdx) {
  return [=](const LegalityQuery &Query) {
    const LLT &QueryTy = Query.Types[TypeIdx];
    return QueryTy.isScalar() && !isPowerOf2_32(QueryTy.getSizeInBits());
  };
}

LegalityPredicate LegalityPredicates::numElementsNotPow2(unsigned TypeIdx) {
  return [=](const LegalityQuery &Query) {
    const LLT &QueryTy = Query.Types[TypeIdx];
    return QueryTy.isVector() && isPowerOf2_32(QueryTy.getNumElements());
  };
}
