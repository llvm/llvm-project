//===- AliasAnalysis.cpp - Alias Analysis for MLIR ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AliasResult
//===----------------------------------------------------------------------===//

/// Merge this alias result with `other` and return a new result that
/// represents the conservative merge of both results.
AliasResult AliasResult::merge(AliasResult other) const {
  if (kind == other.kind)
    return *this;
  // A mix of PartialAlias and MustAlias is PartialAlias.
  if ((isPartial() && other.isMust()) || (other.isPartial() && isMust()))
    return PartialAlias;
  // Otherwise, don't assume anything.
  return MayAlias;
}

void AliasResult::print(raw_ostream &os) const {
  switch (kind) {
  case Kind::NoAlias:
    os << "NoAlias";
    break;
  case Kind::MayAlias:
    os << "MayAlias";
    break;
  case Kind::PartialAlias:
    os << "PartialAlias";
    break;
  case Kind::MustAlias:
    os << "MustAlias";
    break;
  }
}

//===----------------------------------------------------------------------===//
// ModRefResult
//===----------------------------------------------------------------------===//

void ModRefResult::print(raw_ostream &os) const {
  switch (kind) {
  case Kind::NoModRef:
    os << "NoModRef";
    break;
  case Kind::Ref:
    os << "Ref";
    break;
  case Kind::Mod:
    os << "Mod";
    break;
  case Kind::ModRef:
    os << "ModRef";
    break;
  }
}

//===----------------------------------------------------------------------===//
// AliasAnalysis
//===----------------------------------------------------------------------===//

AliasAnalysis::AliasAnalysis(Operation *op) {
  addAnalysisImplementation(LocalAliasAnalysis());
}

AliasResult AliasAnalysis::alias(Value lhs, Value rhs) {
  // Check each of the alias analysis implemenations for an alias result.
  for (const std::unique_ptr<Concept> &aliasImpl : aliasImpls) {
    AliasResult result = aliasImpl->alias(lhs, rhs);
    if (!result.isMay())
      return result;
  }
  return AliasResult::MayAlias;
}

ModRefResult AliasAnalysis::getModRef(Operation *op, Value location) {
  // Compute the mod-ref behavior by refining a top `ModRef` result with each of
  // the alias analysis implementations. We early exit at the point where we
  // refine down to a `NoModRef`.
  ModRefResult result = ModRefResult::getModAndRef();
  for (const std::unique_ptr<Concept> &aliasImpl : aliasImpls) {
    result = result.intersect(aliasImpl->getModRef(op, location));
    if (result.isNoModRef())
      return result;
  }
  return result;
}
