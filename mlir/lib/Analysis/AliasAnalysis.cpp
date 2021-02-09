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

//===----------------------------------------------------------------------===//
// AliasAnalysis
//===----------------------------------------------------------------------===//

AliasAnalysis::AliasAnalysis(Operation *op) {
  addAnalysisImplementation(LocalAliasAnalysis());
}

/// Given the two values, return their aliasing behavior.
AliasResult AliasAnalysis::alias(Value lhs, Value rhs) {
  // Check each of the alias analysis implemenations for an alias result.
  for (const std::unique_ptr<Concept> &aliasImpl : aliasImpls) {
    AliasResult result = aliasImpl->alias(lhs, rhs);
    if (!result.isMay())
      return result;
  }
  return AliasResult::MayAlias;
}
