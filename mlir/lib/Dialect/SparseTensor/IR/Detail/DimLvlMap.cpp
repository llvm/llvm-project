//===- DimLvlMap.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DimLvlMap.h"

using namespace mlir;
using namespace mlir::sparse_tensor;
using namespace mlir::sparse_tensor::ir_detail;

//===----------------------------------------------------------------------===//
// `DimLvlExpr` implementation.
//===----------------------------------------------------------------------===//

SymVar DimLvlExpr::castSymVar() const {
  return SymVar(llvm::cast<AffineSymbolExpr>(expr));
}

std::optional<SymVar> DimLvlExpr::dyn_castSymVar() const {
  if (const auto s = dyn_cast_or_null<AffineSymbolExpr>(expr))
    return SymVar(s);
  return std::nullopt;
}

Var DimLvlExpr::castDimLvlVar() const {
  return Var(getAllowedVarKind(), llvm::cast<AffineDimExpr>(expr));
}

std::optional<Var> DimLvlExpr::dyn_castDimLvlVar() const {
  if (const auto x = dyn_cast_or_null<AffineDimExpr>(expr))
    return Var(getAllowedVarKind(), x);
  return std::nullopt;
}

std::tuple<DimLvlExpr, AffineExprKind, DimLvlExpr>
DimLvlExpr::unpackBinop() const {
  const auto ak = getAffineKind();
  const auto binop = llvm::dyn_cast<AffineBinaryOpExpr>(expr);
  const DimLvlExpr lhs(kind, binop ? binop.getLHS() : nullptr);
  const DimLvlExpr rhs(kind, binop ? binop.getRHS() : nullptr);
  return {lhs, ak, rhs};
}

//===----------------------------------------------------------------------===//
// `DimSpec` implementation.
//===----------------------------------------------------------------------===//

DimSpec::DimSpec(DimVar var, DimExpr expr, SparseTensorDimSliceAttr slice)
    : var(var), expr(expr), slice(slice) {}

bool DimSpec::isValid(Ranks const &ranks) const {
  // Nothing in `slice` needs additional validation.
  // We explicitly consider null-expr to be vacuously valid.
  return ranks.isValid(var) && (!expr || ranks.isValid(expr));
}

//===----------------------------------------------------------------------===//
// `LvlSpec` implementation.
//===----------------------------------------------------------------------===//

LvlSpec::LvlSpec(LvlVar var, LvlExpr expr, LevelType type)
    : var(var), expr(expr), type(type) {
  assert(expr);
  assert(isValidLT(type) && !isUndefLT(type));
}

bool LvlSpec::isValid(Ranks const &ranks) const {
  // Nothing in `type` needs additional validation.
  return ranks.isValid(var) && ranks.isValid(expr);
}

//===----------------------------------------------------------------------===//
// `DimLvlMap` implementation.
//===----------------------------------------------------------------------===//

DimLvlMap::DimLvlMap(unsigned symRank, ArrayRef<DimSpec> dimSpecs,
                     ArrayRef<LvlSpec> lvlSpecs)
    : symRank(symRank), dimSpecs(dimSpecs), lvlSpecs(lvlSpecs),
      mustPrintLvlVars(false) {
  // First, check integrity of the variable-binding structure.
  // NOTE: This establishes the invariant that calls to `VarSet::add`
  // below cannot cause OOB errors.
  assert(isWF());

  VarSet usedVars(getRanks());
  for (const auto &dimSpec : dimSpecs)
    if (!dimSpec.canElideExpr())
      usedVars.add(dimSpec.getExpr());
  for (auto &lvlSpec : this->lvlSpecs) {
    // Is this LvlVar used in any overt expression?
    const bool isUsed = usedVars.contains(lvlSpec.getBoundVar());
    // This LvlVar can be elided iff it isn't overtly used.
    lvlSpec.setElideVar(!isUsed);
    // If any LvlVar cannot be elided, then must forward-declare all LvlVars.
    mustPrintLvlVars = mustPrintLvlVars || isUsed;
  }
}

bool DimLvlMap::isWF() const {
  const auto ranks = getRanks();
  unsigned dimNum = 0;
  for (const auto &dimSpec : dimSpecs)
    if (dimSpec.getBoundVar().getNum() != dimNum++ || !dimSpec.isValid(ranks))
      return false;
  assert(dimNum == ranks.getDimRank());
  unsigned lvlNum = 0;
  for (const auto &lvlSpec : lvlSpecs)
    if (lvlSpec.getBoundVar().getNum() != lvlNum++ || !lvlSpec.isValid(ranks))
      return false;
  assert(lvlNum == ranks.getLvlRank());
  return true;
}

AffineMap DimLvlMap::getDimToLvlMap(MLIRContext *context) const {
  SmallVector<AffineExpr> lvlAffines;
  lvlAffines.reserve(getLvlRank());
  for (const auto &lvlSpec : lvlSpecs)
    lvlAffines.push_back(lvlSpec.getExpr().getAffineExpr());
  auto map = AffineMap::get(getDimRank(), getSymRank(), lvlAffines, context);
  return map;
}

AffineMap DimLvlMap::getLvlToDimMap(MLIRContext *context) const {
  SmallVector<AffineExpr> dimAffines;
  dimAffines.reserve(getDimRank());
  for (const auto &dimSpec : dimSpecs) {
    auto expr = dimSpec.getExpr().getAffineExpr();
    if (expr) {
      dimAffines.push_back(expr);
    }
  }
  auto map = AffineMap::get(getLvlRank(), getSymRank(), dimAffines, context);
  // If no lvlToDim map was passed in, returns a null AffineMap and infers it
  // in SparseTensorEncodingAttr::parse.
  if (dimAffines.empty())
    return AffineMap();
  return map;
}

//===----------------------------------------------------------------------===//
