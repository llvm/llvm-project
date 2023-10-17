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

Var DimLvlExpr::castAnyVar() const {
  assert(expr && "uninitialized DimLvlExpr");
  const auto var = dyn_castAnyVar();
  assert(var && "expected DimLvlExpr to be a Var");
  return *var;
}

std::optional<Var> DimLvlExpr::dyn_castAnyVar() const {
  if (const auto s = expr.dyn_cast_or_null<AffineSymbolExpr>())
    return SymVar(s);
  if (const auto x = expr.dyn_cast_or_null<AffineDimExpr>())
    return Var(getAllowedVarKind(), x);
  return std::nullopt;
}

SymVar DimLvlExpr::castSymVar() const {
  return SymVar(expr.cast<AffineSymbolExpr>());
}

std::optional<SymVar> DimLvlExpr::dyn_castSymVar() const {
  if (const auto s = expr.dyn_cast_or_null<AffineSymbolExpr>())
    return SymVar(s);
  return std::nullopt;
}

Var DimLvlExpr::castDimLvlVar() const {
  return Var(getAllowedVarKind(), expr.cast<AffineDimExpr>());
}

std::optional<Var> DimLvlExpr::dyn_castDimLvlVar() const {
  if (const auto x = expr.dyn_cast_or_null<AffineDimExpr>())
    return Var(getAllowedVarKind(), x);
  return std::nullopt;
}

int64_t DimLvlExpr::castConstantValue() const {
  return expr.cast<AffineConstantExpr>().getValue();
}

std::optional<int64_t> DimLvlExpr::dyn_castConstantValue() const {
  const auto k = expr.dyn_cast_or_null<AffineConstantExpr>();
  return k ? std::make_optional(k.getValue()) : std::nullopt;
}

// This helper method is akin to `AffineExpr::operator==(int64_t)`
// except it uses a different implementation, namely the implementation
// used within `AsmPrinter::Impl::printAffineExprInternal`.
//
// wrengr guesses that `AsmPrinter::Impl::printAffineExprInternal` uses
// this implementation because it avoids constructing the intermediate
// `AffineConstantExpr(val)` and thus should in theory be a bit faster.
// However, if it is indeed faster, then the `AffineExpr::operator==`
// method should be updated to do this instead.  And if it isn't any
// faster, then we should be using `AffineExpr::operator==` instead.
bool DimLvlExpr::hasConstantValue(int64_t val) const {
  const auto k = expr.dyn_cast_or_null<AffineConstantExpr>();
  return k && k.getValue() == val;
}

DimLvlExpr DimLvlExpr::getLHS() const {
  const auto binop = expr.dyn_cast_or_null<AffineBinaryOpExpr>();
  return DimLvlExpr(kind, binop ? binop.getLHS() : nullptr);
}

DimLvlExpr DimLvlExpr::getRHS() const {
  const auto binop = expr.dyn_cast_or_null<AffineBinaryOpExpr>();
  return DimLvlExpr(kind, binop ? binop.getRHS() : nullptr);
}

std::tuple<DimLvlExpr, AffineExprKind, DimLvlExpr>
DimLvlExpr::unpackBinop() const {
  const auto ak = getAffineKind();
  const auto binop = expr.dyn_cast<AffineBinaryOpExpr>();
  const DimLvlExpr lhs(kind, binop ? binop.getLHS() : nullptr);
  const DimLvlExpr rhs(kind, binop ? binop.getRHS() : nullptr);
  return {lhs, ak, rhs};
}

void DimLvlExpr::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}
std::string DimLvlExpr::str() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}
void DimLvlExpr::print(AsmPrinter &printer) const {
  print(printer.getStream());
}
void DimLvlExpr::print(llvm::raw_ostream &os) const {
  if (!expr)
    os << "<<NULL AFFINE EXPR>>";
  else
    printWeak(os);
}

namespace {
struct MatchNeg final : public std::pair<DimLvlExpr, int64_t> {
  using Base = std::pair<DimLvlExpr, int64_t>;
  using Base::Base;
  constexpr DimLvlExpr getLHS() const { return first; }
  constexpr int64_t getRHS() const { return second; }
};
} // namespace

static std::optional<MatchNeg> matchNeg(DimLvlExpr expr) {
  const auto [lhs, op, rhs] = expr.unpackBinop();
  if (op == AffineExprKind::Constant) {
    const auto val = expr.castConstantValue();
    if (val < 0)
      return MatchNeg{DimLvlExpr{expr.getExprKind(), AffineExpr()}, val};
  }
  if (op == AffineExprKind::Mul)
    if (const auto rval = rhs.dyn_castConstantValue(); rval && *rval < 0)
      return MatchNeg{lhs, *rval};
  return std::nullopt;
}

// A heavily revised version of `AsmPrinter::Impl::printAffineExprInternal`.
void DimLvlExpr::printAffineExprInternal(
    llvm::raw_ostream &os, BindingStrength enclosingTightness) const {
  const char *binopSpelling = nullptr;
  switch (getAffineKind()) {
  case AffineExprKind::SymbolId:
    os << castSymVar();
    return;
  case AffineExprKind::DimId:
    os << castDimLvlVar();
    return;
  case AffineExprKind::Constant:
    os << castConstantValue();
    return;
  case AffineExprKind::Add:
    binopSpelling = " + "; // N.B., this is unused
    break;
  case AffineExprKind::Mul:
    binopSpelling = " * ";
    break;
  case AffineExprKind::FloorDiv:
    binopSpelling = " floordiv ";
    break;
  case AffineExprKind::CeilDiv:
    binopSpelling = " ceildiv ";
    break;
  case AffineExprKind::Mod:
    binopSpelling = " mod ";
    break;
  }

  if (enclosingTightness == BindingStrength::Strong)
    os << '(';

  const auto [lhs, op, rhs] = unpackBinop();
  if (op == AffineExprKind::Mul && rhs.hasConstantValue(-1)) {
    // Pretty print `(lhs * -1)` as "-lhs".
    os << '-';
    lhs.printStrong(os);
  } else if (op != AffineExprKind::Add) {
    // Default rule for tightly binding binary operators.
    // (Including `Mul` that didn't match the previous rule.)
    lhs.printStrong(os);
    os << binopSpelling;
    rhs.printStrong(os);
  } else {
    // Combination of all the special rules for addition/subtraction.
    lhs.printWeak(os);
    const auto rx = matchNeg(rhs);
    os << (rx ? " - " : " + ");
    const auto &rlhs = rx ? rx->getLHS() : rhs;
    const auto rrhs = rx ? rx->getRHS() : -1; // value irrelevant when `!rx`
    const bool nonunit = rrhs != -1;          // value irrelevant when `!rx`
    const bool isStrong =
        rx && rlhs && (nonunit || rlhs.getAffineKind() == AffineExprKind::Add);
    if (rlhs)
      rlhs.printAffineExprInternal(os, BindingStrength{isStrong});
    if (rx && rlhs && nonunit)
      os << " * ";
    if (rx && (!rlhs || nonunit))
      os << -rrhs;
  }

  if (enclosingTightness == BindingStrength::Strong)
    os << ')';
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

bool DimSpec::isFunctionOf(VarSet const &vars) const {
  return vars.occursIn(expr);
}

void DimSpec::getFreeVars(VarSet &vars) const { vars.add(expr); }

void DimSpec::dump() const {
  print(llvm::errs(), /*wantElision=*/false);
  llvm::errs() << "\n";
}
std::string DimSpec::str(bool wantElision) const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os, wantElision);
  return os.str();
}
void DimSpec::print(AsmPrinter &printer, bool wantElision) const {
  print(printer.getStream(), wantElision);
}
void DimSpec::print(llvm::raw_ostream &os, bool wantElision) const {
  os << var;
  if (expr && (!wantElision || !elideExpr))
    os << " = " << expr;
  if (slice) {
    os << " : ";
    // Call `SparseTensorDimSliceAttr::print` directly, to avoid
    // printing the mnemonic.
    slice.print(os);
  }
}

//===----------------------------------------------------------------------===//
// `LvlSpec` implementation.
//===----------------------------------------------------------------------===//

LvlSpec::LvlSpec(LvlVar var, LvlExpr expr, DimLevelType type)
    : var(var), expr(expr), type(type) {
  assert(expr);
  assert(isValidDLT(type) && !isUndefDLT(type));
}

bool LvlSpec::isValid(Ranks const &ranks) const {
  // Nothing in `type` needs additional validation.
  return ranks.isValid(var) && ranks.isValid(expr);
}

bool LvlSpec::isFunctionOf(VarSet const &vars) const {
  return vars.occursIn(expr);
}

void LvlSpec::getFreeVars(VarSet &vars) const { vars.add(expr); }

void LvlSpec::dump() const {
  print(llvm::errs(), /*wantElision=*/false);
  llvm::errs() << "\n";
}
std::string LvlSpec::str(bool wantElision) const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os, wantElision);
  return os.str();
}
void LvlSpec::print(AsmPrinter &printer, bool wantElision) const {
  print(printer.getStream(), wantElision);
}
void LvlSpec::print(llvm::raw_ostream &os, bool wantElision) const {
  if (!wantElision || !elideVar)
    os << var << " = ";
  os << expr;
  os << ": " << toMLIRString(type);
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

  // TODO: Second, we need to infer/validate the `lvlToDim` mapping.
  // Along the way we should set every `DimSpec::elideExpr` according
  // to whether the given expression is inferable or not.  Notably, this
  // needs to happen before the code for setting every `LvlSpec::elideVar`,
  // since if the LvlVar is only used in elided DimExpr, then the
  // LvlVar should also be elided.
  // NOTE: Be sure to use `DimLvlMap::setDimExpr` for setting the new exprs,
  // to ensure that we maintain the invariant established by `isWF` above.

  // Third, we set every `LvlSpec::elideVar` according to whether that
  // LvlVar occurs in a non-elided DimExpr (TODO: or CountingExpr).
  // NOTE: The invariant established by `isWF` ensures that the following
  // calls to `VarSet::add` cannot raise OOB errors.
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
  auto map =  AffineMap::get(getDimRank(), getSymRank(), lvlAffines, context);
  if (map.isIdentity()) return AffineMap();
  return map;
}

AffineMap DimLvlMap::getLvlToDimMap(MLIRContext *context) const {
  SmallVector<AffineExpr> dimAffines;
  dimAffines.reserve(getDimRank());
  for (const auto &dimSpec : dimSpecs) {
    auto expr = dimSpec.getExpr().getAffineExpr();
    if (expr) {
      auto exprTuple = dimSpec.getExpr().unpackBinop();
      if (std::get<1>(exprTuple) == AffineExprKind::FloorDiv ||
          std::get<1>(exprTuple) == AffineExprKind::Mod) {
        dimAffines.push_back(expr);
      }
    }
  }
  auto map = AffineMap::get(getLvlRank(), getSymRank(), dimAffines, context);
  if (dimAffines.empty() || map.isIdentity()) return AffineMap();
  return map;
}

void DimLvlMap::dump() const {
  print(llvm::errs(), /*wantElision=*/false);
  llvm::errs() << "\n";
}
std::string DimLvlMap::str(bool wantElision) const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os, wantElision);
  return os.str();
}
void DimLvlMap::print(AsmPrinter &printer, bool wantElision) const {
  print(printer.getStream(), wantElision);
}
void DimLvlMap::print(llvm::raw_ostream &os, bool wantElision) const {
  // Symbolic identifiers.
  // NOTE: Unlike `AffineMap` we place the SymVar bindings before the DimVar
  // bindings, since the SymVars may occur within DimExprs and thus this
  // ordering helps reduce potential user confusion about the scope of bidings
  // (since it means SymVars and DimVars both bind-forward in the usual way,
  // whereas only LvlVars have different binding rules).
  if (symRank != 0) {
    os << "[s0";
    for (unsigned i = 1; i < symRank; ++i)
      os << ", s" << i;
    os << ']';
  }

  // LvlVar forward-declarations.
  if (mustPrintLvlVars) {
    os << '{';
    llvm::interleaveComma(
        lvlSpecs, os, [&](LvlSpec const &spec) { os << spec.getBoundVar(); });
    os << "} ";
  }

  // Dimension specifiers.
  os << '(';
  llvm::interleaveComma(
      dimSpecs, os, [&](DimSpec const &spec) { spec.print(os, wantElision); });
  os << ") -> (";
  // Level specifiers.
  wantElision = wantElision && !mustPrintLvlVars;
  llvm::interleaveComma(
      lvlSpecs, os, [&](LvlSpec const &spec) { spec.print(os, wantElision); });
  os << ')';
}

//===----------------------------------------------------------------------===//
