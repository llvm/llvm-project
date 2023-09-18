//===- DimLvlMap.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// FIXME(wrengr): The `DimLvlMap` class must be public so that it can
// be named as the storage representation of the parameter for the tblgen
// defn of STEA.  We may well need to make the other classes public too,
// so that the rest of the compiler can use them when necessary.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_DIMLVLMAP_H
#define MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_DIMLVLMAP_H

#include "Var.h"

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

namespace mlir {
namespace sparse_tensor {
namespace ir_detail {

//===----------------------------------------------------------------------===//
// TODO(wrengr): Give this enum a better name, so that it fits together
// with the name of the `DimLvlExpr` class (which may also want a better
// name).  Perhaps make this a nested-type too.
//
// NOTE: In the future we will extend this enum to include "counting
// expressions" required for supporting ITPACK/ELL.  Therefore the current
// underlying-type and representation values should not be relied upon.
enum class ExprKind : bool { Dimension = false, Level = true };

// TODO(wrengr): still needs a better name....
constexpr VarKind getVarKindAllowedInExpr(ExprKind ek) {
  using VK = std::underlying_type_t<VarKind>;
  return VarKind{2 * static_cast<VK>(!to_underlying(ek))};
}
static_assert(getVarKindAllowedInExpr(ExprKind::Dimension) == VarKind::Level &&
              getVarKindAllowedInExpr(ExprKind::Level) == VarKind::Dimension);

//===----------------------------------------------------------------------===//
// TODO(wrengr): The goal of this class is to capture a proof that
// we've verified that the given `AffineExpr` only has variables of the
// appropriate kind(s).  So we need to actually prove/verify that in the
// ctor or all its callsites!
class DimLvlExpr {
private:
  // FIXME(wrengr): Per <https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html>,
  // the `kind` field should be private and const.  However, beware
  // that if we mark any field as `const` or if the fields have differing
  // `private`/`protected` privileges then the `IsZeroCostAbstraction`
  // assertion will fail!
  // (Also, iirc, if we end up moving the `expr` to the subclasses
  // instead, that'll also cause `IsZeroCostAbstraction` to fail.)
  ExprKind kind;
  AffineExpr expr;

public:
  constexpr DimLvlExpr(ExprKind ek, AffineExpr expr) : kind(ek), expr(expr) {}

  //
  // Boolean operators.
  //
  constexpr bool operator==(DimLvlExpr other) const {
    return kind == other.kind && expr == other.expr;
  }
  constexpr bool operator!=(DimLvlExpr other) const {
    return !(*this == other);
  }
  explicit operator bool() const { return static_cast<bool>(expr); }

  //
  // RTTI support (for the `DimLvlExpr` class itself).
  //
  template <typename U>
  constexpr bool isa() const;
  template <typename U>
  constexpr U cast() const;
  template <typename U>
  constexpr U dyn_cast() const;

  //
  // Simple getters.
  //
  constexpr ExprKind getExprKind() const { return kind; }
  constexpr VarKind getAllowedVarKind() const {
    return getVarKindAllowedInExpr(kind);
  }
  constexpr AffineExpr getAffineExpr() const { return expr; }
  AffineExprKind getAffineKind() const {
    assert(expr);
    return expr.getKind();
  }
  MLIRContext *tryGetContext() const {
    return expr ? expr.getContext() : nullptr;
  }

  //
  // Getters for handling `AffineExpr` subclasses.
  //
  // TODO(wrengr): is there any way to make these typesafe without too much
  // templating?
  // TODO(wrengr): Most if not all of these don't actually need to be
  // methods, they could be free-functions instead.
  //
  Var castAnyVar() const;
  std::optional<Var> dyn_castAnyVar() const;
  SymVar castSymVar() const;
  std::optional<SymVar> dyn_castSymVar() const;
  Var castDimLvlVar() const;
  std::optional<Var> dyn_castDimLvlVar() const;
  int64_t castConstantValue() const;
  std::optional<int64_t> dyn_castConstantValue() const;
  bool hasConstantValue(int64_t val) const;
  DimLvlExpr getLHS() const;
  DimLvlExpr getRHS() const;
  std::tuple<DimLvlExpr, AffineExprKind, DimLvlExpr> unpackBinop() const;

  /// Checks whether the variables bound/used by this spec are valid
  /// with respect to the given ranks.
  [[nodiscard]] bool isValid(Ranks const &ranks) const;

  std::string str() const;
  void print(llvm::raw_ostream &os) const;
  void print(AsmPrinter &printer) const;
  void dump() const;

protected:
  // Variant of `mlir::AsmPrinter::Impl::BindingStrength`
  enum class BindingStrength : bool { Weak = false, Strong = true };

  // TODO(wrengr): Does our version of `printAffineExprInternal` really
  // need to be a method, or could it be a free-function instead? (assuming
  // `BindingStrength` goes with it).
  void printAffineExprInternal(llvm::raw_ostream &os,
                               BindingStrength enclosingTightness) const;
  void printStrong(llvm::raw_ostream &os) const {
    printAffineExprInternal(os, BindingStrength::Strong);
  }
  void printWeak(llvm::raw_ostream &os) const {
    printAffineExprInternal(os, BindingStrength::Weak);
  }
};
static_assert(IsZeroCostAbstraction<DimLvlExpr>);

// FUTURE_CL(wrengr): It would be nice to have the subclasses override
// `getRHS`, `getLHS`, `unpackBinop`, and `castDimLvlVar` to give them
// the proper covariant return types.
//
class DimExpr final : public DimLvlExpr {
  // FIXME(wrengr): These two are needed for the current RTTI implementation.
  friend class DimLvlExpr;
  constexpr explicit DimExpr(DimLvlExpr expr) : DimLvlExpr(expr) {}

public:
  static constexpr ExprKind Kind = ExprKind::Dimension;
  static constexpr bool classof(DimLvlExpr const *expr) {
    return expr->getExprKind() == Kind;
  }
  constexpr explicit DimExpr(AffineExpr expr) : DimLvlExpr(Kind, expr) {}

  LvlVar castLvlVar() const { return castDimLvlVar().cast<LvlVar>(); }
  std::optional<LvlVar> dyn_castLvlVar() const {
    const auto var = dyn_castDimLvlVar();
    return var ? std::make_optional(var->cast<LvlVar>()) : std::nullopt;
  }
};
static_assert(IsZeroCostAbstraction<DimExpr>);

class LvlExpr final : public DimLvlExpr {
  // FIXME(wrengr): These two are needed for the current RTTI implementation.
  friend class DimLvlExpr;
  constexpr explicit LvlExpr(DimLvlExpr expr) : DimLvlExpr(expr) {}

public:
  static constexpr ExprKind Kind = ExprKind::Level;
  static constexpr bool classof(DimLvlExpr const *expr) {
    return expr->getExprKind() == Kind;
  }
  constexpr explicit LvlExpr(AffineExpr expr) : DimLvlExpr(Kind, expr) {}

  DimVar castDimVar() const { return castDimLvlVar().cast<DimVar>(); }
  std::optional<DimVar> dyn_castDimVar() const {
    const auto var = dyn_castDimLvlVar();
    return var ? std::make_optional(var->cast<DimVar>()) : std::nullopt;
  }
};
static_assert(IsZeroCostAbstraction<LvlExpr>);

// FIXME(wrengr): See comments elsewhere re RTTI implementation issues/questions
template <typename U>
constexpr bool DimLvlExpr::isa() const {
  if constexpr (std::is_same_v<U, DimExpr>)
    return getExprKind() == ExprKind::Dimension;
  if constexpr (std::is_same_v<U, LvlExpr>)
    return getExprKind() == ExprKind::Level;
}

template <typename U>
constexpr U DimLvlExpr::cast() const {
  assert(isa<U>());
  return U(*this);
}

template <typename U>
constexpr U DimLvlExpr::dyn_cast() const {
  return isa<U>() ? U(*this) : U();
}

//===----------------------------------------------------------------------===//
/// The full `dimVar = dimExpr : dimSlice` specification for a given dimension.
class DimSpec final {
  /// The dimension-variable bound by this specification.
  DimVar var;
  /// The dimension-expression.  The `DimSpec` ctor treats this field
  /// as optional; whereas the `DimLvlMap` ctor will fill in (or verify)
  /// the expression via function-inversion inference.
  DimExpr expr;
  /// Can the `expr` be elided when printing? The `DimSpec` ctor assumes
  /// not (though if `expr` is null it will elide printing that); whereas
  /// the `DimLvlMap` ctor will reset it as appropriate.
  bool elideExpr = false;
  /// The dimension-slice; optional, default is null.
  SparseTensorDimSliceAttr slice;

public:
  DimSpec(DimVar var, DimExpr expr, SparseTensorDimSliceAttr slice);

  MLIRContext *tryGetContext() const { return expr.tryGetContext(); }

  constexpr DimVar getBoundVar() const { return var; }
  bool hasExpr() const { return static_cast<bool>(expr); }
  constexpr DimExpr getExpr() const { return expr; }
  void setExpr(DimExpr newExpr) {
    assert(!hasExpr());
    expr = newExpr;
  }
  constexpr bool canElideExpr() const { return elideExpr; }
  void setElideExpr(bool b) { elideExpr = b; }
  constexpr SparseTensorDimSliceAttr getSlice() const { return slice; }

  /// Checks whether the variables bound/used by this spec are valid with
  /// respect to the given ranks.  Note that null `DimExpr` is considered
  /// to be vacuously valid, and therefore calling `setExpr` invalidates
  /// the result of this predicate.
  [[nodiscard]] bool isValid(Ranks const &ranks) const;

  // TODO(wrengr): Use it or loose it.
  bool isFunctionOf(Var var) const;
  bool isFunctionOf(VarSet const &vars) const;
  void getFreeVars(VarSet &vars) const;

  std::string str(bool wantElision = true) const;
  void print(llvm::raw_ostream &os, bool wantElision = true) const;
  void print(AsmPrinter &printer, bool wantElision = true) const;
  void dump() const;
};
// Although this class is more than just a newtype/wrapper, we do want
// to ensure that storing them into `SmallVector` is efficient.
static_assert(IsZeroCostAbstraction<DimSpec>);

//===----------------------------------------------------------------------===//
/// The full `lvlVar = lvlExpr : lvlType` specification for a given level.
class LvlSpec final {
  /// The level-variable bound by this specification.
  LvlVar var;
  /// Can the `var` be elided when printing?  The `LvlSpec` ctor assumes not;
  /// whereas the `DimLvlMap` ctor will reset this as appropriate.
  bool elideVar = false;
  /// The level-expression.
  //
  // NOTE: For now we use `LvlExpr` because all level-expressions must be
  // `AffineExpr`; however, in the future we will also want to allow "counting
  // expressions", and potentially other kinds of non-affine level-expressions.
  // Which kinds of `DimLvlExpr` are allowed will depend on the `DimLevelType`,
  // so we may consider defining another class for pairing those two together
  // to ensure that the pair is well-formed.
  LvlExpr expr;
  /// The level-type (== level-format + lvl-properties).
  DimLevelType type;

public:
  LvlSpec(LvlVar var, LvlExpr expr, DimLevelType type);

  MLIRContext *getContext() const {
    MLIRContext *ctx = expr.tryGetContext();
    assert(ctx);
    return ctx;
  }

  constexpr LvlVar getBoundVar() const { return var; }
  constexpr bool canElideVar() const { return elideVar; }
  void setElideVar(bool b) { elideVar = b; }
  constexpr LvlExpr getExpr() const { return expr; }
  constexpr DimLevelType getType() const { return type; }

  /// Checks whether the variables bound/used by this spec are valid
  /// with respect to the given ranks.
  //
  // NOTE: Once we introduce "counting expressions" this will need
  // a more sophisticated implementation than `DimSpec::isValid` does.
  [[nodiscard]] bool isValid(Ranks const &ranks) const;

  // TODO(wrengr): Use it or loose it.
  bool isFunctionOf(Var var) const;
  bool isFunctionOf(VarSet const &vars) const;
  void getFreeVars(VarSet &vars) const;

  std::string str(bool wantElision = true) const;
  void print(llvm::raw_ostream &os, bool wantElision = true) const;
  void print(AsmPrinter &printer, bool wantElision = true) const;
  void dump() const;
};
// Although this class is more than just a newtype/wrapper, we do want
// to ensure that storing them into `SmallVector` is efficient.
static_assert(IsZeroCostAbstraction<LvlSpec>);

//===----------------------------------------------------------------------===//
class DimLvlMap final {
public:
  DimLvlMap(unsigned symRank, ArrayRef<DimSpec> dimSpecs,
            ArrayRef<LvlSpec> lvlSpecs);

  unsigned getSymRank() const { return symRank; }
  unsigned getDimRank() const { return dimSpecs.size(); }
  unsigned getLvlRank() const { return lvlSpecs.size(); }
  unsigned getRank(VarKind vk) const { return getRanks().getRank(vk); }
  Ranks getRanks() const { return {getSymRank(), getDimRank(), getLvlRank()}; }

  ArrayRef<DimSpec> getDims() const { return dimSpecs; }
  const DimSpec &getDim(Dimension dim) const { return dimSpecs[dim]; }
  SparseTensorDimSliceAttr getDimSlice(Dimension dim) const {
    return getDim(dim).getSlice();
  }

  ArrayRef<LvlSpec> getLvls() const { return lvlSpecs; }
  const LvlSpec &getLvl(Level lvl) const { return lvlSpecs[lvl]; }
  DimLevelType getLvlType(Level lvl) const { return getLvl(lvl).getType(); }

  AffineMap getDimToLvlMap(MLIRContext *context) const;
  AffineMap getLvlToDimMap(MLIRContext *context) const;

  std::string str(bool wantElision = true) const;
  void print(llvm::raw_ostream &os, bool wantElision = true) const;
  void print(AsmPrinter &printer, bool wantElision = true) const;
  void dump() const;

private:
  /// Checks for integrity of variable-binding structure.
  /// This is already called by the ctor.
  [[nodiscard]] bool isWF() const;

  /// Helper function to call `DimSpec::setExpr` while asserting that
  /// the invariant established by `DimLvlMap:isWF` is maintained.
  /// This is used by the ctor.
  void setDimExpr(Dimension dim, DimExpr expr) {
    assert(expr && getRanks().isValid(expr));
    dimSpecs[dim].setExpr(expr);
  }

  // All these fields are const-after-ctor.
  unsigned symRank;
  SmallVector<DimSpec> dimSpecs;
  SmallVector<LvlSpec> lvlSpecs;
  bool mustPrintLvlVars;
};

//===----------------------------------------------------------------------===//

} // namespace ir_detail
} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_DIMLVLMAP_H
