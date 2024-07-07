//===- ParseStructs.h - Presburger Parse Structrures ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PARSER_PARSESTRUCTS_H
#define MLIR_ANALYSIS_PRESBURGER_PARSER_PARSESTRUCTS_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace mlir::presburger {
using llvm::ArrayRef;
using llvm::SmallVector;
using llvm::SmallVectorImpl;

/// Helper for storing coefficients in canonical form: dims followed by symbols,
/// followed by locals, and finally the constant term.
///
/// (x, y)[a, b]: y * 91 + x + 3 * a + 7
/// coefficients: [1, 91, 3, 0, 7]
struct CoefficientVector {
  PresburgerSpace space;
  SmallVector<int64_t, 8> coefficients;

  CoefficientVector(const PresburgerSpace &space, int64_t c = 0)
      : space(space) {
    coefficients.resize(space.getNumCols());
    coefficients[space.getConstantIdx()] = c;
  }

  // Copyable and movable
  CoefficientVector(const CoefficientVector &o) = default;
  CoefficientVector &operator=(const CoefficientVector &o) = default;
  CoefficientVector(CoefficientVector &&o)
      : space(o.space), coefficients(std::move(o.coefficients)) {
    o.coefficients.clear();
  }

  ArrayRef<int64_t> getCoefficients() const { return coefficients; }
  int64_t getConstant() const { return coefficients[space.getConstantIdx()]; }
  size_t size() const { return coefficients.size(); }
  operator ArrayRef<int64_t>() const { return coefficients; }
  void resize(size_t size) { coefficients.resize(size); }
  operator bool() const {
    return any_of(coefficients, [](int64_t c) { return c; });
  }
  int64_t &operator[](unsigned i) {
    assert(i < coefficients.size());
    return coefficients[i];
  }
  int64_t &back() { return coefficients.back(); }
  int64_t back() const { return coefficients.back(); }
  void clear() {
    for_each(coefficients, [](auto &coeff) { coeff = 0; });
  }

  CoefficientVector &operator+=(const CoefficientVector &l) {
    coefficients.resize(l.size());
    for (auto [idx, c] : enumerate(l.getCoefficients()))
      coefficients[idx] += c;
    return *this;
  }
  CoefficientVector &operator*=(int64_t c) {
    for_each(coefficients, [c](auto &coeff) { coeff *= c; });
    return *this;
  }
  CoefficientVector &operator/=(int64_t c) {
    assert(c && "Division by zero");
    for_each(coefficients, [c](auto &coeff) { coeff /= c; });
    return *this;
  }

  CoefficientVector operator+(const CoefficientVector &l) const {
    CoefficientVector ret(*this);
    return ret += l;
  }
  CoefficientVector operator*(int64_t c) const {
    CoefficientVector ret(*this);
    return ret *= c;
  }
  CoefficientVector operator/(int64_t c) const {
    CoefficientVector ret(*this);
    return ret /= c;
  }

  bool isConstant() const {
    return all_of(drop_end(coefficients), [](int64_t c) { return !c; });
  }
  CoefficientVector getPadded(size_t newSize) const {
    assert(newSize >= size() &&
           "Padding size should be greater than expr size");
    CoefficientVector ret(space);
    ret.resize(newSize);

    // Start constructing the result by taking the dims and symbols of the
    // coefficients.
    for (const auto &[col, coeff] : enumerate(drop_end(coefficients)))
      ret[col] = coeff;

    // Put the constant at the end.
    ret.back() = back();
    return ret;
  }

  uint64_t factorMulFromLinearTerm() const {
    uint64_t gcd = 1;
    for (int64_t val : coefficients)
      gcd = std::gcd(gcd, std::abs(val));
    return gcd;
  }
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  bool hasMultipleCoefficients() const {
    return count_if(coefficients, [](auto &coeff) { return coeff; }) > 1;
  }
  LLVM_DUMP_METHOD void dump() const;
#endif
};

enum class DimOrSymbolKind {
  DimId,
  Symbol,
};

using DimOrSymbolExpr = std::pair<DimOrSymbolKind, unsigned>;
enum class DivKind { FloorDiv, Mod };

/// Represents a pure Affine expression. Linear expressions are represented with
/// divisor = 1, and no nestedDivTerms.
///
/// 3 - a * (3 + (3 - x div 3) div 4 + y div 7) div 4
/// ^ linearDivident = 3, mulFactor = 1, divisor = 1
///   ^ nest: 1, mulFactor: -a
///         ^ nest: 1, linearDividend
///              ^ nest: 2, linearDividend
///                     ^ nest: 3
///                          ^ nest: 2
///                                      ^ nest: 2
///                               nest: 1, divisor ^
///
/// Where div = floordiv|mod; ceildiv is pre-reduced
struct PureAffineExprImpl {
  PresburgerSpace space;
  DivKind kind = DivKind::FloorDiv;
  using PureAffineExpr = std::unique_ptr<PureAffineExprImpl>;

  int64_t mulFactor = 1;
  CoefficientVector linearDividend;
  int64_t divisor = 1;
  SmallVector<PureAffineExpr, 4> nestedDivTerms;

  PureAffineExprImpl(const PresburgerSpace &space, int64_t c = 0)
      : space(space), linearDividend(space, c) {}
  PureAffineExprImpl(const PresburgerSpace &space, DimOrSymbolExpr idExpr)
      : PureAffineExprImpl(space) {
    auto [kind, pos] = idExpr;
    unsigned startIdx = kind == DimOrSymbolKind::Symbol
                            ? space.getSymbolStartIdx()
                            : space.getSetDimStartIdx();
    linearDividend.coefficients[startIdx + pos] = 1;
  }
  PureAffineExprImpl(const CoefficientVector &linearDividend,
                     int64_t divisor = 1, DivKind kind = DivKind::FloorDiv)
      : space(linearDividend.space), kind(kind), linearDividend(linearDividend),
        divisor(divisor) {}
  PureAffineExprImpl(PureAffineExprImpl &&div, int64_t divisor, DivKind kind)
      : space(div.space), kind(kind), linearDividend(div.space),
        divisor(divisor) {
    addDivTerm(std::move(div));
  }

  // Non-copyable, only movable
  PureAffineExprImpl(const PureAffineExprImpl &) = delete;
  PureAffineExprImpl(PureAffineExprImpl &&o)
      : space(o.space), kind(o.kind), mulFactor(o.mulFactor),
        linearDividend(std::move(o.linearDividend)), divisor(o.divisor),
        nestedDivTerms(std::move(o.nestedDivTerms)) {
    o.nestedDivTerms.clear();
    o.divisor = o.mulFactor = 1;
  }

  const CoefficientVector &getLinearDividend() const { return linearDividend; }
  CoefficientVector collectLinearTerms() const;
  SmallVector<std::tuple<size_t, int64_t, CoefficientVector>, 8>
  getNonLinearCoeffs() const;

  constexpr bool isMod() const { return kind == DivKind::Mod; }
  constexpr bool hasDivisor() const { return divisor != 1; }
  bool isLinear() const { return nestedDivTerms.empty() && !hasDivisor(); }
  constexpr int64_t getDivisor() const { return divisor; }
  constexpr int64_t getMulFactor() const { return mulFactor; }
  bool isConstant() const {
    return nestedDivTerms.empty() && getLinearDividend().isConstant();
  }
  int64_t getConstant() const { return getLinearDividend().getConstant(); }
  size_t hash() const { return std::hash<const PureAffineExprImpl *>{}(this); }
  ArrayRef<PureAffineExpr> getNestedDivTerms() const { return nestedDivTerms; }

  PureAffineExprImpl &mulConstant(int64_t c) {
    // Canonicalize mulFactors in div terms without divisors.
    mulFactor *= c;
    distributeMulFactor();
    return *this;
  }
  PureAffineExprImpl &addLinearTerm(const CoefficientVector &l) {
    if (hasDivisor())
      nestedDivTerms.emplace_back(
          std::make_unique<PureAffineExprImpl>(std::move(*this)));
    linearDividend += l;
    return *this;
  }
  PureAffineExprImpl &addDivTerm(PureAffineExprImpl &&d) {
    nestedDivTerms.emplace_back(
        std::make_unique<PureAffineExprImpl>(std::move(d)));
    return *this;
  }
  PureAffineExprImpl &divLinearDividend(int64_t c) {
    linearDividend /= c;
    return *this;
  }

  void distributeMulFactor() {
    // Canonicalize the -1 mulFactor for divs without divisor.
    if (mulFactor != -1 || hasDivisor())
      return;
    linearDividend *= -1;
    for_each(nestedDivTerms,
             [](const PureAffineExpr &div) { div->mulFactor *= -1; });
    mulFactor *= -1;
  }
  unsigned countNestedDivs() const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif
};

using PureAffineExpr = std::unique_ptr<PureAffineExprImpl>;

/// This structure holds the final parse result, and is constructed
/// non-trivially to compute the total number of divs, which is then used to
/// compute the total number of columns, and construct the IntMatrix in the
/// Flattener.
struct FinalParseResult {
  PresburgerSpace space;
  SmallVector<PureAffineExpr, 8> exprs;
  SmallVector<bool, 8> eqFlags;
  IntegerPolyhedron cst;

  FinalParseResult(const PresburgerSpace &space,
                   SmallVectorImpl<PureAffineExpr> &&exprStack,
                   ArrayRef<bool> eqFlagStack)
      : space(space), exprs(std::move(exprStack)), eqFlags(eqFlagStack),
        cst(0, 0, space.getNumSetDimVars() + space.getNumSymbolVars() + 1,
            PresburgerSpace::getSetSpace(space.getNumSetDimVars(),
                                         space.getNumSymbolVars(), 0)) {
    this->space.numLocalVars() =
        std::accumulate(exprs.begin(), exprs.end(), 0,
                        [](unsigned acc, const PureAffineExpr &expr) {
                          return acc + expr->countNestedDivs();
                        });
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif
};

// The main interface to the parser is a bunch of operators, which is used to
// successively build the final AffineExpr.
PureAffineExpr operator+(PureAffineExpr &&lhs, PureAffineExpr &&rhs);
PureAffineExpr operator*(PureAffineExpr &&expr, int64_t c);
PureAffineExpr operator+(PureAffineExpr &&expr, int64_t c);
PureAffineExpr div(PureAffineExpr &&dividend, int64_t divisor, DivKind kind);
inline PureAffineExpr floordiv(PureAffineExpr &&expr, int64_t c) {
  return div(std::move(expr), c, DivKind::FloorDiv);
}
inline PureAffineExpr operator%(PureAffineExpr &&expr, int64_t c) {
  return div(std::move(expr), c, DivKind::Mod);
}
inline PureAffineExpr operator*(int64_t c, PureAffineExpr &&expr) {
  return std::move(expr) * c;
}
inline PureAffineExpr operator-(PureAffineExpr &&lhs, PureAffineExpr &&rhs) {
  return std::move(lhs) + std::move(rhs) * -1;
}
inline PureAffineExpr operator+(int64_t c, PureAffineExpr &&expr) {
  return std::move(expr) + c;
}
inline PureAffineExpr operator-(PureAffineExpr &&expr, int64_t c) {
  return std::move(expr) + (-c);
}
inline PureAffineExpr operator-(int64_t c, PureAffineExpr &&expr) {
  return -1 * std::move(expr) + c;
}
inline PureAffineExpr ceildiv(PureAffineExpr &&expr, int64_t c) {
  // expr ceildiv c <=> (expr + c - 1) floordiv c
  return floordiv(std::move(expr) + (c - 1), c);
}

// Our final canonical expression, the outermost div, should have a divisor
// of 1.
inline PureAffineExpr canonicalize(PureAffineExpr &&expr) {
  if (expr->hasDivisor())
    expr->addLinearTerm(CoefficientVector(expr->space));
  return expr;
}
} // namespace mlir::presburger

#endif // MLIR_ANALYSIS_PRESBURGER_PARSER_PARSESTRUCTS_H
