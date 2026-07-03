//===- InferIntDivisibilityOpInterfaceImpl.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Direct implementations of `InferIntDivisibilityOpInterface` for affine ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/InferIntDivisibilityOpInterface.h"

#include <cstdlib>
#include <numeric>

using namespace mlir;
using namespace mlir::affine;

namespace {

static ConstantIntDivisibility
getDivisibilityOfOperand(Value v, IntegerDivisibility divisibility) {
  if (!divisibility.isUninitialized())
    return divisibility.getValue();
  APInt intVal;
  if (matchPattern(v, m_ConstantInt(&intVal))) {
    uint64_t udiv = intVal.getZExtValue();
    uint64_t sdiv = std::abs(intVal.getSExtValue());
    return ConstantIntDivisibility(udiv, sdiv);
  }
  return ConstantIntDivisibility(1, 1);
}

/// Visits affine expressions and recursively calculates the divisibilities of
/// each subexpression. The final divisibilities of the expression and its
/// subexpressions will be stored in the map for which a reference is provided
/// to the AffineExprDivisibilityFinder (i.e., `divisibilityMap`).
class AffineExprDivisibilityFinder
    : public AffineExprVisitor<AffineExprDivisibilityFinder,
                               ConstantIntDivisibility> {
public:
  using ExprDivisibilityMap =
      llvm::DenseMap<AffineExpr, ConstantIntDivisibility>;
  AffineExprDivisibilityFinder(ExprDivisibilityMap &divisibilityMap)
      : divisibilityMap(divisibilityMap) {}

  ConstantIntDivisibility visitConstantExpr(AffineConstantExpr expr) {
    // Constant expressions are trivial, since they are always static.
    uint64_t constValue = std::abs(expr.getValue());
    return ConstantIntDivisibility(constValue, constValue);
  }

  ConstantIntDivisibility visitDimExpr(AffineDimExpr expr) {
    // Dim expressions cannot be analyzed further, so return the divisibility
    // in `divisibilityMap` if it has been populated by the caller, or fallback
    // to the minimum divisibility.
    if (divisibilityMap.contains(expr))
      return divisibilityMap[expr];
    return IntegerDivisibility::getMinDivisibility().getValue();
  }

  ConstantIntDivisibility visitSymbolExpr(AffineSymbolExpr expr) {
    // Symbol expressions cannot be analyzed further, so return the divisibility
    // in `divisibilityMap` if it has been populated by the caller, or fallback
    // to the minimum divisibility.
    if (divisibilityMap.contains(expr))
      return divisibilityMap[expr];
    return IntegerDivisibility::getMinDivisibility().getValue();
  }

  /// Infer the divisibility of an addition or subtraction expression by
  /// recursively visiting the LHS and RHS, and then unioning the results.
  ConstantIntDivisibility visitAddExpr(AffineBinaryOpExpr expr) {
    if (divisibilityMap.contains(expr))
      return divisibilityMap[expr];
    // The divisibility of an addition is the GCD of its constituents'
    // divisibilities.
    ConstantIntDivisibility lhsDiv = visit(expr.getLHS());
    ConstantIntDivisibility rhsDiv = visit(expr.getRHS());
    return lhsDiv.getUnion(rhsDiv);
  }

  /// Infer the divisibility of a multiplication expression by recursively
  /// visiting the LHS and RHS, and then multiplying the results.
  ConstantIntDivisibility visitMulExpr(AffineBinaryOpExpr expr) {
    if (divisibilityMap.contains(expr))
      return divisibilityMap[expr];
    // The divisibility of a multiplication is the product of its constituents'
    // divisibilities.
    ConstantIntDivisibility lhsDiv = visit(expr.getLHS());
    ConstantIntDivisibility rhsDiv = visit(expr.getRHS());
    return ConstantIntDivisibility(lhsDiv.udiv() * rhsDiv.udiv(),
                                   lhsDiv.sdiv() * rhsDiv.sdiv());
  }

  ConstantIntDivisibility visitFloorDivExpr(AffineBinaryOpExpr expr) {
    return visitDivExpr(expr);
  }

  ConstantIntDivisibility visitCeilDivExpr(AffineBinaryOpExpr expr) {
    return visitDivExpr(expr);
  }

  /// Infer the divisibility of a mod expression. If the RHS is a constant,
  /// the result divisibility is gcd(lhs_divisibility, rhs_constant), since
  /// (d * k) mod c is always divisible by gcd(d, c). Furthermore, if the
  /// LHS divisibility is itself divisible by the constant (i.e., d % c == 0),
  /// then (d * k) mod c is always zero, represented as divisibility 0.
  ConstantIntDivisibility visitModExpr(AffineBinaryOpExpr expr) {
    if (divisibilityMap.contains(expr))
      return divisibilityMap[expr];
    auto constRhs = dyn_cast<AffineConstantExpr>(expr.getRHS());
    if (!constRhs || constRhs.getValue() == 0)
      return ConstantIntDivisibility(1, 1);
    auto constValue = static_cast<uint64_t>(std::abs(constRhs.getValue()));
    ConstantIntDivisibility lhsDiv = visit(expr.getLHS());
    // If the LHS is always a multiple of constValue, x mod constValue is
    // always zero. Divisibility 0 is the lattice top ("divides everything").
    uint64_t modUDiv = (lhsDiv.udiv() % constValue == 0)
                           ? 0
                           : std::gcd(lhsDiv.udiv(), constValue);
    uint64_t modSDiv = (lhsDiv.sdiv() % constValue == 0)
                           ? 0
                           : std::gcd(lhsDiv.sdiv(), constValue);
    return ConstantIntDivisibility(modUDiv, modSDiv);
  }

private:
  ConstantIntDivisibility visitInvalidExpr(AffineBinaryOpExpr expr) {
    return IntegerDivisibility::getMinDivisibility().getValue();
  }

  /// Helper shared by ceildiv and floordiv implementations. Returns the minimum
  /// divisibility as a fallback if the divisor is not a constant, because the
  /// divisibility cannot be inferred in this case. If the divisor is a
  /// constant, then this function recursively visits the dividend, and returns
  /// the quotient of the dividend's divisibility with the divisor.
  ConstantIntDivisibility visitDivExpr(AffineBinaryOpExpr expr) {
    if (divisibilityMap.contains(expr))
      return divisibilityMap[expr];
    auto constRhs = dyn_cast<AffineConstantExpr>(expr.getRHS());
    // Division by zero is undefined, so return the minimum divisibility.
    if (!constRhs || constRhs.getValue() == 0)
      return ConstantIntDivisibility(1, 1);
    auto constValue = static_cast<uint64_t>(std::abs(constRhs.getValue()));
    ConstantIntDivisibility lhsDiv = visit(expr.getLHS());
    uint64_t divUDiv =
        lhsDiv.udiv() % constValue == 0 ? lhsDiv.udiv() / constValue : 1;
    uint64_t divSDiv =
        lhsDiv.sdiv() % constValue == 0 ? lhsDiv.sdiv() / constValue : 1;
    return ConstantIntDivisibility(divUDiv, divSDiv);
  }

  ExprDivisibilityMap &divisibilityMap;
};

/// Returns the divisibilities of each AffineMap result based on the
/// divisibilities of its dims and symbols. The `dimAndSymbolDivisibilities`
/// should contain the divisibilities of the dims, followed by the
/// divisibilities of the symbols in ascending order by their positions.
SmallVector<ConstantIntDivisibility> getResultDivisibilities(
    AffineMap map,
    ArrayRef<ConstantIntDivisibility> dimAndSymbolDivisibilities) {
  // Seed the AffineExprDivisibilityFinder with the dimAndSymbolDivisibilities.
  llvm::DenseMap<AffineExpr, ConstantIntDivisibility> exprDivisibilityMap;
  SmallVector<AffineExpr> inputExprs;
  inputExprs.append(llvm::map_to_vector(
      llvm::seq<int64_t>(map.getNumDims()),
      [&](int64_t dim) { return getAffineDimExpr(dim, map.getContext()); }));
  inputExprs.append(llvm::map_to_vector(
      llvm::seq<int64_t>(map.getNumSymbols()),
      [&](int64_t sym) { return getAffineSymbolExpr(sym, map.getContext()); }));
  for (auto [expr, divisibility] :
       llvm::zip_equal(inputExprs, dimAndSymbolDivisibilities)) {
    exprDivisibilityMap[expr] = divisibility;
  }
  AffineExprDivisibilityFinder divisibilityFinder(exprDivisibilityMap);

  // Walk each result expression and compute their divisibilities.
  SmallVector<ConstantIntDivisibility> resultDivisibilities;
  for (AffineExpr resultExpr : map.getResults())
    resultDivisibilities.push_back(divisibilityFinder.visit(resultExpr));
  return resultDivisibilities;
}

/// Infer the result divisibility of an affine.min or affine.max operation
/// based on its operand divisibilities. The result divisibility is the GCD
/// of the divisibilities of each of the affine map results, because the result
/// of the affine.min/max op could be any of these results.
template <typename MinOrMaxTy>
void inferAffineMinOrMaxResultDivisibility(
    MinOrMaxTy minOrMaxOp, ArrayRef<IntegerDivisibility> argDivs,
    SetIntDivisibilityFn setResultDivs) {
  static_assert(llvm::is_one_of<MinOrMaxTy, AffineMinOp, AffineMaxOp>::value,
                "MinOrMaxTy must be AffineMinOp or AffineMaxOp");
  SmallVector<ConstantIntDivisibility> operandDivisibilities;
  for (auto [operand, divisibility] :
       llvm::zip(minOrMaxOp.getOperands(), argDivs)) {
    operandDivisibilities.push_back(
        getDivisibilityOfOperand(operand, divisibility));
  }

  SmallVector<ConstantIntDivisibility> resultDivisibilities =
      getResultDivisibilities(minOrMaxOp.getMap(), operandDivisibilities);

  ConstantIntDivisibility resultDivisibility =
      resultDivisibilities.pop_back_val();
  for (auto divisibility : resultDivisibilities)
    resultDivisibility = resultDivisibility.getUnion(divisibility);
  setResultDivs(minOrMaxOp.getResult(), resultDivisibility);
}

} // namespace

void AffineApplyOp::inferResultDivisibility(
    ArrayRef<IntegerDivisibility> argDivs, SetIntDivisibilityFn setResultDivs) {
  SmallVector<ConstantIntDivisibility> operandDivisibilities;
  for (auto [operand, divisibility] : llvm::zip(getOperands(), argDivs)) {
    operandDivisibilities.push_back(
        getDivisibilityOfOperand(operand, divisibility));
  }

  SmallVector<ConstantIntDivisibility> resultDivisibilities =
      getResultDivisibilities(getMap(), operandDivisibilities);
  for (auto [result, divisibility] :
       llvm::zip_equal(getOperation()->getResults(), resultDivisibilities)) {
    setResultDivs(result, divisibility);
  }
}

void AffineMinOp::inferResultDivisibility(ArrayRef<IntegerDivisibility> argDivs,
                                          SetIntDivisibilityFn setResultDivs) {
  inferAffineMinOrMaxResultDivisibility(*this, argDivs, setResultDivs);
}

void AffineMaxOp::inferResultDivisibility(ArrayRef<IntegerDivisibility> argDivs,
                                          SetIntDivisibilityFn setResultDivs) {
  inferAffineMinOrMaxResultDivisibility(*this, argDivs, setResultDivs);
}

void AffineDelinearizeIndexOp::inferResultDivisibility(
    ArrayRef<IntegerDivisibility> argDivs, SetIntDivisibilityFn setResultDivs) {
  MLIRContext *ctx = getContext();

  // Operands are: [linear_index, dynamic_basis_values...]
  ConstantIntDivisibility linearDiv =
      getDivisibilityOfOperand(getLinearIndex(), argDivs[0]);

  ArrayRef<int64_t> staticBasis = getStaticBasis();
  int64_t numResults = getNumResults();

  // Build affine expressions for each result.
  // Dim 0 = linear index, symbols = dynamic basis values.
  AffineExpr linearExpr = getAffineDimExpr(0, ctx);

  // Collect operand divisibilities: [linear_index_div, dynamic_basis_divs...]
  SmallVector<ConstantIntDivisibility> operandDivs;
  operandDivs.push_back(linearDiv);

  // Map static/dynamic basis values to affine expressions.
  int64_t dynIdx = 0;
  SmallVector<AffineExpr> basisExprs;
  for (int64_t i = 0, e = static_cast<int64_t>(staticBasis.size()); i < e;
       ++i) {
    if (ShapedType::isDynamic(staticBasis[i])) {
      basisExprs.push_back(getAffineSymbolExpr(dynIdx, ctx));
      operandDivs.push_back(getDivisibilityOfOperand(getDynamicBasis()[dynIdx],
                                                     argDivs[1 + dynIdx]));
      dynIdx++;
    } else {
      basisExprs.push_back(getAffineConstantExpr(staticBasis[i], ctx));
    }
  }

  // The computation basis skips the outer bound if present.
  bool hasOuter = hasOuterBound();
  int64_t basisStart = hasOuter ? 1 : 0;

  // Each result[i] can be expressed as an affine expression of the linear
  // index using the effective basis (after dropping outer bound if present).
  // Effective basis B[k] = basisExprs[basisStart + k], for k = 0..N-2.
  // Stride s[i] = product of B[i..N-2] = product of
  //               basisExprs[basisStart+i .. end].
  //
  // result[0]   = x floordiv s[0]
  // result[i>0] = (x floordiv s[i]) mod B[i-1]
  // For i=N-1, s[N-1]=1, so result[N-1] = x mod B[N-2].

  AffineExpr stride = getAffineConstantExpr(1, ctx);
  for (int64_t i = numResults - 1; i >= 0; --i) {
    AffineExpr resultExpr;
    if (i == 0) {
      resultExpr = linearExpr.floorDiv(stride);
    } else {
      resultExpr =
          (linearExpr.floorDiv(stride)) % basisExprs[basisStart + i - 1];
    }

    AffineMap resultMap = AffineMap::get(1, dynIdx, resultExpr, ctx);
    SmallVector<ConstantIntDivisibility> divs =
        getResultDivisibilities(resultMap, operandDivs);
    setResultDivs(getResult(i), divs[0]);

    if (i > 0)
      stride = basisExprs[basisStart + i - 1] * stride;
  }
}
