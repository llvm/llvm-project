//===- Utils.cpp - General utilities for Presburger library ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions required by the Presburger Library.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/Analysis/Presburger/IntegerPolyhedron.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"

using namespace mlir;
using namespace presburger_utils;

/// Normalize a division's `dividend` and the `divisor` by their GCD. For
/// example: if the dividend and divisor are [2,0,4] and 4 respectively,
/// they get normalized to [1,0,2] and 2.
static void normalizeDivisionByGCD(SmallVectorImpl<int64_t> &dividend,
                                   unsigned &divisor) {
  if (divisor == 0 || dividend.empty())
    return;
  // We take the absolute value of dividend's coefficients to make sure that
  // `gcd` is positive.
  int64_t gcd =
      llvm::greatestCommonDivisor(std::abs(dividend.front()), int64_t(divisor));

  // The reason for ignoring the constant term is as follows.
  // For a division:
  //      floor((a + m.f(x))/(m.d))
  // It can be replaced by:
  //      floor((floor(a/m) + f(x))/d)
  // Since `{a/m}/d` in the dividend satisfies 0 <= {a/m}/d < 1/d, it will not
  // influence the result of the floor division and thus, can be ignored.
  for (size_t i = 1, m = dividend.size() - 1; i < m; i++) {
    gcd = llvm::greatestCommonDivisor(std::abs(dividend[i]), gcd);
    if (gcd == 1)
      return;
  }

  // Normalize the dividend and the denominator.
  std::transform(dividend.begin(), dividend.end(), dividend.begin(),
                 [gcd](int64_t &n) { return floor(n / gcd); });
  divisor /= gcd;
}

/// Check if the pos^th identifier can be represented as a division using upper
/// bound inequality at position `ubIneq` and lower bound inequality at position
/// `lbIneq`.
///
/// Let `id` be the pos^th identifier, then `id` is equivalent to
/// `expr floordiv divisor` if there are constraints of the form:
///      0 <= expr - divisor * id <= divisor - 1
/// Rearranging, we have:
///       divisor * id - expr + (divisor - 1) >= 0  <-- Lower bound for 'id'
///      -divisor * id + expr                 >= 0  <-- Upper bound for 'id'
///
/// For example:
///     32*k >= 16*i + j - 31                 <-- Lower bound for 'k'
///     32*k  <= 16*i + j                     <-- Upper bound for 'k'
///     expr = 16*i + j, divisor = 32
///     k = ( 16*i + j ) floordiv 32
///
///     4q >= i + j - 2                       <-- Lower bound for 'q'
///     4q <= i + j + 1                       <-- Upper bound for 'q'
///     expr = i + j + 1, divisor = 4
///     q = (i + j + 1) floordiv 4
//
/// This function also supports detecting divisions from bounds that are
/// strictly tighter than the division bounds described above, since tighter
/// bounds imply the division bounds. For example:
///     4q - i - j + 2 >= 0                       <-- Lower bound for 'q'
///    -4q + i + j     >= 0                       <-- Tight upper bound for 'q'
///
/// To extract floor divisions with tighter bounds, we assume that that the
/// constraints are of the form:
///     c <= expr - divisior * id <= divisor - 1, where 0 <= c <= divisor - 1
/// Rearranging, we have:
///     divisor * id - expr + (divisor - 1) >= 0  <-- Lower bound for 'id'
///    -divisor * id + expr - c             >= 0  <-- Upper bound for 'id'
///
/// If successful, `expr` is set to dividend of the division and `divisor` is
/// set to the denominator of the division. The final division expression is
/// normalized by GCD.
static LogicalResult getDivRepr(const IntegerPolyhedron &cst, unsigned pos,
                                unsigned ubIneq, unsigned lbIneq,
                                SmallVector<int64_t, 8> &expr,
                                unsigned &divisor) {

  assert(pos <= cst.getNumIds() && "Invalid identifier position");
  assert(ubIneq <= cst.getNumInequalities() &&
         "Invalid upper bound inequality position");
  assert(lbIneq <= cst.getNumInequalities() &&
         "Invalid upper bound inequality position");

  // Extract divisor from the lower bound.
  divisor = cst.atIneq(lbIneq, pos);

  // First, check if the constraints are opposite of each other except the
  // constant term.
  unsigned i = 0, e = 0;
  for (i = 0, e = cst.getNumIds(); i < e; ++i)
    if (cst.atIneq(ubIneq, i) != -cst.atIneq(lbIneq, i))
      break;

  if (i < e)
    return failure();

  // Then, check if the constant term is of the proper form.
  // Due to the form of the upper/lower bound inequalities, the sum of their
  // constants is `divisor - 1 - c`. From this, we can extract c:
  int64_t constantSum = cst.atIneq(lbIneq, cst.getNumCols() - 1) +
                        cst.atIneq(ubIneq, cst.getNumCols() - 1);
  int64_t c = divisor - 1 - constantSum;

  // Check if `c` satisfies the condition `0 <= c <= divisor - 1`. This also
  // implictly checks that `divisor` is positive.
  if (!(c >= 0 && c <= divisor - 1))
    return failure();

  // The inequality pair can be used to extract the division.
  // Set `expr` to the dividend of the division except the constant term, which
  // is set below.
  expr.resize(cst.getNumCols(), 0);
  for (i = 0, e = cst.getNumIds(); i < e; ++i)
    if (i != pos)
      expr[i] = cst.atIneq(ubIneq, i);

  // From the upper bound inequality's form, its constant term is equal to the
  // constant term of `expr`, minus `c`. From this,
  // constant term of `expr` = constant term of upper bound + `c`.
  expr.back() = cst.atIneq(ubIneq, cst.getNumCols() - 1) + c;
  normalizeDivisionByGCD(expr, divisor);

  return success();
}

/// Check if the pos^th identifier can be represented as a division using
/// equality at position `eqInd`.
///
/// For example:
///     32*k == 16*i + j - 31                 <-- `eqInd` for 'k'
///     expr = 16*i + j - 31, divisor = 32
///     k = (16*i + j - 31) floordiv 32
///
/// If successful, `expr` is set to dividend of the division and `divisor` is
/// set to the denominator of the division. The final division expression is
/// normalized by GCD.
static LogicalResult getDivRepr(const IntegerPolyhedron &cst, unsigned pos,
                                unsigned eqInd, SmallVector<int64_t, 8> &expr,
                                unsigned &divisor) {

  assert(pos <= cst.getNumIds() && "Invalid identifier position");
  assert(eqInd <= cst.getNumEqualities() && "Invalid equality position");

  // Extract divisor, the divisor can be negative and hence its sign information
  // is stored in `signDiv` to reverse the sign of dividend's coefficients.
  // Equality must involve the pos-th variable and hence `tempDiv` != 0.
  int64_t tempDiv = cst.atEq(eqInd, pos);
  if (tempDiv == 0)
    return failure();
  int64_t signDiv = tempDiv < 0 ? -1 : 1;

  // The divisor is always a positive integer.
  divisor = tempDiv * signDiv;

  expr.resize(cst.getNumCols(), 0);
  for (unsigned i = 0, e = cst.getNumIds(); i < e; ++i)
    if (i != pos)
      expr[i] = signDiv * cst.atEq(eqInd, i);

  expr.back() = signDiv * cst.atEq(eqInd, cst.getNumCols() - 1);
  normalizeDivisionByGCD(expr, divisor);

  return success();
}

// Returns `false` if the constraints depends on a variable for which an
// explicit representation has not been found yet, otherwise returns `true`.
static bool checkExplicitRepresentation(const IntegerPolyhedron &cst,
                                        ArrayRef<bool> foundRepr,
                                        ArrayRef<int64_t> dividend,
                                        unsigned pos) {
  // Exit to avoid circular dependencies between divisions.
  for (unsigned c = 0, e = cst.getNumIds(); c < e; ++c) {
    if (c == pos)
      continue;

    if (!foundRepr[c] && dividend[c] != 0) {
      // Expression can't be constructed as it depends on a yet unknown
      // identifier.
      //
      // TODO: Visit/compute the identifiers in an order so that this doesn't
      // happen. More complex but much more efficient.
      return false;
    }
  }

  return true;
}

/// Check if the pos^th identifier can be expressed as a floordiv of an affine
/// function of other identifiers (where the divisor is a positive constant).
/// `foundRepr` contains a boolean for each identifier indicating if the
/// explicit representation for that identifier has already been computed.
/// Returns the `MaybeLocalRepr` struct which contains the indices of the
/// constraints that can be expressed as a floordiv of an affine function. If
/// the representation could be computed, `dividend` and `denominator` are set.
/// If the representation could not be computed, the kind attribute in
/// `MaybeLocalRepr` is set to None.
MaybeLocalRepr presburger_utils::computeSingleVarRepr(
    const IntegerPolyhedron &cst, ArrayRef<bool> foundRepr, unsigned pos,
    SmallVector<int64_t, 8> &dividend, unsigned &divisor) {
  assert(pos < cst.getNumIds() && "invalid position");
  assert(foundRepr.size() == cst.getNumIds() &&
         "Size of foundRepr does not match total number of variables");

  SmallVector<unsigned, 4> lbIndices, ubIndices, eqIndices;
  cst.getLowerAndUpperBoundIndices(pos, &lbIndices, &ubIndices, &eqIndices);
  MaybeLocalRepr repr{};

  for (unsigned ubPos : ubIndices) {
    for (unsigned lbPos : lbIndices) {
      // Attempt to get divison representation from ubPos, lbPos.
      if (failed(getDivRepr(cst, pos, ubPos, lbPos, dividend, divisor)))
        continue;

      if (!checkExplicitRepresentation(cst, foundRepr, dividend, pos))
        continue;

      repr.kind = ReprKind::Inequality;
      repr.repr.inequalityPair = {ubPos, lbPos};
      return repr;
    }
  }
  for (unsigned eqPos : eqIndices) {
    // Attempt to get divison representation from eqPos.
    if (failed(getDivRepr(cst, pos, eqPos, dividend, divisor)))
      continue;

    if (!checkExplicitRepresentation(cst, foundRepr, dividend, pos))
      continue;

    repr.kind = ReprKind::Equality;
    repr.repr.equalityIdx = eqPos;
    return repr;
  }
  return repr;
}

void presburger_utils::removeDuplicateDivs(
    std::vector<SmallVector<int64_t, 8>> &divs,
    SmallVectorImpl<unsigned> &denoms, unsigned localOffset,
    llvm::function_ref<bool(unsigned i, unsigned j)> merge) {

  // Find and merge duplicate divisions.
  // TODO: Add division normalization to support divisions that differ by
  // a constant.
  // TODO: Add division ordering such that a division representation for local
  // identifier at position `i` only depends on local identifiers at position <
  // `i`. This would make sure that all divisions depending on other local
  // variables that can be merged, are merged.
  for (unsigned i = 0; i < divs.size(); ++i) {
    // Check if a division representation exists for the `i^th` local id.
    if (denoms[i] == 0)
      continue;
    // Check if a division exists which is a duplicate of the division at `i`.
    for (unsigned j = i + 1; j < divs.size(); ++j) {
      // Check if a division representation exists for the `j^th` local id.
      if (denoms[j] == 0)
        continue;
      // Check if the denominators match.
      if (denoms[i] != denoms[j])
        continue;
      // Check if the representations are equal.
      if (divs[i] != divs[j])
        continue;

      // Merge divisions at position `j` into division at position `i`. If
      // merge fails, do not merge these divs.
      bool mergeResult = merge(i, j);
      if (!mergeResult)
        continue;

      // Update division information to reflect merging.
      for (unsigned k = 0, g = divs.size(); k < g; ++k) {
        SmallVector<int64_t, 8> &div = divs[k];
        if (denoms[k] != 0) {
          div[localOffset + i] += div[localOffset + j];
          div.erase(div.begin() + localOffset + j);
        }
      }

      divs.erase(divs.begin() + j);
      denoms.erase(denoms.begin() + j);
      // Since `j` can never be zero, we do not need to worry about overflows.
      --j;
    }
  }
}
