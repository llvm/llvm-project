//===- Polynomial.cpp - MLIR storage type for static Polynomial -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polynomial/IR/Polynomial.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace polynomial {

template <typename PolyT, typename MonomialT>
FailureOr<PolyT> fromMonomialsImpl(ArrayRef<MonomialT> monomials) {
  // A polynomial's terms are canonically stored in order of increasing degree.
  auto monomialsCopy = llvm::SmallVector<MonomialT>(monomials);
  std::sort(monomialsCopy.begin(), monomialsCopy.end());

  // Ensure non-unique exponents are not present. Since we sorted the list by
  // exponent, a linear scan of adjancent monomials suffices.
  if (std::adjacent_find(monomialsCopy.begin(), monomialsCopy.end(),
                         [](const MonomialT &lhs, const MonomialT &rhs) {
                           return lhs.getExponent() == rhs.getExponent();
                         }) != monomialsCopy.end()) {
    return failure();
  }

  return PolyT(monomialsCopy);
}

FailureOr<IntPolynomial>
IntPolynomial::fromMonomials(ArrayRef<IntMonomial> monomials) {
  return fromMonomialsImpl<IntPolynomial, IntMonomial>(monomials);
}

FailureOr<FloatPolynomial>
FloatPolynomial::fromMonomials(ArrayRef<FloatMonomial> monomials) {
  return fromMonomialsImpl<FloatPolynomial, FloatMonomial>(monomials);
}

template <typename PolyT, typename MonomialT, typename CoeffT>
PolyT fromCoefficientsImpl(ArrayRef<CoeffT> coeffs) {
  llvm::SmallVector<MonomialT> monomials;
  auto size = coeffs.size();
  monomials.reserve(size);
  for (size_t i = 0; i < size; i++) {
    monomials.emplace_back(coeffs[i], i);
  }
  auto result = PolyT::fromMonomials(monomials);
  // Construction guarantees unique exponents, so the failure mode of
  // fromMonomials can be bypassed.
  assert(succeeded(result));
  return result.value();
}

IntPolynomial IntPolynomial::fromCoefficients(ArrayRef<int64_t> coeffs) {
  return fromCoefficientsImpl<IntPolynomial, IntMonomial, int64_t>(coeffs);
}

FloatPolynomial FloatPolynomial::fromCoefficients(ArrayRef<double> coeffs) {
  return fromCoefficientsImpl<FloatPolynomial, FloatMonomial, double>(coeffs);
}

} // namespace polynomial
} // namespace mlir
