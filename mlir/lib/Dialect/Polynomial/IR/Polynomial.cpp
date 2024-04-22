//===- Polynomial.cpp - MLIR storage type for static Polynomial -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polynomial/IR/Polynomial.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace polynomial {

FailureOr<Polynomial> Polynomial::fromMonomials(ArrayRef<Monomial> monomials) {
  // A polynomial's terms are canonically stored in order of increasing degree.
  auto monomialsCopy = llvm::SmallVector<Monomial>(monomials);
  std::sort(monomialsCopy.begin(), monomialsCopy.end());

  // Ensure non-unique exponents are not present. Since we sorted the list by
  // exponent, a linear scan of adjancent monomials suffices.
  if (std::adjacent_find(monomialsCopy.begin(), monomialsCopy.end(),
                         [](const Monomial &lhs, const Monomial &rhs) {
                           return lhs.exponent == rhs.exponent;
                         }) != monomialsCopy.end()) {
    return failure();
  }

  return Polynomial(monomialsCopy);
}

Polynomial Polynomial::fromCoefficients(ArrayRef<int64_t> coeffs) {
  llvm::SmallVector<Monomial> monomials;
  auto size = coeffs.size();
  monomials.reserve(size);
  for (size_t i = 0; i < size; i++) {
    monomials.emplace_back(coeffs[i], i);
  }
  auto result = Polynomial::fromMonomials(monomials);
  // Construction guarantees unique exponents, so the failure mode of
  // fromMonomials can be bypassed.
  assert(succeeded(result));
  return result.value();
}

void Polynomial::print(raw_ostream &os, ::llvm::StringRef separator,
                       ::llvm::StringRef exponentiation) const {
  bool first = true;
  for (const Monomial &term : terms) {
    if (first) {
      first = false;
    } else {
      os << separator;
    }
    std::string coeffToPrint;
    if (term.coefficient == 1 && term.exponent.uge(1)) {
      coeffToPrint = "";
    } else {
      llvm::SmallString<16> coeffString;
      term.coefficient.toStringSigned(coeffString);
      coeffToPrint = coeffString.str();
    }

    if (term.exponent == 0) {
      os << coeffToPrint;
    } else if (term.exponent == 1) {
      os << coeffToPrint << "x";
    } else {
      llvm::SmallString<16> expString;
      term.exponent.toStringSigned(expString);
      os << coeffToPrint << "x" << exponentiation << expString;
    }
  }
}

void Polynomial::print(raw_ostream &os) const { print(os, " + ", "**"); }

std::string Polynomial::toIdentifier() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  print(os, "_", "");
  return os.str();
}

unsigned Polynomial::getDegree() const {
  return terms.back().exponent.getZExtValue();
}

} // namespace polynomial
} // namespace mlir
