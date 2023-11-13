//===- Polynomial.cpp - MLIR storage type for static Polynomial -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polynomial/IR/Polynomial.h"

#include "PolynomialDetail.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace polynomial {

MLIRContext *Polynomial::getContext() const { return terms->context; }

ArrayRef<Monomial> Polynomial::getTerms() const { return terms->terms(); }

Polynomial Polynomial::fromMonomials(ArrayRef<Monomial> monomials,
                                     MLIRContext *context) {
  auto assignCtx = [context](detail::PolynomialStorage *storage) {
    storage->context = context;
  };

  // A polynomial's terms are canonically stored in order of increasing degree.
  auto monomialsCopy = llvm::OwningArrayRef<Monomial>(monomials);
  std::sort(monomialsCopy.begin(), monomialsCopy.end());

  StorageUniquer &uniquer = context->getAttributeUniquer();
  return Polynomial(uniquer.get<detail::PolynomialStorage>(
      assignCtx, monomials.size(), monomialsCopy));
}

Polynomial Polynomial::fromCoefficients(ArrayRef<int64_t> coeffs,
                                        MLIRContext *context) {
  std::vector<Monomial> monomials;
  for (size_t i = 0; i < coeffs.size(); i++) {
    monomials.emplace_back(coeffs[i], i);
  }
  return Polynomial::fromMonomials(monomials, context);
}

void Polynomial::print(raw_ostream &os, ::llvm::StringRef separator,
                       ::llvm::StringRef exponentiation) const {
  bool first = true;
  for (const auto &term : terms->terms()) {
    if (first) {
      first = false;
    } else {
      os << separator;
    }
    std::string coeffToPrint;
    if (term.coefficient == 1 && term.exponent.uge(1)) {
      coeffToPrint = "";
    } else {
      llvm::SmallString<512> coeffString;
      term.coefficient.toStringSigned(coeffString);
      coeffToPrint = coeffString.str();
    }

    if (term.exponent == 0) {
      os << coeffToPrint;
    } else if (term.exponent == 1) {
      os << coeffToPrint << "x";
    } else {
      llvm::SmallString<512> expString;
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
  return terms->terms().back().exponent.getZExtValue();
}

} // namespace polynomial
} // namespace mlir

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::polynomial::detail::PolynomialStorage);
