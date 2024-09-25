//===- PolynomialAttributes.cpp - Polynomial dialect attrs ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"

#include "mlir/Dialect/Polynomial/IR/Polynomial.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
namespace polynomial {

void IntPolynomialAttr::print(AsmPrinter &p) const {
  p << '<' << getPolynomial() << '>';
}

void FloatPolynomialAttr::print(AsmPrinter &p) const {
  p << '<' << getPolynomial() << '>';
}

/// A callable that parses the coefficient using the appropriate method for the
/// given monomial type, and stores the parsed coefficient value on the
/// monomial.
template <typename MonomialType>
using ParseCoefficientFn = std::function<OptionalParseResult(MonomialType &)>;

/// Try to parse a monomial. If successful, populate the fields of the outparam
/// `monomial` with the results, and the `variable` outparam with the parsed
/// variable name. Sets shouldParseMore to true if the monomial is followed by
/// a '+'.
///
template <typename Monomial>
ParseResult
parseMonomial(AsmParser &parser, Monomial &monomial, llvm::StringRef &variable,
              bool &isConstantTerm, bool &shouldParseMore,
              ParseCoefficientFn<Monomial> parseAndStoreCoefficient) {
  OptionalParseResult parsedCoeffResult = parseAndStoreCoefficient(monomial);

  isConstantTerm = false;
  shouldParseMore = false;

  // A + indicates it's a constant term with more to go, as in `1 + x`.
  if (succeeded(parser.parseOptionalPlus())) {
    // If no coefficient was parsed, and there's a +, then it's effectively
    // parsing an empty string.
    if (!parsedCoeffResult.has_value()) {
      return failure();
    }
    monomial.setExponent(APInt(apintBitWidth, 0));
    isConstantTerm = true;
    shouldParseMore = true;
    return success();
  }

  // A monomial can be a trailing constant term, as in `x + 1`.
  if (failed(parser.parseOptionalKeyword(&variable))) {
    // If neither a coefficient nor a variable was found, then it's effectively
    // parsing an empty string.
    if (!parsedCoeffResult.has_value()) {
      return failure();
    }

    monomial.setExponent(APInt(apintBitWidth, 0));
    isConstantTerm = true;
    return success();
  }

  // Parse exponentiation symbol as `**`. We can't use caret because it's
  // reserved for basic block identifiers If no star is present, it's treated
  // as a polynomial with exponent 1.
  if (succeeded(parser.parseOptionalStar())) {
    // If there's one * there must be two.
    if (failed(parser.parseStar())) {
      return failure();
    }

    // If there's a **, then the integer exponent is required.
    APInt parsedExponent(apintBitWidth, 0);
    if (failed(parser.parseInteger(parsedExponent))) {
      parser.emitError(parser.getCurrentLocation(),
                       "found invalid integer exponent");
      return failure();
    }

    monomial.setExponent(parsedExponent);
  } else {
    monomial.setExponent(APInt(apintBitWidth, 1));
  }

  if (succeeded(parser.parseOptionalPlus())) {
    shouldParseMore = true;
  }
  return success();
}

template <typename Monomial>
LogicalResult
parsePolynomialAttr(AsmParser &parser, llvm::SmallVector<Monomial> &monomials,
                    llvm::StringSet<> &variables,
                    ParseCoefficientFn<Monomial> parseAndStoreCoefficient) {
  while (true) {
    Monomial parsedMonomial;
    llvm::StringRef parsedVariableRef;
    bool isConstantTerm;
    bool shouldParseMore;
    if (failed(parseMonomial<Monomial>(
            parser, parsedMonomial, parsedVariableRef, isConstantTerm,
            shouldParseMore, parseAndStoreCoefficient))) {
      parser.emitError(parser.getCurrentLocation(), "expected a monomial");
      return failure();
    }

    if (!isConstantTerm) {
      std::string parsedVariable = parsedVariableRef.str();
      variables.insert(parsedVariable);
    }
    monomials.push_back(parsedMonomial);

    if (shouldParseMore)
      continue;

    if (succeeded(parser.parseOptionalGreater())) {
      break;
    }
    parser.emitError(
        parser.getCurrentLocation(),
        "expected + and more monomials, or > to end polynomial attribute");
    return failure();
  }

  if (variables.size() > 1) {
    std::string vars = llvm::join(variables.keys(), ", ");
    parser.emitError(
        parser.getCurrentLocation(),
        "polynomials must have one indeterminate, but there were multiple: " +
            vars);
    return failure();
  }

  return success();
}

Attribute IntPolynomialAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};

  llvm::SmallVector<IntMonomial> monomials;
  llvm::StringSet<> variables;

  if (failed(parsePolynomialAttr<IntMonomial>(
          parser, monomials, variables,
          [&](IntMonomial &monomial) -> OptionalParseResult {
            APInt parsedCoeff(apintBitWidth, 1);
            OptionalParseResult result =
                parser.parseOptionalInteger(parsedCoeff);
            monomial.setCoefficient(parsedCoeff);
            return result;
          }))) {
    return {};
  }

  auto result = IntPolynomial::fromMonomials(monomials);
  if (failed(result)) {
    parser.emitError(parser.getCurrentLocation())
        << "parsed polynomial must have unique exponents among monomials";
    return {};
  }
  return IntPolynomialAttr::get(parser.getContext(), result.value());
}
Attribute FloatPolynomialAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};

  llvm::SmallVector<FloatMonomial> monomials;
  llvm::StringSet<> variables;

  ParseCoefficientFn<FloatMonomial> parseAndStoreCoefficient =
      [&](FloatMonomial &monomial) -> OptionalParseResult {
    double coeffValue = 1.0;
    ParseResult result = parser.parseFloat(coeffValue);
    monomial.setCoefficient(APFloat(coeffValue));
    return OptionalParseResult(result);
  };

  if (failed(parsePolynomialAttr<FloatMonomial>(parser, monomials, variables,
                                                parseAndStoreCoefficient))) {
    return {};
  }

  auto result = FloatPolynomial::fromMonomials(monomials);
  if (failed(result)) {
    parser.emitError(parser.getCurrentLocation())
        << "parsed polynomial must have unique exponents among monomials";
    return {};
  }
  return FloatPolynomialAttr::get(parser.getContext(), result.value());
}

} // namespace polynomial
} // namespace mlir
