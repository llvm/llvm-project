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
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
namespace polynomial {

void PolynomialAttr::print(AsmPrinter &p) const {
  p << '<';
  p << getPolynomial();
  p << '>';
}

/// Try to parse a monomial. If successful, populate the fields of the outparam
/// `monomial` with the results, and the `variable` outparam with the parsed
/// variable name. Sets shouldParseMore to true if the monomial is followed by
/// a '+'.
ParseResult parseMonomial(AsmParser &parser, Monomial &monomial,
                          llvm::StringRef &variable, bool *isConstantTerm,
                          bool *shouldParseMore) {
  APInt parsedCoeff(apintBitWidth, 1);
  auto parsedCoeffResult = parser.parseOptionalInteger(parsedCoeff);
  monomial.coefficient = parsedCoeff;

  *isConstantTerm = false;
  *shouldParseMore = false;

  // A + indicates it's a constant term with more to go, as in `1 + x`.
  if (succeeded(parser.parseOptionalPlus())) {
    // If no coefficient was parsed, and there's a +, then it's effectively
    // parsing an empty string.
    if (!parsedCoeffResult.has_value()) {
      return failure();
    }
    monomial.exponent = APInt(apintBitWidth, 0);
    *isConstantTerm = true;
    *shouldParseMore = true;
    return success();
  }

  // A monomial can be a trailing constant term, as in `x + 1`
  if (failed(parser.parseOptionalKeyword(&variable))) {
    // If neither a coefficient nor a variable was found, then it's effectively
    // parsing an empty string.
    if (!parsedCoeffResult.has_value()) {
      return failure();
    }

    monomial.exponent = APInt(apintBitWidth, 0);
    *isConstantTerm = true;
    return success();
  }

  // Parse exponentiation symbol as **
  // We can't use caret because it's reserved for basic block identifiers
  // If no star is present, it's treated as a polynomial with exponent 1
  if (succeeded(parser.parseOptionalStar())) {
    // If there's one * there must be two
    if (failed(parser.parseStar())) {
      parser.emitError(parser.getCurrentLocation(),
                       "exponents must be specified as a double-asterisk `**`");
      return failure();
    }

    // If there's a **, then the integer exponent is required.
    APInt parsedExponent(apintBitWidth, 0);
    if (failed(parser.parseInteger(parsedExponent))) {
      parser.emitError(parser.getCurrentLocation(),
                       "found invalid integer exponent");
      return failure();
    }

    monomial.exponent = parsedExponent;
  } else {
    monomial.exponent = APInt(apintBitWidth, 1);
  }

  if (succeeded(parser.parseOptionalPlus())) {
    *shouldParseMore = true;
  }
  return success();
}

Attribute PolynomialAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};

  llvm::SmallVector<Monomial> monomials;
  llvm::StringSet<> variables;
  llvm::DenseSet<APInt> exponents;

  while (true) {
    Monomial parsedMonomial;
    llvm::StringRef parsedVariableRef;
    bool isConstantTerm;
    bool shouldParseMore;
    if (failed(parseMonomial(parser, parsedMonomial, parsedVariableRef,
                             &isConstantTerm, &shouldParseMore))) {
      parser.emitError(parser.getCurrentLocation(), "expected a monomial");
      return {};
    }

    if (!isConstantTerm) {
      std::string parsedVariable = parsedVariableRef.str();
      variables.insert(parsedVariable);
    }
    monomials.push_back(parsedMonomial);

    if (llvm::is_contained(exponents, parsedMonomial.exponent)) {
      llvm::SmallString<16> coeffString;
      parsedMonomial.exponent.toStringSigned(coeffString);
      parser.emitError(parser.getCurrentLocation())
          << "at most one monomial may have exponent " << coeffString
          << ", but found multiple";
      return {};
    }
    exponents.insert(parsedMonomial.exponent);

    if (shouldParseMore)
      continue;

    if (succeeded(parser.parseOptionalGreater())) {
      break;
    }
    parser.emitError(
        parser.getCurrentLocation(),
        "expected + and more monomials, or > to end polynomial attribute");
    return {};
  }

  if (variables.size() > 1) {
    std::string vars = llvm::join(variables.keys(), ", ");
    parser.emitError(
        parser.getCurrentLocation(),
        "polynomials must have one indeterminate, but there were multiple: " +
            vars);
  }

  Polynomial poly = Polynomial::fromMonomials(monomials, parser.getContext());
  return PolynomialAttr::get(poly);
}

void RingAttr::print(AsmPrinter &p) const {
  p << "#polynomial.ring<coefficientType=" << getCoefficientType()
    << ", coefficientModulus=" << getCoefficientModulus()
    << ", polynomialModulus=" << getPolynomialModulus() << '>';
}

Attribute RingAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};

  if (failed(parser.parseKeyword("coefficientType")))
    return {};

  if (failed(parser.parseEqual()))
    return {};

  TypeAttr typeAttr;
  if (failed(parser.parseAttribute<TypeAttr>(typeAttr)))
    return {};

  if (failed(parser.parseComma()))
    return {};

  std::optional<IntegerAttr> coefficientModulusAttr = std::nullopt;
  if (succeeded(parser.parseKeyword("coefficientModulus"))) {
    if (failed(parser.parseEqual()))
      return {};

    IntegerType iType = typeAttr.getValue().dyn_cast<IntegerType>();
    if (!iType) {
      parser.emitError(parser.getCurrentLocation(),
                       "coefficientType must specify an integer type");
      return {};
    }
    APInt coefficientModulus(iType.getWidth(), 0);
    auto result = parser.parseInteger(coefficientModulus);
    if (failed(result)) {
      parser.emitError(parser.getCurrentLocation(),
                       "invalid coefficient modulus");
      return {};
    }
    coefficientModulusAttr = IntegerAttr::get(iType, coefficientModulus);

    if (failed(parser.parseComma()))
      return {};
  }

  std::optional<PolynomialAttr> polyAttr = std::nullopt;
  if (succeeded(parser.parseKeyword("polynomialModulus"))) {
    if (failed(parser.parseEqual()))
      return {};

    PolynomialAttr attr;
    if (failed(parser.parseAttribute<PolynomialAttr>(attr)))
      return {};
    polyAttr = attr;
  }

  if (failed(parser.parseGreater()))
    return {};

  return RingAttr::get(parser.getContext(), typeAttr, coefficientModulusAttr,
                       polyAttr);
}

} // namespace polynomial
} // namespace mlir
