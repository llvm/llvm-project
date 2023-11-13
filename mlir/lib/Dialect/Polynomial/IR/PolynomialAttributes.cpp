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

namespace mlir {
namespace polynomial {

void PolynomialAttr::print(AsmPrinter &p) const {
  p << '<';
  p << getPolynomial();
  p << '>';
}

/// Try to parse a monomial. If successful, populate the fields of the outparam
/// `monomial` with the results, and the `variable` outparam with the parsed
/// variable name.
ParseResult parseMonomial(AsmParser &parser, Monomial &monomial,
                          llvm::StringRef *variable, bool *isConstantTerm) {
  APInt parsedCoeff(apintBitWidth, 1);
  auto result = parser.parseOptionalInteger(parsedCoeff);
  if (result.has_value()) {
    if (failed(*result)) {
      parser.emitError(parser.getCurrentLocation(),
                       "invalid integer coefficient");
      return failure();
    }
  }

  // Variable name
  result = parser.parseOptionalKeyword(variable);
  if (!result.has_value() || failed(*result)) {
    // we allow "failed" because it triggers when the next token is a +,
    // which is allowed when the input is the constant term.
    monomial.coefficient = parsedCoeff;
    monomial.exponent = APInt(apintBitWidth, 0);
    *isConstantTerm = true;
    return success();
  }

  // Parse exponentiation symbol as **
  // We can't use caret because it's reserved for basic block identifiers
  // If no star is present, it's treated as a polynomial with exponent 1
  if (failed(parser.parseOptionalStar())) {
    monomial.coefficient = parsedCoeff;
    monomial.exponent = APInt(apintBitWidth, 1);
    return success();
  }

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

  monomial.coefficient = parsedCoeff;
  monomial.exponent = parsedExponent;
  return success();
}

mlir::Attribute mlir::polynomial::PolynomialAttr::parse(AsmParser &parser,
                                                        Type type) {
  if (failed(parser.parseLess()))
    return {};

  std::vector<Monomial> monomials;
  llvm::SmallSet<std::string, 2> variables;
  llvm::DenseSet<APInt> exponents;

  while (true) {
    Monomial parsedMonomial;
    llvm::StringRef parsedVariableRef;
    bool isConstantTerm = false;
    if (failed(parseMonomial(parser, parsedMonomial, &parsedVariableRef,
                             &isConstantTerm))) {
      return {};
    }

    if (!isConstantTerm) {
      std::string parsedVariable = parsedVariableRef.str();
      variables.insert(parsedVariable);
    }
    monomials.push_back(parsedMonomial);

    if (exponents.count(parsedMonomial.exponent) > 0) {
      llvm::SmallString<16> coeffString;
      parsedMonomial.exponent.toStringSigned(coeffString);
      parser.emitError(parser.getCurrentLocation(),
                       "at most one monomial may have exponent " +
                           coeffString + ", but found multiple");
      return {};
    }
    exponents.insert(parsedMonomial.exponent);

    // Parse optional +. If a + is absent, require > and break, otherwise forbid
    // > and continue with the next monomial.
    // ParseOptional{Plus, Greater} does not return an OptionalParseResult, so
    // failed means that the token was not found.
    if (failed(parser.parseOptionalPlus())) {
      if (succeeded(parser.parseGreater())) {
        break;
      } else {
        parser.emitError(
            parser.getCurrentLocation(),
            "expected + and more monomials, or > to end polynomial attribute");
        return {};
      }
    } else if (succeeded(parser.parseOptionalGreater())) {
      parser.emitError(
          parser.getCurrentLocation(),
          "expected another monomial after +, but found > ending attribute");
      return {};
    }
  }

  // insert necessary constant ops to give as input to extract_slice.
  if (variables.size() > 1) {
    std::string vars = llvm::join(variables.begin(), variables.end(), ", ");
    parser.emitError(
        parser.getCurrentLocation(),
        "polynomials must have one indeterminate, but there were multiple: " +
            vars);
  }

  Polynomial poly = Polynomial::fromMonomials(monomials, parser.getContext());
  return PolynomialAttr::get(poly);
}

void RingAttr::print(AsmPrinter &p) const {
  p << "#polynomial.ring<ctype=" << getCoefficientType()
    << ", cmod=" << getCoefficientModulus()
    << ", ideal=" << getPolynomialModulus() << '>';
}

mlir::Attribute mlir::polynomial::RingAttr::parse(AsmParser &parser,
                                                  Type type) {
  if (failed(parser.parseLess()))
    return {};

  if (failed(parser.parseKeyword("ctype")))
    return {};

  if (failed(parser.parseEqual()))
    return {};

  TypeAttr typeAttr;
  if (failed(parser.parseAttribute<TypeAttr>(typeAttr)))
    return {};

  if (failed(parser.parseComma()))
    return {};

  std::optional<IntegerAttr> cmodAttr = std::nullopt;
  if (succeeded(parser.parseKeyword("cmod"))) {
    if (failed(parser.parseEqual()))
      return {};

    IntegerType iType = llvm::dyn_cast<IntegerType>(typeAttr.getValue());
    if (!iType) {
      parser.emitError(
          parser.getCurrentLocation(),
          "invalid coefficient modulus, ctype must specify an integer type");
      return {};
    }
    APInt cmod(iType.getWidth(), 0);
    auto result = parser.parseInteger(cmod);
    if (failed(result)) {
      parser.emitError(parser.getCurrentLocation(),
                       "invalid coefficient modulus");
      return {};
    }
    cmodAttr = IntegerAttr::get(iType, cmod);

    if (failed(parser.parseComma()))
      return {};
  }

  std::optional<PolynomialAttr> polyAttr = std::nullopt;
  if (succeeded(parser.parseKeyword("ideal"))) {
    if (failed(parser.parseEqual()))
      return {};

    PolynomialAttr attr;
    if (failed(parser.parseAttribute<PolynomialAttr>(attr)))
      return {};
    polyAttr = attr;
  }

  if (failed(parser.parseGreater()))
    return {};

  return RingAttr::get(parser.getContext(), typeAttr, cmodAttr, polyAttr);
}

} // namespace polynomial
} // namespace mlir
