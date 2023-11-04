//===- PolynomialAttributes.td - Attribute definitions for the polynomial dialect ------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef POLYNOMIAL_ATTRIBUTES
#define POLYNOMIAL_ATTRIBUTES

include "PolynomialDialect.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

class Polynomial_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Polynomial_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Polynomial_PolynomialAttr : Polynomial_Attr<"Polynomial", "polynomial"> {
  let summary = "An attribute containing a single-variable polynomial.";
  let description = [{
     #poly = #polynomial.poly<x**1024 + 1>
  }];

  let parameters = (ins "Polynomial":$polynomial);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Polynomial":$polynomial), [{
      return $_get(polynomial.getContext(), polynomial);
    }]>
  ];

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
}

def Polynomial_RingAttr : Polynomial_Attr<"Ring", "ring"> {
  let summary = "An attribute specifying a polynomial ring.";
  let description = [{
    A ring describes the domain in which polynomial arithmetic occurs. The ring
    attribute in `polynomial` represents the more specific case of polynomials
    with a single indeterminate; whose coefficients can be represented by
    another MLIR type (`ctype`); and, if the coefficient type is integral,
    whose coefficients are taken modulo some statically known modulus (`cmod`).

    Additionally, a polynomial ring can specify an _ideal_, which converts
    polynomial arithmetic to the analogue of modular integer arithmetic, where
    each polynomial is represented as its remainder when dividing by the
    modulus. For single-variable polynomials, an "ideal" is always specificed
    via a single polynomial, which we call `polynomialModulus`.

    An expressive example is polynomials with i32 coefficients, whose
    coefficients are taken modulo `2**32 - 5`, with a polynomial modulus of
    `x**1024 - 1`.

    ```
    #poly_mod = #polynomial.polynomial<-1 + x**1024>
    #ring = #polynomial.ring<ctype=i32, cmod=4294967291, ideal=#poly_mod>

    %0 = ... : polynomial.polynomial<#ring>
    ```

    In this case, the value of a polynomial is always ``converted'' to a
    canonical form by applying repeated reductions by setting `x**1024 = 1`
    and simplifying.

    The coefficient and polynomial modulus parameters are optional, and the
    coefficient modulus is only allowed if the coefficient type is integral.
  }];

  let parameters = (ins
    Builtin_TypeAttr: $coefficientType,
    OptionalParameter<"std::optional<IntegerAttr>">: $coefficientModulus,
    OptionalParameter<"std::optional<PolynomialAttr>">: $polynomialModulus
  );

  let hasCustomAssemblyFormat = 1;
}

#endif // POLYNOMIAL_ATTRIBUTES
