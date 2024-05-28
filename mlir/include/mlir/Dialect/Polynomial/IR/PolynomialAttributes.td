//===- PolynomialOps.td - Polynomial dialect ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYNOMIAL_ATTRIBUTES
#define POLYNOMIAL_ATTRIBUTES

include "mlir/IR/BuiltinAttributes.td"
include "mlir/Dialect/Polynomial/IR/PolynomialDialect.td"

class Polynomial_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Polynomial_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Polynomial_IntPolynomialAttr : Polynomial_Attr<"IntPolynomial", "int_polynomial"> {
  let summary = "an attribute containing a single-variable polynomial with integer coefficients";
  let description = [{
    A polynomial attribute represents a single-variable polynomial with integer
    coefficients, which is used to define the modulus of a `RingAttr`, as well
    as to define constants and perform constant folding for `polynomial` ops.

    The polynomial must be expressed as a list of monomial terms, with addition
    or subtraction between them. The choice of variable name is arbitrary, but
    must be consistent across all the monomials used to define a single
    attribute. The order of monomial terms is arbitrary, each monomial degree
    must occur at most once.

    Example:

    ```mlir
    #poly = #polynomial.int_polynomial<x**1024 + 1>
    ```
  }];
  let parameters = (ins "::mlir::polynomial::IntPolynomial":$polynomial);
  let hasCustomAssemblyFormat = 1;
}

def Polynomial_FloatPolynomialAttr : Polynomial_Attr<"FloatPolynomial", "float_polynomial"> {
  let summary = "an attribute containing a single-variable polynomial with double precision floating point coefficients";
  let description = [{
    A polynomial attribute represents a single-variable polynomial with double
    precision floating point coefficients.

    The polynomial must be expressed as a list of monomial terms, with addition
    or subtraction between them. The choice of variable name is arbitrary, but
    must be consistent across all the monomials used to define a single
    attribute. The order of monomial terms is arbitrary, each monomial degree
    must occur at most once.

    Example:

    ```mlir
    #poly = #polynomial.float_polynomial<0.5 x**7 + 1.5>
    ```
  }];
  let parameters = (ins "FloatPolynomial":$polynomial);
  let hasCustomAssemblyFormat = 1;
}

def Polynomial_TypedIntPolynomialAttr : Polynomial_Attr<
    "TypedIntPolynomial", "typed_int_polynomial", [TypedAttrInterface]> {
  let summary = "a typed int_polynomial";
  let description = [{
    Example:

    ```mlir
    !poly_ty = !polynomial.polynomial<ring=<coefficientType=i32>>
    #poly = int<1 x**7 + 4> : !poly_ty
    #poly_verbose = #polynomial.typed_int_polynomial<1 x**7 + 4> : !poly_ty
    ```
  }];
  let parameters = (ins "::mlir::Type":$type, "::mlir::polynomial::IntPolynomialAttr":$value);
  let assemblyFormat = "$value `:` $type";
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const IntPolynomial &":$value), [{
      return $_get(
        type.getContext(),
        type,
        IntPolynomialAttr::get(type.getContext(), value));
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const Attribute &":$value), [{
      return $_get(type.getContext(), type, ::llvm::cast<IntPolynomialAttr>(value));
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = ::mlir::Attribute;
  }];
}

def Polynomial_TypedFloatPolynomialAttr : Polynomial_Attr<
    "TypedFloatPolynomial", "typed_float_polynomial", [TypedAttrInterface]> {
  let summary = "a typed float_polynomial";
  let description = [{
    Example:

    ```mlir
    !poly_ty = !polynomial.polynomial<ring=<coefficientType=f32>>
    #poly = float<1.4 x**7 + 4.5> : !poly_ty
    #poly_verbose = #polynomial.typed_float_polynomial<1.4 x**7 + 4.5> : !poly_ty
    ```
  }];
  let parameters = (ins "::mlir::Type":$type, "::mlir::polynomial::FloatPolynomialAttr":$value);
  let assemblyFormat = "$value `:` $type";
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const FloatPolynomial &":$value), [{
      return $_get(
        type.getContext(),
        type,
        FloatPolynomialAttr::get(type.getContext(), value));
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const Attribute &":$value), [{
      return $_get(type.getContext(), type, ::llvm::cast<FloatPolynomialAttr>(value));
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = ::mlir::Attribute;
  }];
}

def Polynomial_RingAttr : Polynomial_Attr<"Ring", "ring"> {
  let summary = "an attribute specifying a polynomial ring";
  let description = [{
    A ring describes the domain in which polynomial arithmetic occurs. The ring
    attribute in `polynomial` represents the more specific case of polynomials
    with a single indeterminate; whose coefficients can be represented by
    another MLIR type (`coefficientType`); and, if the coefficient type is
    integral, whose coefficients are taken modulo some statically known modulus
    (`coefficientModulus`).

    Additionally, a polynomial ring can specify a _polynomialModulus_, which converts
    polynomial arithmetic to the analogue of modular integer arithmetic, where
    each polynomial is represented as its remainder when dividing by the
    modulus. For single-variable polynomials, an "polynomialModulus" is always specificed
    via a single polynomial, which we call `polynomialModulus`.

    An expressive example is polynomials with i32 coefficients, whose
    coefficients are taken modulo `2**32 - 5`, with a polynomial modulus of
    `x**1024 - 1`.

    ```mlir
    #poly_mod = #polynomial.int_polynomial<-1 + x**1024>
    #ring = #polynomial.ring<coefficientType=i32,
                             coefficientModulus=4294967291:i32,
                             polynomialModulus=#poly_mod>

    %0 = ... : polynomial.polynomial<#ring>
    ```

    In this case, the value of a polynomial is always "converted" to a
    canonical form by applying repeated reductions by setting `x**1024 = 1`
    and simplifying.

    The coefficient and polynomial modulus parameters are optional, and the
    coefficient modulus is only allowed if the coefficient type is integral.
  }];

  let parameters = (ins
    "Type": $coefficientType,
    OptionalParameter<"::mlir::IntegerAttr">: $coefficientModulus,
    OptionalParameter<"::mlir::polynomial::IntPolynomialAttr">: $polynomialModulus,
    OptionalParameter<"::mlir::IntegerAttr">: $primitiveRoot
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let builders = [
    AttrBuilderWithInferredContext<
        (ins "::mlir::Type":$coefficientTy,
              CArg<"::mlir::IntegerAttr", "nullptr"> :$coefficientModulusAttr,
              CArg<"::mlir::polynomial::IntPolynomialAttr", "nullptr"> :$polynomialModulusAttr,
              CArg<"::mlir::IntegerAttr", "nullptr"> :$primitiveRootAttr), [{
      return $_get(
        coefficientTy.getContext(),
        coefficientTy,
        coefficientModulusAttr,
        polynomialModulusAttr,
        primitiveRootAttr);
    }]>,
  ];
}

#endif // POLYNOMIAL_ATTRIBUTES
