//===- PolynomialOps.td - Polynomial dialect ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYNOMIAL_OPS
#define POLYNOMIAL_OPS

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Polynomial_Dialect : Dialect {
  let name = "polynomial";
  let cppNamespace = "::mlir::polynomial";
  let description = [{
    The Polynomial dialect defines single-variable polynomial types and
    operations.

    The simplest use of `polynomial` is to represent mathematical operations in
    a polynomial ring `R[x]`, where `R` is another MLIR type like `i32`.

    More generally, this dialect supports representing polynomial operations in a
    quotient ring `R[X]/(f(x))` for some statically fixed polynomial `f(x)`.
    Two polyomials `p(x), q(x)` are considered equal in this ring if they have the
    same remainder when dividing by `f(x)`. When a modulus is given, ring operations
    are performed with reductions modulo `f(x)` and relative to the coefficient ring
    `R`.

    Examples:

    ```mlir
    // A constant polynomial in a ring with i32 coefficients and no polynomial modulus
    #ring = #polynomial.ring<ctype=i32>
    %a = polynomial.constant <1 + x**2 - 3x**3> : polynomial.polynomial<#ring>

    // A constant polynomial in a ring with i32 coefficients, modulo (x^1024 + 1)
    #modulus = #polynomial.polynomial<1 + x**1024>
    #ring = #polynomial.ring<ctype=i32, ideal=#modulus>
    %a = polynomial.constant <1 + x**2 - 3x**3> : polynomial.polynomial<#ring>

    // A constant polynomial in a ring with i32 coefficients, with a polynomial
    // modulus of (x^1024 + 1) and a coefficient modulus of 17.
    #modulus = #polynomial.polynomial<1 + x**1024>
    #ring = #polynomial.ring<ctype=i32, cmod=17, ideal=#modulus>
    %a = polynomial.constant <1 + x**2 - 3x**3> : polynomial.polynomial<#ring>
    ```
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

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
  let hasCustomAssemblyFormat = 1;
}

def Polynomial_RingAttr : Polynomial_Attr<"Ring", "ring"> {
  let summary = "An attribute specifying a polynomial ring.";
  let description = [{
    A ring describes the domain in which polynomial arithmetic occurs. The ring
    attribute in `polynomial` represents the more specific case of polynomials
    with a single indeterminate; whose coefficients can be represented by
    another MLIR type (`coefficientType`); and, if the coefficient type is
    integral, whose coefficients are taken modulo some statically known modulus
    (`coefficientModulus`).

    Additionally, a polynomial ring can specify an _ideal_, which converts
    polynomial arithmetic to the analogue of modular integer arithmetic, where
    each polynomial is represented as its remainder when dividing by the
    modulus. For single-variable polynomials, an "ideal" is always specificed
    via a single polynomial, which we call `polynomialModulus`.

    An expressive example is polynomials with i32 coefficients, whose
    coefficients are taken modulo `2**32 - 5`, with a polynomial modulus of
    `x**1024 - 1`.

    ```mlir
    #poly_mod = #polynomial.polynomial<-1 + x**1024>
    #ring = #polynomial.ring<coefficientType=i32,
                             coefficientModulus=4294967291,
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
    OptionalParameter<"IntegerAttr">: $coefficientModulus,
    OptionalParameter<"PolynomialAttr">: $polynomialModulus
  );

  let hasCustomAssemblyFormat = 1;
}

class Polynomial_Type<string name, string typeMnemonic>
    : TypeDef<Polynomial_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Polynomial_PolynomialType : Polynomial_Type<"Polynomial", "polynomial"> {
  let summary = "An element of a polynomial ring.";

  let description = [{
    A type for polynomials in a polynomial quotient ring.
  }];

  let parameters = (ins Polynomial_RingAttr:$ring);
  let assemblyFormat = "`<` $ring `>`";
}

class Polynomial_Op<string mnemonic, list<Trait> traits = []> :
    Op<Polynomial_Dialect, mnemonic, traits # [Pure]>;

class Polynomial_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Polynomial_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins Polynomial_PolynomialType:$operand);
  let results = (outs Polynomial_PolynomialType:$result);

  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";
}

class Polynomial_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Polynomial_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins Polynomial_PolynomialType:$lhs, Polynomial_PolynomialType:$rhs);
  let results = (outs Polynomial_PolynomialType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
}

#endif // POLYNOMIAL_OPS
