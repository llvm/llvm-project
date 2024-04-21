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

def PolynomialLike: TypeOrContainer<Polynomial_PolynomialType, "polynomial-like">;

class Polynomial_Op<string mnemonic, list<Trait> traits = []> :
    Op<Polynomial_Dialect, mnemonic, traits # [Pure]> {
  let assemblyFormat = [{
    operands attr-dict `:` `(` qualified(type(operands)) `)` `->` qualified(type(results))
  }];
}

class Polynomial_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Polynomial_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins Polynomial_PolynomialType:$operand);
  let results = (outs Polynomial_PolynomialType:$result);
}

class Polynomial_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Polynomial_Op<mnemonic, !listconcat(traits, [Pure, SameOperandsAndResultType, ElementwiseMappable])> {
  let arguments = (ins PolynomialLike:$lhs, PolynomialLike:$rhs);
  let results = (outs PolynomialLike:$result);
  let assemblyFormat = "operands attr-dict `:` qualified(type($result))";
}

def Polynomial_AddOp : Polynomial_BinaryOp<"add", [Commutative]> {
  let summary = "Addition operation between polynomials.";
}

def Polynomial_SubOp : Polynomial_BinaryOp<"sub"> {
  let summary = "Subtraction operation between polynomials.";
}

def Polynomial_MulOp : Polynomial_BinaryOp<"mul", [Commutative]> {
  let summary = "Multiplication operation between polynomials.";
}

def Polynomial_MulScalarOp : Polynomial_Op<"mul_scalar", [
      ElementwiseMappable, AllTypesMatch<["polynomial", "output"]>]> {
  let summary = "Multiplication by a scalar of the field.";

  let arguments = (ins
    PolynomialLike:$polynomial,
    AnyInteger:$scalar
  );

  let results = (outs
    PolynomialLike:$output
  );

  let assemblyFormat = "operands attr-dict `:` qualified(type($polynomial)) `,` type($scalar)";
}

def Polynomial_LeadingTermOp: Polynomial_Op<"leading_term"> {
  let summary = "Compute the leading term of the polynomial.";
  let description = [{
    The degree of a polynomial is the largest $k$ for which the coefficient
    $a_k$ of $x^k$ is nonzero. The leading term is the term $a_k x^k$, which
    this op represents as a pair of results.
  }];
  let arguments = (ins Polynomial_PolynomialType:$input);
  let results = (outs Index:$degree, AnyInteger:$coefficient);
  let assemblyFormat = "operands attr-dict `:` qualified(type($input)) `->` `(` type($degree) `,` type($coefficient) `)`";
}

def Polynomial_MonomialOp: Polynomial_Op<"monomial"> {
  let summary = "Create a polynomial that consists of a single monomial.";
  let arguments = (ins AnyInteger:$coefficient, Index:$degree);
  let results = (outs Polynomial_PolynomialType:$output);
}

def Polynomial_MonomialMulOp: Polynomial_Op<"monomial_mul", [AllTypesMatch<["input", "output"]>]> {
  let summary = "Multiply a polynomial by a monic monomial.";
  let description = [{
    In the ring of polynomials mod $x^n - 1$, `monomial_mul` can be interpreted
    as a cyclic shift of the coefficients of the polynomial. For some rings,
    this results in optimized lowerings that involve rotations and rescaling
    of the coefficients of the input.
  }];
  let arguments = (ins Polynomial_PolynomialType:$input, Index:$monomialDegree);
  let results = (outs Polynomial_PolynomialType:$output);
  let hasVerifier = 1;
}

def Polynomial_FromTensorOp : Polynomial_Op<"from_tensor", [Pure]> {
  let summary = "Creates a polynomial from integer coefficients stored in a tensor.";
  let description = [{
    `polynomial.from_tensor` creates a polynomial value from a tensor of coefficients.
    The input tensor must list the coefficients in degree-increasing order.

    The input one-dimensional tensor may have size at most the degree of the
    ring's ideal generator polynomial, with smaller dimension implying that
    all higher-degree terms have coefficient zero.
  }];
  let arguments = (ins RankedTensorOf<[AnyInteger]>:$input);
  let results = (outs Polynomial_PolynomialType:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` qualified(type($output))";

  let builders = [
    // Builder that infers coefficient modulus from tensor bit width,
    // and uses whatever input ring is provided by the caller.
    OpBuilder<(ins "::mlir::Value":$input, "RingAttr":$ring)>
  ];
  let hasVerifier = 1;
}

def Polynomial_ToTensorOp : Polynomial_Op<"to_tensor", [Pure]> {
  let summary = "Creates a tensor containing the coefficients of a polynomial.";
  let description = [{
    `polynomial.to_tensor` creates a tensor value containing the coefficients of the
    input polynomial. The output tensor contains the coefficients in
    degree-increasing order.

    Operations that act on the coefficients of a polynomial, such as extracting
    a specific coefficient or extracting a range of coefficients, should be
    implemented by composing `to_tensor` with the relevant `tensor` dialect
    ops.

    The output tensor has shape equal to the degree of the ring's ideal
    generator polynomial, including zeroes.
  }];
  let arguments = (ins Polynomial_PolynomialType:$input);
  let results = (outs RankedTensorOf<[AnyInteger]>:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($output)";

  let hasVerifier = 1;
}

def Polynomial_ConstantOp : Polynomial_Op<"constant", [Pure]> {
  let summary = "Define a constant polynomial via an attribute.";
  let arguments = (ins Polynomial_PolynomialAttr:$input);
  let results = (outs Polynomial_PolynomialType:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($output))";
}

#endif // POLYNOMIAL_OPS
