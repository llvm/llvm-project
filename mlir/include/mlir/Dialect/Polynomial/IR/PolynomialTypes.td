//===- PolynomialTypes.td - Polynomial types ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYNOMIAL_TYPES
#define POLYNOMIAL_TYPES

include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "mlir/Dialect/Polynomial/IR/PolynomialDialect.td"

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
  let assemblyFormat = "`<` struct(params) `>`";
}

def PolynomialLike: TypeOrContainer<Polynomial_PolynomialType, "polynomial-like">;


#endif // POLYNOMIAL_TYPES
