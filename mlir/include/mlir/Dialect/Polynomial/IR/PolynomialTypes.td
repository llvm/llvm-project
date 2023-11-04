//===- PolynomialTypes.td - Type definitions for polynomial dialect ------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef POLYNOMIAL_TYPES
#define POLYNOMIAL_TYPES

include "PolynomialAttributes.td"
include "PolynomialDialect.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

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

#endif // POLYNOMIAL_TYPES
