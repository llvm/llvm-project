//===- PolynomialOps.td - Polynomial op definitions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYNOMIAL_OPS
#define POLYNOMIAL_OPS

include "PolynomialDialect.td"
include "PolynomialTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for polynomial dialect ops. Ops in this dialect have no side
// effects.
class Polynomial_Op<string mnemonic, list<Trait> traits = []> :
    Op<Polynomial_Dialect, mnemonic, traits # [Pure]>;

// Base class for unary polynomial operations.
class Polynomial_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Polynomial_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins Polynomial_PolynomialType:$operand);
  let results = (outs Polynomial_PolynomialType:$result);

  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";
}

// Base class for binary polynomial operations.
class Polynomial_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Polynomial_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins Polynomial_PolynomialType:$lhs, Polynomial_PolynomialType:$rhs);
  let results = (outs Polynomial_PolynomialType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
}

#endif // POLYNOMIAL_OPS
