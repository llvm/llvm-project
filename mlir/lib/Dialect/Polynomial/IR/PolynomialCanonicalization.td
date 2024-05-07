//===- PolynomialCanonicalization.td - Polynomial patterns -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYNOMIAL_CANONICALIZATION
#define POLYNOMIAL_CANONICALIZATION

include "mlir/Dialect/Polynomial/IR/Polynomial.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

// TODO: get the proper scalar type from the operand polynomial ring attribute
def SubAsAdd : Pat<
  (Polynomial_SubOp $f, $g),
  (Polynomial_AddOp $f,
    (Polynomial_MulScalarOp $g,
      (Arith_ConstantOp
        ConstantAttr<I32Attr, "-1">)))>;

def INTTAfterNTT : Pat<
  (Polynomial_INTTOp (Polynomial_NTTOp $poly)),
  (replaceWithValue $poly),
  []
>;

def NTTAfterINTT : Pat<
  (Polynomial_NTTOp (Polynomial_INTTOp $tensor)),
  (replaceWithValue $tensor),
  []
>;

#endif  // POLYNOMIAL_CANONICALIZATION
