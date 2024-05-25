//===- PolynomialCanonicalization.td - Polynomial patterns -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYNOMIAL_CANONICALIZATION
#define POLYNOMIAL_CANONICALIZATION

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Polynomial/IR/Polynomial.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

defvar DefOverflow = ConstantEnumCase<Arith_IntegerOverflowAttr, "none">;

// Get a -1 integer attribute of the same type as the polynomial SSA value's
// ring coefficient type.
def getMinusOne
  : NativeCodeCall<
      "$_builder.getIntegerAttr("
        "cast<PolynomialType>($0.getType()).getRing().getCoefficientType(), -1)">;

def SubAsAdd : Pat<
  (Polynomial_SubOp $f, $g),
  (Polynomial_AddOp $f,
    (Polynomial_MulScalarOp $g,
      (Arith_ConstantOp (getMinusOne $g))))>;

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

// NTTs are expensive, and addition in coefficient or NTT domain should be
// equivalently expensive, so reducing the number of NTTs is optimal.
// ntt(a) + ntt(b) -> ntt(a + b)
def NTTOfAdd : Pat<
  (Arith_AddIOp
    (Polynomial_NTTOp $p1),
    (Polynomial_NTTOp $p2),
    $overflow),
  (Polynomial_NTTOp (Polynomial_AddOp $p1, $p2)),
  []
>;
// intt(a) + intt(b) -> intt(a + b)
def INTTOfAdd : Pat<
  (Polynomial_AddOp
    (Polynomial_INTTOp $t1),
    (Polynomial_INTTOp $t2)),
  (Polynomial_INTTOp (Arith_AddIOp $t1, $t2, DefOverflow)),
  []
>;
// repeated for sub
def NTTOfSub : Pat<
  (Arith_SubIOp
    (Polynomial_NTTOp $p1),
    (Polynomial_NTTOp $p2),
    $overflow),
  (Polynomial_NTTOp (Polynomial_SubOp $p1, $p2)),
  []
>;
def INTTOfSub : Pat<
  (Polynomial_SubOp
    (Polynomial_INTTOp $t1),
    (Polynomial_INTTOp $t2)),
  (Polynomial_INTTOp (Arith_SubIOp $t1, $t2, DefOverflow)),
  []
>;

#endif  // POLYNOMIAL_CANONICALIZATION
