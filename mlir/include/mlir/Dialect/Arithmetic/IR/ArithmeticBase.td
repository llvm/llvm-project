//===- ArithmeticBase.td - Base defs for arith dialect ------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARITHMETIC_BASE
#define ARITHMETIC_BASE

include "mlir/IR/OpBase.td"

def Arithmetic_Dialect : Dialect {
  let name = "arith";
  let cppNamespace = "::mlir::arith";
  let description = [{
    The arithmetic dialect is intended to hold basic integer and floating point
    mathematical operations. This includes unary, binary, and ternary arithmetic
    ops, bitwise and shift ops, cast ops, and compare ops. Operations in this
    dialect also accept vectors and tensors of integers or floats.
  }];

  let hasConstantMaterializer = 1;
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

// The predicate indicates the type of the comparison to perform:
// (un)orderedness, (in)equality and less/greater than (or equal to) as
// well as predicates that are always true or false.
def Arith_CmpFPredicateAttr : I64EnumAttr<
    "CmpFPredicate", "",
    [
      I64EnumAttrCase<"AlwaysFalse", 0, "false">,
      I64EnumAttrCase<"OEQ", 1, "oeq">,
      I64EnumAttrCase<"OGT", 2, "ogt">,
      I64EnumAttrCase<"OGE", 3, "oge">,
      I64EnumAttrCase<"OLT", 4, "olt">,
      I64EnumAttrCase<"OLE", 5, "ole">,
      I64EnumAttrCase<"ONE", 6, "one">,
      I64EnumAttrCase<"ORD", 7, "ord">,
      I64EnumAttrCase<"UEQ", 8, "ueq">,
      I64EnumAttrCase<"UGT", 9, "ugt">,
      I64EnumAttrCase<"UGE", 10, "uge">,
      I64EnumAttrCase<"ULT", 11, "ult">,
      I64EnumAttrCase<"ULE", 12, "ule">,
      I64EnumAttrCase<"UNE", 13, "une">,
      I64EnumAttrCase<"UNO", 14, "uno">,
      I64EnumAttrCase<"AlwaysTrue", 15, "true">,
    ]> {
  let cppNamespace = "::mlir::arith";
}

def Arith_CmpIPredicateAttr : I64EnumAttr<
    "CmpIPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"slt", 2>,
      I64EnumAttrCase<"sle", 3>,
      I64EnumAttrCase<"sgt", 4>,
      I64EnumAttrCase<"sge", 5>,
      I64EnumAttrCase<"ult", 6>,
      I64EnumAttrCase<"ule", 7>,
      I64EnumAttrCase<"ugt", 8>,
      I64EnumAttrCase<"uge", 9>,
    ]> {
  let cppNamespace = "::mlir::arith";
}

#endif // ARITHMETIC_BASE
