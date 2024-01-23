//===- ArithBase.td - Base defs for arith dialect -----------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARITH_BASE
#define ARITH_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Arith_Dialect : Dialect {
  let name = "arith";
  let cppNamespace = "::mlir::arith";
  let description = [{
    The arith dialect is intended to hold basic integer and floating point
    mathematical operations. This includes unary, binary, and ternary arithmetic
    ops, bitwise and shift ops, cast ops, and compare ops. Operations in this
    dialect also accept vectors and tensors of integers or floats. The dialect
    assumes integers are represented by bitvectors with a two's complement 
    representation. Unless otherwise stated, the operations within this dialect 
    propagate poison values, i.e., if any of its inputs are poison, then the 
    output is poison. Unless otherwise stated, operations applied to `vector` 
    and `tensor` values propagates poison elementwise.
  }];

  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
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

def ATOMIC_RMW_KIND_ADDF     : I64EnumAttrCase<"addf", 0>;
def ATOMIC_RMW_KIND_ADDI     : I64EnumAttrCase<"addi", 1>;
def ATOMIC_RMW_KIND_ASSIGN   : I64EnumAttrCase<"assign", 2>;
def ATOMIC_RMW_KIND_MAXIMUMF : I64EnumAttrCase<"maximumf", 3>;
def ATOMIC_RMW_KIND_MAXS     : I64EnumAttrCase<"maxs", 4>;
def ATOMIC_RMW_KIND_MAXU     : I64EnumAttrCase<"maxu", 5>;
def ATOMIC_RMW_KIND_MINIMUMF : I64EnumAttrCase<"minimumf", 6>;
def ATOMIC_RMW_KIND_MINS     : I64EnumAttrCase<"mins", 7>;
def ATOMIC_RMW_KIND_MINU     : I64EnumAttrCase<"minu", 8>;
def ATOMIC_RMW_KIND_MULF     : I64EnumAttrCase<"mulf", 9>;
def ATOMIC_RMW_KIND_MULI     : I64EnumAttrCase<"muli", 10>;
def ATOMIC_RMW_KIND_ORI      : I64EnumAttrCase<"ori", 11>;
def ATOMIC_RMW_KIND_ANDI     : I64EnumAttrCase<"andi", 12>;
def ATOMIC_RMW_KIND_MAXNUMF  : I64EnumAttrCase<"maxnumf", 13>;
def ATOMIC_RMW_KIND_MINNUMF  : I64EnumAttrCase<"minnumf", 14>;

def AtomicRMWKindAttr : I64EnumAttr<
    "AtomicRMWKind", "",
    [ATOMIC_RMW_KIND_ADDF, ATOMIC_RMW_KIND_ADDI, ATOMIC_RMW_KIND_ASSIGN,
     ATOMIC_RMW_KIND_MAXIMUMF, ATOMIC_RMW_KIND_MAXS, ATOMIC_RMW_KIND_MAXU,
     ATOMIC_RMW_KIND_MINIMUMF, ATOMIC_RMW_KIND_MINS, ATOMIC_RMW_KIND_MINU,
     ATOMIC_RMW_KIND_MULF, ATOMIC_RMW_KIND_MULI, ATOMIC_RMW_KIND_ORI,
     ATOMIC_RMW_KIND_ANDI, ATOMIC_RMW_KIND_MAXNUMF, ATOMIC_RMW_KIND_MINNUMF]> {
  let cppNamespace = "::mlir::arith";
}

def FASTMATH_NONE            : I32BitEnumAttrCaseNone<"none"      >;
def FASTMATH_REASSOC         : I32BitEnumAttrCaseBit<"reassoc",  0>;
def FASTMATH_NO_NANS         : I32BitEnumAttrCaseBit<"nnan",     1>;
def FASTMATH_NO_INFS         : I32BitEnumAttrCaseBit<"ninf",     2>;
def FASTMATH_NO_SIGNED_ZEROS : I32BitEnumAttrCaseBit<"nsz",      3>;
def FASTMATH_ALLOW_RECIP     : I32BitEnumAttrCaseBit<"arcp",     4>;
def FASTMATH_ALLOW_CONTRACT  : I32BitEnumAttrCaseBit<"contract", 5>;
def FASTMATH_APPROX_FUNC     : I32BitEnumAttrCaseBit<"afn",      6>;
def FASTMATH_FAST            : I32BitEnumAttrCaseGroup<
    "fast",
    [
      FASTMATH_REASSOC,         FASTMATH_NO_NANS,     FASTMATH_NO_INFS,
      FASTMATH_NO_SIGNED_ZEROS, FASTMATH_ALLOW_RECIP, FASTMATH_ALLOW_CONTRACT,
      FASTMATH_APPROX_FUNC]>;

def FastMathFlags : I32BitEnumAttr<
    "FastMathFlags",
    "Floating point fast math flags",
    [
      FASTMATH_NONE,           FASTMATH_REASSOC,         FASTMATH_NO_NANS,
      FASTMATH_NO_INFS,        FASTMATH_NO_SIGNED_ZEROS, FASTMATH_ALLOW_RECIP,
      FASTMATH_ALLOW_CONTRACT, FASTMATH_APPROX_FUNC,     FASTMATH_FAST]> {
  let separator = ",";
  let cppNamespace = "::mlir::arith";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def Arith_FastMathAttr :
    EnumAttr<Arith_Dialect, FastMathFlags, "fastmath"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Arith_IntegerOverflowFlags
//===----------------------------------------------------------------------===//

def Arith_IOFnone : I32BitEnumAttrCaseNone<"none">;
def Arith_IOFnsw  : I32BitEnumAttrCaseBit<"nsw", 0>;
def Arith_IOFnuw  : I32BitEnumAttrCaseBit<"nuw", 1>;

def Arith_IntegerOverflowFlags : I32BitEnumAttr<
    "IntegerOverflowFlags",
    "Integer overflow arith flags",
    [Arith_IOFnone, Arith_IOFnsw, Arith_IOFnuw]> {
  let separator = ", ";
  let cppNamespace = "::mlir::arith";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def Arith_IntegerOverflowAttr :
    EnumAttr<Arith_Dialect, Arith_IntegerOverflowFlags, "overflow"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // ARITH_BASE
