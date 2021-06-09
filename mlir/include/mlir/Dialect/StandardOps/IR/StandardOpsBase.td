//===- StandardOpsBase.td - Standard ops definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines base support for standard operations.
//
//===----------------------------------------------------------------------===//

#ifndef STANDARD_OPS_BASE
#define STANDARD_OPS_BASE

include "mlir/IR/OpBase.td"

def ATOMIC_RMW_KIND_ADDF    : I64EnumAttrCase<"addf", 0>;
def ATOMIC_RMW_KIND_ADDI    : I64EnumAttrCase<"addi", 1>;
def ATOMIC_RMW_KIND_ASSIGN  : I64EnumAttrCase<"assign", 2>;
def ATOMIC_RMW_KIND_MAXF    : I64EnumAttrCase<"maxf", 3>;
def ATOMIC_RMW_KIND_MAXS    : I64EnumAttrCase<"maxs", 4>;
def ATOMIC_RMW_KIND_MAXU    : I64EnumAttrCase<"maxu", 5>;
def ATOMIC_RMW_KIND_MINF    : I64EnumAttrCase<"minf", 6>;
def ATOMIC_RMW_KIND_MINS    : I64EnumAttrCase<"mins", 7>;
def ATOMIC_RMW_KIND_MINU    : I64EnumAttrCase<"minu", 8>;
def ATOMIC_RMW_KIND_MULF    : I64EnumAttrCase<"mulf", 9>;
def ATOMIC_RMW_KIND_MULI    : I64EnumAttrCase<"muli", 10>;

def AtomicRMWKindAttr : I64EnumAttr<
    "AtomicRMWKind", "",
    [ATOMIC_RMW_KIND_ADDF, ATOMIC_RMW_KIND_ADDI, ATOMIC_RMW_KIND_ASSIGN,
     ATOMIC_RMW_KIND_MAXF, ATOMIC_RMW_KIND_MAXS, ATOMIC_RMW_KIND_MAXU,
     ATOMIC_RMW_KIND_MINF, ATOMIC_RMW_KIND_MINS, ATOMIC_RMW_KIND_MINU,
     ATOMIC_RMW_KIND_MULF, ATOMIC_RMW_KIND_MULI]> {
  let cppNamespace = "::mlir";
}

// The predicate indicates the type of the comparison to perform:
// (un)orderedness, (in)equality and less/greater than (or equal to) as
// well as predicates that are always true or false.
def CMPF_P_FALSE   : I64EnumAttrCase<"AlwaysFalse", 0, "false">;
def CMPF_P_OEQ     : I64EnumAttrCase<"OEQ", 1, "oeq">;
def CMPF_P_OGT     : I64EnumAttrCase<"OGT", 2, "ogt">;
def CMPF_P_OGE     : I64EnumAttrCase<"OGE", 3, "oge">;
def CMPF_P_OLT     : I64EnumAttrCase<"OLT", 4, "olt">;
def CMPF_P_OLE     : I64EnumAttrCase<"OLE", 5, "ole">;
def CMPF_P_ONE     : I64EnumAttrCase<"ONE", 6, "one">;
def CMPF_P_ORD     : I64EnumAttrCase<"ORD", 7, "ord">;
def CMPF_P_UEQ     : I64EnumAttrCase<"UEQ", 8, "ueq">;
def CMPF_P_UGT     : I64EnumAttrCase<"UGT", 9, "ugt">;
def CMPF_P_UGE     : I64EnumAttrCase<"UGE", 10, "uge">;
def CMPF_P_ULT     : I64EnumAttrCase<"ULT", 11, "ult">;
def CMPF_P_ULE     : I64EnumAttrCase<"ULE", 12, "ule">;
def CMPF_P_UNE     : I64EnumAttrCase<"UNE", 13, "une">;
def CMPF_P_UNO     : I64EnumAttrCase<"UNO", 14, "uno">;
def CMPF_P_TRUE    : I64EnumAttrCase<"AlwaysTrue", 15, "true">;

def CmpFPredicateAttr : I64EnumAttr<
    "CmpFPredicate", "",
    [CMPF_P_FALSE, CMPF_P_OEQ, CMPF_P_OGT, CMPF_P_OGE, CMPF_P_OLT, CMPF_P_OLE,
     CMPF_P_ONE, CMPF_P_ORD, CMPF_P_UEQ, CMPF_P_UGT, CMPF_P_UGE, CMPF_P_ULT,
     CMPF_P_ULE, CMPF_P_UNE, CMPF_P_UNO, CMPF_P_TRUE]> {
  let cppNamespace = "::mlir";
}

def CMPI_P_EQ  : I64EnumAttrCase<"eq", 0>;
def CMPI_P_NE  : I64EnumAttrCase<"ne", 1>;
def CMPI_P_SLT : I64EnumAttrCase<"slt", 2>;
def CMPI_P_SLE : I64EnumAttrCase<"sle", 3>;
def CMPI_P_SGT : I64EnumAttrCase<"sgt", 4>;
def CMPI_P_SGE : I64EnumAttrCase<"sge", 5>;
def CMPI_P_ULT : I64EnumAttrCase<"ult", 6>;
def CMPI_P_ULE : I64EnumAttrCase<"ule", 7>;
def CMPI_P_UGT : I64EnumAttrCase<"ugt", 8>;
def CMPI_P_UGE : I64EnumAttrCase<"uge", 9>;

def CmpIPredicateAttr : I64EnumAttr<
    "CmpIPredicate", "",
    [CMPI_P_EQ, CMPI_P_NE, CMPI_P_SLT, CMPI_P_SLE, CMPI_P_SGT,
     CMPI_P_SGE, CMPI_P_ULT, CMPI_P_ULE, CMPI_P_UGT, CMPI_P_UGE]> {
  let cppNamespace = "::mlir";
}

#endif // STANDARD_OPS_BASE
