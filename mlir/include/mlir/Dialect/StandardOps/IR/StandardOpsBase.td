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

#endif // STANDARD_OPS_BASE
