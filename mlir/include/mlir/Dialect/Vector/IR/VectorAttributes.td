//===- VectorAttributes.td - Vector Dialect ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes used in the Vector dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_IR_VECTOR_ATTRIBUTES
#define MLIR_DIALECT_VECTOR_IR_VECTOR_ATTRIBUTES

include "mlir/Dialect/Vector/IR/Vector.td"
include "mlir/IR/EnumAttr.td"

// The "kind" of combining function for contractions and reductions.
def COMBINING_KIND_ADD : I32BitEnumAttrCaseBit<"ADD", 0, "add">;
def COMBINING_KIND_MUL : I32BitEnumAttrCaseBit<"MUL", 1, "mul">;
def COMBINING_KIND_MINUI : I32BitEnumAttrCaseBit<"MINUI", 2, "minui">;
def COMBINING_KIND_MINSI : I32BitEnumAttrCaseBit<"MINSI", 3, "minsi">;
def COMBINING_KIND_MINNUMF : I32BitEnumAttrCaseBit<"MINNUMF", 4, "minnumf">;
def COMBINING_KIND_MAXUI : I32BitEnumAttrCaseBit<"MAXUI", 5, "maxui">;
def COMBINING_KIND_MAXSI : I32BitEnumAttrCaseBit<"MAXSI", 6, "maxsi">;
def COMBINING_KIND_MAXNUMF : I32BitEnumAttrCaseBit<"MAXNUMF", 7, "maxnumf">;
def COMBINING_KIND_AND : I32BitEnumAttrCaseBit<"AND", 8, "and">;
def COMBINING_KIND_OR  : I32BitEnumAttrCaseBit<"OR", 9, "or">;
def COMBINING_KIND_XOR : I32BitEnumAttrCaseBit<"XOR", 10, "xor">;
def COMBINING_KIND_MINIMUMF : I32BitEnumAttrCaseBit<"MINIMUMF", 11, "minimumf">;
def COMBINING_KIND_MAXIMUMF : I32BitEnumAttrCaseBit<"MAXIMUMF", 12, "maximumf">;

def CombiningKind : I32BitEnumAttr<
    "CombiningKind",
    "Kind of combining function for contractions and reductions",
    [COMBINING_KIND_ADD, COMBINING_KIND_MUL, COMBINING_KIND_MINUI,
     COMBINING_KIND_MINSI, COMBINING_KIND_MINNUMF, COMBINING_KIND_MAXUI,
     COMBINING_KIND_MAXSI, COMBINING_KIND_MAXNUMF, COMBINING_KIND_AND,
     COMBINING_KIND_OR, COMBINING_KIND_XOR,
     COMBINING_KIND_MAXIMUMF, COMBINING_KIND_MINIMUMF]> {
  let cppNamespace = "::mlir::vector";
  let genSpecializedAttr = 0;
}

/// An attribute that specifies the combining function for `vector.contract`,
/// and `vector.reduction`.
def Vector_CombiningKindAttr : EnumAttr<Vector_Dialect, CombiningKind, "kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def Vector_IteratorType : I32EnumAttr<"IteratorType", "Iterator type", [
  I32EnumAttrCase<"parallel", 0>,
  I32EnumAttrCase<"reduction", 1>
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::vector";
}

def Vector_IteratorTypeEnum
    : EnumAttr<Vector_Dialect, Vector_IteratorType, "iterator_type"> {
    let assemblyFormat = "`<` $value `>`";
}

def Vector_IteratorTypeArrayAttr
    : TypedArrayAttrBase<Vector_IteratorTypeEnum,
                         "Iterator type should be an enum.">;

def PrintPunctuation : I32EnumAttr<"PrintPunctuation",
                                  "Punctuation for separating vectors or vector elements", [
  I32EnumAttrCase<"NoPunctuation", 0, "no_punctuation">,
  I32EnumAttrCase<"NewLine", 1, "newline">,
  I32EnumAttrCase<"Comma", 2, "comma">,
  I32EnumAttrCase<"Open", 3, "open">,
  I32EnumAttrCase<"Close", 4, "close">
]> {
  let cppNamespace = "::mlir::vector";
  let genSpecializedAttr = 0;
}

def Vector_PrintPunctuation : EnumAttr<Vector_Dialect, PrintPunctuation, "punctuation"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // MLIR_DIALECT_VECTOR_IR_VECTOR_ATTRIBUTES
