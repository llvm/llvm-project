//===-- enums.td - EnumsGen test definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def CaseA: I32EnumAttrCase<"CaseA", 0>;
def CaseB: I32EnumAttrCase<"CaseB", 1>;

def FooEnum: I32EnumAttr<"FooEnum", "A test enum", [CaseA, CaseB]> {
  let cppNamespace = "Outer::Inner";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
  let genSpecializedAttr = 1;
}

def Case5: I32EnumAttrCase<"Case5", 5>;
def Case10: I32EnumAttrCase<"Case10", 10>;

def I32Enum: I32EnumAttr<"I32Enum", "A test enum", [Case5, Case10]>;

def NoBits : I32BitEnumAttrCaseNone<"None">;
def Bit0 : I32BitEnumAttrCaseBit<"Bit0", 0>;
def Bit1 : I32BitEnumAttrCaseBit<"Bit1", 1>;
def Bit2 : I32BitEnumAttrCaseBit<"Bit2", 2>;
def Bit3 : I32BitEnumAttrCaseBit<"Bit3", 3>;
def Bit4 : I32BitEnumAttrCaseBit<"Bit4", 4>;
def Bit5 : I32BitEnumAttrCaseBit<"Bit5", 5>;

def BitEnumWithNone : I32BitEnumAttr<"BitEnumWithNone", "A test enum",
                                     [NoBits, Bit0, Bit3]> {
  let separator = " | ";
}

def BitEnumWithoutNone : I32BitEnumAttr<"BitEnumWithoutNone", "A test enum",
                                        [Bit0, Bit3]>;

def Bits0To3 : I32BitEnumAttrCaseGroup<"Bits0To3",
                                       [Bit0, Bit1, Bit2, Bit3]>;
def Bits4And5 : I32BitEnumAttrCaseGroup<"Bits4And5",
                                       [Bit4, Bit5]>;
def Bits0To5 : I32BitEnumAttrCaseGroup<"Bits0To5",
                                       [Bits0To3, Bits4And5]>;

def BitEnumWithGroup : I32BitEnumAttr<"BitEnumWithGroup", "A test enum",
                                      [Bit0, Bit1, Bit2, Bit3, Bit4, Bits0To3]> {
  let separator = "|";
}

def BitEnumPrimaryGroup : I32BitEnumAttr<"BitEnumPrimaryGroup", "test enum",
                                        [Bit0, Bit1, Bit2, Bit3, Bit4, Bit5,
                                         Bits0To3, Bits4And5, Bits0To5]> {
  let separator = ", ";
  let printBitEnumPrimaryGroups = 1;
}

def BitEnum64_None : I64BitEnumAttrCaseNone<"None">;
def BitEnum64_57   : I64BitEnumAttrCaseBit<"Bit57", 57>;
def BitEnum64_1    : I64BitEnumAttrCaseBit<"Bit1", 1>;
def BitEnum64_Test : I64BitEnumAttr<"BitEnum64_Test", "A 64-bit test enum",
                                    [BitEnum64_None, BitEnum64_1, BitEnum64_57]>;

def PrettyIntEnumCase1: I32EnumAttrCase<"Case1", 1, "case_one">;
def PrettyIntEnumCase2: I32EnumAttrCase<"Case2", 2, "case_two">;

def PrettyIntEnum: I32EnumAttr<"PrettyIntEnum", "A test enum",
                               [PrettyIntEnumCase1, PrettyIntEnumCase2]>;
