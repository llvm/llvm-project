//===-- enums.td - EnumsGen test definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def CaseA: StrEnumAttrCase<"CaseA">;
def CaseB: StrEnumAttrCase<"CaseB", 10>;

def StrEnum: StrEnumAttr<"StrEnum", "A test enum", [CaseA, CaseB]> {
  let cppNamespace = "Outer::Inner";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
  let genSpecializedAttr = 1;
}

def Case5: I32EnumAttrCase<"Case5", 5>;
def Case10: I32EnumAttrCase<"Case10", 10>;

def I32Enum: I32EnumAttr<"I32Enum", "A test enum", [Case5, Case10]>;

def NoBits : BitEnumAttrCaseNone<"None">;
def Bit0 : BitEnumAttrCaseBit<"Bit0", 0>;
def Bit1 : BitEnumAttrCaseBit<"Bit1", 1>;
def Bit2 : BitEnumAttrCaseBit<"Bit2", 2>;
def Bit3 : BitEnumAttrCaseBit<"Bit3", 3>;
def Bit4 : BitEnumAttrCaseBit<"Bit4", 4>;
def Bit5 : BitEnumAttrCaseBit<"Bit5", 5>;

def BitEnumWithNone : BitEnumAttr<"BitEnumWithNone", "A test enum",
                                  [NoBits, Bit0, Bit3]>;

def BitEnumWithoutNone : BitEnumAttr<"BitEnumWithoutNone", "A test enum",
                                     [Bit0, Bit3]>;

def Bits0To3 : BitEnumAttrCaseGroup<"Bits0To3",
                                    [Bit0, Bit1, Bit2, Bit3]>;

def BitEnumWithGroup : BitEnumAttr<"BitEnumWithGroup", "A test enum",
                                   [Bit0, Bit1, Bit2, Bit3, Bit4, Bits0To3]>;

def PrettyIntEnumCase1: I32EnumAttrCase<"Case1", 1, "case_one">;
def PrettyIntEnumCase2: I32EnumAttrCase<"Case2", 2, "case_two">;

def PrettyIntEnum: I32EnumAttr<"PrettyIntEnum", "A test enum",
                               [PrettyIntEnumCase1, PrettyIntEnumCase2]>;
