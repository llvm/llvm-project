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
}

def Case5: I32EnumAttrCase<"Case5", 5>;
def Case10: I32EnumAttrCase<"Case10", 10>;

def I32Enum: I32EnumAttr<"I32Enum", "A test enum", [Case5, Case10]>;

def Bit0 : BitEnumAttrCase<"None", 0x0000>;
def Bit1 : BitEnumAttrCase<"Bit1", 0x0001>;
def Bit3 : BitEnumAttrCase<"Bit3", 0x0004>;

def BitEnumWithNone : BitEnumAttr<"BitEnumWithNone", "A test enum",
                                  [Bit0, Bit1, Bit3]>;

def BitEnumWithoutNone : BitEnumAttr<"BitEnumWithoutNone", "A test enum",
                                     [Bit1, Bit3]>;

def PrettyIntEnumCase1: I32EnumAttrCase<"Case1", 1, "case_one">;
def PrettyIntEnumCase2: I32EnumAttrCase<"Case2", 2, "case_two">;

def PrettyIntEnum: I32EnumAttr<"PrettyIntEnum", "A test enum",
                               [PrettyIntEnumCase1, PrettyIntEnumCase2]>;
