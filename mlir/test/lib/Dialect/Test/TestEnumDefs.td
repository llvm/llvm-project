//===-- TestEnumDefs.td - Test dialect enum definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TableGen enum definitions for Test dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_ENUMDEFS
#define TEST_ENUMDEFS

include "mlir/IR/EnumAttr.td"

def I32Case5:  I32EnumAttrCase<"case5", 5>;
def I32Case10: I32EnumAttrCase<"case10", 10>;
def I32CaseSignedMaxPlusOne
    : I32EnumAttrCase<"caseSignedMaxPlusOne", 2147483648>;
def I32CaseUnsignedMax : I32EnumAttrCase<"caseUnsignedMax", 4294967295>;

def SomeI32Enum : I32EnumAttr<"SomeI32Enum", "",
                              [I32Case5, I32Case10, I32CaseSignedMaxPlusOne,
                               I32CaseUnsignedMax]>;

def I64Case5:  I64EnumAttrCase<"case5", 5>;
def I64Case10: I64EnumAttrCase<"case10", 10>;
def I64Case1p32 : I64EnumAttrCase<"caseLarse", 4294967296>;

def SomeI64Enum: I64EnumAttr<
  "SomeI64Enum", "", [I64Case5, I64Case10, I64Case1p32]>;

//===----------------------------------------------------------------------===//
// Test Enum
//===----------------------------------------------------------------------===//

// Define the C++ enum.
def TestEnum
    : I32EnumAttr<"TestEnum", "a test enum", [
        I32EnumAttrCase<"First", 0, "first">,
        I32EnumAttrCase<"Second", 1, "second">,
        I32EnumAttrCase<"Third", 2, "third">,
      ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "test";
}

def TestSimpleEnum : I32Enum<"SimpleEnum", "", [
    I32EnumCase<"a", 0>,
    I32EnumCase<"b", 1>
  ]> {
  let cppNamespace = "::test";
}

def TestSimpleEnum64 : I64Enum<"SimpleEnum64", "", [
    I64EnumCase<"a", 4294967296>,
    I64EnumCase<"b", 4294967297>
  ]> {
  let cppNamespace = "::test";
}

//===----------------------------------------------------------------------===//
// Test Bit Enum
//===----------------------------------------------------------------------===//

// Define the C++ enum.
def TestBitEnum
    : I32BitEnum<"TestBitEnum", "a test bit enum", [
        I32BitEnumCaseBit<"Read", 0, "read">,
        I32BitEnumCaseBit<"Write", 1, "write">,
        I32BitEnumCaseBit<"Execute", 2, "execute">,
      ]> {
  let cppNamespace = "test";
  let separator = ", ";
}

// Define an enum with a different separator
def TestBitEnumVerticalBar
    : I32BitEnum<"TestBitEnumVerticalBar", "another test bit enum", [
        I32BitEnumCaseBit<"User", 0, "user">,
        I32BitEnumCaseBit<"Group", 1, "group">,
        I32BitEnumCaseBit<"Other", 2, "other">,
      ]> {
  let cppNamespace = "test";
  let separator = " | ";
}

//===----------------------------------------------------------------------===//
// Test Patterns (Multi-result Ops)
//===----------------------------------------------------------------------===//

def MultiResultOpKind1: I64EnumAttrCase<"kind1", 1>;
def MultiResultOpKind2: I64EnumAttrCase<"kind2", 2>;
def MultiResultOpKind3: I64EnumAttrCase<"kind3", 3>;
def MultiResultOpKind4: I64EnumAttrCase<"kind4", 4>;
def MultiResultOpKind5: I64EnumAttrCase<"kind5", 5>;
def MultiResultOpKind6: I64EnumAttrCase<"kind6", 6>;

def MultiResultOpEnum: I64EnumAttr<
  "MultiResultOpEnum", "Multi-result op kinds", [
    MultiResultOpKind1, MultiResultOpKind2, MultiResultOpKind3,
    MultiResultOpKind4, MultiResultOpKind5, MultiResultOpKind6
  ]>;

#endif // TEST_ENUMDEFS
