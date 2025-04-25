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

def SomeI32Enum: I32EnumAttr<
  "SomeI32Enum", "", [I32Case5, I32Case10]>;

def I64Case5:  I64EnumAttrCase<"case5", 5>;
def I64Case10: I64EnumAttrCase<"case10", 10>;

def SomeI64Enum: I64EnumAttr<
  "SomeI64Enum", "", [I64Case5, I64Case10]>;

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

def TestSimpleEnum : I32EnumAttr<"SimpleEnum", "", [
    I32EnumAttrCase<"a", 0>,
    I32EnumAttrCase<"b", 1>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::test";
}

//===----------------------------------------------------------------------===//
// Test Bit Enum
//===----------------------------------------------------------------------===//

// Define the C++ enum.
def TestBitEnum
    : I32BitEnumAttr<"TestBitEnum", "a test bit enum", [
        I32BitEnumAttrCaseBit<"Read", 0, "read">,
        I32BitEnumAttrCaseBit<"Write", 1, "write">,
        I32BitEnumAttrCaseBit<"Execute", 2, "execute">,
      ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "test";
  let separator = ", ";
}

// Define an enum with a different separator
def TestBitEnumVerticalBar
    : I32BitEnumAttr<"TestBitEnumVerticalBar", "another test bit enum", [
        I32BitEnumAttrCaseBit<"User", 0, "user">,
        I32BitEnumAttrCaseBit<"Group", 1, "group">,
        I32BitEnumAttrCaseBit<"Other", 2, "other">,
      ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "test";
  let separator = " | ";
}

//===----------------------------------------------------------------------===//
// Test Patterns (Multi-result Ops)

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
