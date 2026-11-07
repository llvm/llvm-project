//===- TestOpBase.td - Test dialect op base definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Definitions shared by the op definitions of the test dialect, which are
// split by category across the TestOps<Category>.td files.

#ifndef TEST_OP_BASE
#define TEST_OP_BASE

include "TestDialect.td"
include "TestInterfaces.td"
include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Dialect/DLTI/DLTIOpInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ValueBoundsOpInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.td"

// Include the attribute definitions.
include "TestAttrDefs.td"
// Include the type definitions.
include "TestTypeDefs.td"

class TEST_Op<string mnemonic, list<Trait> traits = []> :
    Op<Test_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Test Types
//===----------------------------------------------------------------------===//

def TEST_TestType : DialectType<Test_Dialect,
    CPred<"::llvm::isa<::test::TestType>($_self)">, "test">,
    BuildableType<"$_builder.getType<::test::TestType>()">;

//===----------------------------------------------------------------------===//
// Test Enum Attributes
//===----------------------------------------------------------------------===//

// Define the enum attribute.
def TestEnumAttr : EnumAttr<Test_Dialect, TestEnum, "enum">;

def TestPrettyEnumAttr
    : EnumAttr<Test_Dialect, TestPrettyEnum, "pretty_enum">;

//===----------------------------------------------------------------------===//
// Test Bit Enum Attributes
//===----------------------------------------------------------------------===//

// Define the enum attribute.
def TestBitEnumAttr : EnumAttr<Test_Dialect, TestBitEnum, "bit_enum">;

def TestBitEnumVerticalBarAttr
    : EnumAttr<Test_Dialect, TestBitEnumVerticalBar, "bit_enum_vbar">;

def TestBitEnumSpaceAttr
    : EnumAttr<Test_Dialect, TestBitEnumSpace, "bit_enum_space">;

#endif // TEST_OP_BASE
