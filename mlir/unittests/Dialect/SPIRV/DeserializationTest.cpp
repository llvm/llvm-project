//===- DeserializationTest.cpp - SPIR-V Deserialization Tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The purpose of this file is to provide negative deserialization tests.
// For positive deserialization tests, please use serialization and
// deserialization for roundtripping, unless the serializer cannot produce the
// SPIR-V binary in question.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/SPIRV/Deserialization.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/Target/SPIRV/SPIRVExtInstSets.h"
#include "gmock/gmock.h"

#include <memory>

using namespace mlir;

using ::testing::StrEq;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

/// A deserialization test fixture providing minimal SPIR-V building and
/// diagnostic checking utilities.
class DeserializationTest : public ::testing::Test {
protected:
  DeserializationTest() {
    context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();
    // Register a diagnostic handler to capture the diagnostic so that we can
    // check it later.
    context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
      diagnostic = std::make_unique<Diagnostic>(std::move(diag));
    });
  }

  /// Performs deserialization and returns the constructed spirv.module op.
  OwningOpRef<spirv::ModuleOp> deserialize() {
    return spirv::deserialize(binary, &context);
  }

  /// Checks there is a diagnostic generated with the given `errorMessage`.
  void expectDiagnostic(StringRef errorMessage) {
    ASSERT_NE(nullptr, diagnostic.get());

    // TODO: check error location too.
    EXPECT_THAT(diagnostic->str(), StrEq(std::string(errorMessage)));
  }

  /// Checks the module contains exactly one constant with the expected value.
  void expectConstant(spirv::ModuleOp module, Type type, Attribute value) {
    EXPECT_TRUE(succeeded(verify(module.getOperation())));

    unsigned numConstants = 0;
    module.walk([&](spirv::ConstantOp op) {
      EXPECT_EQ(op.getType(), type);
      EXPECT_EQ(op.getValue(), value);
      ++numConstants;
    });
    EXPECT_EQ(numConstants, 1u);
  }

  //===--------------------------------------------------------------------===//
  // SPIR-V builder methods
  //===--------------------------------------------------------------------===//

  /// Adds the SPIR-V module header to `binary`.
  void addHeader() {
    spirv::appendModuleHeader(binary, spirv::Version::V_1_0, /*idBound=*/0);
  }

  /// Adds the SPIR-V instruction into `binary`.
  void addInstruction(spirv::Opcode op, ArrayRef<uint32_t> operands) {
    uint32_t wordCount = 1 + operands.size();
    binary.push_back(spirv::getPrefixedOpcode(wordCount, op));
    binary.append(operands.begin(), operands.end());
  }

  void addMemoryModel() {
    addInstruction(spirv::Opcode::OpMemoryModel,
                   {static_cast<uint32_t>(spirv::AddressingModel::Logical),
                    static_cast<uint32_t>(spirv::MemoryModel::GLSL450)});
  }

  void addName(uint32_t id, StringRef name) {
    SmallVector<uint32_t, 4> operands = {id};
    spirv::encodeStringLiteralInto(operands, name);
    addInstruction(spirv::Opcode::OpName, operands);
  }

  uint32_t addDebugInfoExtInstImport() {
    SmallVector<uint32_t, 4> importOperands = {nextID++};
    uint32_t extInstSetID = importOperands[0];
    spirv::encodeStringLiteralInto(importOperands, spirv::extDebugInfo);
    addInstruction(spirv::Opcode::OpExtInstImport, importOperands);
    return extInstSetID;
  }

  uint32_t addVoidType() {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpTypeVoid, {id});
    return id;
  }

  uint32_t addBoolType() {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpTypeBool, {id});
    return id;
  }

  uint32_t addIntType(uint32_t bitwidth) {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpTypeInt, {id, bitwidth, /*signedness=*/1});
    return id;
  }

  uint32_t addStructType(ArrayRef<uint32_t> memberTypes) {
    auto id = nextID++;
    SmallVector<uint32_t, 2> words;
    words.push_back(id);
    words.append(memberTypes.begin(), memberTypes.end());
    addInstruction(spirv::Opcode::OpTypeStruct, words);
    return id;
  }

  uint32_t addVectorType(uint32_t elementType, uint32_t count) {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpTypeVector, {id, elementType, count});
    return id;
  }

  uint32_t addArrayType(uint32_t elementType, uint32_t countID) {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpTypeArray, {id, elementType, countID});
    return id;
  }

  uint32_t addPointerType(uint32_t pointeeType) {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpTypePointer,
                   {id, static_cast<uint32_t>(spirv::StorageClass::Function),
                    pointeeType});
    return id;
  }

  uint32_t addIntConstant(uint32_t type, uint32_t value) {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpConstant, {type, id, value});
    return id;
  }

  uint32_t addConstantNull(uint32_t type) {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpConstantNull, {type, id});
    return id;
  }

  uint32_t addConstantComposite(uint32_t type,
                                ArrayRef<uint32_t> constituents) {
    auto id = nextID++;
    SmallVector<uint32_t, 4> operands;
    operands.push_back(type);
    operands.push_back(id);
    operands.append(constituents.begin(), constituents.end());
    addInstruction(spirv::Opcode::OpConstantComposite, operands);
    return id;
  }

  uint32_t addUndef(uint32_t type) {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpUndef, {type, id});
    return id;
  }

  uint32_t addFunctionType(uint32_t retType, ArrayRef<uint32_t> paramTypes) {
    auto id = nextID++;
    SmallVector<uint32_t, 4> operands;
    operands.push_back(id);
    operands.push_back(retType);
    operands.append(paramTypes.begin(), paramTypes.end());
    addInstruction(spirv::Opcode::OpTypeFunction, operands);
    return id;
  }

  uint32_t addFunction(uint32_t retType, uint32_t fnType) {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpFunction,
                   {retType, id,
                    static_cast<uint32_t>(spirv::FunctionControl::None),
                    fnType});
    return id;
  }

  void addFunctionEnd() { addInstruction(spirv::Opcode::OpFunctionEnd, {}); }

  void addReturn() { addInstruction(spirv::Opcode::OpReturn, {}); }

  /// Adds a use of the constant so it is materialized during deserialization.
  void addFunctionReturning(uint32_t type, uint32_t value) {
    auto fnType = addFunctionType(type, {});
    addFunction(type, fnType);
    addInstruction(spirv::Opcode::OpLabel, {nextID++});
    addInstruction(spirv::Opcode::OpReturnValue, {value});
    addFunctionEnd();
  }

protected:
  SmallVector<uint32_t, 5> binary;
  uint32_t nextID = 1;
  MLIRContext context;
  std::unique_ptr<Diagnostic> diagnostic;
};

//===----------------------------------------------------------------------===//
// Basics
//===----------------------------------------------------------------------===//

TEST_F(DeserializationTest, EmptyModuleFailure) {
  ASSERT_FALSE(deserialize());
  expectDiagnostic("SPIR-V binary module must have a 5-word header");
}

TEST_F(DeserializationTest, WrongMagicNumberFailure) {
  addHeader();
  binary.front() = 0xdeadbeef; // Change to a wrong magic number
  ASSERT_FALSE(deserialize());
  expectDiagnostic("incorrect magic number");
}

TEST_F(DeserializationTest, OnlyHeaderSuccess) {
  addHeader();
  EXPECT_TRUE(deserialize());
}

TEST_F(DeserializationTest, ZeroWordCountFailure) {
  addHeader();
  binary.push_back(0); // OpNop with zero word count

  ASSERT_FALSE(deserialize());
  expectDiagnostic("word count cannot be zero");
}

TEST_F(DeserializationTest, InsufficientWordFailure) {
  addHeader();
  binary.push_back((2u << 16) |
                   static_cast<uint32_t>(spirv::Opcode::OpTypeVoid));
  // Missing word for type <id>.

  ASSERT_FALSE(deserialize());
  expectDiagnostic("insufficient words for the last instruction");
}

TEST_F(DeserializationTest, FunctionParameterAttributeMissingOperand) {
  addHeader();
  addInstruction(spirv::Opcode::OpDecorate,
                 {1, static_cast<uint32_t>(spirv::Decoration::FuncParamAttr)});
  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpDecorate with FuncParamAttr needs a single function "
                   "parameter attribute");
}

TEST_F(DeserializationTest, FunctionParameterAttributeExcessOperands) {
  addHeader();
  addInstruction(
      spirv::Opcode::OpDecorate,
      {1, static_cast<uint32_t>(spirv::Decoration::FuncParamAttr),
       static_cast<uint32_t>(spirv::FunctionParameterAttribute::NoAlias),
       static_cast<uint32_t>(spirv::FunctionParameterAttribute::NoCapture)});
  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpDecorate with FuncParamAttr needs a single function "
                   "parameter attribute");
}

TEST_F(DeserializationTest, FunctionParameterAttributeUnknownValue) {
  addHeader();
  addInstruction(
      spirv::Opcode::OpDecorate,
      {1, static_cast<uint32_t>(spirv::Decoration::FuncParamAttr), 100});
  ASSERT_FALSE(deserialize());
  expectDiagnostic("invalid FunctionParameterAttribute code: 100");
}

TEST_F(DeserializationTest, FunctionReturnAttributeOnVoid) {
  addHeader();
  auto fnID = nextID++;
  addInstruction(
      spirv::Opcode::OpDecorate,
      {fnID, static_cast<uint32_t>(spirv::Decoration::FuncParamAttr),
       static_cast<uint32_t>(spirv::FunctionParameterAttribute::Zext)});
  auto voidType = addVoidType();
  auto fnType = addFunctionType(voidType, {});
  addInstruction(spirv::Opcode::OpFunction,
                 {voidType, fnID,
                  static_cast<uint32_t>(spirv::FunctionControl::None), fnType});
  addInstruction(spirv::Opcode::OpLabel, {nextID++});
  addInstruction(spirv::Opcode::OpReturn, {});
  addFunctionEnd();

  ASSERT_FALSE(deserialize());
  expectDiagnostic(
      "FuncParamAttr decoration requires a non-void function result");
}

TEST_F(DeserializationTest, FunctionParameterAttributeOnGlobalUnsupported) {
  addHeader();
  auto varID = nextID++;
  addInstruction(
      spirv::Opcode::OpDecorate,
      {varID, static_cast<uint32_t>(spirv::Decoration::FuncParamAttr),
       static_cast<uint32_t>(spirv::FunctionParameterAttribute::NoAlias)});
  auto intType = addIntType(32);
  auto ptrType = nextID++;
  auto storageClass =
      static_cast<uint32_t>(spirv::StorageClass::CrossWorkgroup);
  addInstruction(spirv::Opcode::OpTypePointer,
                 {ptrType, storageClass, intType});
  addInstruction(spirv::Opcode::OpVariable, {ptrType, varID, storageClass});

  ASSERT_FALSE(deserialize());
  expectDiagnostic(
      "FuncParamAttr decoration on result <id> 1 is not supported: "
      "expected OpFunctionParameter or OpFunction");
}

TEST_F(DeserializationTest, FunctionParameterAttributeUnknownTarget) {
  addHeader();
  addInstruction(
      spirv::Opcode::OpDecorate,
      {42, static_cast<uint32_t>(spirv::Decoration::FuncParamAttr),
       static_cast<uint32_t>(spirv::FunctionParameterAttribute::NoAlias)});

  ASSERT_FALSE(deserialize());
  expectDiagnostic(
      "FuncParamAttr decoration on result <id> 42 is not supported: "
      "expected OpFunctionParameter or OpFunction");
}

class MultipleParameterDecorationsTest
    : public DeserializationTest,
      public ::testing::WithParamInterface<spirv::Decoration> {};

TEST_P(MultipleParameterDecorationsTest, RejectCombinedWithOtherDecorations) {
  addHeader();
  auto argID = nextID++;
  SmallVector<uint32_t> operands = {argID, static_cast<uint32_t>(GetParam())};
  if (GetParam() == spirv::Decoration::Alignment)
    operands.push_back(16);
  else if (GetParam() == spirv::Decoration::FuncParamAttr)
    operands.push_back(
        static_cast<uint32_t>(spirv::FunctionParameterAttribute::NoWrite));
  addInstruction(spirv::Opcode::OpDecorate, operands);
  addInstruction(spirv::Opcode::OpDecorate,
                 {argID, static_cast<uint32_t>(spirv::Decoration::Constant)});
  addInstruction(spirv::Opcode::OpDecorate,
                 {argID, static_cast<uint32_t>(spirv::Decoration::Volatile)});

  auto voidType = addVoidType();
  auto intType = addIntType(32);
  auto ptrType = nextID++;
  addInstruction(spirv::Opcode::OpTypePointer,
                 {ptrType,
                  static_cast<uint32_t>(spirv::StorageClass::CrossWorkgroup),
                  intType});
  auto fnType = addFunctionType(voidType, {ptrType});
  addFunction(voidType, fnType);
  addInstruction(spirv::Opcode::OpFunctionParameter, {ptrType, argID});
  addInstruction(spirv::Opcode::OpLabel, {nextID++});
  addReturn();
  addFunctionEnd();

  ASSERT_FALSE(deserialize());
  expectDiagnostic(
      "already found a decoration for function argument with result <id> 1");
}

INSTANTIATE_TEST_SUITE_P(
    SPIRV, MultipleParameterDecorationsTest,
    ::testing::Values(spirv::Decoration::Alignment,
                      spirv::Decoration::FuncParamAttr,
                      spirv::Decoration::Restrict,
                      spirv::Decoration::RelaxedPrecision));

TEST_F(DeserializationTest, NonUnitParameterDecoration) {
  addHeader();
  auto argID = nextID++;
  addInstruction(
      spirv::Opcode::OpDecorate,
      {argID, static_cast<uint32_t>(spirv::Decoration::Location), 0});

  auto voidType = addVoidType();
  auto intType = addIntType(32);
  auto fnType = addFunctionType(voidType, {intType});
  addFunction(voidType, fnType);
  addInstruction(spirv::Opcode::OpFunctionParameter, {intType, argID});
  addInstruction(spirv::Opcode::OpLabel, {nextID++});
  addReturn();
  addFunctionEnd();

  ASSERT_FALSE(deserialize());
  expectDiagnostic("unhandled decoration 'location' on function argument with "
                   "result <id> 1");
}

TEST_F(DeserializationTest, AlignmentMissingOperand) {
  addHeader();
  addInstruction(spirv::Opcode::OpDecorate,
                 {1, static_cast<uint32_t>(spirv::Decoration::Alignment)});
  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpDecorate with Alignment needs a single integer literal");
}

TEST_F(DeserializationTest, AlignmentExcessOperands) {
  addHeader();
  addInstruction(
      spirv::Opcode::OpDecorate,
      {1, static_cast<uint32_t>(spirv::Decoration::Alignment), 4, 8});
  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpDecorate with Alignment needs a single integer literal");
}

TEST_F(DeserializationTest, OpExtInstMissingOperands) {
  addHeader();
  addInstruction(spirv::Opcode::OpExtInst, {1, 2});

  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpExtInst must have at least 4 operands, result type <id>, "
                   "result <id>, set <id> and instruction opcode");
}

TEST_F(DeserializationTest, DebugInfoExtInstMissingOperands) {
  addHeader();
  uint32_t extInstSetID = addDebugInfoExtInstImport();

  uint32_t voidType = addVoidType();
  addInstruction(
      spirv::Opcode::OpExtInst,
      {voidType, nextID++, extInstSetID,
       static_cast<uint32_t>(spirv::GraphDebugInfoExtInst::DebugTensor)});

  ASSERT_FALSE(deserialize());
  expectDiagnostic("DebugTensor must have tensor and string IDs");
}

TEST_F(DeserializationTest, DebugOperationMissingInstructionIDs) {
  addHeader();
  uint32_t extInstSetID = addDebugInfoExtInstImport();

  uint32_t voidType = addVoidType();
  addInstruction(
      spirv::Opcode::OpExtInst,
      {voidType, nextID++, extInstSetID,
       static_cast<uint32_t>(spirv::GraphDebugInfoExtInst::DebugOperation),
       /*debugGraphID=*/42, /*stringID=*/43});

  ASSERT_FALSE(deserialize());
  expectDiagnostic(
      "DebugOperation must have graph, string and instruction IDs");
}

TEST_F(DeserializationTest, DebugInfoExtInstNonVoidResultType) {
  addHeader();
  uint32_t extInstSetID = addDebugInfoExtInstImport();

  uint32_t boolType = addBoolType();
  addInstruction(
      spirv::Opcode::OpExtInst,
      {boolType, nextID++, extInstSetID,
       static_cast<uint32_t>(spirv::GraphDebugInfoExtInst::DebugTensor), 42,
       43});

  ASSERT_FALSE(deserialize());
  expectDiagnostic("DebugInfo instructions must have OpTypeVoid result type");
}

TEST_F(DeserializationTest, DebugInfoExtInstUnknownOpcode) {
  addHeader();
  uint32_t extInstSetID = addDebugInfoExtInstImport();

  uint32_t voidType = addVoidType();
  addInstruction(spirv::Opcode::OpExtInst,
                 {voidType, nextID++, extInstSetID, 99});

  ASSERT_FALSE(deserialize());
  expectDiagnostic("unknown DebugInfo instruction opcode: 99");
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

TEST_F(DeserializationTest, IntTypeMissingSignednessFailure) {
  addHeader();
  addInstruction(spirv::Opcode::OpTypeInt, {nextID++, 32});

  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpTypeInt must have bitwidth and signedness parameters");
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

TEST_F(DeserializationTest, OpMemberNameSuccess) {
  addHeader();
  SmallVector<uint32_t, 5> typeDecl;
  std::swap(typeDecl, binary);

  auto int32Type = addIntType(32);
  auto structType = addStructType({int32Type, int32Type});
  std::swap(typeDecl, binary);

  SmallVector<uint32_t, 5> operands1 = {structType, 0};
  spirv::encodeStringLiteralInto(operands1, "i1");
  addInstruction(spirv::Opcode::OpMemberName, operands1);

  SmallVector<uint32_t, 5> operands2 = {structType, 1};
  spirv::encodeStringLiteralInto(operands2, "i2");
  addInstruction(spirv::Opcode::OpMemberName, operands2);

  binary.append(typeDecl.begin(), typeDecl.end());
  EXPECT_TRUE(deserialize());
}

TEST_F(DeserializationTest, OpMemberNameMissingOperands) {
  addHeader();
  SmallVector<uint32_t, 5> typeDecl;
  std::swap(typeDecl, binary);

  auto int32Type = addIntType(32);
  auto int64Type = addIntType(64);
  auto structType = addStructType({int32Type, int64Type});
  std::swap(typeDecl, binary);

  SmallVector<uint32_t, 5> operands1 = {structType};
  addInstruction(spirv::Opcode::OpMemberName, operands1);

  binary.append(typeDecl.begin(), typeDecl.end());
  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpMemberName must have at least 3 operands");
}

TEST_F(DeserializationTest, OpMemberNameExcessOperands) {
  addHeader();
  SmallVector<uint32_t, 5> typeDecl;
  std::swap(typeDecl, binary);

  auto int32Type = addIntType(32);
  auto structType = addStructType({int32Type});
  std::swap(typeDecl, binary);

  SmallVector<uint32_t, 5> operands = {structType, 0};
  spirv::encodeStringLiteralInto(operands, "int32");
  operands.push_back(42);
  addInstruction(spirv::Opcode::OpMemberName, operands);

  binary.append(typeDecl.begin(), typeDecl.end());
  ASSERT_FALSE(deserialize());
  expectDiagnostic("unexpected trailing words in OpMemberName instruction");
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// The serializer emits OpConstantComposite for zero-valued arrays and structs,
// so OpConstantNull of these types is only reachable from a SPIR-V binary and
// cannot be covered by round-trip tests.

TEST_F(DeserializationTest, ConstantNullArray) {
  addHeader();
  addMemoryModel();
  auto int32Type = addIntType(32);
  auto two = addIntConstant(int32Type, 2);
  auto arrayType = addArrayType(int32Type, two);
  auto null = addConstantNull(arrayType);
  addFunctionReturning(arrayType, null);

  OwningOpRef<spirv::ModuleOp> module = deserialize();
  ASSERT_TRUE(module);

  Builder builder(&context);
  Type si32 = builder.getIntegerType(32, /*isSigned=*/true);
  Attribute zero = builder.getIntegerAttr(si32, 0);
  expectConstant(module.get(), spirv::ArrayType::get(si32, 2),
                 builder.getArrayAttr({zero, zero}));
}

TEST_F(DeserializationTest, ConstantNullNestedArray) {
  addHeader();
  addMemoryModel();
  auto int32Type = addIntType(32);
  auto two = addIntConstant(int32Type, 2);
  auto three = addIntConstant(int32Type, 3);
  auto innerType = addArrayType(int32Type, three);
  auto outerType = addArrayType(innerType, two);
  auto null = addConstantNull(outerType);
  addFunctionReturning(outerType, null);

  OwningOpRef<spirv::ModuleOp> module = deserialize();
  ASSERT_TRUE(module);

  Builder builder(&context);
  Type si32 = builder.getIntegerType(32, /*isSigned=*/true);
  Attribute zero = builder.getIntegerAttr(si32, 0);
  Attribute inner = builder.getArrayAttr({zero, zero, zero});
  expectConstant(module.get(),
                 spirv::ArrayType::get(spirv::ArrayType::get(si32, 3), 2),
                 builder.getArrayAttr({inner, inner}));
}

TEST_F(DeserializationTest, ConstantNullArrayOfVectors) {
  addHeader();
  addMemoryModel();
  auto int32Type = addIntType(32);
  auto two = addIntConstant(int32Type, 2);
  auto vectorType = addVectorType(int32Type, 4);
  auto arrayType = addArrayType(vectorType, two);
  auto null = addConstantNull(arrayType);
  addFunctionReturning(arrayType, null);

  OwningOpRef<spirv::ModuleOp> module = deserialize();
  ASSERT_TRUE(module);

  Builder builder(&context);
  Type si32 = builder.getIntegerType(32, /*isSigned=*/true);
  VectorType vector = VectorType::get({4}, si32);
  Attribute zero = builder.getZeroAttr(vector);
  expectConstant(module.get(), spirv::ArrayType::get(vector, 2),
                 builder.getArrayAttr({zero, zero}));
}

TEST_F(DeserializationTest, ConstantNullStruct) {
  addHeader();
  addMemoryModel();
  auto int32Type = addIntType(32);
  auto two = addIntConstant(int32Type, 2);
  auto arrayType = addArrayType(int32Type, two);
  auto structType = addStructType({int32Type, arrayType});
  auto null = addConstantNull(structType);
  addFunctionReturning(structType, null);

  OwningOpRef<spirv::ModuleOp> module = deserialize();
  ASSERT_TRUE(module);

  Builder builder(&context);
  Type si32 = builder.getIntegerType(32, /*isSigned=*/true);
  Attribute zero = builder.getIntegerAttr(si32, 0);
  Attribute array = builder.getArrayAttr({zero, zero});
  expectConstant(module.get(),
                 spirv::StructType::get({si32, spirv::ArrayType::get(si32, 2)}),
                 builder.getArrayAttr({zero, array}));
}

TEST_F(DeserializationTest, ConstantNullEmptyStruct) {
  addHeader();
  addMemoryModel();
  auto structType = addStructType({});
  auto null = addConstantNull(structType);
  addFunctionReturning(structType, null);

  OwningOpRef<spirv::ModuleOp> module = deserialize();
  ASSERT_TRUE(module);

  Builder builder(&context);
  expectConstant(module.get(), spirv::StructType::getEmpty(&context),
                 builder.getArrayAttr({}));
}

TEST_F(DeserializationTest, ConstantCompositeMissingConstituentsFailure) {
  addHeader();
  auto int32Type = addIntType(32);
  auto two = addIntConstant(int32Type, 2);
  auto arrayType = addArrayType(int32Type, two);
  addConstantComposite(arrayType, {});

  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpConstantComposite must have at least 1 parameter");
}

TEST_F(DeserializationTest, ConstantCompositeVectorMissingConstituentsFailure) {
  addHeader();
  auto int32Type = addIntType(32);
  auto vectorType = addVectorType(int32Type, 2);
  addConstantComposite(vectorType, {});

  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpConstantComposite must have at least 1 parameter");
}

TEST_F(DeserializationTest, ConstantCompositeNullConstituent) {
  addHeader();
  addMemoryModel();
  auto int32Type = addIntType(32);
  auto one = addIntConstant(int32Type, 1);
  auto two = addIntConstant(int32Type, 2);
  auto innerType = addArrayType(int32Type, two);
  auto outerType = addArrayType(innerType, two);
  auto null = addConstantNull(innerType);
  auto inner = addConstantComposite(innerType, {one, two});
  auto outer = addConstantComposite(outerType, {null, inner});
  addFunctionReturning(outerType, outer);

  OwningOpRef<spirv::ModuleOp> module = deserialize();
  ASSERT_TRUE(module);

  Builder builder(&context);
  Type si32 = builder.getIntegerType(32, /*isSigned=*/true);
  Attribute zero = builder.getIntegerAttr(si32, 0);
  Attribute zeros = builder.getArrayAttr({zero, zero});
  Attribute oneTwo = builder.getArrayAttr(
      {builder.getIntegerAttr(si32, 1), builder.getIntegerAttr(si32, 2)});
  expectConstant(module.get(),
                 spirv::ArrayType::get(spirv::ArrayType::get(si32, 2), 2),
                 builder.getArrayAttr({zeros, oneTwo}));
}

TEST_F(DeserializationTest, ConstantNullPointerFailure) {
  addHeader();
  auto int32Type = addIntType(32);
  auto pointerType = addPointerType(int32Type);
  addConstantNull(pointerType);

  ASSERT_FALSE(deserialize());
  expectDiagnostic(
      "unsupported OpConstantNull type: '!spirv.ptr<si32, Function>'");
}

TEST_F(DeserializationTest, ConstantNullArrayOfPointersFailure) {
  addHeader();
  auto int32Type = addIntType(32);
  auto two = addIntConstant(int32Type, 2);
  auto pointerType = addPointerType(int32Type);
  auto arrayType = addArrayType(pointerType, two);
  addConstantNull(arrayType);

  ASSERT_FALSE(deserialize());
  expectDiagnostic("unsupported OpConstantNull type: "
                   "'!spirv.array<2 x !spirv.ptr<si32, Function>>'");
}

TEST_F(DeserializationTest, ConstantNullIdentifiedStructFailure) {
  addHeader();
  SmallVector<uint32_t, 5> typeDecl;
  std::swap(typeDecl, binary);

  auto int32Type = addIntType(32);
  auto structType = addStructType({int32Type});
  addConstantNull(structType);
  std::swap(typeDecl, binary);

  // A struct with a name is deserialized into an identified struct.
  addName(structType, "S");

  binary.append(typeDecl.begin(), typeDecl.end());
  ASSERT_FALSE(deserialize());
  expectDiagnostic(
      "unsupported OpConstantNull type: '!spirv.struct<S, (si32)>'");
}

TEST_F(DeserializationTest, ConstantCompositeUndefConstituentFailure) {
  addHeader();
  auto int32Type = addIntType(32);
  auto two = addIntConstant(int32Type, 2);
  auto arrayType = addArrayType(int32Type, two);
  auto undef = addUndef(int32Type);
  addConstantComposite(arrayType, {two, undef});

  ASSERT_FALSE(deserialize());
  expectDiagnostic(("OpConstantComposite component #1 <id> " + Twine(undef) +
                    " is an OpUndef, which cannot be represented in "
                    "spirv.Constant")
                       .str());
}

TEST_F(DeserializationTest, ConstantCompositeNonConstantConstituentFailure) {
  addHeader();
  auto int32Type = addIntType(32);
  auto two = addIntConstant(int32Type, 2);
  auto arrayType = addArrayType(int32Type, two);
  addConstantComposite(arrayType, {int32Type, two});

  ASSERT_FALSE(deserialize());
  expectDiagnostic(("OpConstantComposite component #0 <id> " +
                    Twine(int32Type) + " must come from a normal constant")
                       .str());
}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

TEST_F(DeserializationTest, FunctionMissingEndFailure) {
  addHeader();
  auto voidType = addVoidType();
  auto fnType = addFunctionType(voidType, {});
  addFunction(voidType, fnType);
  // Missing OpFunctionEnd.

  ASSERT_FALSE(deserialize());
  expectDiagnostic("expected OpFunctionEnd instruction");
}

TEST_F(DeserializationTest, FunctionMissingParameterFailure) {
  addHeader();
  auto voidType = addVoidType();
  auto i32Type = addIntType(32);
  auto fnType = addFunctionType(voidType, {i32Type});
  addFunction(voidType, fnType);
  // Missing OpFunctionParameter.

  ASSERT_FALSE(deserialize());
  expectDiagnostic("expected OpFunctionParameter instruction");
}

TEST_F(DeserializationTest, FunctionMissingLabelForFirstBlockFailure) {
  addHeader();
  auto voidType = addVoidType();
  auto fnType = addFunctionType(voidType, {});
  addFunction(voidType, fnType);
  // Missing OpLabel.
  addReturn();
  addFunctionEnd();

  ASSERT_FALSE(deserialize());
  expectDiagnostic("a basic block must start with OpLabel");
}

TEST_F(DeserializationTest, FunctionMalformedLabelFailure) {
  addHeader();
  auto voidType = addVoidType();
  auto fnType = addFunctionType(voidType, {});
  addFunction(voidType, fnType);
  addInstruction(spirv::Opcode::OpLabel, {}); // Malformed OpLabel
  addReturn();
  addFunctionEnd();

  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpLabel should only have result <id>");
}

//===----------------------------------------------------------------------===//
// Switch
//===----------------------------------------------------------------------===//

TEST_F(DeserializationTest, SwitchInt64LiteralWordOrder) {
  addHeader();
  auto voidType = addVoidType();
  auto intType = addIntType(64);
  auto fnType = addFunctionType(voidType, {intType});
  addFunction(voidType, fnType);
  auto selectorID = nextID++;
  addInstruction(spirv::Opcode::OpFunctionParameter, {intType, selectorID});
  auto entryID = nextID++;
  auto headerID = nextID++;
  auto defaultID = nextID++;
  auto caseID = nextID++;
  auto mergeID = nextID++;
  addInstruction(spirv::Opcode::OpLabel, {entryID});
  addInstruction(spirv::Opcode::OpBranch, {headerID});
  addInstruction(spirv::Opcode::OpLabel, {headerID});
  addInstruction(
      spirv::Opcode::OpSelectionMerge,
      {mergeID, static_cast<uint32_t>(spirv::SelectionControl::None)});
  // 64-bit literals span two words with the low-order word first:
  // {1, 0} encodes 1, {0, 1} encodes 1 << 32, and {0xffffffff, 0xffffffff}
  // encodes -1.
  addInstruction(spirv::Opcode::OpSwitch,
                 {selectorID, defaultID, 1, 0, caseID, 0, 1, caseID,
                  0xffffffffu, 0xffffffffu, caseID});
  addInstruction(spirv::Opcode::OpLabel, {defaultID});
  addInstruction(spirv::Opcode::OpBranch, {mergeID});
  addInstruction(spirv::Opcode::OpLabel, {caseID});
  addInstruction(spirv::Opcode::OpBranch, {mergeID});
  addInstruction(spirv::Opcode::OpLabel, {mergeID});
  addReturn();
  addFunctionEnd();

  auto module = deserialize();
  ASSERT_TRUE(module);
  SmallVector<int64_t> literals;
  module->walk([&](spirv::SwitchOp switchOp) {
    for (APInt literal : switchOp.getLiterals()->getValues<APInt>())
      literals.push_back(literal.getSExtValue());
  });
  EXPECT_THAT(literals,
              ::testing::ElementsAre(1, static_cast<int64_t>(1) << 32, -1));
}

TEST_F(DeserializationTest, SwitchInt64LiteralOperandCountMismatch) {
  addHeader();
  auto voidType = addVoidType();
  auto intType = addIntType(64);
  auto fnType = addFunctionType(voidType, {intType});
  addFunction(voidType, fnType);
  auto selectorID = nextID++;
  addInstruction(spirv::Opcode::OpFunctionParameter, {intType, selectorID});
  auto entryID = nextID++;
  auto headerID = nextID++;
  auto defaultID = nextID++;
  auto caseID = nextID++;
  auto mergeID = nextID++;
  addInstruction(spirv::Opcode::OpLabel, {entryID});
  addInstruction(spirv::Opcode::OpBranch, {headerID});
  addInstruction(spirv::Opcode::OpLabel, {headerID});
  addInstruction(
      spirv::Opcode::OpSelectionMerge,
      {mergeID, static_cast<uint32_t>(spirv::SelectionControl::None)});
  // A single-word literal is not valid for a 64-bit selector.
  addInstruction(spirv::Opcode::OpSwitch, {selectorID, defaultID, 1, caseID});
  addInstruction(spirv::Opcode::OpLabel, {defaultID});
  addInstruction(spirv::Opcode::OpBranch, {mergeID});
  addInstruction(spirv::Opcode::OpLabel, {caseID});
  addInstruction(spirv::Opcode::OpBranch, {mergeID});
  addInstruction(spirv::Opcode::OpLabel, {mergeID});
  addReturn();
  addFunctionEnd();

  ASSERT_FALSE(deserialize());
  expectDiagnostic("OpSwitch must specify selector, default target and any "
                   "number of literal and label <id> pairs, where each literal "
                   "of a 64-bit selector spans 2 word(s)");
}
