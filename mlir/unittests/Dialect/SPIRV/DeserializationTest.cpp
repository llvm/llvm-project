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
// deserialization for roundtripping.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/SPIRV/Deserialization.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
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
// Memory copies
//===----------------------------------------------------------------------===//

class CopyMemoryDeserializationTest
    : public DeserializationTest,
      public ::testing::WithParamInterface<spirv::Opcode> {
protected:
  void addCopyMemoryFunction() {
    addHeader();
    auto voidType = addVoidType();
    auto intType = addIntType(32);
    auto ptrType = nextID++;
    addInstruction(spirv::Opcode::OpTypePointer,
                   {ptrType,
                    static_cast<uint32_t>(spirv::StorageClass::Function),
                    intType});
    auto fnType = addFunctionType(voidType, {ptrType, ptrType, intType});
    addFunction(voidType, fnType);
    for (auto type : {ptrType, ptrType, intType}) {
      auto id = nextID++;
      addInstruction(spirv::Opcode::OpFunctionParameter, {type, id});
      copyOperands.push_back(id);
    }
    if (GetParam() == spirv::Opcode::OpCopyMemory)
      copyOperands.pop_back();
    addInstruction(spirv::Opcode::OpLabel, {nextID++});
  }

  void addCopyMemory(ArrayRef<uint32_t> memoryOperands) {
    copyOperands.append(memoryOperands.begin(), memoryOperands.end());
    addInstruction(GetParam(), copyOperands);
    addReturn();
    addFunctionEnd();
  }

  SmallVector<uint32_t> copyOperands;
};

TEST_P(CopyMemoryDeserializationTest, MissingOperands) {
  addHeader();
  addInstruction(GetParam(), {});
  ASSERT_FALSE(deserialize());
  expectDiagnostic(
      GetParam() == spirv::Opcode::OpCopyMemory
          ? "expected at least 2 operands for spirv.CopyMemory, but got 0"
          : "expected at least 3 operands for spirv.CopyMemorySized, but got "
            "0");
}

TEST_P(CopyMemoryDeserializationTest, UnknownValue) {
  addCopyMemoryFunction();
  copyOperands[0] = 100;
  addCopyMemory({});
  ASSERT_FALSE(deserialize());
  expectDiagnostic("unknown result <id> : 100");
}

TEST_P(CopyMemoryDeserializationTest, MissingTargetAlignment) {
  addCopyMemoryFunction();
  addCopyMemory({static_cast<uint32_t>(spirv::MemoryAccess::Volatile |
                                       spirv::MemoryAccess::Aligned)});
  ASSERT_FALSE(deserialize());
  expectDiagnostic("missing alignment literal for memory_access");
}

TEST_P(CopyMemoryDeserializationTest, MissingSourceAlignment) {
  addCopyMemoryFunction();
  addCopyMemory({0, static_cast<uint32_t>(spirv::MemoryAccess::Aligned)});
  ASSERT_FALSE(deserialize());
  expectDiagnostic("missing alignment literal for source_memory_access");
}

TEST_P(CopyMemoryDeserializationTest, UnsupportedScopeOperand) {
  addCopyMemoryFunction();
  addCopyMemory(
      {static_cast<uint32_t>(spirv::MemoryAccess::MakePointerAvailable |
                             spirv::MemoryAccess::NonPrivatePointer),
       1});
  ASSERT_FALSE(deserialize());
  expectDiagnostic("unsupported memory access mask: "
                   "MakePointerAvailable|NonPrivatePointer (40)");
}

TEST_P(CopyMemoryDeserializationTest, UnsupportedSourceAliasOperand) {
  addCopyMemoryFunction();
  addCopyMemory(
      {0, static_cast<uint32_t>(spirv::MemoryAccess::AliasScopeINTELMask), 1});
  ASSERT_FALSE(deserialize());
  expectDiagnostic(
      "unsupported memory access mask: AliasScopeINTELMask (65536)");
}

TEST_P(CopyMemoryDeserializationTest, UnknownMemoryAccessMask) {
  addCopyMemoryFunction();
  addCopyMemory({64});
  ASSERT_FALSE(deserialize());
  expectDiagnostic("unsupported memory access mask: unknown (64)");
}

TEST_P(CopyMemoryDeserializationTest, ExcessOperands) {
  addCopyMemoryFunction();
  addCopyMemory({0, 0, 0});
  ASSERT_FALSE(deserialize());
  expectDiagnostic(GetParam() == spirv::Opcode::OpCopyMemory
                       ? "found more operands than expected when deserializing "
                         "spirv.CopyMemory, only 4 of 5 processed"
                       : "found more operands than expected when deserializing "
                         "spirv.CopyMemorySized, only 5 of 6 processed");
}

INSTANTIATE_TEST_SUITE_P(SPIRV, CopyMemoryDeserializationTest,
                         ::testing::Values(spirv::Opcode::OpCopyMemory,
                                           spirv::Opcode::OpCopyMemorySized));

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
