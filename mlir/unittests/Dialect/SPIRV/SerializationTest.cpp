//===- SerializationTest.cpp - SPIR-V Serialization Tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains corner case tests for the SPIR-V serializer that are not
// covered by normal serialization and deserialization roundtripping.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/SPIRV/SPIRVBinaryUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class SerializationTest : public ::testing::Test {
protected:
  SerializationTest() {
    context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();
    initModuleOp();
  }

  /// Initializes an empty SPIR-V module op.
  void initModuleOp() {
    OpBuilder builder(&context);
    OperationState state(UnknownLoc::get(&context),
                         spirv::ModuleOp::getOperationName());
    state.addAttribute("addressing_model",
                       builder.getAttr<spirv::AddressingModelAttr>(
                           spirv::AddressingModel::Logical));
    state.addAttribute("memory_model", builder.getAttr<spirv::MemoryModelAttr>(
                                           spirv::MemoryModel::GLSL450));
    state.addAttribute("vce_triple",
                       spirv::VerCapExtAttr::get(
                           spirv::Version::V_1_0, ArrayRef<spirv::Capability>(),
                           ArrayRef<spirv::Extension>(), &context));
    spirv::ModuleOp::build(builder, state);
    module = cast<spirv::ModuleOp>(Operation::create(state));
  }

  /// Gets the `struct { float }` type.
  spirv::StructType getFloatStructType() {
    OpBuilder builder(module->getRegion());
    llvm::SmallVector<Type, 1> elementTypes{builder.getF32Type()};
    llvm::SmallVector<spirv::StructType::OffsetInfo, 1> offsetInfo{0};
    return spirv::StructType::get(elementTypes, offsetInfo);
  }

  /// Inserts a global variable of the given `type` and `name`.
  spirv::GlobalVariableOp addGlobalVar(Type type, llvm::StringRef name) {
    OpBuilder builder(module->getRegion());
    auto ptrType = spirv::PointerType::get(type, spirv::StorageClass::Uniform);
    return builder.create<spirv::GlobalVariableOp>(
        UnknownLoc::get(&context), TypeAttr::get(ptrType),
        builder.getStringAttr(name), nullptr);
  }

  // Inserts an Integer or a Vector of Integers constant of value 'val'.
  spirv::ConstantOp AddConstInt(Type type, APInt val) {
    OpBuilder builder(module->getRegion());
    auto loc = UnknownLoc::get(&context);

    if (auto intType = dyn_cast<IntegerType>(type)) {
      return builder.create<spirv::ConstantOp>(
          loc, type, builder.getIntegerAttr(type, val));
    }
    if (auto vectorType = dyn_cast<VectorType>(type)) {
      Type elemType = vectorType.getElementType();
      if (auto intType = dyn_cast<IntegerType>(elemType)) {
        return builder.create<spirv::ConstantOp>(
            loc, type,
            DenseElementsAttr::get(vectorType,
                                   IntegerAttr::get(elemType, val).getValue()));
      }
    }
    llvm_unreachable("unimplemented types for AddConstInt()");
  }

  /// Handles a SPIR-V instruction with the given `opcode` and `operand`.
  /// Returns true to interrupt.
  using HandleFn = llvm::function_ref<bool(spirv::Opcode opcode,
                                           ArrayRef<uint32_t> operands)>;

  /// Returns true if we can find a matching instruction in the SPIR-V blob.
  bool scanInstruction(HandleFn handleFn) {
    auto binarySize = binary.size();
    auto *begin = binary.begin();
    auto currOffset = spirv::kHeaderWordCount;

    while (currOffset < binarySize) {
      auto wordCount = binary[currOffset] >> 16;
      if (!wordCount || (currOffset + wordCount > binarySize))
        return false;

      spirv::Opcode opcode =
          static_cast<spirv::Opcode>(binary[currOffset] & 0xffff);
      llvm::ArrayRef<uint32_t> operands(begin + currOffset + 1,
                                        begin + currOffset + wordCount);
      if (handleFn(opcode, operands))
        return true;

      currOffset += wordCount;
    }
    return false;
  }

protected:
  MLIRContext context;
  OwningOpRef<spirv::ModuleOp> module;
  SmallVector<uint32_t, 0> binary;
};

//===----------------------------------------------------------------------===//
// Block decoration
//===----------------------------------------------------------------------===//

TEST_F(SerializationTest, ContainsBlockDecoration) {
  auto structType = getFloatStructType();
  addGlobalVar(structType, "var0");

  ASSERT_TRUE(succeeded(spirv::serialize(module.get(), binary)));

  auto hasBlockDecoration = [](spirv::Opcode opcode,
                               ArrayRef<uint32_t> operands) {
    return opcode == spirv::Opcode::OpDecorate && operands.size() == 2 &&
           operands[1] == static_cast<uint32_t>(spirv::Decoration::Block);
  };
  EXPECT_TRUE(scanInstruction(hasBlockDecoration));
}

TEST_F(SerializationTest, ContainsNoDuplicatedBlockDecoration) {
  auto structType = getFloatStructType();
  // Two global variables using the same type should not decorate the type with
  // duplicated `Block` decorations.
  addGlobalVar(structType, "var0");
  addGlobalVar(structType, "var1");

  ASSERT_TRUE(succeeded(spirv::serialize(module.get(), binary)));

  unsigned count = 0;
  auto countBlockDecoration = [&count](spirv::Opcode opcode,
                                       ArrayRef<uint32_t> operands) {
    if (opcode == spirv::Opcode::OpDecorate && operands.size() == 2 &&
        operands[1] == static_cast<uint32_t>(spirv::Decoration::Block))
      ++count;
    return false;
  };
  ASSERT_FALSE(scanInstruction(countBlockDecoration));
  EXPECT_EQ(count, 1u);
}

TEST_F(SerializationTest, SignlessVsSignedIntegerConstantBitExtension) {

  auto signlessInt16Type =
      IntegerType::get(&context, 16, IntegerType::Signless);
  auto signedInt16Type = IntegerType::get(&context, 16, IntegerType::Signed);
  // Check the bit extension of same value under different signedness semantics.
  APInt signlessIntConstVal(signlessInt16Type.getWidth(), -1,
                            signlessInt16Type.getSignedness());
  APInt signedIntConstVal(signedInt16Type.getWidth(), -1,
                          signedInt16Type.getSignedness());

  AddConstInt(signlessInt16Type, signlessIntConstVal);
  AddConstInt(signedInt16Type, signedIntConstVal);
  ASSERT_TRUE(succeeded(spirv::serialize(module.get(), binary)));

  auto hasSignlessVal = [&](spirv::Opcode opcode, ArrayRef<uint32_t> operands) {
    return opcode == spirv::Opcode::OpConstant && operands.size() == 3 &&
           operands[2] == 65535;
  };
  EXPECT_TRUE(scanInstruction(hasSignlessVal));

  auto hasSignedVal = [&](spirv::Opcode opcode, ArrayRef<uint32_t> operands) {
    return opcode == spirv::Opcode::OpConstant && operands.size() == 3 &&
           operands[2] == 4294967295;
  };
  EXPECT_TRUE(scanInstruction(hasSignedVal));
}

TEST_F(SerializationTest, ContainsSymbolName) {
  auto structType = getFloatStructType();
  addGlobalVar(structType, "var0");

  spirv::SerializationOptions options;
  options.emitSymbolName = true;
  ASSERT_TRUE(succeeded(spirv::serialize(module.get(), binary, options)));

  auto hasVarName = [](spirv::Opcode opcode, ArrayRef<uint32_t> operands) {
    unsigned index = 1; // Skip the result <id>
    return opcode == spirv::Opcode::OpName &&
           spirv::decodeStringLiteral(operands, index) == "var0";
  };
  EXPECT_TRUE(scanInstruction(hasVarName));
}

TEST_F(SerializationTest, DoesNotContainSymbolName) {
  auto structType = getFloatStructType();
  addGlobalVar(structType, "var0");

  spirv::SerializationOptions options;
  options.emitSymbolName = false;
  ASSERT_TRUE(succeeded(spirv::serialize(module.get(), binary, options)));

  auto hasVarName = [](spirv::Opcode opcode, ArrayRef<uint32_t> operands) {
    unsigned index = 1; // Skip the result <id>
    return opcode == spirv::Opcode::OpName &&
           spirv::decodeStringLiteral(operands, index) == "var0";
  };
  EXPECT_FALSE(scanInstruction(hasVarName));
}
