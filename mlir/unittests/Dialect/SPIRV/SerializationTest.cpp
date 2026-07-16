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
#include "mlir/Target/SPIRV/Deserialization.h"
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
    return spirv::GlobalVariableOp::create(
        builder, UnknownLoc::get(&context), TypeAttr::get(ptrType),
        builder.getStringAttr(name), nullptr);
  }

  // Inserts an Integer or a Vector of Integers constant of value 'val'.
  spirv::ConstantOp addConstInt(Type type, const APInt &val) {
    OpBuilder builder(module->getRegion());
    auto loc = UnknownLoc::get(&context);

    if (auto intType = dyn_cast<IntegerType>(type)) {
      return spirv::ConstantOp::create(builder, loc, type,
                                       builder.getIntegerAttr(type, val));
    }
    if (auto vectorType = dyn_cast<VectorType>(type)) {
      Type elemType = vectorType.getElementType();
      if (auto intType = dyn_cast<IntegerType>(elemType)) {
        return spirv::ConstantOp::create(
            builder, loc, type,
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
  APInt signlessIntConstVal(signlessInt16Type.getWidth(), 0xffff,
                            signlessInt16Type.getSignedness());
  APInt signedIntConstVal(signedInt16Type.getWidth(), -1,
                          signedInt16Type.getSignedness());

  addConstInt(signlessInt16Type, signlessIntConstVal);
  addConstInt(signedInt16Type, signedIntConstVal);
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

//===----------------------------------------------------------------------===//
// SPV_INTEL_long_composites: composites whose binary form would exceed the
// SPIR-V 16-bit word-count limit are split into a parent + *ContinuedINTEL ops
// on serialization, and merged back on deserialization. These tests build the
// large composites programmatically so that the IR doesn't have to expand
// thousands of operands literally.
//===----------------------------------------------------------------------===//

namespace {

// Picked to comfortably exceed kMaxWordCount = 65535 for any of the splittable
// composite/struct opcodes -- each one packs at most kMaxWordCount - {1,2,3}
// operands into the parent word, so 65540 always triggers a split.
constexpr unsigned kLongCompositeSize = 65540;

bool hasOpcode(SmallVectorImpl<uint32_t> &binary, spirv::Opcode target) {
  size_t offset = spirv::kHeaderWordCount;
  while (offset < binary.size()) {
    uint32_t wordCount = binary[offset] >> 16;
    if (!wordCount || offset + wordCount > binary.size())
      return false;
    auto op = static_cast<spirv::Opcode>(binary[offset] & 0xffff);
    if (op == target)
      return true;
    offset += wordCount;
  }
  return false;
}

bool hasLongCompositesCapabilityAndExtension(
    SmallVectorImpl<uint32_t> &binary) {
  bool foundCap = false;
  bool foundExt = false;
  size_t offset = spirv::kHeaderWordCount;
  size_t binarySize = binary.size();
  while (offset < binarySize) {
    uint32_t wordCount = binary[offset] >> 16;
    if (!wordCount || offset + wordCount > binarySize)
      break;
    auto op = static_cast<spirv::Opcode>(binary[offset] & 0xffff);
    ArrayRef<uint32_t> operands(binary.data() + offset + 1, wordCount - 1);
    if (op == spirv::Opcode::OpCapability && !operands.empty() &&
        operands[0] ==
            static_cast<uint32_t>(spirv::Capability::LongCompositesINTEL))
      foundCap = true;
    if (op == spirv::Opcode::OpExtension) {
      unsigned idx = 0;
      if (spirv::decodeStringLiteral(operands, idx) ==
          spirv::stringifyExtension(
              spirv::Extension::SPV_INTEL_long_composites))
        foundExt = true;
    }
    offset += wordCount;
  }
  return foundCap && foundExt;
}

// Verifies that no instruction in the binary has a word count exceeding the
// SPIR-V 16-bit limit (which would mean the splitting logic failed).
bool allInstructionsWithinWordLimit(SmallVectorImpl<uint32_t> &binary) {
  size_t offset = spirv::kHeaderWordCount;
  size_t binarySize = binary.size();
  while (offset < binarySize) {
    uint32_t wordCount = binary[offset] >> 16;
    if (!wordCount || wordCount > spirv::kMaxWordCount)
      return false;
    offset += wordCount;
  }
  return true;
}

} // namespace

TEST_F(SerializationTest, LongTypeStructIsSplit) {
  OpBuilder builder(module->getRegion());
  Type i32Type = builder.getIntegerType(32);
  Type f32Type = builder.getF32Type();
  SmallVector<Type> memberTypes;
  memberTypes.reserve(kLongCompositeSize);
  for (unsigned i = 0; i < kLongCompositeSize; ++i)
    memberTypes.push_back((i & 1) ? f32Type : i32Type);
  SmallVector<spirv::StructType::OffsetInfo> offsets(kLongCompositeSize, 0);
  auto structType = spirv::StructType::get(memberTypes, offsets);
  addGlobalVar(structType, "var0");

  ASSERT_TRUE(succeeded(spirv::serialize(module.get(), binary)));
  EXPECT_TRUE(allInstructionsWithinWordLimit(binary));
  EXPECT_TRUE(hasOpcode(binary, spirv::Opcode::OpTypeStruct));
  EXPECT_TRUE(hasOpcode(binary, spirv::Opcode::OpTypeStructContinuedINTEL));
  EXPECT_TRUE(hasLongCompositesCapabilityAndExtension(binary));

  MLIRContext freshContext;
  freshContext.getOrLoadDialect<spirv::SPIRVDialect>();
  OwningOpRef<spirv::ModuleOp> roundTripped =
      spirv::deserialize(binary, &freshContext);
  ASSERT_TRUE(roundTripped);
  bool foundStruct = false;
  roundTripped->walk([&](spirv::GlobalVariableOp gv) {
    auto ptrType = dyn_cast<spirv::PointerType>(gv.getType());
    if (!ptrType)
      return;
    auto rtStruct = dyn_cast<spirv::StructType>(ptrType.getPointeeType());
    if (!rtStruct)
      return;
    ASSERT_EQ(rtStruct.getNumElements(), kLongCompositeSize);
    bool typesMatch = true;
    for (unsigned i = 0; i < kLongCompositeSize; ++i) {
      Type expected = (i & 1) ? Type(Float32Type::get(&freshContext))
                              : Type(IntegerType::get(&freshContext, 32));
      if (rtStruct.getElementType(i) != expected) {
        typesMatch = false;
        break;
      }
    }
    EXPECT_TRUE(typesMatch);
    foundStruct = true;
  });
  EXPECT_TRUE(foundStruct);
}

TEST_F(SerializationTest, LongConstantCompositeIsSplit) {
  OpBuilder builder(module->getRegion());
  Location loc = UnknownLoc::get(&context);
  Type i32Type = builder.getIntegerType(32);
  auto arrayType = spirv::ArrayType::get(i32Type, kLongCompositeSize);
  auto funcType = builder.getFunctionType({}, {arrayType});

  auto funcOp = spirv::FuncOp::create(builder, loc, "long_array_const",
                                      funcType, spirv::FunctionControl::None);
  Block *entry = funcOp.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  SmallVector<Attribute> elements;
  elements.reserve(kLongCompositeSize);
  for (unsigned i = 0; i < kLongCompositeSize; ++i)
    elements.push_back(bodyBuilder.getI32IntegerAttr(i & 0xff));
  auto arrayAttr = bodyBuilder.getArrayAttr(elements);
  auto cst = spirv::ConstantOp::create(bodyBuilder, loc, arrayType, arrayAttr);
  spirv::ReturnValueOp::create(bodyBuilder, loc, cst.getResult());

  ASSERT_TRUE(succeeded(spirv::serialize(module.get(), binary)));
  EXPECT_TRUE(allInstructionsWithinWordLimit(binary));
  EXPECT_TRUE(hasOpcode(binary, spirv::Opcode::OpConstantComposite));
  EXPECT_TRUE(
      hasOpcode(binary, spirv::Opcode::OpConstantCompositeContinuedINTEL));
  EXPECT_TRUE(hasLongCompositesCapabilityAndExtension(binary));

  MLIRContext freshContext;
  freshContext.getOrLoadDialect<spirv::SPIRVDialect>();
  OwningOpRef<spirv::ModuleOp> roundTripped =
      spirv::deserialize(binary, &freshContext);
  ASSERT_TRUE(roundTripped);
  bool foundConst = false;
  roundTripped->walk([&](spirv::ConstantOp op) {
    auto arr = dyn_cast<ArrayAttr>(op.getValue());
    if (!arr)
      return;
    ASSERT_EQ(arr.size(), kLongCompositeSize);
    bool valuesMatch = true;
    for (unsigned i = 0; i < kLongCompositeSize; ++i) {
      auto intAttr = dyn_cast<IntegerAttr>(arr[i]);
      if (!intAttr || intAttr.getInt() != static_cast<int64_t>(i & 0xff)) {
        valuesMatch = false;
        break;
      }
    }
    EXPECT_TRUE(valuesMatch);
    foundConst = true;
  });
  EXPECT_TRUE(foundConst);
}

TEST_F(SerializationTest, LongSpecConstantCompositeIsSplit) {
  OpBuilder builder(module->getRegion());
  Location loc = UnknownLoc::get(&context);
  Type i32Type = builder.getIntegerType(32);
  auto arrayType = spirv::ArrayType::get(i32Type, kLongCompositeSize);

  SmallVector<Attribute> constituents;
  constituents.reserve(kLongCompositeSize);
  for (unsigned i = 0; i < kLongCompositeSize; ++i) {
    std::string name = ("sc" + Twine(i)).str();
    auto sc =
        spirv::SpecConstantOp::create(builder, loc, builder.getStringAttr(name),
                                      builder.getI32IntegerAttr(0));
    constituents.push_back(SymbolRefAttr::get(sc));
  }
  spirv::SpecConstantCompositeOp::create(builder, loc, TypeAttr::get(arrayType),
                                         builder.getStringAttr("long_scc"),
                                         builder.getArrayAttr(constituents));

  ASSERT_TRUE(succeeded(spirv::serialize(module.get(), binary)));
  EXPECT_TRUE(allInstructionsWithinWordLimit(binary));
  EXPECT_TRUE(hasOpcode(binary, spirv::Opcode::OpSpecConstantComposite));
  EXPECT_TRUE(
      hasOpcode(binary, spirv::Opcode::OpSpecConstantCompositeContinuedINTEL));
  EXPECT_TRUE(hasLongCompositesCapabilityAndExtension(binary));

  MLIRContext freshContext;
  freshContext.getOrLoadDialect<spirv::SPIRVDialect>();
  OwningOpRef<spirv::ModuleOp> roundTripped =
      spirv::deserialize(binary, &freshContext);
  ASSERT_TRUE(roundTripped);
  bool foundSCC = false;
  roundTripped->walk([&](spirv::SpecConstantCompositeOp op) {
    ArrayAttr rtConstituents = op.getConstituents();
    ASSERT_EQ(rtConstituents.size(), kLongCompositeSize);
    bool namesMatch = true;
    for (unsigned i = 0; i < kLongCompositeSize; ++i) {
      auto symRef = dyn_cast<SymbolRefAttr>(rtConstituents[i]);
      if (!symRef ||
          symRef.getLeafReference().getValue() != ("sc" + Twine(i)).str()) {
        namesMatch = false;
        break;
      }
    }
    EXPECT_TRUE(namesMatch);
    foundSCC = true;
  });
  EXPECT_TRUE(foundSCC);
}

TEST_F(SerializationTest, LongCompositeConstructIsSplit) {
  OpBuilder builder(module->getRegion());
  Location loc = UnknownLoc::get(&context);
  Type i32Type = builder.getIntegerType(32);
  auto arrayType = spirv::ArrayType::get(i32Type, kLongCompositeSize);
  auto funcType = builder.getFunctionType({}, {arrayType});

  auto funcOp = spirv::FuncOp::create(builder, loc, "long_composite_construct",
                                      funcType, spirv::FunctionControl::None);
  Block *entry = funcOp.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  SmallVector<Value> constituents;
  constituents.reserve(kLongCompositeSize);
  for (unsigned i = 0; i < kLongCompositeSize; ++i) {
    auto cst = spirv::ConstantOp::create(
        bodyBuilder, loc, i32Type, bodyBuilder.getI32IntegerAttr(i & 0xff));
    constituents.push_back(cst.getResult());
  }
  auto cc = spirv::CompositeConstructOp::create(bodyBuilder, loc, arrayType,
                                                constituents);
  spirv::ReturnValueOp::create(bodyBuilder, loc, cc.getResult());

  ASSERT_TRUE(succeeded(spirv::serialize(module.get(), binary)));
  EXPECT_TRUE(allInstructionsWithinWordLimit(binary));
  EXPECT_TRUE(hasOpcode(binary, spirv::Opcode::OpCompositeConstruct));
  EXPECT_TRUE(
      hasOpcode(binary, spirv::Opcode::OpCompositeConstructContinuedINTEL));
  EXPECT_TRUE(hasLongCompositesCapabilityAndExtension(binary));

  MLIRContext freshContext;
  freshContext.getOrLoadDialect<spirv::SPIRVDialect>();
  OwningOpRef<spirv::ModuleOp> roundTripped =
      spirv::deserialize(binary, &freshContext);
  ASSERT_TRUE(roundTripped);
  bool foundCC = false;
  roundTripped->walk([&](spirv::CompositeConstructOp op) {
    auto rtConstituents = op.getConstituents();
    ASSERT_EQ(rtConstituents.size(), kLongCompositeSize);
    bool valuesMatch = true;
    for (unsigned i = 0; i < kLongCompositeSize; ++i) {
      auto definingCst = rtConstituents[i].getDefiningOp<spirv::ConstantOp>();
      if (!definingCst) {
        valuesMatch = false;
        break;
      }
      auto intAttr = dyn_cast<IntegerAttr>(definingCst.getValue());
      if (!intAttr || intAttr.getInt() != static_cast<int64_t>(i & 0xff)) {
        valuesMatch = false;
        break;
      }
    }
    EXPECT_TRUE(valuesMatch);
    foundCC = true;
  });
  EXPECT_TRUE(foundCC);
}

namespace {
unsigned countOpcode(SmallVectorImpl<uint32_t> &binary, spirv::Opcode target) {
  unsigned count = 0;
  size_t offset = spirv::kHeaderWordCount;
  size_t binarySize = binary.size();
  while (offset < binarySize) {
    uint32_t wordCount = binary[offset] >> 16;
    if (!wordCount || offset + wordCount > binarySize)
      break;
    auto op = static_cast<spirv::Opcode>(binary[offset] & 0xffff);
    if (op == target)
      ++count;
    offset += wordCount;
  }
  return count;
}
} // namespace

TEST_F(SerializationTest, LongCompositeDoesNotDuplicateDeclaredCapability) {
  // Pre-declare LongCompositesINTEL / SPV_INTEL_long_composites in the VCE
  // triple. The serializer must not emit a second OpCapability/OpExtension
  // when a long composite triggers `addLongCompositesCapability()`.
  module->getOperation()->setAttr(
      spirv::ModuleOp::getVCETripleAttrName(),
      spirv::VerCapExtAttr::get(
          spirv::Version::V_1_0, {spirv::Capability::LongCompositesINTEL},
          {spirv::Extension::SPV_INTEL_long_composites}, &context));

  OpBuilder builder(module->getRegion());
  Type i32Type = builder.getIntegerType(32);
  SmallVector<Type> memberTypes(kLongCompositeSize, i32Type);
  SmallVector<spirv::StructType::OffsetInfo> offsets(kLongCompositeSize, 0);
  auto structType = spirv::StructType::get(memberTypes, offsets);
  addGlobalVar(structType, "var0");

  ASSERT_TRUE(succeeded(spirv::serialize(module.get(), binary)));
  EXPECT_TRUE(allInstructionsWithinWordLimit(binary));
  EXPECT_TRUE(hasOpcode(binary, spirv::Opcode::OpTypeStructContinuedINTEL));
  EXPECT_EQ(countOpcode(binary, spirv::Opcode::OpCapability), 1u);
  EXPECT_EQ(countOpcode(binary, spirv::Opcode::OpExtension), 1u);
}
