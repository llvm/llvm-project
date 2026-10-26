//===- LLVMTypeTest.cpp - Tests for LLVM types ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLVMTestBase.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"

using namespace mlir;
using namespace mlir::LLVM;

TEST_F(LLVMIRTest, IsStructTypeMutable) {
  auto structTy = LLVMStructType::getIdentified(&context, "foo");
  ASSERT_TRUE(bool(structTy));
  ASSERT_TRUE(structTy.hasTrait<TypeTrait::IsMutable>());
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
using LLVMIRDeathTest = LLVMIRTest;

TEST_F(LLVMIRDeathTest, StructTypeGetVerifiesElementTypes) {
  EXPECT_DEATH(
      LLVMStructType::getLiteral(&context, {LLVMVoidType::get(&context)}),
      "invalid LLVM structure element type");
}
#endif

TEST_F(LLVMIRTest, StructElementOffsets) {
  DataLayout dataLayout;
  Type i8 = IntegerType::get(&context, 8);
  Type i32 = IntegerType::get(&context, 32);
  Type f64 = Float64Type::get(&context);
  Type f80 = Float80Type::get(&context);

  // Check ABI alignment for unpacked structures.
  auto unpacked = LLVMStructType::getLiteral(&context, {i8, f64, i32});
  EXPECT_EQ(unpacked.getElementOffset(dataLayout, 0), 0u);
  EXPECT_EQ(unpacked.getElementOffset(dataLayout, 1), 8u);
  EXPECT_EQ(unpacked.getElementOffset(dataLayout, 2), 16u);
  EXPECT_EQ(dataLayout.getTypeSize(unpacked), 24u);

  // Check no alignment for packed structures.
  auto packed = LLVMStructType::getLiteral(&context, {i8, f64, i32},
                                           /*isPacked=*/true);
  EXPECT_EQ(packed.getElementOffset(dataLayout, 0), 0u);
  EXPECT_EQ(packed.getElementOffset(dataLayout, 1), 1u);
  EXPECT_EQ(packed.getElementOffset(dataLayout, 2), 9u);
  EXPECT_EQ(dataLayout.getTypeSize(packed), 13u);

  // Check that fields get their allocation size, even in packed
  // structures.
  auto alloc = LLVMStructType::getLiteral(&context, {f80, i8});
  EXPECT_EQ(alloc.getElementOffset(dataLayout, 1), 16u);
  EXPECT_EQ(dataLayout.getTypeSize(alloc), 32u);
  auto packedAlloc =
      LLVMStructType::getLiteral(&context, {f80, i8}, /*isPacked=*/true);
  EXPECT_EQ(packedAlloc.getElementOffset(dataLayout, 1), 16u);
  EXPECT_EQ(dataLayout.getTypeSize(packedAlloc), 17u);
}

TEST_F(LLVMIRTest, TargetExtensionLayoutMatchesLLVM) {
  DataLayout layout;
  llvm::LLVMContext llvmContext;
  llvm::DataLayout llvmLayout("");
  auto check = [&](StringRef name, ArrayRef<Type> typeParams,
                   ArrayRef<llvm::Type *> llvmTypeParams,
                   ArrayRef<unsigned> intParams) {
    SCOPED_TRACE(name.str());
    auto type = LLVMTargetExtType::get(&context, name, typeParams, intParams);
    auto *llvmType =
        llvm::TargetExtType::get(llvmContext, name, llvmTypeParams, intParams);
    llvm::TypeSize size = layout.getTypeSizeInBits(type);
    llvm::TypeSize llvmSize = llvmLayout.getTypeSizeInBits(llvmType);
    EXPECT_EQ(size.getKnownMinValue(), llvmSize.getKnownMinValue());
    EXPECT_EQ(size.isScalable(), llvmSize.isScalable());
    EXPECT_EQ(layout.getTypeABIAlignment(type),
              llvmLayout.getABITypeAlign(llvmType).value());
    EXPECT_EQ(layout.getTypePreferredAlignment(type),
              llvmLayout.getPrefTypeAlign(llvmType).value());
  };

  check("spirv.Image", {Float32Type::get(&context)},
        {llvm::Type::getFloatTy(llvmContext)}, {1, 0, 0, 0, 0, 0, 0});
  check("spirv.SignedImage", {}, {}, {});
  check("spirv.Sampler", {}, {}, {});
  check("spirv.Pipe", {}, {}, {0});
  check("spirv.Event", {}, {}, {});
  check("dx.RawBuffer", {IntegerType::get(&context, 32)},
        {llvm::Type::getInt32Ty(llvmContext)}, {0, 0});
  check("wasm.funcref", {}, {}, {});
  check("wasm.externref", {}, {}, {});
  check("spirv.Type", {}, {}, {0, 8, 32});
  check("spirv.Type", {}, {}, {0, 0, 0});
  check("spirv.Padding", {}, {}, {3});
  check("dx.Padding", {}, {}, {3});
  check("amdgcn.named.barrier", {}, {}, {0});
  check("llvm.test.vectorelement", {}, {}, {});
  check("aarch64.svcount", {}, {}, {});
  check("riscv.vector.tuple",
        {VectorType::get({4}, IntegerType::get(&context, 64), {true})},
        {llvm::ScalableVectorType::get(llvm::Type::getInt64Ty(llvmContext), 4)},
        {3});
}

TEST_F(LLVMIRTest, UnsizedTargetExtensionLayout) {
  for (StringRef name : {"spirv.IntegralConstant", "spirv.Literal",
                         "amdgpu.stridemark", "unknown"}) {
    SCOPED_TRACE(name.str());
    auto type = LLVMTargetExtType::get(&context, name, {}, {});
    EXPECT_TRUE(isa<LLVMVoidType>(type.getLayoutType()));
  }
}

TEST_F(LLVMIRTest, TargetExtensionLayoutInAggregates) {
  DataLayout layout;
  auto sampler = LLVMTargetExtType::get(&context, "spirv.Sampler", {}, {});
  auto array = LLVMArrayType::get(sampler, 3);
  EXPECT_EQ(layout.getTypeSize(array).getFixedValue(), 24u);
  EXPECT_EQ(layout.getTypeABIAlignment(array), 8u);
  auto record = LLVMStructType::getLiteral(
      &context, {IntegerType::get(&context, 8), sampler});
  EXPECT_EQ(record.getElementOffset(layout, 1), 8u);
  EXPECT_EQ(layout.getTypeSize(record).getFixedValue(), 16u);
}

TEST_F(LLVMIRTest, TargetExtensionCustomPointerLayout) {
  context.getOrLoadDialect<DLTIDialect>();
  Builder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  auto pointerEntry = DataLayoutEntryAttr::get(
      LLVMPointerType::get(&context), builder.getI64VectorAttr({32, 32, 64}));
  (*module)->setAttr(DLTIDialect::kDataLayoutAttrName,
                     DataLayoutSpecAttr::get(&context, {pointerEntry}));
  DataLayout layout(module.get());
  auto sampler = LLVMTargetExtType::get(&context, "spirv.Sampler", {}, {});
  EXPECT_EQ(layout.getTypeSizeInBits(sampler).getFixedValue(), 32u);
  EXPECT_EQ(layout.getTypeABIAlignment(sampler), 4u);
  EXPECT_EQ(layout.getTypePreferredAlignment(sampler), 8u);
}
