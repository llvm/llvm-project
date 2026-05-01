//===- ABITypeMapperTest.cpp - Unit tests for ABITypeMapper ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/ABI/ABITypeMapper.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ABI/Types.h"

#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::abi;

namespace {

class ABITypeMapperTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<DLTIDialect>();
    module = ModuleOp::create(UnknownLoc::get(&ctx));
  }

  void TearDown() override { module->destroy(); }

  MLIRContext ctx;
  ModuleOp module;
};

TEST_F(ABITypeMapperTest, MapI32) {
  DataLayout dl(module);
  ABITypeMapper mapper(dl);

  auto i32 = IntegerType::get(&ctx, 32);
  const llvm::abi::Type *result = mapper.map(i32);

  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->isInteger());

  auto *intTy = llvm::cast<llvm::abi::IntegerType>(result);
  EXPECT_EQ(intTy->getSizeInBits().getFixedValue(), 32u);
}

TEST_F(ABITypeMapperTest, MapI1) {
  DataLayout dl(module);
  ABITypeMapper mapper(dl);

  auto i1 = IntegerType::get(&ctx, 1);
  const llvm::abi::Type *result = mapper.map(i1);

  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->isInteger());

  auto *intTy = llvm::cast<llvm::abi::IntegerType>(result);
  EXPECT_EQ(intTy->getSizeInBits().getFixedValue(), 1u);
}

TEST_F(ABITypeMapperTest, MapI64) {
  DataLayout dl(module);
  ABITypeMapper mapper(dl);

  auto i64 = IntegerType::get(&ctx, 64);
  const llvm::abi::Type *result = mapper.map(i64);

  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->isInteger());

  auto *intTy = llvm::cast<llvm::abi::IntegerType>(result);
  EXPECT_EQ(intTy->getSizeInBits().getFixedValue(), 64u);
}

TEST_F(ABITypeMapperTest, MapF32) {
  DataLayout dl(module);
  ABITypeMapper mapper(dl);

  auto f32 = Float32Type::get(&ctx);
  const llvm::abi::Type *result = mapper.map(f32);

  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->isFloat());

  auto *floatTy = llvm::cast<llvm::abi::FloatType>(result);
  EXPECT_EQ(floatTy->getSizeInBits().getFixedValue(), 32u);
}

TEST_F(ABITypeMapperTest, MapF64) {
  DataLayout dl(module);
  ABITypeMapper mapper(dl);

  auto f64 = Float64Type::get(&ctx);
  const llvm::abi::Type *result = mapper.map(f64);

  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->isFloat());

  auto *floatTy = llvm::cast<llvm::abi::FloatType>(result);
  EXPECT_EQ(floatTy->getSizeInBits().getFixedValue(), 64u);
}

TEST_F(ABITypeMapperTest, MapF16) {
  DataLayout dl(module);
  ABITypeMapper mapper(dl);

  auto f16 = Float16Type::get(&ctx);
  const llvm::abi::Type *result = mapper.map(f16);

  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->isFloat());

  auto *floatTy = llvm::cast<llvm::abi::FloatType>(result);
  EXPECT_EQ(floatTy->getSizeInBits().getFixedValue(), 16u);
}

TEST_F(ABITypeMapperTest, MapNoneType) {
  DataLayout dl(module);
  ABITypeMapper mapper(dl);

  auto none = NoneType::get(&ctx);
  const llvm::abi::Type *result = mapper.map(none);

  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->isVoid());
}

TEST_F(ABITypeMapperTest, MapVectorOf4xF32) {
  DataLayout dl(module);
  ABITypeMapper mapper(dl);

  auto f32 = Float32Type::get(&ctx);
  auto vec = VectorType::get({4}, f32);
  const llvm::abi::Type *result = mapper.map(vec);

  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->isVector());

  auto *vecTy = llvm::cast<llvm::abi::VectorType>(result);
  EXPECT_EQ(vecTy->getNumElements().getFixedValue(), 4u);
  EXPECT_TRUE(vecTy->getElementType()->isFloat());
}

TEST_F(ABITypeMapperTest, MapSignedI32) {
  DataLayout dl(module);
  ABITypeMapper mapper(dl);

  auto si32 = IntegerType::get(&ctx, 32, IntegerType::Signed);
  const llvm::abi::Type *result = mapper.map(si32);

  ASSERT_NE(result, nullptr);
  auto *intTy = llvm::cast<llvm::abi::IntegerType>(result);
  EXPECT_TRUE(intTy->isSigned());
}

TEST_F(ABITypeMapperTest, MapUnsignedI32) {
  DataLayout dl(module);
  ABITypeMapper mapper(dl);

  auto ui32 = IntegerType::get(&ctx, 32, IntegerType::Unsigned);
  const llvm::abi::Type *result = mapper.map(ui32);

  ASSERT_NE(result, nullptr);
  auto *intTy = llvm::cast<llvm::abi::IntegerType>(result);
  EXPECT_FALSE(intTy->isSigned());
}

} // namespace
