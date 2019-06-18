//===--- llvm/unittest/IR/VectorTypesTest.cpp - vector types unit tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ScalableSize.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {
TEST(VectorTypesTest, FixedLength) {
  LLVMContext Ctx;

  Type *Int16Ty = Type::getInt16Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *Float64Ty = Type::getDoubleTy(Ctx);

  VectorType *V8Int32Ty = VectorType::get(Int32Ty, 8);
  ASSERT_FALSE(V8Int32Ty->isScalable());
  EXPECT_EQ(V8Int32Ty->getNumElements(), 8U);
  EXPECT_EQ(V8Int32Ty->getElementType()->getScalarSizeInBits(), 32U);

  VectorType *V8Int16Ty = VectorType::get(Int16Ty, {8, false});
  ASSERT_FALSE(V8Int16Ty->isScalable());
  EXPECT_EQ(V8Int16Ty->getNumElements(), 8U);
  EXPECT_EQ(V8Int16Ty->getElementType()->getScalarSizeInBits(), 16U);

  ElementCount EltCnt(4, false);
  VectorType *V4Int64Ty = VectorType::get(Int64Ty, EltCnt);
  ASSERT_FALSE(V4Int64Ty->isScalable());
  EXPECT_EQ(V4Int64Ty->getNumElements(), 4U);
  EXPECT_EQ(V4Int64Ty->getElementType()->getScalarSizeInBits(), 64U);

  VectorType *V2Int64Ty = VectorType::get(Int64Ty, EltCnt/2);
  ASSERT_FALSE(V2Int64Ty->isScalable());
  EXPECT_EQ(V2Int64Ty->getNumElements(), 2U);
  EXPECT_EQ(V2Int64Ty->getElementType()->getScalarSizeInBits(), 64U);

  VectorType *V8Int64Ty = VectorType::get(Int64Ty, EltCnt*2);
  ASSERT_FALSE(V8Int64Ty->isScalable());
  EXPECT_EQ(V8Int64Ty->getNumElements(), 8U);
  EXPECT_EQ(V8Int64Ty->getElementType()->getScalarSizeInBits(), 64U);

  VectorType *V4Float64Ty = VectorType::get(Float64Ty, EltCnt);
  ASSERT_FALSE(V4Float64Ty->isScalable());
  EXPECT_EQ(V4Float64Ty->getNumElements(), 4U);
  EXPECT_EQ(V4Float64Ty->getElementType()->getScalarSizeInBits(), 64U);

  VectorType *ExtTy = VectorType::getExtendedElementVectorType(V8Int16Ty);
  EXPECT_EQ(ExtTy, V8Int32Ty);
  ASSERT_FALSE(ExtTy->isScalable());
  EXPECT_EQ(ExtTy->getNumElements(), 8U);
  EXPECT_EQ(ExtTy->getElementType()->getScalarSizeInBits(), 32U);

  VectorType *TruncTy = VectorType::getTruncatedElementVectorType(V8Int32Ty);
  EXPECT_EQ(TruncTy, V8Int16Ty);
  ASSERT_FALSE(TruncTy->isScalable());
  EXPECT_EQ(TruncTy->getNumElements(), 8U);
  EXPECT_EQ(TruncTy->getElementType()->getScalarSizeInBits(), 16U);

  VectorType *HalvedTy = VectorType::getHalfElementsVectorType(V4Int64Ty);
  EXPECT_EQ(HalvedTy, V2Int64Ty);
  ASSERT_FALSE(HalvedTy->isScalable());
  EXPECT_EQ(HalvedTy->getNumElements(), 2U);
  EXPECT_EQ(HalvedTy->getElementType()->getScalarSizeInBits(), 64U);

  VectorType *DoubledTy = VectorType::getDoubleElementsVectorType(V4Int64Ty);
  EXPECT_EQ(DoubledTy, V8Int64Ty);
  ASSERT_FALSE(DoubledTy->isScalable());
  EXPECT_EQ(DoubledTy->getNumElements(), 8U);
  EXPECT_EQ(DoubledTy->getElementType()->getScalarSizeInBits(), 64U);

  VectorType *ConvTy = VectorType::getInteger(V4Float64Ty);
  EXPECT_EQ(ConvTy, V4Int64Ty);
  ASSERT_FALSE(ConvTy->isScalable());
  EXPECT_EQ(ConvTy->getNumElements(), 4U);
  EXPECT_EQ(ConvTy->getElementType()->getScalarSizeInBits(), 64U);

  EltCnt = V8Int64Ty->getElementCount();
  EXPECT_EQ(EltCnt.Min, 8U);
  ASSERT_FALSE(EltCnt.Scalable);
}

TEST(VectorTypesTest, Scalable) {
  LLVMContext Ctx;

  Type *Int16Ty = Type::getInt16Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *Float64Ty = Type::getDoubleTy(Ctx);

  VectorType *ScV8Int32Ty = VectorType::get(Int32Ty, 8, true);
  ASSERT_TRUE(ScV8Int32Ty->isScalable());
  EXPECT_EQ(ScV8Int32Ty->getNumElements(), 8U);
  EXPECT_EQ(ScV8Int32Ty->getElementType()->getScalarSizeInBits(), 32U);

  VectorType *ScV8Int16Ty = VectorType::get(Int16Ty, {8, true});
  ASSERT_TRUE(ScV8Int16Ty->isScalable());
  EXPECT_EQ(ScV8Int16Ty->getNumElements(), 8U);
  EXPECT_EQ(ScV8Int16Ty->getElementType()->getScalarSizeInBits(), 16U);

  ElementCount EltCnt(4, true);
  VectorType *ScV4Int64Ty = VectorType::get(Int64Ty, EltCnt);
  ASSERT_TRUE(ScV4Int64Ty->isScalable());
  EXPECT_EQ(ScV4Int64Ty->getNumElements(), 4U);
  EXPECT_EQ(ScV4Int64Ty->getElementType()->getScalarSizeInBits(), 64U);

  VectorType *ScV2Int64Ty = VectorType::get(Int64Ty, EltCnt/2);
  ASSERT_TRUE(ScV2Int64Ty->isScalable());
  EXPECT_EQ(ScV2Int64Ty->getNumElements(), 2U);
  EXPECT_EQ(ScV2Int64Ty->getElementType()->getScalarSizeInBits(), 64U);

  VectorType *ScV8Int64Ty = VectorType::get(Int64Ty, EltCnt*2);
  ASSERT_TRUE(ScV8Int64Ty->isScalable());
  EXPECT_EQ(ScV8Int64Ty->getNumElements(), 8U);
  EXPECT_EQ(ScV8Int64Ty->getElementType()->getScalarSizeInBits(), 64U);

  VectorType *ScV4Float64Ty = VectorType::get(Float64Ty, EltCnt);
  ASSERT_TRUE(ScV4Float64Ty->isScalable());
  EXPECT_EQ(ScV4Float64Ty->getNumElements(), 4U);
  EXPECT_EQ(ScV4Float64Ty->getElementType()->getScalarSizeInBits(), 64U);

  VectorType *ExtTy = VectorType::getExtendedElementVectorType(ScV8Int16Ty);
  EXPECT_EQ(ExtTy, ScV8Int32Ty);
  ASSERT_TRUE(ExtTy->isScalable());
  EXPECT_EQ(ExtTy->getNumElements(), 8U);
  EXPECT_EQ(ExtTy->getElementType()->getScalarSizeInBits(), 32U);

  VectorType *TruncTy = VectorType::getTruncatedElementVectorType(ScV8Int32Ty);
  EXPECT_EQ(TruncTy, ScV8Int16Ty);
  ASSERT_TRUE(TruncTy->isScalable());
  EXPECT_EQ(TruncTy->getNumElements(), 8U);
  EXPECT_EQ(TruncTy->getElementType()->getScalarSizeInBits(), 16U);

  VectorType *HalvedTy = VectorType::getHalfElementsVectorType(ScV4Int64Ty);
  EXPECT_EQ(HalvedTy, ScV2Int64Ty);
  ASSERT_TRUE(HalvedTy->isScalable());
  EXPECT_EQ(HalvedTy->getNumElements(), 2U);
  EXPECT_EQ(HalvedTy->getElementType()->getScalarSizeInBits(), 64U);

  VectorType *DoubledTy = VectorType::getDoubleElementsVectorType(ScV4Int64Ty);
  EXPECT_EQ(DoubledTy, ScV8Int64Ty);
  ASSERT_TRUE(DoubledTy->isScalable());
  EXPECT_EQ(DoubledTy->getNumElements(), 8U);
  EXPECT_EQ(DoubledTy->getElementType()->getScalarSizeInBits(), 64U);

  VectorType *ConvTy = VectorType::getInteger(ScV4Float64Ty);
  EXPECT_EQ(ConvTy, ScV4Int64Ty);
  ASSERT_TRUE(ConvTy->isScalable());
  EXPECT_EQ(ConvTy->getNumElements(), 4U);
  EXPECT_EQ(ConvTy->getElementType()->getScalarSizeInBits(), 64U);

  EltCnt = ScV8Int64Ty->getElementCount();
  EXPECT_EQ(EltCnt.Min, 8U);
  ASSERT_TRUE(EltCnt.Scalable);
}

} // end anonymous namespace
