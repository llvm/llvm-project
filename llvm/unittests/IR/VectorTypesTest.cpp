//===--- llvm/unittest/IR/VectorTypesTest.cpp - vector types unit tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/TypeSize.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

#define EXPECT_VTY_EQ(LHS, RHS)                                                \
  ASSERT_NE(LHS, nullptr) << #LHS << " must not be null";                      \
  ASSERT_NE(RHS, nullptr) << #RHS << " must not be null";                      \
  EXPECT_EQ(LHS, RHS) << "Expect that " << #LHS << " == " << #RHS << " where " \
                      << #LHS << " = " << *LHS << " and " << #RHS << " = "     \
                      << *RHS;

#define EXPECT_VTY_NE(LHS, RHS)                                                \
  ASSERT_NE(LHS, nullptr) << #LHS << " must not be null";                      \
  ASSERT_NE(RHS, nullptr) << #RHS << " must not be null";                      \
  EXPECT_NE(LHS, RHS) << "Expect that " << #LHS << " != " << #RHS << " where " \
                      << #LHS << " = " << *LHS << " and " << #RHS << " = "     \
                      << *RHS;

TEST(VectorTypesTest, FixedLength) {
  LLVMContext Ctx;

  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int16Ty = Type::getInt16Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *Float64Ty = Type::getDoubleTy(Ctx);

  auto *V16Int8Ty = FixedVectorType::get(Int8Ty, 16);
  ASSERT_NE(nullptr, V16Int8Ty);
  EXPECT_EQ(V16Int8Ty->getNumElements(), 16U);
  EXPECT_EQ(V16Int8Ty->getElementType()->getScalarSizeInBits(), 8U);

  auto *V8Int32Ty =
      dyn_cast<FixedVectorType>(VectorType::get(Int32Ty, 8, false));
  ASSERT_NE(nullptr, V8Int32Ty);
  EXPECT_EQ(V8Int32Ty->getNumElements(), 8U);
  EXPECT_EQ(V8Int32Ty->getElementType()->getScalarSizeInBits(), 32U);

  auto *V8Int8Ty =
      dyn_cast<FixedVectorType>(VectorType::get(Int8Ty, V8Int32Ty));
  EXPECT_VTY_NE(V8Int32Ty, V8Int8Ty);
  EXPECT_EQ(V8Int8Ty->getElementCount(), V8Int32Ty->getElementCount());
  EXPECT_EQ(V8Int8Ty->getElementType()->getScalarSizeInBits(), 8U);

  auto *V8Int32Ty2 =
      dyn_cast<FixedVectorType>(VectorType::get(Int32Ty, V8Int32Ty));
  EXPECT_VTY_EQ(V8Int32Ty, V8Int32Ty2);

  auto *V8Int16Ty = dyn_cast<FixedVectorType>(
      VectorType::get(Int16Ty, ElementCount::getFixed(8)));
  ASSERT_NE(nullptr, V8Int16Ty);
  EXPECT_EQ(V8Int16Ty->getNumElements(), 8U);
  EXPECT_EQ(V8Int16Ty->getElementType()->getScalarSizeInBits(), 16U);

  auto EltCnt = ElementCount::getFixed(4);
  auto *V4Int64Ty = dyn_cast<FixedVectorType>(VectorType::get(Int64Ty, EltCnt));
  ASSERT_NE(nullptr, V4Int64Ty);
  EXPECT_EQ(V4Int64Ty->getNumElements(), 4U);
  EXPECT_EQ(V4Int64Ty->getElementType()->getScalarSizeInBits(), 64U);

  auto *V2Int64Ty = dyn_cast<FixedVectorType>(
      VectorType::get(Int64Ty, EltCnt.divideCoefficientBy(2)));
  ASSERT_NE(nullptr, V2Int64Ty);
  EXPECT_EQ(V2Int64Ty->getNumElements(), 2U);
  EXPECT_EQ(V2Int64Ty->getElementType()->getScalarSizeInBits(), 64U);

  auto *V8Int64Ty =
      dyn_cast<FixedVectorType>(VectorType::get(Int64Ty, EltCnt * 2));
  ASSERT_NE(nullptr, V8Int64Ty);
  EXPECT_EQ(V8Int64Ty->getNumElements(), 8U);
  EXPECT_EQ(V8Int64Ty->getElementType()->getScalarSizeInBits(), 64U);

  auto *V4Float64Ty =
      dyn_cast<FixedVectorType>(VectorType::get(Float64Ty, EltCnt));
  ASSERT_NE(nullptr, V4Float64Ty);
  EXPECT_EQ(V4Float64Ty->getNumElements(), 4U);
  EXPECT_EQ(V4Float64Ty->getElementType()->getScalarSizeInBits(), 64U);

  auto *ExtTy = dyn_cast<FixedVectorType>(
      VectorType::getExtendedElementVectorType(V8Int16Ty));
  EXPECT_VTY_EQ(ExtTy, V8Int32Ty);
  EXPECT_EQ(ExtTy->getNumElements(), 8U);
  EXPECT_EQ(ExtTy->getElementType()->getScalarSizeInBits(), 32U);

  auto *TruncTy = dyn_cast<FixedVectorType>(
      VectorType::getTruncatedElementVectorType(V8Int32Ty));
  EXPECT_VTY_EQ(TruncTy, V8Int16Ty);
  EXPECT_EQ(TruncTy->getNumElements(), 8U);
  EXPECT_EQ(TruncTy->getElementType()->getScalarSizeInBits(), 16U);

  auto *HalvedTy = dyn_cast<FixedVectorType>(
      VectorType::getHalfElementsVectorType(V4Int64Ty));
  EXPECT_VTY_EQ(HalvedTy, V2Int64Ty);
  EXPECT_EQ(HalvedTy->getNumElements(), 2U);
  EXPECT_EQ(HalvedTy->getElementType()->getScalarSizeInBits(), 64U);

  auto *DoubledTy = dyn_cast<FixedVectorType>(
      VectorType::getDoubleElementsVectorType(V4Int64Ty));
  EXPECT_VTY_EQ(DoubledTy, V8Int64Ty);
  EXPECT_EQ(DoubledTy->getNumElements(), 8U);
  EXPECT_EQ(DoubledTy->getElementType()->getScalarSizeInBits(), 64U);

  auto *ConvTy = dyn_cast<FixedVectorType>(VectorType::getInteger(V4Float64Ty));
  EXPECT_VTY_EQ(ConvTy, V4Int64Ty);
  EXPECT_EQ(ConvTy->getNumElements(), 4U);
  EXPECT_EQ(ConvTy->getElementType()->getScalarSizeInBits(), 64U);

  EltCnt = V8Int64Ty->getElementCount();
  EXPECT_EQ(EltCnt.getKnownMinValue(), 8U);
  ASSERT_FALSE(EltCnt.isScalable());
}

TEST(VectorTypesTest, Scalable) {
  LLVMContext Ctx;

  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int16Ty = Type::getInt16Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *Float64Ty = Type::getDoubleTy(Ctx);

  auto *ScV16Int8Ty = ScalableVectorType::get(Int8Ty, 16);
  ASSERT_NE(nullptr, ScV16Int8Ty);
  EXPECT_EQ(ScV16Int8Ty->getMinNumElements(), 16U);
  EXPECT_EQ(ScV16Int8Ty->getScalarSizeInBits(), 8U);

  auto *ScV8Int32Ty =
      dyn_cast<ScalableVectorType>(VectorType::get(Int32Ty, 8, true));
  ASSERT_NE(nullptr, ScV8Int32Ty);
  EXPECT_EQ(ScV8Int32Ty->getMinNumElements(), 8U);
  EXPECT_EQ(ScV8Int32Ty->getElementType()->getScalarSizeInBits(), 32U);

  auto *ScV8Int8Ty =
      dyn_cast<ScalableVectorType>(VectorType::get(Int8Ty, ScV8Int32Ty));
  EXPECT_VTY_NE(ScV8Int32Ty, ScV8Int8Ty);
  EXPECT_EQ(ScV8Int8Ty->getElementCount(), ScV8Int32Ty->getElementCount());
  EXPECT_EQ(ScV8Int8Ty->getElementType()->getScalarSizeInBits(), 8U);

  auto *ScV8Int32Ty2 =
      dyn_cast<ScalableVectorType>(VectorType::get(Int32Ty, ScV8Int32Ty));
  EXPECT_VTY_EQ(ScV8Int32Ty, ScV8Int32Ty2);

  auto *ScV8Int16Ty = dyn_cast<ScalableVectorType>(
      VectorType::get(Int16Ty, ElementCount::getScalable(8)));
  ASSERT_NE(nullptr, ScV8Int16Ty);
  EXPECT_EQ(ScV8Int16Ty->getMinNumElements(), 8U);
  EXPECT_EQ(ScV8Int16Ty->getElementType()->getScalarSizeInBits(), 16U);

  auto EltCnt = ElementCount::getScalable(4);
  auto *ScV4Int64Ty =
      dyn_cast<ScalableVectorType>(VectorType::get(Int64Ty, EltCnt));
  ASSERT_NE(nullptr, ScV4Int64Ty);
  EXPECT_EQ(ScV4Int64Ty->getMinNumElements(), 4U);
  EXPECT_EQ(ScV4Int64Ty->getElementType()->getScalarSizeInBits(), 64U);

  auto *ScV2Int64Ty = dyn_cast<ScalableVectorType>(
      VectorType::get(Int64Ty, EltCnt.divideCoefficientBy(2)));
  ASSERT_NE(nullptr, ScV2Int64Ty);
  EXPECT_EQ(ScV2Int64Ty->getMinNumElements(), 2U);
  EXPECT_EQ(ScV2Int64Ty->getElementType()->getScalarSizeInBits(), 64U);

  auto *ScV8Int64Ty =
      dyn_cast<ScalableVectorType>(VectorType::get(Int64Ty, EltCnt * 2));
  ASSERT_NE(nullptr, ScV8Int64Ty);
  EXPECT_EQ(ScV8Int64Ty->getMinNumElements(), 8U);
  EXPECT_EQ(ScV8Int64Ty->getElementType()->getScalarSizeInBits(), 64U);

  auto *ScV4Float64Ty =
      dyn_cast<ScalableVectorType>(VectorType::get(Float64Ty, EltCnt));
  ASSERT_NE(nullptr, ScV4Float64Ty);
  EXPECT_EQ(ScV4Float64Ty->getMinNumElements(), 4U);
  EXPECT_EQ(ScV4Float64Ty->getElementType()->getScalarSizeInBits(), 64U);

  auto *ExtTy = dyn_cast<ScalableVectorType>(
      VectorType::getExtendedElementVectorType(ScV8Int16Ty));
  EXPECT_VTY_EQ(ExtTy, ScV8Int32Ty);
  EXPECT_EQ(ExtTy->getMinNumElements(), 8U);
  EXPECT_EQ(ExtTy->getElementType()->getScalarSizeInBits(), 32U);

  auto *TruncTy = dyn_cast<ScalableVectorType>(
      VectorType::getTruncatedElementVectorType(ScV8Int32Ty));
  EXPECT_VTY_EQ(TruncTy, ScV8Int16Ty);
  EXPECT_EQ(TruncTy->getMinNumElements(), 8U);
  EXPECT_EQ(TruncTy->getElementType()->getScalarSizeInBits(), 16U);

  auto *HalvedTy = dyn_cast<ScalableVectorType>(
      VectorType::getHalfElementsVectorType(ScV4Int64Ty));
  EXPECT_VTY_EQ(HalvedTy, ScV2Int64Ty);
  EXPECT_EQ(HalvedTy->getMinNumElements(), 2U);
  EXPECT_EQ(HalvedTy->getElementType()->getScalarSizeInBits(), 64U);

  auto *DoubledTy = dyn_cast<ScalableVectorType>(
      VectorType::getDoubleElementsVectorType(ScV4Int64Ty));
  EXPECT_VTY_EQ(DoubledTy, ScV8Int64Ty);
  EXPECT_EQ(DoubledTy->getMinNumElements(), 8U);
  EXPECT_EQ(DoubledTy->getElementType()->getScalarSizeInBits(), 64U);

  auto *ConvTy =
      dyn_cast<ScalableVectorType>(VectorType::getInteger(ScV4Float64Ty));
  EXPECT_VTY_EQ(ConvTy, ScV4Int64Ty);
  EXPECT_EQ(ConvTy->getMinNumElements(), 4U);
  EXPECT_EQ(ConvTy->getElementType()->getScalarSizeInBits(), 64U);

  EltCnt = ScV8Int64Ty->getElementCount();
  EXPECT_EQ(EltCnt.getKnownMinValue(), 8U);
  ASSERT_TRUE(EltCnt.isScalable());
}

TEST(VectorTypesTest, BaseVectorType) {
  LLVMContext Ctx;

  Type *Int16Ty = Type::getInt16Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);

  std::array<VectorType *, 8> VTys = {
      VectorType::get(Int16Ty, ElementCount::getScalable(4)),
      VectorType::get(Int16Ty, ElementCount::getFixed(4)),
      VectorType::get(Int16Ty, ElementCount::getScalable(2)),
      VectorType::get(Int16Ty, ElementCount::getFixed(2)),
      VectorType::get(Int32Ty, ElementCount::getScalable(4)),
      VectorType::get(Int32Ty, ElementCount::getFixed(4)),
      VectorType::get(Int32Ty, ElementCount::getScalable(2)),
      VectorType::get(Int32Ty, ElementCount::getFixed(2))};

  /*
    The comparison matrix is symmetric, so we only check the upper triangle:

    (0,0) (0,1) (0,2) ... (0,7)
    (1,0) (1,1) (1,2)         .
    (2,0) (2,1) (2,2)         .
    .                 .       .
    .                  .
    .                   .
    (7,0) ...             (7,7)
  */
  for (size_t I = 0, IEnd = VTys.size(); I < IEnd; ++I) {
    // test I == J
    VectorType *VI = VTys[I];
    ElementCount ECI = VI->getElementCount();
    EXPECT_EQ(isa<ScalableVectorType>(VI), ECI.isScalable());

    for (size_t J = I + 1, JEnd = VTys.size(); J < JEnd; ++J) {
      // test I < J
      VectorType *VJ = VTys[J];
      EXPECT_VTY_NE(VI, VJ);

      VectorType *VJPrime = VectorType::get(VI->getElementType(), VJ);
      if (VI->getElementType() == VJ->getElementType()) {
        EXPECT_VTY_EQ(VJ, VJPrime);
      } else {
        EXPECT_VTY_NE(VJ, VJPrime);
      }

      EXPECT_EQ(VJ->getTypeID(), VJPrime->getTypeID())
          << "VJ and VJPrime are the same sort of vector";
    }
  }
}

TEST(VectorTypesTest, FixedLenComparisons) {
  LLVMContext Ctx;
  DataLayout DL("");

  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  auto *V2Int32Ty = FixedVectorType::get(Int32Ty, 2);
  auto *V4Int32Ty = FixedVectorType::get(Int32Ty, 4);

  auto *V2Int64Ty = FixedVectorType::get(Int64Ty, 2);

  TypeSize V2I32Len = V2Int32Ty->getPrimitiveSizeInBits();
  EXPECT_EQ(V2I32Len.getKnownMinSize(), 64U);
  EXPECT_FALSE(V2I32Len.isScalable());

  EXPECT_LT(V2Int32Ty->getPrimitiveSizeInBits().getFixedSize(),
            V4Int32Ty->getPrimitiveSizeInBits().getFixedSize());
  EXPECT_GT(V2Int64Ty->getPrimitiveSizeInBits().getFixedSize(),
            V2Int32Ty->getPrimitiveSizeInBits().getFixedSize());
  EXPECT_EQ(V4Int32Ty->getPrimitiveSizeInBits(),
            V2Int64Ty->getPrimitiveSizeInBits());
  EXPECT_NE(V2Int32Ty->getPrimitiveSizeInBits(),
            V2Int64Ty->getPrimitiveSizeInBits());

  // Check that a fixed-only comparison works for fixed size vectors.
  EXPECT_EQ(V2Int64Ty->getPrimitiveSizeInBits().getFixedSize(),
            V4Int32Ty->getPrimitiveSizeInBits().getFixedSize());

  // Check the DataLayout interfaces.
  EXPECT_EQ(DL.getTypeSizeInBits(V2Int64Ty), DL.getTypeSizeInBits(V4Int32Ty));
  EXPECT_EQ(DL.getTypeSizeInBits(V2Int32Ty), 64U);
  EXPECT_EQ(DL.getTypeSizeInBits(V2Int64Ty), 128U);
  EXPECT_EQ(DL.getTypeStoreSize(V2Int64Ty), DL.getTypeStoreSize(V4Int32Ty));
  EXPECT_NE(DL.getTypeStoreSizeInBits(V2Int32Ty),
            DL.getTypeStoreSizeInBits(V2Int64Ty));
  EXPECT_EQ(DL.getTypeStoreSizeInBits(V2Int32Ty), 64U);
  EXPECT_EQ(DL.getTypeStoreSize(V2Int64Ty), 16U);
  EXPECT_EQ(DL.getTypeAllocSize(V4Int32Ty), DL.getTypeAllocSize(V2Int64Ty));
  EXPECT_NE(DL.getTypeAllocSizeInBits(V2Int32Ty),
            DL.getTypeAllocSizeInBits(V2Int64Ty));
  EXPECT_EQ(DL.getTypeAllocSizeInBits(V4Int32Ty), 128U);
  EXPECT_EQ(DL.getTypeAllocSize(V2Int32Ty), 8U);
  ASSERT_TRUE(DL.typeSizeEqualsStoreSize(V4Int32Ty));
}

TEST(VectorTypesTest, ScalableComparisons) {
  LLVMContext Ctx;
  DataLayout DL("");

  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  auto *ScV2Int32Ty = ScalableVectorType::get(Int32Ty, 2);
  auto *ScV4Int32Ty = ScalableVectorType::get(Int32Ty, 4);

  auto *ScV2Int64Ty = ScalableVectorType::get(Int64Ty, 2);

  TypeSize ScV2I32Len = ScV2Int32Ty->getPrimitiveSizeInBits();
  EXPECT_EQ(ScV2I32Len.getKnownMinSize(), 64U);
  EXPECT_TRUE(ScV2I32Len.isScalable());

  EXPECT_LT(ScV2Int32Ty->getPrimitiveSizeInBits().getKnownMinSize(),
            ScV4Int32Ty->getPrimitiveSizeInBits().getKnownMinSize());
  EXPECT_GT(ScV2Int64Ty->getPrimitiveSizeInBits().getKnownMinSize(),
            ScV2Int32Ty->getPrimitiveSizeInBits().getKnownMinSize());
  EXPECT_EQ(ScV4Int32Ty->getPrimitiveSizeInBits().getKnownMinSize(),
            ScV2Int64Ty->getPrimitiveSizeInBits().getKnownMinSize());
  EXPECT_NE(ScV2Int32Ty->getPrimitiveSizeInBits().getKnownMinSize(),
            ScV2Int64Ty->getPrimitiveSizeInBits().getKnownMinSize());

  // Check the DataLayout interfaces.
  EXPECT_EQ(DL.getTypeSizeInBits(ScV2Int64Ty),
            DL.getTypeSizeInBits(ScV4Int32Ty));
  EXPECT_EQ(DL.getTypeSizeInBits(ScV2Int32Ty).getKnownMinSize(), 64U);
  EXPECT_EQ(DL.getTypeStoreSize(ScV2Int64Ty), DL.getTypeStoreSize(ScV4Int32Ty));
  EXPECT_NE(DL.getTypeStoreSizeInBits(ScV2Int32Ty),
            DL.getTypeStoreSizeInBits(ScV2Int64Ty));
  EXPECT_EQ(DL.getTypeStoreSizeInBits(ScV2Int32Ty).getKnownMinSize(), 64U);
  EXPECT_EQ(DL.getTypeStoreSize(ScV2Int64Ty).getKnownMinSize(), 16U);
  EXPECT_EQ(DL.getTypeAllocSize(ScV4Int32Ty), DL.getTypeAllocSize(ScV2Int64Ty));
  EXPECT_NE(DL.getTypeAllocSizeInBits(ScV2Int32Ty),
            DL.getTypeAllocSizeInBits(ScV2Int64Ty));
  EXPECT_EQ(DL.getTypeAllocSizeInBits(ScV4Int32Ty).getKnownMinSize(), 128U);
  EXPECT_EQ(DL.getTypeAllocSize(ScV2Int32Ty).getKnownMinSize(), 8U);
  ASSERT_TRUE(DL.typeSizeEqualsStoreSize(ScV4Int32Ty));
}

TEST(VectorTypesTest, CrossComparisons) {
  LLVMContext Ctx;

  Type *Int32Ty = Type::getInt32Ty(Ctx);

  auto *V4Int32Ty = FixedVectorType::get(Int32Ty, 4);
  auto *ScV4Int32Ty = ScalableVectorType::get(Int32Ty, 4);

  // Even though the minimum size is the same, a scalable vector could be
  // larger so we don't consider them to be the same size.
  EXPECT_NE(V4Int32Ty->getPrimitiveSizeInBits(),
            ScV4Int32Ty->getPrimitiveSizeInBits());
  // If we are only checking the minimum, then they are the same size.
  EXPECT_EQ(V4Int32Ty->getPrimitiveSizeInBits().getKnownMinSize(),
            ScV4Int32Ty->getPrimitiveSizeInBits().getKnownMinSize());

  // We can't use ordering comparisons (<,<=,>,>=) between scalable and
  // non-scalable vector sizes.
}

} // end anonymous namespace
