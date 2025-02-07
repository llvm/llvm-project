//===------- VectorTypeUtilsTest.cpp - Vector utils tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/VectorTypeUtils.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class VectorTypeUtilsTest : public ::testing::Test {};

TEST(VectorTypeUtilsTest, TestToVectorizedTy) {
  LLVMContext C;

  Type *ITy = Type::getInt32Ty(C);
  Type *FTy = Type::getFloatTy(C);
  Type *HomogeneousStructTy = StructType::get(FTy, FTy, FTy);
  Type *MixedStructTy = StructType::get(FTy, ITy);
  Type *VoidTy = Type::getVoidTy(C);

  for (ElementCount VF :
       {ElementCount::getFixed(4), ElementCount::getScalable(2)}) {
    Type *IntVec = toVectorizedTy(ITy, VF);
    EXPECT_TRUE(isa<VectorType>(IntVec));
    EXPECT_EQ(IntVec, VectorType::get(ITy, VF));

    Type *FloatVec = toVectorizedTy(FTy, VF);
    EXPECT_TRUE(isa<VectorType>(FloatVec));
    EXPECT_EQ(FloatVec, VectorType::get(FTy, VF));

    Type *WideHomogeneousStructTy = toVectorizedTy(HomogeneousStructTy, VF);
    EXPECT_TRUE(isa<StructType>(WideHomogeneousStructTy));
    EXPECT_TRUE(
        cast<StructType>(WideHomogeneousStructTy)->containsHomogeneousTypes());
    EXPECT_TRUE(cast<StructType>(WideHomogeneousStructTy)->getNumElements() ==
                3);
    EXPECT_TRUE(cast<StructType>(WideHomogeneousStructTy)->getElementType(0) ==
                VectorType::get(FTy, VF));

    Type *WideMixedStructTy = toVectorizedTy(MixedStructTy, VF);
    EXPECT_TRUE(isa<StructType>(WideMixedStructTy));
    EXPECT_TRUE(cast<StructType>(WideMixedStructTy)->getNumElements() == 2);
    EXPECT_TRUE(cast<StructType>(WideMixedStructTy)->getElementType(0) ==
                VectorType::get(FTy, VF));
    EXPECT_TRUE(cast<StructType>(WideMixedStructTy)->getElementType(1) ==
                VectorType::get(ITy, VF));

    EXPECT_EQ(toVectorizedTy(VoidTy, VF), VoidTy);
  }

  ElementCount ScalarVF = ElementCount::getFixed(1);
  for (Type *Ty : {ITy, FTy, HomogeneousStructTy, MixedStructTy, VoidTy}) {
    EXPECT_EQ(toVectorizedTy(Ty, ScalarVF), Ty);
  }
}

TEST(VectorTypeUtilsTest, TestToScalarizedTy) {
  LLVMContext C;

  Type *ITy = Type::getInt32Ty(C);
  Type *FTy = Type::getFloatTy(C);
  Type *HomogeneousStructTy = StructType::get(FTy, FTy, FTy);
  Type *MixedStructTy = StructType::get(FTy, ITy);
  Type *VoidTy = Type::getVoidTy(C);

  for (ElementCount VF : {ElementCount::getFixed(1), ElementCount::getFixed(4),
                          ElementCount::getScalable(2)}) {
    for (Type *Ty : {ITy, FTy, HomogeneousStructTy, MixedStructTy, VoidTy}) {
      // toScalarizedTy should be the inverse of toVectorizedTy.
      EXPECT_EQ(toScalarizedTy(toVectorizedTy(Ty, VF)), Ty);
    };
  }
}

TEST(VectorTypeUtilsTest, TestGetContainedTypes) {
  LLVMContext C;

  Type *ITy = Type::getInt32Ty(C);
  Type *FTy = Type::getFloatTy(C);
  Type *HomogeneousStructTy = StructType::get(FTy, FTy, FTy);
  Type *MixedStructTy = StructType::get(FTy, ITy);
  Type *VoidTy = Type::getVoidTy(C);

  EXPECT_EQ(getContainedTypes(ITy), ArrayRef<Type *>({ITy}));
  EXPECT_EQ(getContainedTypes(FTy), ArrayRef<Type *>({FTy}));
  EXPECT_EQ(getContainedTypes(VoidTy), ArrayRef<Type *>({VoidTy}));
  EXPECT_EQ(getContainedTypes(HomogeneousStructTy),
            ArrayRef<Type *>({FTy, FTy, FTy}));
  EXPECT_EQ(getContainedTypes(MixedStructTy), ArrayRef<Type *>({FTy, ITy}));
}

TEST(VectorTypeUtilsTest, TestIsVectorizedTy) {
  LLVMContext C;

  Type *ITy = Type::getInt32Ty(C);
  Type *FTy = Type::getFloatTy(C);
  Type *NarrowStruct = StructType::get(FTy, ITy);
  Type *VoidTy = Type::getVoidTy(C);

  EXPECT_FALSE(isVectorizedTy(ITy));
  EXPECT_FALSE(isVectorizedTy(NarrowStruct));
  EXPECT_FALSE(isVectorizedTy(VoidTy));

  ElementCount VF = ElementCount::getFixed(4);
  EXPECT_TRUE(isVectorizedTy(toVectorizedTy(ITy, VF)));
  EXPECT_TRUE(isVectorizedTy(toVectorizedTy(NarrowStruct, VF)));

  Type *MixedVFStruct =
      StructType::get(VectorType::get(ITy, ElementCount::getFixed(2)),
                      VectorType::get(ITy, ElementCount::getFixed(4)));
  EXPECT_FALSE(isVectorizedTy(MixedVFStruct));

  // Currently only literals types are considered wide.
  Type *NamedWideStruct = StructType::create("Named", VectorType::get(ITy, VF),
                                             VectorType::get(ITy, VF));
  EXPECT_FALSE(isVectorizedTy(NamedWideStruct));

  // Currently only unpacked types are considered wide.
  Type *PackedWideStruct = StructType::get(
      C, ArrayRef<Type *>{VectorType::get(ITy, VF), VectorType::get(ITy, VF)},
      /*isPacked=*/true);
  EXPECT_FALSE(isVectorizedTy(PackedWideStruct));
}

TEST(VectorTypeUtilsTest, TestGetVectorizedTypeVF) {
  LLVMContext C;

  Type *ITy = Type::getInt32Ty(C);
  Type *FTy = Type::getFloatTy(C);
  Type *HomogeneousStructTy = StructType::get(FTy, FTy, FTy);
  Type *MixedStructTy = StructType::get(FTy, ITy);

  for (ElementCount VF :
       {ElementCount::getFixed(4), ElementCount::getScalable(2)}) {
    for (Type *Ty : {ITy, FTy, HomogeneousStructTy, MixedStructTy}) {
      EXPECT_EQ(getVectorizedTypeVF(toVectorizedTy(Ty, VF)), VF);
    };
  }
}

} // namespace
