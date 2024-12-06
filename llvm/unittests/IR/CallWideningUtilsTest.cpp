//===------- CallWideningUtilsTest.cpp - Call widening utils tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/CallWideningUtils.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class CallWideningUtilsTest : public ::testing::Test {};

TEST(CallWideningUtilsTest, TestToWideTy) {
  LLVMContext C;

  Type *ITy = Type::getInt32Ty(C);
  Type *FTy = Type::getFloatTy(C);
  Type *HomogeneousStructTy = StructType::get(FTy, FTy, FTy);
  Type *MixedStructTy = StructType::get(FTy, ITy);
  Type *VoidTy = Type::getVoidTy(C);

  for (ElementCount VF :
       {ElementCount::getFixed(4), ElementCount::getScalable(2)}) {
    Type *IntVec = ToWideTy(ITy, VF);
    EXPECT_TRUE(isa<VectorType>(IntVec));
    EXPECT_EQ(IntVec, VectorType::get(ITy, VF));

    Type *FloatVec = ToWideTy(FTy, VF);
    EXPECT_TRUE(isa<VectorType>(FloatVec));
    EXPECT_EQ(FloatVec, VectorType::get(FTy, VF));

    Type *WideHomogeneousStructTy = ToWideTy(HomogeneousStructTy, VF);
    EXPECT_TRUE(isa<StructType>(WideHomogeneousStructTy));
    EXPECT_TRUE(
        cast<StructType>(WideHomogeneousStructTy)->containsHomogeneousTypes());
    EXPECT_TRUE(cast<StructType>(WideHomogeneousStructTy)->getNumElements() ==
                3);
    EXPECT_TRUE(cast<StructType>(WideHomogeneousStructTy)->getElementType(0) ==
                VectorType::get(FTy, VF));

    Type *WideMixedStructTy = ToWideTy(MixedStructTy, VF);
    EXPECT_TRUE(isa<StructType>(WideMixedStructTy));
    EXPECT_TRUE(cast<StructType>(WideMixedStructTy)->getNumElements() == 2);
    EXPECT_TRUE(cast<StructType>(WideMixedStructTy)->getElementType(0) ==
                VectorType::get(FTy, VF));
    EXPECT_TRUE(cast<StructType>(WideMixedStructTy)->getElementType(1) ==
                VectorType::get(ITy, VF));

    EXPECT_EQ(ToWideTy(VoidTy, VF), VoidTy);
  }

  ElementCount ScalarVF = ElementCount::getFixed(1);
  for (Type *Ty : {ITy, FTy, HomogeneousStructTy, MixedStructTy, VoidTy}) {
    EXPECT_EQ(ToWideTy(Ty, ScalarVF), Ty);
  }
}

TEST(CallWideningUtilsTest, TestToNarrowTy) {
  LLVMContext C;

  Type *ITy = Type::getInt32Ty(C);
  Type *FTy = Type::getFloatTy(C);
  Type *HomogeneousStructTy = StructType::get(FTy, FTy, FTy);
  Type *MixedStructTy = StructType::get(FTy, ITy);
  Type *VoidTy = Type::getVoidTy(C);

  for (ElementCount VF : {ElementCount::getFixed(1), ElementCount::getFixed(4),
                          ElementCount::getScalable(2)}) {
    for (Type *Ty : {ITy, FTy, HomogeneousStructTy, MixedStructTy, VoidTy}) {
      // ToNarrowTy should be the inverse of ToWideTy.
      EXPECT_EQ(ToNarrowTy(ToWideTy(Ty, VF)), Ty);
    };
  }
}

TEST(CallWideningUtilsTest, TestGetContainedTypes) {
  LLVMContext C;

  Type *ITy = Type::getInt32Ty(C);
  Type *FTy = Type::getFloatTy(C);
  Type *HomogeneousStructTy = StructType::get(FTy, FTy, FTy);
  Type *MixedStructTy = StructType::get(FTy, ITy);
  Type *VoidTy = Type::getVoidTy(C);

  EXPECT_EQ(getContainedTypes(ITy), SmallVector<Type *>({ITy}));
  EXPECT_EQ(getContainedTypes(FTy), SmallVector<Type *>({FTy}));
  EXPECT_EQ(getContainedTypes(VoidTy), SmallVector<Type *>({VoidTy}));
  EXPECT_EQ(getContainedTypes(HomogeneousStructTy),
            SmallVector<Type *>({FTy, FTy, FTy}));
  EXPECT_EQ(getContainedTypes(MixedStructTy), SmallVector<Type *>({FTy, ITy}));
}

TEST(CallWideningUtilsTest, TestIsWideTy) {
  LLVMContext C;

  Type *ITy = Type::getInt32Ty(C);
  Type *FTy = Type::getFloatTy(C);
  Type *NarrowStruct = StructType::get(FTy, ITy);
  Type *VoidTy = Type::getVoidTy(C);

  EXPECT_FALSE(isWideTy(ITy));
  EXPECT_FALSE(isWideTy(NarrowStruct));
  EXPECT_FALSE(isWideTy(VoidTy));

  ElementCount VF = ElementCount::getFixed(4);
  EXPECT_TRUE(isWideTy(ToWideTy(ITy, VF)));
  EXPECT_TRUE(isWideTy(ToWideTy(NarrowStruct, VF)));

  Type *MixedVFStruct =
      StructType::get(VectorType::get(ITy, ElementCount::getFixed(2)),
                      VectorType::get(ITy, ElementCount::getFixed(4)));
  EXPECT_FALSE(isWideTy(MixedVFStruct));

  // Currently only literals types are considered wide.
  Type *NamedWideStruct = StructType::create("Named", VectorType::get(ITy, VF),
                                             VectorType::get(ITy, VF));
  EXPECT_FALSE(isWideTy(NamedWideStruct));

  // Currently only unpacked types are considered wide.
  Type *PackedWideStruct = StructType::get(
      C, ArrayRef<Type *>{VectorType::get(ITy, VF), VectorType::get(ITy, VF)},
      /*isPacked=*/true);
  EXPECT_FALSE(isWideTy(PackedWideStruct));
}

TEST(CallWideningUtilsTest, TestGetWideTypeVF) {
  LLVMContext C;

  Type *ITy = Type::getInt32Ty(C);
  Type *FTy = Type::getFloatTy(C);
  Type *HomogeneousStructTy = StructType::get(FTy, FTy, FTy);
  Type *MixedStructTy = StructType::get(FTy, ITy);

  for (ElementCount VF :
       {ElementCount::getFixed(4), ElementCount::getScalable(2)}) {
    for (Type *Ty : {ITy, FTy, HomogeneousStructTy, MixedStructTy}) {
      EXPECT_EQ(getWideTypeVF(ToWideTy(Ty, VF)), VF);
    };
  }
}

} // namespace
