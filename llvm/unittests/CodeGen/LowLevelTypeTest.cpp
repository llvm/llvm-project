//===- llvm/unittest/CodeGen/GlobalISel/LowLevelTypeTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/TypeSize.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(LowLevelTypeTest, Scalar) {
  LLVMContext C;
  DataLayout DL("");

  for (unsigned S : {1U, 17U, 32U, 64U, 0xfffffU}) {
    const LLT Ty = LLT::scalar(S);

    // Test kind.
    ASSERT_TRUE(Ty.isValid());
    ASSERT_TRUE(Ty.isScalar());

    ASSERT_FALSE(Ty.isPointer());
    ASSERT_FALSE(Ty.isVector());

    // Test sizes.
    EXPECT_EQ(S, Ty.getSizeInBits());
    EXPECT_EQ(S, Ty.getScalarSizeInBits());

    // Test equality operators.
    EXPECT_TRUE(Ty == Ty);
    EXPECT_FALSE(Ty != Ty);

    // Test Type->LLT conversion.
    Type *IRTy = IntegerType::get(C, S);
    EXPECT_EQ(Ty, getLLTForType(*IRTy, DL));
  }
}

TEST(LowLevelTypeTest, Vector) {
  LLVMContext C;
  DataLayout DL("");

  for (unsigned S : {1U, 17U, 32U, 64U, 0xfffU}) {
    for (auto EC :
         {ElementCount::getFixed(2), ElementCount::getFixed(3),
          ElementCount::getFixed(4), ElementCount::getFixed(32),
          ElementCount::getFixed(0xff), ElementCount::getScalable(2),
          ElementCount::getScalable(3), ElementCount::getScalable(4),
          ElementCount::getScalable(32), ElementCount::getScalable(0xff)}) {
      const LLT STy = LLT::scalar(S);
      const LLT VTy = LLT::vector(EC, S);

      // Test the alternative vector().
      {
        const LLT VSTy = LLT::vector(EC, STy);
        EXPECT_EQ(VTy, VSTy);
      }

      // Test getElementType().
      EXPECT_EQ(STy, VTy.getElementType());

      // Test kind.
      ASSERT_TRUE(VTy.isValid());
      ASSERT_TRUE(VTy.isVector());

      ASSERT_FALSE(VTy.isScalar());
      ASSERT_FALSE(VTy.isPointer());

      // Test sizes.
      EXPECT_EQ(S, VTy.getScalarSizeInBits());
      EXPECT_EQ(EC, VTy.getElementCount());
      if (!EC.isScalable())
        EXPECT_EQ(S * EC.getFixedValue(), VTy.getSizeInBits());

      // Test equality operators.
      EXPECT_TRUE(VTy == VTy);
      EXPECT_FALSE(VTy != VTy);

      // Test inequality operators on..
      // ..different kind.
      EXPECT_NE(VTy, STy);

      // Test Type->LLT conversion.
      Type *IRSTy = IntegerType::get(C, S);
      Type *IRTy = VectorType::get(IRSTy, EC);
      EXPECT_EQ(VTy, getLLTForType(*IRTy, DL));
    }
  }
}

TEST(LowLevelTypeTest, ScalarOrVector) {
  // Test version with number of bits for scalar type.
  EXPECT_EQ(LLT::scalar(32),
            LLT::scalarOrVector(ElementCount::getFixed(1), 32));
  EXPECT_EQ(LLT::fixed_vector(2, 32),
            LLT::scalarOrVector(ElementCount::getFixed(2), 32));
  EXPECT_EQ(LLT::scalable_vector(1, 32),
            LLT::scalarOrVector(ElementCount::getScalable(1), 32));

  // Test version with LLT for scalar type.
  EXPECT_EQ(LLT::scalar(32),
            LLT::scalarOrVector(ElementCount::getFixed(1), LLT::scalar(32)));
  EXPECT_EQ(LLT::fixed_vector(2, 32),
            LLT::scalarOrVector(ElementCount::getFixed(2), LLT::scalar(32)));

  // Test with pointer elements.
  EXPECT_EQ(LLT::pointer(1, 32), LLT::scalarOrVector(ElementCount::getFixed(1),
                                                     LLT::pointer(1, 32)));
  EXPECT_EQ(
      LLT::fixed_vector(2, LLT::pointer(1, 32)),
      LLT::scalarOrVector(ElementCount::getFixed(2), LLT::pointer(1, 32)));
}

TEST(LowLevelTypeTest, ChangeElementType) {
  const LLT P0 = LLT::pointer(0, 32);
  const LLT P1 = LLT::pointer(1, 64);

  const LLT S32 = LLT::scalar(32);
  const LLT S64 = LLT::scalar(64);

  const LLT V2S32 = LLT::fixed_vector(2, 32);
  const LLT V2S64 = LLT::fixed_vector(2, 64);

  const LLT V2P0 = LLT::fixed_vector(2, P0);
  const LLT V2P1 = LLT::fixed_vector(2, P1);

  EXPECT_EQ(S64, S32.changeElementType(S64));
  EXPECT_EQ(S32, S32.changeElementType(S32));

  EXPECT_EQ(S32, S64.changeElementSize(32));
  EXPECT_EQ(S32, S32.changeElementSize(32));

  EXPECT_EQ(V2S64, V2S32.changeElementType(S64));
  EXPECT_EQ(V2S32, V2S64.changeElementType(S32));

  EXPECT_EQ(V2S64, V2S32.changeElementSize(64));
  EXPECT_EQ(V2S32, V2S64.changeElementSize(32));

  EXPECT_EQ(P0, S32.changeElementType(P0));
  EXPECT_EQ(S32, P0.changeElementType(S32));

  EXPECT_EQ(V2P1, V2P0.changeElementType(P1));
  EXPECT_EQ(V2S32, V2P0.changeElementType(S32));

  // Similar tests for for scalable vectors.
  const LLT NXV2S32 = LLT::scalable_vector(2, 32);
  const LLT NXV2S64 = LLT::scalable_vector(2, 64);

  const LLT NXV2P0 = LLT::scalable_vector(2, P0);
  const LLT NXV2P1 = LLT::scalable_vector(2, P1);

  EXPECT_EQ(NXV2S64, NXV2S32.changeElementType(S64));
  EXPECT_EQ(NXV2S32, NXV2S64.changeElementType(S32));

  EXPECT_EQ(NXV2S64, NXV2S32.changeElementSize(64));
  EXPECT_EQ(NXV2S32, NXV2S64.changeElementSize(32));

  EXPECT_EQ(NXV2P1, NXV2P0.changeElementType(P1));
  EXPECT_EQ(NXV2S32, NXV2P0.changeElementType(S32));
}

TEST(LowLevelTypeTest, ChangeNumElements) {
  const LLT P0 = LLT::pointer(0, 32);
  const LLT V2P0 = LLT::fixed_vector(2, P0);
  const LLT V3P0 = LLT::fixed_vector(3, P0);

  const LLT S64 = LLT::scalar(64);
  const LLT V2S64 = LLT::fixed_vector(2, 64);
  const LLT V3S64 = LLT::fixed_vector(3, 64);

  // Vector to scalar
  EXPECT_EQ(S64, V2S64.changeElementCount(ElementCount::getFixed(1)));

  // Vector to vector
  EXPECT_EQ(V3S64, V2S64.changeElementCount(ElementCount::getFixed(3)));

  // Scalar to vector
  EXPECT_EQ(V2S64, S64.changeElementCount(ElementCount::getFixed(2)));

  EXPECT_EQ(P0, V2P0.changeElementCount(ElementCount::getFixed(1)));
  EXPECT_EQ(V3P0, V2P0.changeElementCount(ElementCount::getFixed(3)));
  EXPECT_EQ(V2P0, P0.changeElementCount(ElementCount::getFixed(2)));

  const LLT NXV2S64 = LLT::scalable_vector(2, 64);
  const LLT NXV3S64 = LLT::scalable_vector(3, 64);
  const LLT NXV2P0 = LLT::scalable_vector(2, P0);

  // Scalable vector to scalar
  EXPECT_EQ(S64, NXV2S64.changeElementCount(ElementCount::getFixed(1)));
  EXPECT_EQ(P0, NXV2P0.changeElementCount(ElementCount::getFixed(1)));

  // Fixed-width vector to scalable vector
  EXPECT_EQ(NXV3S64, V2S64.changeElementCount(ElementCount::getScalable(3)));

  // Scalable vector to fixed-width vector
  EXPECT_EQ(V3P0, NXV2P0.changeElementCount(ElementCount::getFixed(3)));

  // Scalar to scalable vector
  EXPECT_EQ(NXV2S64, S64.changeElementCount(ElementCount::getScalable(2)));
  EXPECT_EQ(NXV2P0, P0.changeElementCount(ElementCount::getScalable(2)));
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG

// Invalid to directly change the element size for pointers.
TEST(LowLevelTypeTest, ChangeElementTypeDeath) {
  const LLT P0 = LLT::pointer(0, 32);
  const LLT V2P0 = LLT::fixed_vector(2, P0);

  EXPECT_DEATH(P0.changeElementSize(64),
               "invalid to directly change element size for pointers");
  EXPECT_DEATH(V2P0.changeElementSize(64),
               "invalid to directly change element size for pointers");

  // Make sure this still fails even without a change in size.
  EXPECT_DEATH(P0.changeElementSize(32),
               "invalid to directly change element size for pointers");
  EXPECT_DEATH(V2P0.changeElementSize(32),
               "invalid to directly change element size for pointers");
}

#endif
#endif

TEST(LowLevelTypeTest, Pointer) {
  LLVMContext C;
  DataLayout DL("p64:64:64-p127:512:512:512-p16777215:65528:8");

  for (unsigned AS : {0U, 1U, 127U, 0xffffU,
        static_cast<unsigned>(maxUIntN(23)),
        static_cast<unsigned>(maxUIntN(24))}) {
    for (ElementCount EC :
         {ElementCount::getFixed(2), ElementCount::getFixed(3),
          ElementCount::getFixed(4), ElementCount::getFixed(256),
          ElementCount::getFixed(65535), ElementCount::getScalable(2),
          ElementCount::getScalable(3), ElementCount::getScalable(4),
          ElementCount::getScalable(256), ElementCount::getScalable(65535)}) {
      const LLT Ty = LLT::pointer(AS, DL.getPointerSizeInBits(AS));
      const LLT VTy = LLT::vector(EC, Ty);

      // Test kind.
      ASSERT_TRUE(Ty.isValid());
      ASSERT_TRUE(Ty.isPointer());

      ASSERT_FALSE(Ty.isScalar());
      ASSERT_FALSE(Ty.isVector());

      ASSERT_TRUE(VTy.isValid());
      ASSERT_TRUE(VTy.isVector());
      ASSERT_TRUE(VTy.getElementType().isPointer());

      EXPECT_EQ(Ty, VTy.getElementType());
      EXPECT_EQ(Ty.getSizeInBits(), VTy.getScalarSizeInBits());

      // Test address space.
      EXPECT_EQ(AS, Ty.getAddressSpace());
      EXPECT_EQ(AS, VTy.getElementType().getAddressSpace());

      // Test equality operators.
      EXPECT_TRUE(Ty == Ty);
      EXPECT_FALSE(Ty != Ty);
      EXPECT_TRUE(VTy == VTy);
      EXPECT_FALSE(VTy != VTy);

      // Test Type->LLT conversion.
      Type *IRTy = PointerType::get(IntegerType::get(C, 8), AS);
      EXPECT_EQ(Ty, getLLTForType(*IRTy, DL));
      Type *IRVTy =
          VectorType::get(PointerType::get(IntegerType::get(C, 8), AS), EC);
      EXPECT_EQ(VTy, getLLTForType(*IRVTy, DL));
    }
  }
}

TEST(LowLevelTypeTest, Invalid) {
  const LLT Ty;

  ASSERT_FALSE(Ty.isValid());
  ASSERT_FALSE(Ty.isScalar());
  ASSERT_FALSE(Ty.isPointer());
  ASSERT_FALSE(Ty.isVector());
}

TEST(LowLevelTypeTest, Divide) {
  // Test basic scalar->scalar cases.
  EXPECT_EQ(LLT::scalar(16), LLT::scalar(32).divide(2));
  EXPECT_EQ(LLT::scalar(8), LLT::scalar(32).divide(4));
  EXPECT_EQ(LLT::scalar(8), LLT::scalar(32).divide(4));

  // Test pointer->scalar
  EXPECT_EQ(LLT::scalar(32), LLT::pointer(0, 64).divide(2));

  // Test dividing vectors.
  EXPECT_EQ(LLT::scalar(32), LLT::fixed_vector(2, 32).divide(2));
  EXPECT_EQ(LLT::fixed_vector(2, 32), LLT::fixed_vector(4, 32).divide(2));

  // Test vector of pointers
  EXPECT_EQ(LLT::pointer(1, 64),
            LLT::fixed_vector(4, LLT::pointer(1, 64)).divide(4));
  EXPECT_EQ(LLT::fixed_vector(2, LLT::pointer(1, 64)),
            LLT::fixed_vector(4, LLT::pointer(1, 64)).divide(2));
}

}
