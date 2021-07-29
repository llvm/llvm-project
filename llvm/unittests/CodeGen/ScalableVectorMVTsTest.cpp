//===-------- llvm/unittest/CodeGen/ScalableVectorMVTsTest.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Support/TypeSize.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(ScalableVectorMVTsTest, IntegerMVTs) {
  for (MVT VecTy : MVT::integer_scalable_vector_valuetypes()) {
    ASSERT_TRUE(VecTy.isValid());
    ASSERT_TRUE(VecTy.isInteger());
    ASSERT_TRUE(VecTy.isVector());
    ASSERT_TRUE(VecTy.isScalableVector());
    ASSERT_TRUE(VecTy.getScalarType().isValid());

    ASSERT_FALSE(VecTy.isFloatingPoint());
  }
}

TEST(ScalableVectorMVTsTest, FloatMVTs) {
  for (MVT VecTy : MVT::fp_scalable_vector_valuetypes()) {
    ASSERT_TRUE(VecTy.isValid());
    ASSERT_TRUE(VecTy.isFloatingPoint());
    ASSERT_TRUE(VecTy.isVector());
    ASSERT_TRUE(VecTy.isScalableVector());
    ASSERT_TRUE(VecTy.getScalarType().isValid());

    ASSERT_FALSE(VecTy.isInteger());
  }
}

TEST(ScalableVectorMVTsTest, HelperFuncs) {
  LLVMContext Ctx;

  // Create with scalable flag
  EVT Vnx4i32 = EVT::getVectorVT(Ctx, MVT::i32, 4, /*Scalable=*/true);
  ASSERT_TRUE(Vnx4i32.isScalableVector());

  // Create with separate llvm::ElementCount
  auto EltCnt = ElementCount::getScalable(2);
  EVT Vnx2i32 = EVT::getVectorVT(Ctx, MVT::i32, EltCnt);
  ASSERT_TRUE(Vnx2i32.isScalableVector());

  // Create with inline llvm::ElementCount
  EVT Vnx2i64 = EVT::getVectorVT(Ctx, MVT::i64, ElementCount::getScalable(2));
  ASSERT_TRUE(Vnx2i64.isScalableVector());

  // Check that changing scalar types/element count works
  EXPECT_EQ(Vnx2i32.widenIntegerVectorElementType(Ctx), Vnx2i64);
  EXPECT_EQ(Vnx4i32.getHalfNumVectorElementsVT(Ctx), Vnx2i32);

  // Check that operators work
  EXPECT_EQ(EVT::getVectorVT(Ctx, MVT::i64, EltCnt * 2), MVT::nxv4i64);
  EXPECT_EQ(EVT::getVectorVT(Ctx, MVT::i64, EltCnt.divideCoefficientBy(2)),
            MVT::nxv1i64);

  // Check that float->int conversion works
  EVT Vnx2f64 = EVT::getVectorVT(Ctx, MVT::f64, ElementCount::getScalable(2));
  EXPECT_EQ(Vnx2f64.changeTypeToInteger(), Vnx2i64);

  // Check fields inside llvm::ElementCount
  EltCnt = Vnx4i32.getVectorElementCount();
  EXPECT_EQ(EltCnt.getKnownMinValue(), 4U);
  ASSERT_TRUE(EltCnt.isScalable());

  // Check that fixed-length vector types aren't scalable.
  EVT V8i32 = EVT::getVectorVT(Ctx, MVT::i32, 8);
  ASSERT_FALSE(V8i32.isScalableVector());
  EVT V4f64 = EVT::getVectorVT(Ctx, MVT::f64, ElementCount::getFixed(4));
  ASSERT_FALSE(V4f64.isScalableVector());

  // Check that llvm::ElementCount works for fixed-length types.
  EltCnt = V8i32.getVectorElementCount();
  EXPECT_EQ(EltCnt.getKnownMinValue(), 8U);
  ASSERT_FALSE(EltCnt.isScalable());
}

TEST(ScalableVectorMVTsTest, IRToVTTranslation) {
  LLVMContext Ctx;

  Type *Int64Ty = Type::getInt64Ty(Ctx);
  VectorType *ScV8Int64Ty =
      VectorType::get(Int64Ty, ElementCount::getScalable(8));

  // Check that we can map a scalable IR type to an MVT 
  MVT Mnxv8i64 = MVT::getVT(ScV8Int64Ty);
  ASSERT_TRUE(Mnxv8i64.isScalableVector());
  ASSERT_EQ(ScV8Int64Ty->getElementCount(), Mnxv8i64.getVectorElementCount());
  ASSERT_EQ(MVT::getVT(ScV8Int64Ty->getElementType()),
            Mnxv8i64.getScalarType());

  // Check that we can map a scalable IR type to an EVT
  EVT Enxv8i64 = EVT::getEVT(ScV8Int64Ty);
  ASSERT_TRUE(Enxv8i64.isScalableVector());
  ASSERT_EQ(ScV8Int64Ty->getElementCount(), Enxv8i64.getVectorElementCount());
  ASSERT_EQ(EVT::getEVT(ScV8Int64Ty->getElementType()),
            Enxv8i64.getScalarType());
}

TEST(ScalableVectorMVTsTest, VTToIRTranslation) {
  LLVMContext Ctx;

  EVT Enxv4f64 = EVT::getVectorVT(Ctx, MVT::f64, ElementCount::getScalable(4));

  Type *Ty = Enxv4f64.getTypeForEVT(Ctx);
  VectorType *ScV4Float64Ty = cast<VectorType>(Ty);
  ASSERT_TRUE(isa<ScalableVectorType>(ScV4Float64Ty));
  ASSERT_EQ(Enxv4f64.getVectorElementCount(), ScV4Float64Ty->getElementCount());
  ASSERT_EQ(Enxv4f64.getScalarType().getTypeForEVT(Ctx),
            ScV4Float64Ty->getElementType());
}

TEST(ScalableVectorMVTsTest, SizeQueries) {
  LLVMContext Ctx;

  EVT nxv4i32 = EVT::getVectorVT(Ctx, MVT::i32, 4, /*Scalable=*/ true);
  EVT nxv2i32 = EVT::getVectorVT(Ctx, MVT::i32, 2, /*Scalable=*/ true);
  EVT nxv2i64 = EVT::getVectorVT(Ctx, MVT::i64, 2, /*Scalable=*/ true);
  EVT nxv2f64 = EVT::getVectorVT(Ctx, MVT::f64, 2, /*Scalable=*/ true);

  EVT v4i32 = EVT::getVectorVT(Ctx, MVT::i32, 4);
  EVT v2i32 = EVT::getVectorVT(Ctx, MVT::i32, 2);
  EVT v2i64 = EVT::getVectorVT(Ctx, MVT::i64, 2);
  EVT v2f64 = EVT::getVectorVT(Ctx, MVT::f64, 2);

  // Check equivalence and ordering on scalable types.
  EXPECT_EQ(nxv4i32.getSizeInBits(), nxv2i64.getSizeInBits());
  EXPECT_EQ(nxv2f64.getSizeInBits(), nxv2i64.getSizeInBits());
  EXPECT_NE(nxv2i32.getSizeInBits(), nxv4i32.getSizeInBits());
  EXPECT_LT(nxv2i32.getSizeInBits().getKnownMinSize(),
            nxv2i64.getSizeInBits().getKnownMinSize());
  EXPECT_LE(nxv4i32.getSizeInBits().getKnownMinSize(),
            nxv2i64.getSizeInBits().getKnownMinSize());
  EXPECT_GT(nxv4i32.getSizeInBits().getKnownMinSize(),
            nxv2i32.getSizeInBits().getKnownMinSize());
  EXPECT_GE(nxv2i64.getSizeInBits().getKnownMinSize(),
            nxv4i32.getSizeInBits().getKnownMinSize());

  // Check equivalence and ordering on fixed types.
  EXPECT_EQ(v4i32.getSizeInBits(), v2i64.getSizeInBits());
  EXPECT_EQ(v2f64.getSizeInBits(), v2i64.getSizeInBits());
  EXPECT_NE(v2i32.getSizeInBits(), v4i32.getSizeInBits());
  EXPECT_LT(v2i32.getFixedSizeInBits(), v2i64.getFixedSizeInBits());
  EXPECT_LE(v4i32.getFixedSizeInBits(), v2i64.getFixedSizeInBits());
  EXPECT_GT(v4i32.getFixedSizeInBits(), v2i32.getFixedSizeInBits());
  EXPECT_GE(v2i64.getFixedSizeInBits(), v4i32.getFixedSizeInBits());

  // Check that scalable and non-scalable types with the same minimum size
  // are not considered equal.
  ASSERT_TRUE(v4i32.getSizeInBits() != nxv4i32.getSizeInBits());
  ASSERT_FALSE(v2i64.getSizeInBits() == nxv2f64.getSizeInBits());

  // Check that we can obtain a known-exact size from a non-scalable type.
  EXPECT_EQ(v4i32.getFixedSizeInBits(), 128U);
  EXPECT_EQ(v2i64.getFixedSizeInBits(), 128U);

  // Check that we can query the known minimum size for both scalable and
  // fixed length types.
  EXPECT_EQ(nxv2i32.getSizeInBits().getKnownMinSize(), 64U);
  EXPECT_EQ(nxv2f64.getSizeInBits().getKnownMinSize(), 128U);
  EXPECT_EQ(v2i32.getSizeInBits().getKnownMinSize(),
            nxv2i32.getSizeInBits().getKnownMinSize());

  // Check scalable property.
  ASSERT_FALSE(v4i32.getSizeInBits().isScalable());
  ASSERT_TRUE(nxv4i32.getSizeInBits().isScalable());

  // Check convenience size scaling methods.
  EXPECT_EQ(v2i32.getSizeInBits() * 2, v4i32.getSizeInBits());
  EXPECT_EQ(2 * nxv2i32.getSizeInBits(), nxv4i32.getSizeInBits());
  EXPECT_EQ(nxv2f64.getSizeInBits().divideCoefficientBy(2),
            nxv2i32.getSizeInBits());
}

} // end anonymous namespace
