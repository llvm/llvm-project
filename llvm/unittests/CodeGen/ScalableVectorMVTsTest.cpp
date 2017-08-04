//===-------- llvm/unittest/CodeGen/ScalableVectorMVTsTest.cpp ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(ScalableVectorMVTsTest, IntegerMVTs) {
  for (auto VecTy : MVT::integer_scalable_vector_valuetypes()) {
    ASSERT_TRUE(VecTy.isValid());
    ASSERT_TRUE(VecTy.isInteger());
    ASSERT_TRUE(VecTy.isVector());
    ASSERT_TRUE(VecTy.isScalableVector());
    ASSERT_TRUE(VecTy.getScalarType().isValid());

    ASSERT_FALSE(VecTy.isFloatingPoint());
  }
}

TEST(ScalableVectorMVTsTest, FloatMVTs) {
  for (auto VecTy : MVT::fp_scalable_vector_valuetypes()) {
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

  // Create with separate MVT::ElementCount
  auto EltCnt = MVT::ElementCount(2, true);
  EVT Vnx2i32 = EVT::getVectorVT(Ctx, MVT::i32, EltCnt);
  ASSERT_TRUE(Vnx2i32.isScalableVector());

  // Create with inline MVT::ElementCount
  EVT Vnx2i64 = EVT::getVectorVT(Ctx, MVT::i64, {2, true});
  ASSERT_TRUE(Vnx2i64.isScalableVector());

  // Check that changing scalar types/element count works
  EXPECT_EQ(Vnx2i32.widenIntegerVectorElementType(Ctx), Vnx2i64);
  EXPECT_EQ(Vnx4i32.getHalfNumVectorElementsVT(Ctx), Vnx2i32);

  // Check that overloaded '*' and '/' operators work
  EXPECT_EQ(EVT::getVectorVT(Ctx, MVT::i64, EltCnt * 2), MVT::nxv4i64);
  EXPECT_EQ(EVT::getVectorVT(Ctx, MVT::i64, EltCnt / 2), MVT::nxv1i64);

  // Check that float->int conversion works
  EVT Vnx2f64 = EVT::getVectorVT(Ctx, MVT::f64, {2, true});
  EXPECT_EQ(Vnx2f64.changeTypeToInteger(), Vnx2i64);

  // Check fields inside MVT::ElementCount
  EltCnt = Vnx4i32.getVectorElementCount();
  EXPECT_EQ(EltCnt.Min, 4U);
  ASSERT_TRUE(EltCnt.Scalable);

  // Check that fixed-length vector types aren't scalable.
  EVT V8i32 = EVT::getVectorVT(Ctx, MVT::i32, 8);
  ASSERT_FALSE(V8i32.isScalableVector());
  EVT V4f64 = EVT::getVectorVT(Ctx, MVT::f64, {4, false});
  ASSERT_FALSE(V4f64.isScalableVector());

  // Check that MVT::ElementCount works for fixed-length types.
  EltCnt = V8i32.getVectorElementCount();
  EXPECT_EQ(EltCnt.Min, 8U);
  ASSERT_FALSE(EltCnt.Scalable);
}

}
