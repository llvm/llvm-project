//===- llvm/unittest/CodeGen/GlobalISel/LowLevelTypeTest.cpp --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "gtest/gtest.h"

using namespace llvm;

// Define a pretty printer to help debugging when things go wrong.
namespace llvm {
std::ostream &
operator<<(std::ostream &OS, const llvm::LLT Ty) {
  std::string Repr;
  raw_string_ostream SS{Repr};
  Ty.print(SS);
  OS << SS.str();
  return OS;
}
}

namespace {

TEST(LowLevelTypeTest, Scalar) {
  LLVMContext C;
  DataLayout DL("");

  for (unsigned S : {1U, 17U, 32U, 64U, 0xfffffU}) {
    const LLT Ty = LLT::scalar(S);
    const LLT HalfTy = (S % 2) == 0 ? Ty.halfScalarSize() : Ty;
    const LLT DoubleTy = Ty.doubleScalarSize();

    // Test kind.
    for (const LLT TestTy : {Ty, HalfTy, DoubleTy}) {
      ASSERT_TRUE(TestTy.isValid());
      ASSERT_TRUE(TestTy.isScalar());

      ASSERT_FALSE(TestTy.isPointer());
      ASSERT_FALSE(TestTy.isVector());
    }

    // Test sizes.
    EXPECT_EQ(S, Ty.getSizeInBits());
    EXPECT_EQ(S, Ty.getScalarSizeInBits());

    EXPECT_EQ(S*2, DoubleTy.getSizeInBits());
    EXPECT_EQ(S*2, DoubleTy.getScalarSizeInBits());

    if ((S % 2) == 0) {
      EXPECT_EQ(S/2, HalfTy.getSizeInBits());
      EXPECT_EQ(S/2, HalfTy.getScalarSizeInBits());
    }

    // Test equality operators.
    EXPECT_TRUE(Ty == Ty);
    EXPECT_FALSE(Ty != Ty);

    EXPECT_NE(Ty, DoubleTy);

    // Test Type->LLT conversion.
    Type *IRTy = IntegerType::get(C, S);
    EXPECT_EQ(Ty, getLLTForType(*IRTy, DL));
  }
}

TEST(LowLevelTypeTest, Vector) {
  LLVMContext C;
  DataLayout DL("");

  for (unsigned S : {1U, 17U, 32U, 64U, 0xfffU}) {
    for (uint16_t Elts : {2U, 3U, 4U, 32U, 0xffU}) {
      const LLT STy = LLT::scalar(S);
      const LLT VTy = LLT::vector(Elts, S);

      // Test the alternative vector().
      {
        const LLT VSTy = LLT::vector(Elts, STy);
        EXPECT_EQ(VTy, VSTy);
      }

      // Test getElementType().
      EXPECT_EQ(STy, VTy.getElementType());

      const LLT HalfSzTy = ((S % 2) == 0) ? VTy.halfScalarSize() : VTy;
      const LLT DoubleSzTy = VTy.doubleScalarSize();

      // halfElements requires an even number of elements.
      const LLT HalfEltIfEvenTy = ((Elts % 2) == 0) ?  VTy.halfElements() : VTy;
      const LLT DoubleEltTy = VTy.doubleElements();

      // Test kind.
      for (const LLT TestTy : {VTy, HalfSzTy, DoubleSzTy, DoubleEltTy}) {
        ASSERT_TRUE(TestTy.isValid());
        ASSERT_TRUE(TestTy.isVector());

        ASSERT_FALSE(TestTy.isScalar());
        ASSERT_FALSE(TestTy.isPointer());
      }

      // Test halving elements to a scalar.
      {
        ASSERT_TRUE(HalfEltIfEvenTy.isValid());
        ASSERT_FALSE(HalfEltIfEvenTy.isPointer());
        if (Elts > 2) {
          ASSERT_TRUE(HalfEltIfEvenTy.isVector());
        } else {
          ASSERT_FALSE(HalfEltIfEvenTy.isVector());
          EXPECT_EQ(STy, HalfEltIfEvenTy);
        }
      }


      // Test sizes.
      EXPECT_EQ(S * Elts, VTy.getSizeInBits());
      EXPECT_EQ(S, VTy.getScalarSizeInBits());
      EXPECT_EQ(Elts, VTy.getNumElements());

      if ((S % 2) == 0) {
        EXPECT_EQ((S / 2) * Elts, HalfSzTy.getSizeInBits());
        EXPECT_EQ(S / 2, HalfSzTy.getScalarSizeInBits());
        EXPECT_EQ(Elts, HalfSzTy.getNumElements());
      }

      EXPECT_EQ((S * 2) * Elts, DoubleSzTy.getSizeInBits());
      EXPECT_EQ(S * 2, DoubleSzTy.getScalarSizeInBits());
      EXPECT_EQ(Elts, DoubleSzTy.getNumElements());

      if ((Elts % 2) == 0) {
        EXPECT_EQ(S * (Elts / 2), HalfEltIfEvenTy.getSizeInBits());
        EXPECT_EQ(S, HalfEltIfEvenTy.getScalarSizeInBits());
        if (Elts > 2)
          EXPECT_EQ(Elts / 2, HalfEltIfEvenTy.getNumElements());
      }

      EXPECT_EQ(S * (Elts * 2), DoubleEltTy.getSizeInBits());
      EXPECT_EQ(S, DoubleEltTy.getScalarSizeInBits());
      EXPECT_EQ(Elts * 2, DoubleEltTy.getNumElements());

      // Test equality operators.
      EXPECT_TRUE(VTy == VTy);
      EXPECT_FALSE(VTy != VTy);

      // Test inequality operators on..
      // ..different kind.
      EXPECT_NE(VTy, STy);
      // ..different #elts.
      EXPECT_NE(VTy, DoubleEltTy);
      // ..different scalar size.
      EXPECT_NE(VTy, DoubleSzTy);

      // Test Type->LLT conversion.
      Type *IRSTy = IntegerType::get(C, S);
      Type *IRTy = VectorType::get(IRSTy, Elts);
      EXPECT_EQ(VTy, getLLTForType(*IRTy, DL));
    }
  }
}

TEST(LowLevelTypeTest, Pointer) {
  LLVMContext C;
  DataLayout DL("");

  for (unsigned AS : {0U, 1U, 127U, 0xffffU}) {
    const LLT Ty = LLT::pointer(AS, DL.getPointerSizeInBits(AS));
    const LLT VTy = LLT::vector(4, Ty);

    // Test kind.
    ASSERT_TRUE(Ty.isValid());
    ASSERT_TRUE(Ty.isPointer());

    ASSERT_FALSE(Ty.isScalar());
    ASSERT_FALSE(Ty.isVector());

    ASSERT_TRUE(VTy.isValid());
    ASSERT_TRUE(VTy.isVector());
    ASSERT_TRUE(VTy.getElementType().isPointer());

    // Test addressspace.
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
        VectorType::get(PointerType::get(IntegerType::get(C, 8), AS), 4);
    EXPECT_EQ(VTy, getLLTForType(*IRVTy, DL));
  }
}

TEST(LowLevelTypeTest, Invalid) {
  const LLT Ty;

  ASSERT_FALSE(Ty.isValid());
  ASSERT_FALSE(Ty.isScalar());
  ASSERT_FALSE(Ty.isPointer());
  ASSERT_FALSE(Ty.isVector());
}

}
