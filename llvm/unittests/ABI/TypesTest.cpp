//===- TypesTest.cpp - ABI type unit tests --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "gtest/gtest.h"

using llvm::Align;
using llvm::ElementCount;
using llvm::TypeSize;
using llvm::abi::AtomicType;
using llvm::abi::FieldInfo;
using llvm::abi::RecordFlags;
using llvm::abi::RecordType;
using llvm::abi::StructPacking;
using llvm::abi::TupleType;
using llvm::abi::TypeBuilder;
using llvm::abi::VectorKind;
using llvm::abi::VectorType;

namespace {

class ABITypesTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;

  ABITypesTest() : TB(Alloc) {}

  const RecordType *makeRecord(llvm::ArrayRef<FieldInfo> Fields,
                               uint64_t SizeBits, RecordFlags Flags,
                               llvm::ArrayRef<FieldInfo> Bases = {},
                               llvm::ArrayRef<FieldInfo> VBases = {},
                               Align Alignment = Align(1)) {
    return TB.getRecordType(Fields, TypeSize::getFixed(SizeBits), Alignment,
                            /*UnadjustedAlign=*/Alignment,
                            StructPacking::Default, Bases, VBases, Flags);
  }
};

TEST_F(ABITypesTest, AtomicTypeProperties) {
  const llvm::abi::Type *Value =
      TB.getIntegerType(24, Align(1), /*Signed=*/false);
  const AtomicType *Atomic = TB.getAtomicType(Value, 32, Align(4));

  EXPECT_TRUE(Atomic->isAtomic());
  EXPECT_EQ(Atomic->getKind(), llvm::abi::TypeKind::Atomic);
  EXPECT_EQ(Atomic->getValueType(), Value);
  EXPECT_EQ(Atomic->getSizeInBits(), TypeSize::getFixed(32));
  EXPECT_EQ(Atomic->getAlignment(), Align(4));
  EXPECT_TRUE(llvm::isa<AtomicType>(Atomic));
  EXPECT_FALSE(Atomic->isEmptyRecord());
}

TEST_F(ABITypesTest, EmptyCRecord) {
  const RecordType *Empty = makeRecord({}, 0, RecordFlags::CanPassInRegisters);
  EXPECT_TRUE(Empty->isEmpty());
  EXPECT_TRUE(Empty->isEmptyRecord());
}

TEST_F(ABITypesTest, NestedEmptyCRecordField) {
  const RecordType *Empty = makeRecord({}, 8, RecordFlags::CanPassInRegisters);
  const RecordType *Nested =
      makeRecord({FieldInfo(Empty, 0)}, 8, RecordFlags::CanPassInRegisters);
  EXPECT_TRUE(Nested->isEmpty());
}

TEST_F(ABITypesTest, CXXNestedEmptyFieldRequiresNoUniqueAddress) {
  RecordFlags CXXFlags = static_cast<RecordFlags>(
      RecordFlags::CanPassInRegisters | RecordFlags::IsCXXRecord);
  const RecordType *Empty = makeRecord({}, 8, CXXFlags);

  const RecordType *WithoutNUA = makeRecord({FieldInfo(Empty, 0)}, 8, CXXFlags);
  EXPECT_FALSE(WithoutNUA->isEmpty());

  FieldInfo NUAField(Empty, 0, /*IsBitField=*/false, /*BitFieldWidth=*/0,
                     /*IsUnnamedBitField=*/false,
                     /*HasNoUniqueAddress=*/true);
  const RecordType *WithNUA = makeRecord({NUAField}, 8, CXXFlags);
  EXPECT_TRUE(WithNUA->isEmpty());
}

TEST_F(ABITypesTest, ArrayOfEmptyRecords) {
  RecordFlags CFlags = RecordFlags::CanPassInRegisters;
  RecordFlags CXXFlags = static_cast<RecordFlags>(
      RecordFlags::CanPassInRegisters | RecordFlags::IsCXXRecord);
  const RecordType *EmptyC = makeRecord({}, 8, CFlags);
  const RecordType *EmptyCXX = makeRecord({}, 8, CXXFlags);
  const llvm::abi::Type *ArrC = TB.getArrayType(EmptyC, 2, 16);
  const llvm::abi::Type *ArrCXX = TB.getArrayType(EmptyCXX, 2, 16);
  const llvm::abi::Type *ZeroArrCXX = TB.getArrayType(EmptyCXX, 0, 0);

  EXPECT_TRUE(makeRecord({FieldInfo(ArrC, 0)}, 16, CFlags)->isEmpty());
  EXPECT_FALSE(makeRecord({FieldInfo(ArrCXX, 0)}, 16, CXXFlags)->isEmpty());
  EXPECT_TRUE(makeRecord({FieldInfo(ZeroArrCXX, 0)}, 0, CXXFlags)->isEmpty());
}

TEST_F(ABITypesTest, BitfieldsAndFlexibleArrays) {
  const llvm::abi::Type *I32 = TB.getIntegerType(32, Align(4), /*Signed=*/true);
  FieldInfo Unnamed(I32, 0, /*IsBitField=*/true, /*BitFieldWidth=*/3,
                    /*IsUnnamedBitField=*/true);
  FieldInfo NamedZero(I32, 0, /*IsBitField=*/true, /*BitFieldWidth=*/0);

  EXPECT_TRUE(
      makeRecord({Unnamed}, 8, RecordFlags::CanPassInRegisters)->isEmpty());
  EXPECT_FALSE(
      makeRecord({NamedZero}, 8, RecordFlags::CanPassInRegisters)->isEmpty());
  EXPECT_FALSE(
      makeRecord({}, 0, RecordFlags::HasFlexibleArrayMember)->isEmpty());
}

TEST_F(ABITypesTest, DirectVirtualBasesAndVTablePointer) {
  RecordFlags CXXFlags = static_cast<RecordFlags>(
      RecordFlags::CanPassInRegisters | RecordFlags::IsCXXRecord);
  // Polymorphic classes have a non-trivial copy constructor, so they are not
  // passed in registers.
  RecordFlags PolymorphicFlags = static_cast<RecordFlags>(
      RecordFlags::IsCXXRecord | RecordFlags::IsPolymorphic);
  const RecordType *Empty = makeRecord({}, 8, CXXFlags);
  const RecordType *IntField = makeRecord(
      {FieldInfo(TB.getIntegerType(32, Align(4), /*Signed=*/true), 0)}, 32,
      CXXFlags, /*Bases=*/{}, /*VBases=*/{}, Align(4));
  const llvm::abi::Type *VPtr = TB.getPointerType(64, Align(8));
  FieldInfo VTable(VPtr, 0);

  // Empty vbase with vtable
  EXPECT_FALSE(makeRecord({VTable}, 64, PolymorphicFlags, /*Bases=*/{},
                          /*VBases=*/{FieldInfo(Empty, 0)}, Align(8))
                   ->isEmpty());
  // Non-empty vbase with vtable
  EXPECT_FALSE(makeRecord({VTable}, 128, PolymorphicFlags, /*Bases=*/{},
                          /*VBases=*/{FieldInfo(IntField, 64)}, Align(8))
                   ->isEmpty());
  // Empty base with vtable
  EXPECT_FALSE(makeRecord({VTable, FieldInfo(Empty, 64)}, 128, PolymorphicFlags,
                          /*Bases=*/{FieldInfo(Empty, 0)}, /*VBases=*/{},
                          Align(8))
                   ->isEmpty());
}

TEST_F(ABITypesTest, GenericVector) {
  const llvm::abi::Type *I32 = TB.getIntegerType(32, Align(4), /*Signed=*/true);
  const VectorType *V4I32 =
      TB.getVectorType(I32, ElementCount::getFixed(4), Align(16));

  EXPECT_EQ(V4I32->getVectorKind(), VectorKind::Generic);
  EXPECT_FALSE(V4I32->isSVEType());
  EXPECT_FALSE(V4I32->isSVESizelessType());
  EXPECT_TRUE(V4I32->isFixedLength());
  EXPECT_FALSE(V4I32->isTuple());
  EXPECT_EQ(V4I32->getSizeInBits(), TypeSize::getFixed(128));
  EXPECT_EQ(V4I32->getFixedSizeInBitsOrZero(), 128u);
}

// An x87 element holds 80 bits but counts at its 16-byte allocation size
// before the vector is rounded up to a power of two. With 4-byte alignment an
// element counts as 96 bits.
TEST_F(ABITypesTest, X87VectorABISize) {
  const llvm::abi::Type *F80 =
      TB.getFloatType(llvm::APFloat::x87DoubleExtended(), Align(16));
  auto MakeVector = [&](const llvm::abi::Type *Elt, unsigned N) {
    return TB.getVectorType(Elt, ElementCount::getFixed(N), Align(16));
  };

  EXPECT_EQ(MakeVector(F80, 1)->getSizeInBits(), TypeSize::getFixed(80));
  EXPECT_EQ(MakeVector(F80, 1)->getABISizeInBits(), 128u);
  EXPECT_EQ(MakeVector(F80, 2)->getABISizeInBits(), 256u);
  EXPECT_EQ(MakeVector(F80, 3)->getABISizeInBits(), 512u);
  EXPECT_EQ(MakeVector(F80, 4)->getABISizeInBits(), 512u);

  const llvm::abi::Type *F80Align4 =
      TB.getFloatType(llvm::APFloat::x87DoubleExtended(), Align(4));
  EXPECT_EQ(MakeVector(F80Align4, 5)->getABISizeInBits(), 512u);
  EXPECT_EQ(MakeVector(F80Align4, 6)->getABISizeInBits(), 1024u);
}

// An integer takes whole bytes, a _BitInt and a floating-point type are padded
// out to their alignment, and a complex type is twice its element.
TEST_F(ABITypesTest, ScalarABISize) {
  EXPECT_EQ(
      TB.getIntegerType(1, Align(1), /*Signed=*/false)->getABISizeInBits(), 8u);
  EXPECT_EQ(
      TB.getIntegerType(24, Align(4), /*Signed=*/false)->getABISizeInBits(),
      24u);
  auto MakeBitInt = [&](unsigned Bits, unsigned AlignBytes) {
    return TB.getIntegerType(Bits, Align(AlignBytes), /*Signed=*/true,
                             /*IsBitInt=*/true);
  };
  EXPECT_EQ(MakeBitInt(3, 1)->getABISizeInBits(), 8u);
  EXPECT_EQ(MakeBitInt(17, 4)->getABISizeInBits(), 32u);
  EXPECT_EQ(MakeBitInt(33, 8)->getABISizeInBits(), 64u);
  EXPECT_EQ(MakeBitInt(65, 8)->getABISizeInBits(), 128u);
  EXPECT_EQ(MakeBitInt(129, 8)->getABISizeInBits(), 192u);

  const llvm::abi::Type *F80 =
      TB.getFloatType(llvm::APFloat::x87DoubleExtended(), Align(16));
  EXPECT_EQ(F80->getSizeInBits(), TypeSize::getFixed(80));
  EXPECT_EQ(F80->getABISizeInBits(), 128u);
  EXPECT_EQ(TB.getFloatType(llvm::APFloat::x87DoubleExtended(), Align(4))
                ->getABISizeInBits(),
            96u);
  EXPECT_EQ(TB.getComplexType(F80, Align(16))->getABISizeInBits(), 256u);
  const llvm::abi::Type *F32 =
      TB.getFloatType(llvm::APFloat::IEEEsingle(), Align(4));
  EXPECT_EQ(TB.getComplexType(F32, Align(4))->getABISizeInBits(), 64u);
}

// A bool element takes one bit and a one-bit _BitInt element a whole byte.  The
// total is rounded up to a power of two of at least 8 bits.
TEST_F(ABITypesTest, VectorABISize) {
  auto MakeVector = [&](const llvm::abi::Type *Elt, unsigned N,
                        unsigned AlignBytes) {
    return TB.getVectorType(Elt, ElementCount::getFixed(N), Align(AlignBytes));
  };
  const llvm::abi::Type *Bool =
      TB.getIntegerType(1, Align(1), /*Signed=*/false);
  const llvm::abi::Type *UBitInt1 =
      TB.getIntegerType(1, Align(1), /*Signed=*/false, /*IsBitInt=*/true);
  const llvm::abi::Type *F32 =
      TB.getFloatType(llvm::APFloat::IEEEsingle(), Align(4));
  EXPECT_EQ(MakeVector(Bool, 4, 1)->getABISizeInBits(), 8u);
  EXPECT_EQ(MakeVector(Bool, 8, 1)->getABISizeInBits(), 8u);
  EXPECT_EQ(MakeVector(Bool, 12, 2)->getABISizeInBits(), 16u);
  EXPECT_EQ(MakeVector(UBitInt1, 4, 4)->getABISizeInBits(), 32u);
  EXPECT_EQ(MakeVector(UBitInt1, 8, 8)->getABISizeInBits(), 64u);
  EXPECT_EQ(MakeVector(F32, 3, 16)->getABISizeInBits(), 128u);
}

// A record, an array, a pointer, a member pointer, an atomic or a void type
// takes the size it was created with, and a scalable vector or tuple has no
// fixed size.
TEST_F(ABITypesTest, OtherKindsABISize) {
  const llvm::abi::Type *I8 = TB.getIntegerType(8, Align(1), /*Signed=*/true);
  const RecordType *R = makeRecord({FieldInfo(I8, 0)}, 32, RecordFlags::None,
                                   /*Bases=*/{}, /*VBases=*/{}, Align(4));
  EXPECT_EQ(R->getABISizeInBits(), 32u);
  EXPECT_EQ(TB.getArrayType(I8, /*NumElements=*/3, /*SizeInBits=*/24)
                ->getABISizeInBits(),
            24u);
  EXPECT_EQ(TB.getPointerType(64, Align(8))->getABISizeInBits(), 64u);
  EXPECT_EQ(TB.getMemberPointerType(/*IsFunctionPointer=*/true, 128, Align(8))
                ->getABISizeInBits(),
            128u);
  EXPECT_EQ(TB.getAtomicType(R, 32, Align(4))->getABISizeInBits(), 32u);
  EXPECT_EQ(TB.getVoidType()->getABISizeInBits(), 0u);
  const VectorType *SV =
      TB.getVectorType(I8, ElementCount::getScalable(16), Align(16));
  EXPECT_EQ(SV->getABISizeInBits(), 0u);
  EXPECT_EQ(TB.getTupleType(SV, /*NumVectors=*/2)->getABISizeInBits(), 0u);
}

// svint32_t is <vscale x 4 x i32>.
TEST_F(ABITypesTest, SVEDataVector) {
  const llvm::abi::Type *I32 = TB.getIntegerType(32, Align(4), /*Signed=*/true);
  const VectorType *SVInt32 = TB.getVectorType(
      I32, ElementCount::getScalable(4), Align(16), VectorKind::SVEData);

  EXPECT_TRUE(SVInt32->isSVEData());
  EXPECT_TRUE(SVInt32->isSVEType());
  EXPECT_TRUE(SVInt32->isSVESizelessType());
  EXPECT_TRUE(SVInt32->isScalable());
  EXPECT_FALSE(SVInt32->isTuple());
  EXPECT_EQ(SVInt32->getSizeInBits(), TypeSize::getScalable(128));
  EXPECT_EQ(SVInt32->getFixedSizeInBitsOrZero(), 0u);
  EXPECT_EQ(SVInt32->getAlignment(), Align(16));
}

// svint32x3_t is three <vscale x 4 x i32> vectors.
TEST_F(ABITypesTest, SVEDataVectorTuple) {
  const llvm::abi::Type *I32 = TB.getIntegerType(32, Align(4), /*Signed=*/true);
  const VectorType *SVInt32 = TB.getVectorType(
      I32, ElementCount::getScalable(4), Align(16), VectorKind::SVEData);
  const TupleType *SVInt32x3 = TB.getTupleType(SVInt32, /*NumVectors=*/3);

  EXPECT_TRUE(SVInt32x3->isTuple());
  EXPECT_TRUE(SVInt32x3->isSVESizelessType());
  EXPECT_EQ(SVInt32x3->getNumVectors(), 3u);
  EXPECT_EQ(SVInt32x3->getVectorType(), SVInt32);
  EXPECT_EQ(SVInt32x3->getAlignment(), Align(16));
  // The contained vector keeps a per-vector element count; the tuple size
  // covers all of the vectors.
  EXPECT_EQ(SVInt32->getNumElements(), ElementCount::getScalable(4));
  EXPECT_EQ(SVInt32->getSizeInBits(), TypeSize::getScalable(128));
  EXPECT_EQ(SVInt32x3->getSizeInBits(), TypeSize::getScalable(384));
}

// svbool_t is <vscale x 16 x i1>.
TEST_F(ABITypesTest, SVEPredicateVector) {
  const llvm::abi::Type *I1 = TB.getIntegerType(1, Align(1), /*Signed=*/false);
  const VectorType *SVBool = TB.getVectorType(
      I1, ElementCount::getScalable(16), Align(2), VectorKind::SVEPredicate);

  EXPECT_TRUE(SVBool->isSVEPredicate());
  EXPECT_FALSE(SVBool->isSVEData());
  EXPECT_TRUE(SVBool->isSVESizelessType());
  EXPECT_TRUE(SVBool->isScalable());
  EXPECT_EQ(SVBool->getSizeInBits(), TypeSize::getScalable(16));
  EXPECT_EQ(SVBool->getAlignment(), Align(2));
}

TEST_F(ABITypesTest, SVECount) {
  const VectorType *SVCount =
      TB.getScalablePredicateOrCountVectorType(Align(2), VectorKind::SVECount);

  EXPECT_TRUE(SVCount->isSVECount());
  EXPECT_TRUE(SVCount->isSVEType());
  EXPECT_TRUE(SVCount->isSVESizelessType());
  EXPECT_FALSE(SVCount->isSVEPredicate());
  EXPECT_TRUE(SVCount->isScalable());
  EXPECT_FALSE(SVCount->isTuple());
  EXPECT_EQ(SVCount->getSizeInBits(), TypeSize::getScalable(16));
}

// Scalable vectors have no fixed size, so isZeroSize() must not query one.
TEST_F(ABITypesTest, ScalableVectorIsNotZeroSized) {
  const llvm::abi::Type *I32 = TB.getIntegerType(32, Align(4), /*Signed=*/true);
  const VectorType *SVInt32 = TB.getVectorType(
      I32, ElementCount::getScalable(4), Align(16), VectorKind::SVEData);

  EXPECT_FALSE(SVInt32->isZeroSize());
}

// Fixed-length SVE from arm_sve_vector_bits keeps the SVE kind but is not
// a sizeless builtin type.
TEST_F(ABITypesTest, FixedLengthSVEIsNotSizeless) {
  const llvm::abi::Type *I32 = TB.getIntegerType(32, Align(4), /*Signed=*/true);
  const VectorType *FixedInt32 = TB.getVectorType(
      I32, ElementCount::getFixed(4), Align(16), VectorKind::SVEData);
  const llvm::abi::Type *I8 = TB.getIntegerType(8, Align(1), /*Signed=*/false);
  const VectorType *FixedBool = TB.getVectorType(
      I8, ElementCount::getFixed(16), Align(2), VectorKind::SVEPredicate);

  EXPECT_TRUE(FixedInt32->isSVEData());
  EXPECT_FALSE(FixedInt32->isSVESizelessType());
  EXPECT_TRUE(FixedBool->isSVEPredicate());
  EXPECT_FALSE(FixedBool->isSVESizelessType());
}

TEST_F(ABITypesTest, NonSVETypesAreNotSizelessSVE) {
  const llvm::abi::Type *I32 = TB.getIntegerType(32, Align(4), /*Signed=*/true);
  EXPECT_FALSE(I32->isSVESizelessType());
  EXPECT_FALSE(TB.getVoidType()->isSVESizelessType());
}

} // namespace
