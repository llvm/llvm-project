//===- X86TargetInfoTest.cpp - x86 ABI unit tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

namespace {

// RecordFlags' bitmask operators are declared in namespace llvm, so combining
// two of them needs that namespace visible.
using namespace llvm;

using ABIType = llvm::abi::Type;
using llvm::abi::ABICompatInfo;
using llvm::abi::ArgInfo;
using llvm::abi::createX86_64TargetInfo;
using llvm::abi::FieldInfo;
using llvm::abi::FunctionInfo;
using llvm::abi::RecordFlags;
using llvm::abi::StructPacking;
using llvm::abi::TargetInfo;
using llvm::abi::TypeBuilder;
using llvm::abi::X86AVXABILevel;

class X86TargetInfoTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;
  const ABIType *I8;
  const ABIType *I32;
  const ABIType *F32;
  const ABIType *F64;
  const ABIType *Void;
  /// An empty class: a record with no fields, one byte wide.
  const ABIType *Empty;
  /// The same, over-aligned, so it wins the union reduction's alignment
  /// comparison.
  const ABIType *EmptyOver;

  X86TargetInfoTest()
      : TB(Alloc), I8(TB.getIntegerType(8, llvm::Align(1), /*Signed=*/true)),
        I32(TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true)),
        F32(TB.getFloatType(llvm::APFloat::IEEEsingle(), llvm::Align(4))),
        F64(TB.getFloatType(llvm::APFloat::IEEEdouble(), llvm::Align(8))),
        Void(TB.getVoidType()),
        Empty(TB.getRecordType({}, llvm::TypeSize::getFixed(8), llvm::Align(1),
                               StructPacking::Default, {}, {},
                               RecordFlags::CanPassInRegisters)),
        EmptyOver(TB.getRecordType({}, llvm::TypeSize::getFixed(128),
                                   llvm::Align(16), StructPacking::Default, {},
                                   {}, RecordFlags::CanPassInRegisters)) {}

  std::unique_ptr<TargetInfo> target() const {
    return createX86_64TargetInfo(const_cast<TypeBuilder &>(TB),
                                  X86AVXABILevel::None,
                                  /*Has64BitPointers=*/true, ABICompatInfo());
  }

  const ABIType *unionOf(llvm::ArrayRef<FieldInfo> Fields, uint64_t SizeInBits,
                         llvm::Align Alignment,
                         RecordFlags Flags = RecordFlags::None) {
    return TB.getUnionType(Fields, llvm::TypeSize::getFixed(SizeInBits),
                           Alignment, StructPacking::Default,
                           Flags | RecordFlags::CanPassInRegisters);
  }

  /// The argument classification the target computes for a single parameter.
  const ArgInfo &classifyArg(const ABIType *ArgTy,
                             std::unique_ptr<FunctionInfo> &FI,
                             std::unique_ptr<TargetInfo> &TI) {
    TI = target();
    FI = FunctionInfo::create(llvm::CallingConv::C, Void, {ArgTy});
    TI->computeInfo(*FI);
    return FI->getArgInfo(0).Info;
  }
};

static void expectDirectInteger(const ArgInfo &Info, unsigned Bits) {
  ASSERT_TRUE(Info.isDirect());
  const ABIType *Coerce = Info.getCoerceToType();
  ASSERT_NE(Coerce, nullptr);
  const auto *IT = llvm::dyn_cast<llvm::abi::IntegerType>(Coerce);
  ASSERT_NE(IT, nullptr);
  EXPECT_EQ(IT->getSizeInBits().getFixedValue(), Bits);
}

static void expectDirectFloat(const ArgInfo &Info,
                              const llvm::fltSemantics &Sem) {
  ASSERT_TRUE(Info.isDirect());
  const ABIType *Coerce = Info.getCoerceToType();
  ASSERT_NE(Coerce, nullptr);
  const auto *FT = llvm::dyn_cast<llvm::abi::FloatType>(Coerce);
  ASSERT_NE(FT, nullptr);
  EXPECT_EQ(FT->getSemantics(), &Sem);
}

// An empty member supplies no bytes, so the int is the storage the coercion is
// built from and the union coerces to its width.
TEST_F(X86TargetInfoTest, UnionWithEmptyMemberCoercesToDataMember) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *U =
      unionOf({FieldInfo(Empty), FieldInfo(I32)}, 32, llvm::Align(4));
  expectDirectInteger(classifyArg(U, FI, TI), 32);
}

// The empty member's declared alignment outranks the int's, so it wins the
// reduction unless it is skipped.  Classic passes this 16-byte union as i32.
TEST_F(X86TargetInfoTest, UnionWithOverAlignedEmptyMemberCoercesToDataMember) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *U =
      unionOf({FieldInfo(EmptyOver), FieldInfo(I32)}, 128, llvm::Align(16));
  expectDirectInteger(classifyArg(U, FI, TI), 32);
}

// The reduction also breaks alignment ties by size, so an array of empty
// records beats a one-byte member without being wider in data.
TEST_F(X86TargetInfoTest, UnionWithArrayOfEmptyMembersCoercesToDataMember) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *ArrEmpty = TB.getArrayType(Empty, /*NumElements=*/2,
                                            /*SizeInBits=*/16);
  const ABIType *U =
      unionOf({FieldInfo(ArrEmpty), FieldInfo(I8)}, 16, llvm::Align(1));
  expectDirectInteger(classifyArg(U, FI, TI), 8);
}

// The same at a full eightbyte, where the array of empty records spans the
// union and the coercion still narrows to the one byte of data.
TEST_F(X86TargetInfoTest, UnionWithEightbyteArrayOfEmptyMembersNarrows) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *ArrEmpty = TB.getArrayType(Empty, /*NumElements=*/8,
                                            /*SizeInBits=*/64);
  const ABIType *U =
      unionOf({FieldInfo(ArrEmpty), FieldInfo(I8)}, 64, llvm::Align(1));
  expectDirectInteger(classifyArg(U, FI, TI), 8);
}

// A union of nothing but empty members classifies Ignore, the same as an empty
// record does.
TEST_F(X86TargetInfoTest, UnionOfOnlyEmptyMembersIsIgnore) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *U = unionOf({FieldInfo(Empty)}, 8, llvm::Align(1));
  EXPECT_TRUE(classifyArg(U, FI, TI).isIgnore());
}

// Skipping the empty member does not force the coercion to be an integer: the
// remaining member still decides the eightbyte's class.
TEST_F(X86TargetInfoTest, UnionWithEmptyMemberKeepsSSEClass) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *U =
      unionOf({FieldInfo(Empty), FieldInfo(F64)}, 64, llvm::Align(8));
  expectDirectFloat(classifyArg(U, FI, TI), llvm::APFloat::IEEEdouble());
}

// Two floats in one eightbyte still pair into a vector with an empty member
// alongside them.
TEST_F(X86TargetInfoTest, UnionWithEmptyMemberKeepsFloatPair) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *Floats = TB.getRecordType(
      {FieldInfo(F32, 0), FieldInfo(F32, 32)}, llvm::TypeSize::getFixed(64),
      llvm::Align(4), StructPacking::Default, {}, {},
      RecordFlags::CanPassInRegisters);
  const ABIType *U =
      unionOf({FieldInfo(Empty), FieldInfo(Floats)}, 64, llvm::Align(4));
  const ArgInfo &Info = classifyArg(U, FI, TI);
  ASSERT_TRUE(Info.isDirect());
  const auto *VT =
      llvm::dyn_cast_or_null<llvm::abi::VectorType>(Info.getCoerceToType());
  ASSERT_NE(VT, nullptr);
  EXPECT_EQ(VT->getNumElements().getFixedValue(), 2u);
  const auto *ElemFT =
      llvm::dyn_cast<llvm::abi::FloatType>(VT->getElementType());
  ASSERT_NE(ElemFT, nullptr);
  EXPECT_EQ(ElemFT->getSemantics(), &llvm::APFloat::IEEEsingle());
}

// Where the data member does fill the eightbyte, narrowing must not happen:
// every byte past the first is user data, so the coercion stays i64.
TEST_F(X86TargetInfoTest, UnionWithEmptyMemberDoesNotNarrowOverData) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *Bytes = TB.getArrayType(I8, /*NumElements=*/8,
                                         /*SizeInBits=*/64);
  const ABIType *U =
      unionOf({FieldInfo(Empty), FieldInfo(Bytes)}, 64, llvm::Align(1));
  expectDirectInteger(classifyArg(U, FI, TI), 64);
}

// A transparent union is classified as its first field, and skipping empty
// members leaves that alone.  The empty-first case is decided by
// useFirstFieldIfTransparentUnion before the reduction runs, so it reaches
// Ignore rather than the reduction's storage-type choice.
TEST_F(X86TargetInfoTest, TransparentUnionTakesFirstField) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *DataFirst =
      unionOf({FieldInfo(I32), FieldInfo(F32)}, 32, llvm::Align(4),
              RecordFlags::IsTransparent);
  expectDirectInteger(classifyArg(DataFirst, FI, TI), 32);

  const ABIType *EmptyFirst =
      unionOf({FieldInfo(Empty), FieldInfo(I8)}, 8, llvm::Align(1),
              RecordFlags::IsTransparent);
  EXPECT_TRUE(classifyArg(EmptyFirst, FI, TI).isIgnore());
}

// An unnamed zero-width bit-field is skipped as it was before, so a union of
// nothing else still has no storage type to reduce to.
TEST_F(X86TargetInfoTest, UnionOfZeroWidthBitFieldIsIgnore) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  FieldInfo ZeroWidth(I32, 0, /*IsBitField=*/true, /*BitFieldWidth=*/0,
                      /*IsUnnamedBitField=*/true);
  const ABIType *U = unionOf({ZeroWidth}, 8, llvm::Align(1));
  EXPECT_TRUE(classifyArg(U, FI, TI).isIgnore());
}

} // namespace
