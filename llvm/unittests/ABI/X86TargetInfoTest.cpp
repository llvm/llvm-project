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
using llvm::abi::ArgInfo;
using llvm::abi::createX86_64TargetInfo;
using llvm::abi::FieldInfo;
using llvm::abi::FunctionInfo;
using llvm::abi::RecordFlags;
using llvm::abi::RequiredArgs;
using llvm::abi::StructPacking;
using llvm::abi::TargetInfo;
using llvm::abi::TypeBuilder;
using llvm::abi::X86ABICompatInfo;
using llvm::abi::X86AVXABILevel;

class X86TargetInfoTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;
  const ABIType *I8;
  const ABIType *I32;
  const ABIType *I64;
  const ABIType *F32;
  const ABIType *F64;
  /// An x87 long double: 80 bits of value in 16 bytes of storage.
  const ABIType *F80;
  /// A bool: one bit of value in a byte of storage.
  const ABIType *Bool;
  const ABIType *Void;
  /// An empty class: a record with no fields, one byte wide.
  const ABIType *Empty;
  /// The same, over-aligned, so it wins the union reduction's alignment
  /// comparison.
  const ABIType *EmptyOver;

  X86TargetInfoTest()
      : TB(Alloc), I8(TB.getIntegerType(8, llvm::Align(1), /*Signed=*/true)),
        I32(TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true)),
        I64(TB.getIntegerType(64, llvm::Align(8), /*Signed=*/true)),
        F32(TB.getFloatType(llvm::APFloat::IEEEsingle(), llvm::Align(4))),
        F64(TB.getFloatType(llvm::APFloat::IEEEdouble(), llvm::Align(8))),
        F80(TB.getFloatType(llvm::APFloat::x87DoubleExtended(),
                            llvm::Align(16))),
        Bool(TB.getIntegerType(1, llvm::Align(1), /*Signed=*/false)),
        Void(TB.getVoidType()),
        Empty(TB.getRecordType({}, llvm::TypeSize::getFixed(8), llvm::Align(1),
                               /*UnadjustedAlign=*/llvm::Align(1),
                               StructPacking::Default, {}, {},
                               RecordFlags::CanPassInRegisters)),
        EmptyOver(TB.getRecordType(
            {}, llvm::TypeSize::getFixed(128), llvm::Align(16),
            /*UnadjustedAlign=*/llvm::Align(16), StructPacking::Default, {}, {},
            RecordFlags::CanPassInRegisters)) {}

  std::unique_ptr<TargetInfo>
  target(X86AVXABILevel AVXLevel = X86AVXABILevel::None) const {
    return createX86_64TargetInfo(const_cast<TypeBuilder &>(TB), AVXLevel,
                                  /*Has64BitPointers=*/true,
                                  X86ABICompatInfo());
  }

  const ABIType *makeVector(const ABIType *Elt, unsigned NumElements,
                            llvm::Align Alignment) {
    return TB.getVectorType(Elt, llvm::ElementCount::getFixed(NumElements),
                            Alignment);
  }

  /// A register-passable record with the given fields, size and alignment.
  const ABIType *makeRecord(llvm::ArrayRef<FieldInfo> Fields,
                            uint64_t SizeInBits, llvm::Align Alignment) {
    return TB.getRecordType(Fields, llvm::TypeSize::getFixed(SizeInBits),
                            Alignment, Alignment, StructPacking::Default, {},
                            {}, RecordFlags::CanPassInRegisters);
  }

  const ABIType *unionOf(llvm::ArrayRef<FieldInfo> Fields, uint64_t SizeInBits,
                         llvm::Align Alignment,
                         RecordFlags Flags = RecordFlags::None) {
    return TB.getUnionType(Fields, llvm::TypeSize::getFixed(SizeInBits),
                           Alignment, Alignment, StructPacking::Default,
                           Flags | RecordFlags::CanPassInRegisters);
  }

  /// The argument classification the target computes for a single parameter.
  const ArgInfo &classifyArg(const ABIType *ArgTy,
                             std::unique_ptr<FunctionInfo> &FI,
                             std::unique_ptr<TargetInfo> &TI,
                             X86AVXABILevel AVXLevel = X86AVXABILevel::None) {
    TI = target(AVXLevel);
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

static void expectInteger(const ABIType *Ty, unsigned Bits) {
  const auto *IT = llvm::dyn_cast<llvm::abi::IntegerType>(Ty);
  ASSERT_NE(IT, nullptr);
  EXPECT_EQ(IT->getSizeInBits().getFixedValue(), Bits);
}

/// The {low, high} halves of a two-eightbyte coercion.
static llvm::ArrayRef<FieldInfo> directPair(const ArgInfo &Info) {
  EXPECT_TRUE(Info.isDirect());
  const auto *RT =
      llvm::dyn_cast_or_null<llvm::abi::RecordType>(Info.getCoerceToType());
  EXPECT_NE(RT, nullptr);
  if (!RT)
    return {};
  return RT->getFields();
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

// A scalar atomic has its underlying type's scalar evaluation kind. Although
// the SysV classifier assigns it Memory, it remains a direct LLVM argument.
TEST_F(X86TargetInfoTest, AtomicFloatScalarIsDirect) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *AtomicF32 = TB.getAtomicType(F32, 32, llvm::Align(4));
  const ArgInfo &Info = classifyArg(AtomicF32, FI, TI);

  EXPECT_TRUE(Info.isDirect());
  EXPECT_EQ(Info.getCoerceToType(), nullptr);
}

// An atomic whose value type has aggregate evaluation kind remains aggregate
// for the indirect-result decision.
TEST_F(X86TargetInfoTest, AtomicRecordIsIndirect) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *Value = TB.getRecordType(
      {FieldInfo(I32, 0)}, llvm::TypeSize::getFixed(32), llvm::Align(4),
      /*UnadjustedAlign=*/llvm::Align(4), StructPacking::Default, {}, {},
      RecordFlags::CanPassInRegisters);
  const ABIType *Atomic = TB.getAtomicType(Value, 32, llvm::Align(4));
  const ArgInfo &Info = classifyArg(Atomic, FI, TI);

  ASSERT_TRUE(Info.isIndirect());
  EXPECT_TRUE(Info.getIndirectByVal());
  EXPECT_EQ(Info.getIndirectAlign(), llvm::Align(8));
  EXPECT_EQ(Info.getIndirectAddrSpace(), 0u);
}

// Atomic fields classify Memory rather than inheriting the underlying float's
// SSE class, forcing the containing record to the stack.
TEST_F(X86TargetInfoTest, RecordOfAtomicFloatsIsIndirect) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *AtomicF32 = TB.getAtomicType(F32, 32, llvm::Align(4));
  const ABIType *Record = TB.getRecordType(
      {FieldInfo(AtomicF32, 0), FieldInfo(AtomicF32, 32)},
      llvm::TypeSize::getFixed(64), llvm::Align(4),
      /*UnadjustedAlign=*/llvm::Align(4), StructPacking::Default, {}, {},
      RecordFlags::CanPassInRegisters);
  const ArgInfo &Info = classifyArg(Record, FI, TI);

  ASSERT_TRUE(Info.isIndirect());
  EXPECT_TRUE(Info.getIndirectByVal());
  EXPECT_EQ(Info.getIndirectAlign(), llvm::Align(8));
  EXPECT_EQ(Info.getIndirectAddrSpace(), 0u);
}

// Without the atomic wrappers, the same record is passed in an SSE register.
TEST_F(X86TargetInfoTest, RecordOfFloatsIsDirectSSE) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *Record = TB.getRecordType(
      {FieldInfo(F32, 0), FieldInfo(F32, 32)}, llvm::TypeSize::getFixed(64),
      llvm::Align(4), /*UnadjustedAlign=*/llvm::Align(4),
      StructPacking::Default, {}, {}, RecordFlags::CanPassInRegisters);
  const ArgInfo &Info = classifyArg(Record, FI, TI);

  ASSERT_TRUE(Info.isDirect());
  const auto *Vector =
      llvm::dyn_cast_or_null<llvm::abi::VectorType>(Info.getCoerceToType());
  ASSERT_NE(Vector, nullptr);
  EXPECT_EQ(Vector->getNumElements().getFixedValue(), 2u);
  EXPECT_EQ(Vector->getElementType(), F32);
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
      llvm::Align(4), llvm::Align(4), StructPacking::Default, {}, {},
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

// No member reaches the union's second eightbyte: the array covers 12 of the
// 16 bytes and the pointer supplies the alignment that rounds the size up.
// The high half is sized from the bytes the union has there, so the four bytes
// of array plus the four of padding make it an i64.
TEST_F(X86TargetInfoTest, UnionTailPaddingSizesHighHalfFromUnion) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *U32 = TB.getIntegerType(32, llvm::Align(4), /*Signed=*/false);
  const ABIType *Words = TB.getArrayType(U32, /*NumElements=*/3,
                                         /*SizeInBits=*/96);
  const ABIType *Ptr = TB.getPointerType(64, llvm::Align(8));
  const ABIType *U =
      unionOf({FieldInfo(Words), FieldInfo(Ptr)}, 128, llvm::Align(8));
  llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(U, FI, TI));
  ASSERT_EQ(Pair.size(), 2u);
  EXPECT_TRUE(Pair[0].FieldType->isPointer());
  expectInteger(Pair[1].FieldType, 64);
}

// One byte of the second eightbyte holds data and the rest is padding, so the
// high half narrows to that byte instead of spanning the union's tail.
TEST_F(X86TargetInfoTest, UnionTailPaddingNarrowsHighHalfToByte) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *Bytes = TB.getArrayType(I8, /*NumElements=*/9,
                                         /*SizeInBits=*/72);
  const ABIType *Ptr = TB.getPointerType(64, llvm::Align(8));
  const ABIType *U =
      unionOf({FieldInfo(Bytes), FieldInfo(Ptr)}, 128, llvm::Align(8));
  llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(U, FI, TI));
  ASSERT_EQ(Pair.size(), 2u);
  EXPECT_TRUE(Pair[0].FieldType->isPointer());
  expectInteger(Pair[1].FieldType, 8);
}

// One more byte of data is enough to stop the narrowing, so the high half
// covers the union's remaining bytes rather than the two that hold data.
TEST_F(X86TargetInfoTest, UnionTailPaddingKeepsHighHalfPastOneByte) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *Bytes = TB.getArrayType(I8, /*NumElements=*/10,
                                         /*SizeInBits=*/80);
  const ABIType *Ptr = TB.getPointerType(64, llvm::Align(8));
  const ABIType *U =
      unionOf({FieldInfo(Bytes), FieldInfo(Ptr)}, 128, llvm::Align(8));
  llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(U, FI, TI));
  ASSERT_EQ(Pair.size(), 2u);
  EXPECT_TRUE(Pair[0].FieldType->isPointer());
  expectInteger(Pair[1].FieldType, 64);
}

// A union that stops short of two full eightbytes sizes the high half from
// what it has left rather than from a whole eightbyte.
TEST_F(X86TargetInfoTest, UnionTailPaddingClampsHighHalfToUnionSize) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *Bytes = TB.getArrayType(I8, /*NumElements=*/12,
                                         /*SizeInBits=*/96);
  const ABIType *U =
      unionOf({FieldInfo(Bytes), FieldInfo(I32)}, 96, llvm::Align(4));
  llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(U, FI, TI));
  ASSERT_EQ(Pair.size(), 2u);
  expectInteger(Pair[0].FieldType, 64);
  expectInteger(Pair[1].FieldType, 32);
}

// Narrowing still applies inside such a union, where only one byte past the
// first eightbyte holds data.
TEST_F(X86TargetInfoTest, UnionTailPaddingNarrowsInsideShortUnion) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *Bytes = TB.getArrayType(I8, /*NumElements=*/9,
                                         /*SizeInBits=*/72);
  const ABIType *U =
      unionOf({FieldInfo(Bytes), FieldInfo(I32)}, 96, llvm::Align(4));
  llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(U, FI, TI));
  ASSERT_EQ(Pair.size(), 2u);
  expectInteger(Pair[0].FieldType, 64);
  expectInteger(Pair[1].FieldType, 8);
}

// A one-element x87 vector takes 16 bytes, so a struct or union wrapping it,
// directly or in a one-element array, is passed as the vector.
TEST_F(X86TargetInfoTest, X87VectorWrapperCoercesToVector) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *V1F80 = makeVector(F80, 1, llvm::Align(16));
  const ABIType *Arr =
      TB.getArrayType(V1F80, /*NumElements=*/1, /*SizeInBits=*/128);
  const ABIType *Wrappers[] = {
      makeRecord({FieldInfo(V1F80, 0)}, 128, llvm::Align(16)),
      makeRecord({FieldInfo(Arr, 0)}, 128, llvm::Align(16)),
      unionOf({FieldInfo(V1F80)}, 128, llvm::Align(16))};
  for (const ABIType *Wrapper : Wrappers) {
    const ArgInfo &Info = classifyArg(Wrapper, FI, TI);
    ASSERT_TRUE(Info.isDirect());
    EXPECT_EQ(Info.getCoerceToType(), V1F80);
  }
}

// Two x87 elements fill 256 bits, so with AVX a struct or union wrapping the
// vector, directly or in a one-element array, is passed directly as the vector.
TEST_F(X86TargetInfoTest, X87VectorWrapperIsDirectWithAVX) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *V2F80 = makeVector(F80, 2, llvm::Align(32));
  const ABIType *Arr =
      TB.getArrayType(V2F80, /*NumElements=*/1, /*SizeInBits=*/256);
  const ABIType *Wrappers[] = {
      makeRecord({FieldInfo(V2F80, 0)}, 256, llvm::Align(32)),
      makeRecord({FieldInfo(Arr, 0)}, 256, llvm::Align(32)),
      unionOf({FieldInfo(V2F80)}, 256, llvm::Align(32))};
  for (const ABIType *Wrapper : Wrappers) {
    const ArgInfo &Info = classifyArg(Wrapper, FI, TI, X86AVXABILevel::AVX);
    ASSERT_TRUE(Info.isDirect());
    EXPECT_EQ(Info.getCoerceToType(), V2F80);
  }
}

// Three x87 elements round up to 512 bits, wider than 256 bits, so the
// vector is passed in memory with AVX and directly with AVX-512.
TEST_F(X86TargetInfoTest, ThreeElementX87VectorNeedsAVX512) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *V3F80 = makeVector(F80, 3, llvm::Align(64));

  {
    const ArgInfo &Info = classifyArg(V3F80, FI, TI, X86AVXABILevel::AVX);
    ASSERT_TRUE(Info.isIndirect());
    EXPECT_TRUE(Info.getIndirectByVal());
    EXPECT_EQ(Info.getIndirectAlign(), llvm::Align(64));
  }
  const ArgInfo &Info = classifyArg(V3F80, FI, TI, X86AVXABILevel::AVX512);
  ASSERT_TRUE(Info.isDirect());
  EXPECT_EQ(Info.getCoerceToType(), V3F80);
}

// Three 128-bit _BitInt elements round up to 512 bits, so with AVX-512 a struct
// wrapping the vector is passed as an <8 x i64> vector.
TEST_F(X86TargetInfoTest, BitInt128VectorWrapperIsI64VectorWithAVX512) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *I128 = TB.getIntegerType(128, llvm::Align(16), /*Signed=*/true,
                                          /*IsBitInt=*/true);
  const ABIType *V3I128 = makeVector(I128, 3, llvm::Align(64));
  const ABIType *S = makeRecord({FieldInfo(V3I128, 0)}, 512, llvm::Align(64));
  const ArgInfo &Info = classifyArg(S, FI, TI, X86AVXABILevel::AVX512);
  ASSERT_TRUE(Info.isDirect());
  const auto *VT =
      llvm::dyn_cast_or_null<llvm::abi::VectorType>(Info.getCoerceToType());
  ASSERT_NE(VT, nullptr);
  EXPECT_EQ(VT->getNumElements().getFixedValue(), 8u);
  expectInteger(VT->getElementType(), 64);
}

// A one-element x87 vector takes 128 bits, so even as an unnamed argument it
// and a struct holding it count against one SSE register.
TEST_F(X86TargetInfoTest, VariadicX87VectorUsesSSERegister) {
  const ABIType *V1F80 = makeVector(F80, 1, llvm::Align(16));
  const ABIType *S = makeRecord({FieldInfo(V1F80, 0)}, 128, llvm::Align(16));
  for (const ABIType *ArgTy : {V1F80, S}) {
    std::unique_ptr<TargetInfo> TI = target();
    std::unique_ptr<FunctionInfo> FI = FunctionInfo::create(
        llvm::CallingConv::C, Void, {ArgTy}, RequiredArgs(0));
    TI->computeInfo(*FI);
    const ArgInfo &Info = FI->getArgInfo(0).Info;
    ASSERT_TRUE(Info.isDirect());
    EXPECT_EQ(Info.getNeededSseRegs(), 1u);
  }
}

// The x87 vector covers all 16 bytes of the union, so the struct's padding
// after the int counts as data and the high eightbyte is a whole i64.
TEST_F(X86TargetInfoTest, X87VectorPaddingIsDataInUnion) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *V1F80 = makeVector(F80, 1, llvm::Align(16));
  const ABIType *LongAndInt =
      makeRecord({FieldInfo(I64, 0), FieldInfo(I32, 64)}, 128, llvm::Align(16));
  const ABIType *U =
      unionOf({FieldInfo(LongAndInt), FieldInfo(V1F80)}, 128, llvm::Align(16));
  llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(U, FI, TI));
  ASSERT_EQ(Pair.size(), 2u);
  expectInteger(Pair[0].FieldType, 64);
  expectInteger(Pair[1].FieldType, 64);
}

// Three chars take 32 bits, so the vector and a struct holding it are passed
// as an i32. Three shorts take 64 bits and are passed as a double.
TEST_F(X86TargetInfoTest, ThreeElementVectorsTakeTheirABISize) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *I16 = TB.getIntegerType(16, llvm::Align(2), /*Signed=*/true);
  const ABIType *V3I8 = makeVector(I8, 3, llvm::Align(4));

  expectDirectInteger(classifyArg(V3I8, FI, TI), 32);
  expectDirectInteger(
      classifyArg(makeRecord({FieldInfo(V3I8, 0)}, 32, llvm::Align(4)), FI, TI),
      32);
  expectDirectFloat(classifyArg(makeVector(I16, 3, llvm::Align(8)), FI, TI),
                    llvm::APFloat::IEEEdouble());
}

// Each bool takes one bit, so four bools round up to an i8 and seventeen to
// an i32. A struct holding the four is passed as an i8 as well.
TEST_F(X86TargetInfoTest, BoolVectorIsPassedAsInteger) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *I1 = TB.getIntegerType(1, llvm::Align(1), /*Signed=*/false);
  const ABIType *V4I1 = makeVector(I1, 4, llvm::Align(1));
  const ABIType *V17I1 = makeVector(I1, 17, llvm::Align(4));

  expectDirectInteger(classifyArg(V4I1, FI, TI), 8);
  expectDirectInteger(
      classifyArg(makeRecord({FieldInfo(V4I1, 0)}, 8, llvm::Align(1)), FI, TI),
      8);
  expectDirectInteger(classifyArg(V17I1, FI, TI), 32);
}

// Each two-element vector of 4-bit _BitInt takes 16 bits, so the fifth sits in
// the high eightbyte and the struct is passed as an i64 and an i16.
TEST_F(X86TargetInfoTest, VectorArrayStepsByVectorSize) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *I4 = TB.getIntegerType(4, llvm::Align(1), /*Signed=*/true,
                                        /*IsBitInt=*/true);
  const ABIType *V2I4 = makeVector(I4, 2, llvm::Align(2));
  const ABIType *Arr =
      TB.getArrayType(V2I4, /*NumElements=*/5, /*SizeInBits=*/80);
  const ABIType *S = makeRecord({FieldInfo(Arr, 0)}, 80, llvm::Align(2));
  llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(S, FI, TI));
  ASSERT_EQ(Pair.size(), 2u);
  expectInteger(Pair[0].FieldType, 64);
  expectInteger(Pair[1].FieldType, 16);
}

// Padding past a three-float vector makes the 256-bit struct larger than the
// vector, so even with AVX it is passed in memory.
TEST_F(X86TargetInfoTest, OverAlignedVectorWrapperIsIndirectWithAVX) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *V3F32 = makeVector(F32, 3, llvm::Align(16));
  const ABIType *S = makeRecord({FieldInfo(V3F32, 0)}, 256, llvm::Align(32));
  EXPECT_TRUE(classifyArg(S, FI, TI, X86AVXABILevel::AVX).isIndirect());
}

// The third three-char vector of the array starts at bit 64 and reaches past
// the short, so the high eightbyte of the union is a whole i64.
TEST_F(X86TargetInfoTest, VectorArrayStepsByVectorSizeInUnion) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *I16 = TB.getIntegerType(16, llvm::Align(2), /*Signed=*/true);
  const ABIType *V3I8 = makeVector(I8, 3, llvm::Align(4));
  const ABIType *Arr =
      TB.getArrayType(V3I8, /*NumElements=*/3, /*SizeInBits=*/96);
  const ABIType *LongAndShort =
      makeRecord({FieldInfo(I64, 0), FieldInfo(I16, 64)}, 128, llvm::Align(8));
  const ABIType *U =
      unionOf({FieldInfo(LongAndShort), FieldInfo(Arr)}, 128, llvm::Align(8));
  llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(U, FI, TI));
  ASSERT_EQ(Pair.size(), 2u);
  expectInteger(Pair[0].FieldType, 64);
  expectInteger(Pair[1].FieldType, 64);
}

// Sixteen 4-bit _BitInt elements take 128 bits, so once the SSE registers run
// out the vector is still passed directly rather than in memory.
TEST_F(X86TargetInfoTest, NibbleVectorIsDirectAfterSSERegistersRunOut) {
  const ABIType *I4 = TB.getIntegerType(4, llvm::Align(1), /*Signed=*/true,
                                        /*IsBitInt=*/true);
  const ABIType *V16I4 = makeVector(I4, 16, llvm::Align(16));
  std::unique_ptr<TargetInfo> TI = target();
  std::unique_ptr<FunctionInfo> FI =
      FunctionInfo::create(llvm::CallingConv::C, Void,
                           {F64, F64, F64, F64, F64, F64, F64, F64, V16I4});
  TI->computeInfo(*FI);
  const ArgInfo &Info = FI->getArgInfo(8).Info;
  ASSERT_TRUE(Info.isDirect());
  EXPECT_EQ(Info.getCoerceToType(), nullptr);
}

// Once the SSE registers run out, a vector of _BitInt(128) is still passed
// directly, while a vector of __int128 is passed in memory.
TEST_F(X86TargetInfoTest, Int128VectorOnlyIsIndirectAfterSSERegistersRunOut) {
  const ABIType *BitInt128 = TB.getIntegerType(
      128, llvm::Align(16), /*Signed=*/true, /*IsBitInt=*/true);
  const ABIType *Int128 =
      TB.getIntegerType(128, llvm::Align(16), /*Signed=*/true);
  for (const ABIType *Elt : {BitInt128, Int128}) {
    const ABIType *V = makeVector(Elt, 1, llvm::Align(16));
    std::unique_ptr<TargetInfo> TI = target();
    std::unique_ptr<FunctionInfo> FI =
        FunctionInfo::create(llvm::CallingConv::C, Void,
                             {F64, F64, F64, F64, F64, F64, F64, F64, V});
    TI->computeInfo(*FI);
    const ArgInfo &Info = FI->getArgInfo(8).Info;
    if (Elt == BitInt128) {
      EXPECT_TRUE(Info.isDirect());
    } else {
      ASSERT_TRUE(Info.isIndirect());
      EXPECT_TRUE(Info.getIndirectByVal());
    }
  }
}

// Each _BitInt(17) takes 32 bits, so in each struct the last two elements fill
// the high eightbyte.
TEST_F(X86TargetInfoTest, BitIntArrayStepsByStorageSize) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *BitInt17 = TB.getIntegerType(17, llvm::Align(4),
                                              /*Signed=*/true,
                                              /*IsBitInt=*/true);
  const ABIType *Arr4 =
      TB.getArrayType(BitInt17, /*NumElements=*/4, /*SizeInBits=*/128);
  const ABIType *Arr3 =
      TB.getArrayType(BitInt17, /*NumElements=*/3, /*SizeInBits=*/96);
  const ABIType *Records[] = {
      makeRecord({FieldInfo(Arr4, 0)}, 128, llvm::Align(4)),
      makeRecord({FieldInfo(I32, 0), FieldInfo(Arr3, 32)}, 128,
                 llvm::Align(4))};
  for (const ABIType *S : Records) {
    SCOPED_TRACE(S == Records[0] ? "_BitInt(17) a[4]"
                                 : "int x; _BitInt(17) a[3]");
    llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(S, FI, TI));
    ASSERT_EQ(Pair.size(), 2u);
    expectInteger(Pair[0].FieldType, 64);
    expectInteger(Pair[1].FieldType, 64);
  }
}

// _BitInt elements are padded out to their alignment and bools take a byte, so
// these arrays reach the high eightbyte.
TEST_F(X86TargetInfoTest, NarrowElementArraysStepByStorageSize) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *BitInt3 = TB.getIntegerType(3, llvm::Align(1), /*Signed=*/true,
                                             /*IsBitInt=*/true);
  const ABIType *BitInt9 = TB.getIntegerType(9, llvm::Align(2), /*Signed=*/true,
                                             /*IsBitInt=*/true);
  const ABIType *BitInt33 = TB.getIntegerType(33, llvm::Align(8),
                                              /*Signed=*/true,
                                              /*IsBitInt=*/true);
  const ABIType *BitInt3x16 =
      TB.getArrayType(BitInt3, /*NumElements=*/16, /*SizeInBits=*/128);
  const ABIType *BitInt9x5 =
      TB.getArrayType(BitInt9, /*NumElements=*/5, /*SizeInBits=*/80);
  const ABIType *BitInt33x2 =
      TB.getArrayType(BitInt33, /*NumElements=*/2, /*SizeInBits=*/128);
  const ABIType *Boolx16 =
      TB.getArrayType(Bool, /*NumElements=*/16, /*SizeInBits=*/128);
  const ABIType *Boolx9 =
      TB.getArrayType(Bool, /*NumElements=*/9, /*SizeInBits=*/72);
  struct {
    const char *Decl;
    const ABIType *S;
    unsigned HighBits;
  } Cases[] = {
      {"_BitInt(3) a[16]",
       makeRecord({FieldInfo(BitInt3x16, 0)}, 128, llvm::Align(1)), 64},
      {"_BitInt(9) a[5]",
       makeRecord({FieldInfo(BitInt9x5, 0)}, 80, llvm::Align(2)), 16},
      {"_BitInt(33) a[2]",
       makeRecord({FieldInfo(BitInt33x2, 0)}, 128, llvm::Align(8)), 64},
      {"_Bool b[16]", makeRecord({FieldInfo(Boolx16, 0)}, 128, llvm::Align(1)),
       64},
      {"_Bool b[9]", makeRecord({FieldInfo(Boolx9, 0)}, 72, llvm::Align(1)),
       8}};
  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Decl);
    llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(Case.S, FI, TI));
    ASSERT_EQ(Pair.size(), 2u);
    expectInteger(Pair[0].FieldType, 64);
    expectInteger(Pair[1].FieldType, Case.HighBits);
  }
}

// The third element of each array starts at bit 16, so the union's data reaches
// past the short and the union is passed as an i32.
TEST_F(X86TargetInfoTest, NarrowArrayInUnionReachesPastShort) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *I16 = TB.getIntegerType(16, llvm::Align(2), /*Signed=*/true);
  const ABIType *UBitInt3 = TB.getIntegerType(3, llvm::Align(1),
                                              /*Signed=*/false,
                                              /*IsBitInt=*/true);
  for (const ABIType *Elt : {UBitInt3, Bool}) {
    SCOPED_TRACE(Elt == Bool ? "_Bool a[3]" : "unsigned _BitInt(3) a[3]");
    const ABIType *Arr =
        TB.getArrayType(Elt, /*NumElements=*/3, /*SizeInBits=*/24);
    const ABIType *U =
        unionOf({FieldInfo(Arr), FieldInfo(I16)}, 32, llvm::Align(2));
    expectDirectInteger(classifyArg(U, FI, TI), 32);
  }
}

// The one-byte element after the long is followed only by padding, so the high
// eightbyte narrows to an i8.
TEST_F(X86TargetInfoTest, OneElementNarrowArrayNarrowsHighHalf) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *UBitInt3 = TB.getIntegerType(3, llvm::Align(1),
                                              /*Signed=*/false,
                                              /*IsBitInt=*/true);
  const ABIType *Arr =
      TB.getArrayType(UBitInt3, /*NumElements=*/1, /*SizeInBits=*/8);
  const ABIType *S =
      makeRecord({FieldInfo(I64, 0), FieldInfo(Arr, 64)}, 128, llvm::Align(8));
  llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(S, FI, TI));
  ASSERT_EQ(Pair.size(), 2u);
  expectInteger(Pair[0].FieldType, 64);
  expectInteger(Pair[1].FieldType, 8);
}

// The ninth element and the float share the high eightbyte, which is therefore
// an integer.
TEST_F(X86TargetInfoTest, NarrowArrayBeforeFloatIsInteger) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *UBitInt3 = TB.getIntegerType(3, llvm::Align(1),
                                              /*Signed=*/false,
                                              /*IsBitInt=*/true);
  const ABIType *Arr =
      TB.getArrayType(UBitInt3, /*NumElements=*/9, /*SizeInBits=*/72);
  const ABIType *S =
      makeRecord({FieldInfo(Arr, 0), FieldInfo(F32, 96)}, 128, llvm::Align(4));
  llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(S, FI, TI));
  ASSERT_EQ(Pair.size(), 2u);
  expectInteger(Pair[0].FieldType, 64);
  expectInteger(Pair[1].FieldType, 64);
}

// A bool or narrow _BitInt bit-field counts at the size of its type, a whole
// byte, so from bit 3 it reaches the second byte of the struct.
TEST_F(X86TargetInfoTest, NarrowBitFieldCountsItsTypeSize) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *U8 = TB.getIntegerType(8, llvm::Align(1), /*Signed=*/false);
  const ABIType *BitInt5 = TB.getIntegerType(5, llvm::Align(1), /*Signed=*/true,
                                             /*IsBitInt=*/true);
  for (const ABIType *Second : {Bool, BitInt5}) {
    SCOPED_TRACE(Second == Bool ? "_Bool b : 1" : "_BitInt(5) y : 1");
    const ABIType *S = makeRecord(
        {FieldInfo(U8, 0, /*IsBitField=*/true, /*BitFieldWidth=*/3),
         FieldInfo(Second, 3, /*IsBitField=*/true, /*BitFieldWidth=*/1)},
        16, llvm::Align(2));
    expectDirectInteger(classifyArg(S, FI, TI), 16);
  }
}

// The long double covers all 16 bytes of the union, so in either member order
// the high eightbyte is a whole i64.
TEST_F(X86TargetInfoTest, X87PaddingIsDataInUnion) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *I16 = TB.getIntegerType(16, llvm::Align(2), /*Signed=*/true);
  const ABIType *LongShort =
      makeRecord({FieldInfo(I64, 0), FieldInfo(I16, 64)}, 128, llvm::Align(16));
  const ABIType *Unions[] = {
      unionOf({FieldInfo(F80), FieldInfo(LongShort)}, 128, llvm::Align(16)),
      unionOf({FieldInfo(LongShort), FieldInfo(F80)}, 128, llvm::Align(16))};
  for (const ABIType *U : Unions) {
    SCOPED_TRACE(U == Unions[0] ? "long double first" : "struct first");
    llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(U, FI, TI));
    ASSERT_EQ(Pair.size(), 2u);
    expectInteger(Pair[0].FieldType, 64);
    expectInteger(Pair[1].FieldType, 64);
  }
}

// A bool vector is stored as an integer with one bit per element, at least a
// byte wide, so the eightbyte holding it narrows to that integer when it is an
// i8, i16 or i32 and the rest of the eightbyte is padding.  Twelve bools make
// an i12, so that eightbyte stays an i64.
TEST_F(X86TargetInfoTest, BoolVectorInMemoryIsItsLaneInteger) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  struct {
    unsigned Lanes;
    llvm::Align Alignment;
    unsigned HighBits;
  } Cases[] = {{3, llvm::Align(1), 8},   {4, llvm::Align(1), 8},
               {8, llvm::Align(1), 8},   {12, llvm::Align(2), 64},
               {16, llvm::Align(2), 16}, {32, llvm::Align(4), 32}};
  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Lanes);
    const ABIType *V = makeVector(Bool, Case.Lanes, Case.Alignment);
    const ABIType *S =
        makeRecord({FieldInfo(I64, 0), FieldInfo(V, 64)}, 128, llvm::Align(8));
    llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(S, FI, TI));
    ASSERT_EQ(Pair.size(), 2u);
    expectInteger(Pair[0].FieldType, 64);
    expectInteger(Pair[1].FieldType, Case.HighBits);
  }
}

// A bool vector narrows its eightbyte to its integer in a one-element array at
// the start of a struct, in a nested struct, in a union, or alone in an
// over-aligned struct.
TEST_F(X86TargetInfoTest, BoolVectorInMemoryNarrowsInEveryPosition) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *V4 = makeVector(Bool, 4, llvm::Align(1));
  const ABIType *V16 = makeVector(Bool, 16, llvm::Align(2));
  {
    SCOPED_TRACE("one-element array first");
    const ABIType *Arr =
        TB.getArrayType(V4, /*NumElements=*/1, /*SizeInBits=*/8);
    const ABIType *ArrayFirst = makeRecord(
        {FieldInfo(Arr, 0), FieldInfo(I64, 64)}, 128, llvm::Align(8));
    llvm::ArrayRef<FieldInfo> Pair =
        directPair(classifyArg(ArrayFirst, FI, TI));
    ASSERT_EQ(Pair.size(), 2u);
    expectInteger(Pair[0].FieldType, 8);
    expectInteger(Pair[1].FieldType, 64);
  }
  {
    SCOPED_TRACE("nested struct");
    const ABIType *Inner = makeRecord({FieldInfo(V4, 0)}, 8, llvm::Align(1));
    const ABIType *Nested = makeRecord(
        {FieldInfo(I64, 0), FieldInfo(Inner, 64)}, 128, llvm::Align(8));
    llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(Nested, FI, TI));
    ASSERT_EQ(Pair.size(), 2u);
    expectInteger(Pair[0].FieldType, 64);
    expectInteger(Pair[1].FieldType, 8);
  }
  {
    SCOPED_TRACE("union aligned 4");
    expectDirectInteger(
        classifyArg(unionOf({FieldInfo(V4)}, 32, llvm::Align(4)), FI, TI), 8);
  }
  {
    SCOPED_TRACE("struct aligned 4");
    expectDirectInteger(
        classifyArg(makeRecord({FieldInfo(V4, 0)}, 32, llvm::Align(4)), FI, TI),
        8);
  }
  {
    SCOPED_TRACE("16 bools, struct aligned 8");
    expectDirectInteger(
        classifyArg(makeRecord({FieldInfo(V16, 0)}, 64, llvm::Align(8)), FI,
                    TI),
        16);
  }
}

// A vector of one-bit _BitInts is not a bool vector, so the eightbyte holding
// it stays an i64.
TEST_F(X86TargetInfoTest, BitIntVectorInMemoryIsNotNarrowed) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *UBitInt1 = TB.getIntegerType(1, llvm::Align(1),
                                              /*Signed=*/false,
                                              /*IsBitInt=*/true);
  const ABIType *V = makeVector(UBitInt1, 1, llvm::Align(1));
  const ABIType *S =
      makeRecord({FieldInfo(I64, 0), FieldInfo(V, 64)}, 128, llvm::Align(8));
  llvm::ArrayRef<FieldInfo> Pair = directPair(classifyArg(S, FI, TI));
  ASSERT_EQ(Pair.size(), 2u);
  expectInteger(Pair[0].FieldType, 64);
  expectInteger(Pair[1].FieldType, 64);
}

} // namespace
