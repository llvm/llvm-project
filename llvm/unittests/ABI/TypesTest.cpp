//===- TypesTest.cpp - ABI type emptiness unit tests ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/Types.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "gtest/gtest.h"

using llvm::Align;
using llvm::TypeSize;
using llvm::abi::FieldInfo;
using llvm::abi::RecordFlags;
using llvm::abi::RecordType;
using llvm::abi::StructPacking;
using llvm::abi::TypeBuilder;

namespace {

class ABITypesTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;

  ABITypesTest() : TB(Alloc) {}

  const RecordType *makeRecord(llvm::ArrayRef<FieldInfo> Fields,
                               uint64_t SizeBits, RecordFlags Flags,
                               llvm::ArrayRef<FieldInfo> Bases = {},
                               llvm::ArrayRef<FieldInfo> DirectVBases = {}) {
    return TB.getRecordType(Fields, TypeSize::getFixed(SizeBits), Align(1),
                            StructPacking::Default, Bases, /*VBases=*/{}, Flags,
                            DirectVBases);
  }
};

TEST_F(ABITypesTest, EmptyCRecord) {
  const RecordType *Empty = makeRecord({}, 0, RecordFlags::CanPassInRegisters);
  EXPECT_TRUE(Empty->isEmpty());
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
  const RecordType *Empty = makeRecord({}, 8, CXXFlags);
  const RecordType *IntField = makeRecord(
      {FieldInfo(TB.getIntegerType(32, Align(4), /*Signed=*/true), 0)}, 32,
      CXXFlags);
  const llvm::abi::Type *VPtr = TB.getPointerType(64, Align(8));
  FieldInfo VTable(VPtr, 0, /*IsBitField=*/false, /*BitFieldWidth=*/0,
                   /*IsUnnamedBitField=*/false,
                   /*HasNoUniqueAddress=*/false,
                   /*IsVTablePointer=*/true);

  EXPECT_TRUE(makeRecord({}, 8, CXXFlags, /*Bases=*/{},
                         /*DirectVBases=*/{FieldInfo(Empty, 0)})
                  ->isEmpty());
  EXPECT_FALSE(makeRecord({}, 32, CXXFlags, /*Bases=*/{},
                          /*DirectVBases=*/{FieldInfo(IntField, 0)})
                   ->isEmpty());
  EXPECT_TRUE(makeRecord({VTable}, 64,
                         static_cast<RecordFlags>(CXXFlags |
                                                  RecordFlags::IsPolymorphic),
                         {FieldInfo(Empty, 0)})
                  ->isEmpty());
}

} // namespace
