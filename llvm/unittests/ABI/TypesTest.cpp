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
                               llvm::ArrayRef<FieldInfo> VBases = {},
                               Align Alignment = Align(1)) {
    return TB.getRecordType(Fields, TypeSize::getFixed(SizeBits), Alignment,
                            StructPacking::Default, Bases, VBases, Flags);
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

} // namespace
