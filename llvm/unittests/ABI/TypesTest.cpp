//===- TypesTest.cpp - Unit tests for ABI Types ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/Types.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::abi;

namespace {

class ABITypesTest : public ::testing::Test {
protected:
  BumpPtrAllocator Alloc;
  TypeBuilder TB{Alloc};
};

TEST_F(ABITypesTest, RecordFlagsNonTrivialCopyConstructor) {
  RecordFlags Flags =
      RecordFlags::IsCXXRecord | RecordFlags::HasNonTrivialCopyConstructor;
  const RecordType *RT =
      TB.getRecordType({}, TypeSize::getFixed(64), Align(8),
                       StructPacking::Default, {}, {}, Flags);
  EXPECT_TRUE(RT->hasNonTrivialCopyConstructor());
  EXPECT_FALSE(RT->hasNonTrivialDestructor());
  EXPECT_TRUE(RT->isCXXRecord());
}

TEST_F(ABITypesTest, RecordFlagsNonTrivialDestructor) {
  RecordFlags Flags =
      RecordFlags::IsCXXRecord | RecordFlags::HasNonTrivialDestructor;
  const RecordType *RT =
      TB.getRecordType({}, TypeSize::getFixed(64), Align(8),
                       StructPacking::Default, {}, {}, Flags);
  EXPECT_FALSE(RT->hasNonTrivialCopyConstructor());
  EXPECT_TRUE(RT->hasNonTrivialDestructor());
}

TEST_F(ABITypesTest, RecordFlagsBothNonTrivial) {
  RecordFlags Flags = RecordFlags::IsCXXRecord |
                      RecordFlags::HasNonTrivialCopyConstructor |
                      RecordFlags::HasNonTrivialDestructor;
  const RecordType *RT =
      TB.getRecordType({}, TypeSize::getFixed(64), Align(8),
                       StructPacking::Default, {}, {}, Flags);
  EXPECT_TRUE(RT->hasNonTrivialCopyConstructor());
  EXPECT_TRUE(RT->hasNonTrivialDestructor());
}

TEST_F(ABITypesTest, RecordFlagsNoneSet) {
  const RecordType *RT = TB.getRecordType({}, TypeSize::getFixed(64), Align(8));
  EXPECT_FALSE(RT->hasNonTrivialCopyConstructor());
  EXPECT_FALSE(RT->hasNonTrivialDestructor());
  EXPECT_FALSE(RT->isCXXRecord());
  EXPECT_FALSE(RT->isUnion());
  EXPECT_FALSE(RT->isPolymorphic());
}

TEST_F(ABITypesTest, RecordFlagsCombineWithExisting) {
  RecordFlags Flags = RecordFlags::CanPassInRegisters |
                      RecordFlags::IsCXXRecord |
                      RecordFlags::HasNonTrivialDestructor;
  const RecordType *RT =
      TB.getRecordType({}, TypeSize::getFixed(64), Align(8),
                       StructPacking::Default, {}, {}, Flags);
  EXPECT_TRUE(RT->canPassInRegisters());
  EXPECT_TRUE(RT->isCXXRecord());
  EXPECT_TRUE(RT->hasNonTrivialDestructor());
  EXPECT_FALSE(RT->hasNonTrivialCopyConstructor());
  EXPECT_FALSE(RT->isUnion());
}

} // namespace
