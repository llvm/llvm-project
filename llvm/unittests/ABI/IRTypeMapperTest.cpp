//===- IRTypeMapperTest.cpp - ABI to LLVM IR type mapping tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/IRTypeMapper.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "gtest/gtest.h"

namespace {

class IRTypeMapperTest : public ::testing::Test {
protected:
  llvm::LLVMContext Context;
  llvm::DataLayout DL{""};
  llvm::BumpPtrAllocator Alloc;
  llvm::abi::TypeBuilder TB{Alloc};
  llvm::abi::IRTypeMapper Mapper{Context, DL};
};

TEST_F(IRTypeMapperTest, SameSizeAtomicMapsToValueType) {
  const llvm::abi::Type *F32 =
      TB.getFloatType(llvm::APFloat::IEEEsingle(), llvm::Align(4));
  const llvm::abi::AtomicType *Atomic =
      TB.getAtomicType(F32, 32, llvm::Align(4));

  EXPECT_TRUE(Mapper.convertType(Atomic)->isFloatTy());
}

TEST_F(IRTypeMapperTest, PaddedAtomicMapsToValueAndTailPadding) {
  const llvm::abi::Type *I8 =
      TB.getIntegerType(8, llvm::Align(1), /*Signed=*/true);
  const llvm::abi::RecordType *ThreeBytes = TB.getRecordType(
      {llvm::abi::FieldInfo(I8, 0), llvm::abi::FieldInfo(I8, 8),
       llvm::abi::FieldInfo(I8, 16)},
      llvm::TypeSize::getFixed(24), llvm::Align(1));
  const llvm::abi::AtomicType *Atomic =
      TB.getAtomicType(ThreeBytes, 32, llvm::Align(4));

  auto *Struct = llvm::dyn_cast<llvm::StructType>(Mapper.convertType(Atomic));
  ASSERT_NE(Struct, nullptr);
  ASSERT_EQ(Struct->getNumElements(), 2u);
  EXPECT_TRUE(Struct->getElementType(0)->isStructTy());

  const auto *Padding =
      llvm::dyn_cast<llvm::ArrayType>(Struct->getElementType(1));
  ASSERT_NE(Padding, nullptr);
  EXPECT_TRUE(Padding->getElementType()->isIntegerTy(8));
  EXPECT_EQ(Padding->getNumElements(), 1u);
  EXPECT_EQ(DL.getTypeAllocSize(Struct), llvm::TypeSize::getFixed(32 / 8));
}

} // namespace
