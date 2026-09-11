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

TEST_F(IRTypeMapperTest, GenericVectorMapsToFixedVector) {
  const llvm::abi::Type *I32 =
      TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true);
  const llvm::abi::VectorType *V4I32 =
      TB.getVectorType(I32, llvm::ElementCount::getFixed(4), llvm::Align(16));

  auto *Vec = llvm::dyn_cast<llvm::VectorType>(Mapper.convertType(V4I32));
  ASSERT_NE(Vec, nullptr);
  EXPECT_FALSE(Vec->isScalableTy());
  EXPECT_EQ(Vec->getElementCount(), llvm::ElementCount::getFixed(4));
  EXPECT_TRUE(Vec->getElementType()->isIntegerTy(32));
}

TEST_F(IRTypeMapperTest, SVEDataVectorMapsToScalableVector) {
  const llvm::abi::Type *I32 =
      TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true);
  const llvm::abi::VectorType *SVInt32 =
      TB.getVectorType(I32, llvm::ElementCount::getScalable(4), llvm::Align(16),
                       llvm::abi::VectorKind::SVEData);

  auto *Vec = llvm::dyn_cast<llvm::VectorType>(Mapper.convertType(SVInt32));
  ASSERT_NE(Vec, nullptr);
  EXPECT_TRUE(Vec->isScalableTy());
  EXPECT_EQ(Vec->getElementCount(), llvm::ElementCount::getScalable(4));
  EXPECT_TRUE(Vec->getElementType()->isIntegerTy(32));
}

TEST_F(IRTypeMapperTest, SVEPredicateVectorMapsToScalableI1Vector) {
  const llvm::abi::Type *I1 =
      TB.getIntegerType(1, llvm::Align(1), /*Signed=*/false);
  const llvm::abi::VectorType *SVBool =
      TB.getVectorType(I1, llvm::ElementCount::getScalable(16), llvm::Align(2),
                       llvm::abi::VectorKind::SVEPredicate);

  auto *Vec = llvm::dyn_cast<llvm::VectorType>(Mapper.convertType(SVBool));
  ASSERT_NE(Vec, nullptr);
  EXPECT_TRUE(Vec->isScalableTy());
  EXPECT_EQ(Vec->getElementCount(), llvm::ElementCount::getScalable(16));
  EXPECT_TRUE(Vec->getElementType()->isIntegerTy(1));
}

TEST_F(IRTypeMapperTest, SVECountMapsToAArch64SVCount) {
  const llvm::abi::VectorType *SVCount = TB.getSVECountType(llvm::Align(2));

  auto *TET = llvm::dyn_cast<llvm::TargetExtType>(Mapper.convertType(SVCount));
  ASSERT_NE(TET, nullptr);
  EXPECT_EQ(TET->getName(), "aarch64.svcount");
}

TEST_F(IRTypeMapperTest, SVEDataTupleMapsToStructOfVectors) {
  const llvm::abi::Type *I32 =
      TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true);
  const llvm::abi::VectorType *SVInt32 =
      TB.getVectorType(I32, llvm::ElementCount::getScalable(4), llvm::Align(16),
                       llvm::abi::VectorKind::SVEData);
  const llvm::abi::TupleType *SVInt32x3 =
      TB.getTupleType(SVInt32, /*NumVectors=*/3);

  auto *Struct =
      llvm::dyn_cast<llvm::StructType>(Mapper.convertType(SVInt32x3));
  ASSERT_NE(Struct, nullptr);
  ASSERT_EQ(Struct->getNumElements(), 3u);

  llvm::Type *ExpectedVec = Mapper.convertType(SVInt32);
  for (unsigned I = 0; I < 3; ++I)
    EXPECT_EQ(Struct->getElementType(I), ExpectedVec);
}

} // namespace
