//===- FunctionInfoTest.cpp - ArgInfo and FunctionInfo unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

namespace {

using ABIType = llvm::abi::Type;
using llvm::abi::ArgEntry;
using llvm::abi::ArgInfo;
using llvm::abi::FieldInfo;
using llvm::abi::FunctionInfo;
using llvm::abi::StructPacking;
using llvm::abi::TypeBuilder;

class FunctionInfoTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;
  const ABIType *I32;
  const ABIType *I64;
  /// A two-i64 record: the shape a classifier coerces a 16-byte struct to when
  /// it lands in two registers, and so the shape a rewriter may flatten.
  const ABIType *TwoI64;

  FunctionInfoTest()
      : TB(Alloc), I32(TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true)),
        I64(TB.getIntegerType(64, llvm::Align(8), /*Signed=*/true)),
        TwoI64(TB.getRecordType({FieldInfo(I64, 0), FieldInfo(I64, 64)},
                                llvm::TypeSize::getFixed(128), llvm::Align(8),
                                StructPacking::Default)) {}
};

TEST_F(FunctionInfoTest, DirectCanBeFlattenedByDefault) {
  EXPECT_TRUE(ArgInfo::getDirect().getCanBeFlattened());
  EXPECT_TRUE(ArgInfo::getDirect(TwoI64).getCanBeFlattened());
  EXPECT_TRUE(ArgInfo::getDirect(I64, /*Offset=*/8).getCanBeFlattened());
}

TEST_F(FunctionInfoTest, SetCanBeFlattenedRoundTrips) {
  ArgInfo Info = ArgInfo::getDirect(TwoI64);
  EXPECT_EQ(&Info.setCanBeFlattened(false), &Info);
  EXPECT_FALSE(Info.getCanBeFlattened());
  // Clearing the flag leaves the rest of the classification alone.
  EXPECT_TRUE(Info.isDirect());
  EXPECT_EQ(Info.getCoerceToType(), TwoI64);
  EXPECT_EQ(Info.getDirectOffset(), 0u);

  Info.setCanBeFlattened(true);
  EXPECT_TRUE(Info.getCanBeFlattened());
}

TEST_F(FunctionInfoTest, SetCanBeFlattenedChainsOffGetDirect) {
  // The spelling a classifier uses to keep an aggregate in one piece.
  ArgInfo Info = ArgInfo::getDirect(TwoI64).setCanBeFlattened(false);
  EXPECT_TRUE(Info.isDirect());
  EXPECT_FALSE(Info.getCanBeFlattened());
}

TEST_F(FunctionInfoTest, CanBeFlattenedSurvivesFunctionInfo) {
  std::unique_ptr<FunctionInfo> FI =
      FunctionInfo::create(llvm::CallingConv::C, TwoI64, {TwoI64, I32});
  FI->getReturnInfo() = ArgInfo::getDirect(TwoI64).setCanBeFlattened(false);
  FI->getArgInfo(0).Info = ArgInfo::getDirect(TwoI64).setCanBeFlattened(false);
  FI->getArgInfo(1).Info = ArgInfo::getDirect(I32);

  const FunctionInfo &ConstFI = *FI;
  EXPECT_FALSE(ConstFI.getReturnInfo().getCanBeFlattened());
  EXPECT_FALSE(ConstFI.arguments()[0].Info.getCanBeFlattened());
  EXPECT_TRUE(ConstFI.arguments()[1].Info.getCanBeFlattened());

  // The flag rides along with the rest of the classification on copy.
  ArgEntry Copy = ConstFI.getArgInfo(0);
  EXPECT_FALSE(Copy.Info.getCanBeFlattened());
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
TEST_F(FunctionInfoTest, CanBeFlattenedIsDirectOnly) {
  EXPECT_DEATH((void)ArgInfo::getIgnore().getCanBeFlattened(), "Invalid Kind");
  EXPECT_DEATH((void)ArgInfo::getExtend(I32).getCanBeFlattened(),
               "Invalid Kind");
  EXPECT_DEATH((void)ArgInfo::getIndirect(llvm::Align(8), /*ByVal=*/true)
                   .getCanBeFlattened(),
               "Invalid Kind");
  EXPECT_DEATH((void)ArgInfo::getIgnore().setCanBeFlattened(false),
               "Invalid Kind");
}
#endif

} // namespace
