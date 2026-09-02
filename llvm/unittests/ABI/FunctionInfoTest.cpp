//===- FunctionInfoTest.cpp - ArgInfo unit tests --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

namespace {

using llvm::abi::ArgInfo;
using llvm::abi::Type;
using llvm::abi::TypeBuilder;

class ArgInfoTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;
  const Type *I32;

  ArgInfoTest()
      : TB(Alloc), I32(TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true)) {
  }
};

TEST_F(ArgInfoTest, DirectCanBeFlattenedDefaultsTrue) {
  EXPECT_TRUE(ArgInfo::getDirect().getCanBeFlattened());
  EXPECT_TRUE(ArgInfo::getDirect(I32).getCanBeFlattened());
}

TEST_F(ArgInfoTest, DirectCanBeFlattenedHonorsFalse) {
  ArgInfo AI = ArgInfo::getDirect(I32, /*Offset=*/0, /*Align=*/std::nullopt,
                                  /*CanBeFlattened=*/false);
  EXPECT_TRUE(AI.isDirect());
  EXPECT_FALSE(AI.getCanBeFlattened());
}

} // namespace
