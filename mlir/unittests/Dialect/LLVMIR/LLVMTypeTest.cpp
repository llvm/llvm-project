//===- LLVMTypeTest.cpp - Tests for LLVM types ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLVMTestBase.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace mlir;
using namespace mlir::LLVM;

TEST_F(LLVMIRTest, IsStructTypeMutable) {
  auto structTy = LLVMStructType::getIdentified(&context, "foo");
  ASSERT_TRUE(bool(structTy));
  ASSERT_TRUE(structTy.hasTrait<TypeTrait::IsMutable>());
}
