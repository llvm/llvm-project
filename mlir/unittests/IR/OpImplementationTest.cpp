//===- OpImplementationTest.cpp - Operation implementation tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OpImplementation.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
TEST(AsmDialectResourceHandleTest, DefaultHandle) {
  AsmDialectResourceHandle handle;
  EXPECT_EQ(handle.getResource(), nullptr);
  EXPECT_EQ(handle.getTypeID(), TypeID::get<void>());
  EXPECT_EQ(handle.getDialect(), nullptr);
}
} // namespace
