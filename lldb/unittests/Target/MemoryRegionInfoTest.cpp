//===-- MemoryRegionInfoTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/MemoryRegionInfo.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(MemoryRegionInfoTest, Formatv) {
  EXPECT_EQ("yes", llvm::formatv("{0}", eLazyBoolYes).str());
  EXPECT_EQ("no", llvm::formatv("{0}", eLazyBoolNo).str());
  EXPECT_EQ("don't know", llvm::formatv("{0}", eLazyBoolDontKnow).str());
}
