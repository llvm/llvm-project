//===-- MemoryRegionInfoTest.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/MemoryRegionInfo.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(MemoryRegionInfoTest, Formatv) {
  EXPECT_EQ("yes", llvm::formatv("{0}", MemoryRegionInfo::eYes).str());
  EXPECT_EQ("no", llvm::formatv("{0}", MemoryRegionInfo::eNo).str());
  EXPECT_EQ("don't know", llvm::formatv("{0}", MemoryRegionInfo::eDontKnow).str());
}
