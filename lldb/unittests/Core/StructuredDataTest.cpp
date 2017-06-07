//===-- StructuredDataTest.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Core/StructuredData.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/BinaryFormat/MachO.h"

using namespace lldb;
using namespace lldb_private;

TEST(StructuredDataTest, StringDump) {
  std::pair<llvm::StringRef, llvm::StringRef> TestCases[] = {
    { R"(asdfg)", R"("asdfg")" },
    { R"(as"df)", R"("as\"df")" },
    { R"(as\df)", R"("as\\df")" },
  };
  for(auto P : TestCases) {
    StreamString S;
    const bool pretty_print = false;
    StructuredData::String(P.first).Dump(S, pretty_print);
    EXPECT_EQ(P.second, S.GetString());
  }
}
