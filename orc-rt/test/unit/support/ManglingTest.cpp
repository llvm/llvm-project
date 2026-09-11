//===- ManglingTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's Mangling.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/support/Mangling.h"
#include "gtest/gtest.h"

using namespace orc_rt;

TEST(ManglingTest, LinkerManglingIsIdentity) {
  EXPECT_EQ(mangledCopy(SymbolNameSpec::linker("")), "");
  EXPECT_EQ(mangledCopy(SymbolNameSpec::linker("foo")), "foo");
}

TEST(ManglingTest, EmptyManglesToEmpty) {
  EXPECT_EQ(mangledCopy(SymbolNameSpec::c("")), "");
}

#if defined(__APPLE__)
TEST(ManglingTest, DarwinShortNameMangling) {
  EXPECT_EQ(mangledCopy(SymbolNameSpec::c("foo")), "_foo");
}

TEST(ManglingTest, DarwinLongNameMangling) {
  std::string LongName(4096, 'x');
  EXPECT_EQ(mangledCopy(SymbolNameSpec::c(LongName)), "_" + LongName);
}
#endif // defined(__APPLE__)
