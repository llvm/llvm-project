//===-- DemangleTest.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"
#include "gmock/gmock.h"

using namespace llvm;

TEST(Demangle, demangleTest) {
  EXPECT_EQ(demangle("_"), "_");
  EXPECT_EQ(demangle("_Z3fooi"), "foo(int)");
  EXPECT_EQ(demangle("__Z3fooi"), "foo(int)");
  EXPECT_EQ(demangle("___Z3fooi_block_invoke"),
            "invocation function for block in foo(int)");
  EXPECT_EQ(demangle("____Z3fooi_block_invoke"),
            "invocation function for block in foo(int)");
  EXPECT_EQ(demangle("?foo@@YAXH@Z"), "void __cdecl foo(int)");
  EXPECT_EQ(demangle("foo"), "foo");
}
