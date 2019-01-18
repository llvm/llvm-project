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
  EXPECT_EQ(demangle("_Z3fooi"), "foo(int)");
  EXPECT_EQ(demangle("?foo@@YAXH@Z"), "void __cdecl foo(int)");
  EXPECT_EQ(demangle("foo"), "foo");
}
