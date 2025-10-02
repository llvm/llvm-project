//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/NativePDB/PdbUtil.h"
#include "gtest/gtest.h"

using namespace lldb_private::npdb;

TEST(PdbUtil, StripCDeclPrefix) {
  ASSERT_EQ(StripCDeclPrefix("main"), "main");

  // __cdecl
  ASSERT_EQ(StripCDeclPrefix("_main"), "main");
  ASSERT_EQ(StripCDeclPrefix("__main"), "_main");
  ASSERT_EQ(StripCDeclPrefix("_main@"), "main@");
  ASSERT_EQ(StripCDeclPrefix("_main@foo"), "main@foo");
  ASSERT_EQ(StripCDeclPrefix("_main@4@foo"), "main@4@foo");

  // __stdcall
  ASSERT_EQ(StripCDeclPrefix("_main@4"), "_main@4");
  ASSERT_EQ(StripCDeclPrefix("_main@foo@4"), "_main@foo@4");
  ASSERT_EQ(StripCDeclPrefix("_main@4@5"), "_main@4@5");

  // __fastcall
  ASSERT_EQ(StripCDeclPrefix("@main@4"), "@main@4");

  // __vectorcall
  ASSERT_EQ(StripCDeclPrefix("main@@4"), "main@@4");
  ASSERT_EQ(StripCDeclPrefix("_main@@4"), "_main@@4");

  // MS C++ mangling
  ASSERT_EQ(StripCDeclPrefix("?a@@YAHD@Z"), "?a@@YAHD@Z");
  // Itanium mangling (e.g. on MinGW)
  ASSERT_EQ(StripCDeclPrefix("__Z7recursei"), "_Z7recursei");

  ASSERT_EQ(StripCDeclPrefix("_"), "");
  ASSERT_EQ(StripCDeclPrefix("_@"), "@");
  ASSERT_EQ(StripCDeclPrefix(""), "");
  ASSERT_EQ(StripCDeclPrefix("_@4"), "_@4");
  ASSERT_EQ(StripCDeclPrefix("@4"), "@4");
  ASSERT_EQ(StripCDeclPrefix("@"), "@");
}
