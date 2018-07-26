//===-- MangledTest.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Core/Mangled.h"

using namespace lldb;
using namespace lldb_private;

TEST(MangledTest, ResultForValidName) {
  ConstString MangledName("_ZN1a1b1cIiiiEEvm");
  bool IsMangled = true;

  Mangled TheMangled(MangledName, IsMangled);
  const ConstString &TheDemangled =
      TheMangled.GetDemangledName(eLanguageTypeC_plus_plus);

  ConstString ExpectedResult("void a::b::c<int, int, int>(unsigned long)");
  EXPECT_STREQ(ExpectedResult.GetCString(), TheDemangled.GetCString());
}

TEST(MangledTest, EmptyForInvalidName) {
  ConstString MangledName("_ZN1a1b1cmxktpEEvm");
  bool IsMangled = true;

  Mangled TheMangled(MangledName, IsMangled);
  const ConstString &TheDemangled =
      TheMangled.GetDemangledName(eLanguageTypeC_plus_plus);

  EXPECT_STREQ("", TheDemangled.GetCString());
}
