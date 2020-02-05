//===-- TestType.cpp ------------------------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Symbol/Type.h"

using namespace lldb;
using namespace lldb_private;

namespace {
void TestGetTypeScopeAndBasenameHelper(const char *full_type,
                                       bool expected_is_scoped,
                                       const char *expected_scope,
                                       const char *expected_name) {
  llvm::StringRef scope, name;
  lldb::TypeClass type_class;
  bool is_scoped =
      Type::GetTypeScopeAndBasename(full_type, scope, name, type_class);
  EXPECT_EQ(is_scoped, expected_is_scoped);
  if (expected_is_scoped) {
    EXPECT_EQ(scope, expected_scope);
    EXPECT_EQ(name, expected_name);
  }
}
}

TEST(Type, GetTypeScopeAndBasename) {
  TestGetTypeScopeAndBasenameHelper("int", false, "", "");
  TestGetTypeScopeAndBasenameHelper("std::string", true, "std::", "string");
  TestGetTypeScopeAndBasenameHelper("std::set<int>", true, "std::", "set<int>");
  TestGetTypeScopeAndBasenameHelper("std::set<int, std::less<int>>", true,
                                    "std::", "set<int, std::less<int>>");
  TestGetTypeScopeAndBasenameHelper("std::string::iterator", true,
                                    "std::string::", "iterator");
  TestGetTypeScopeAndBasenameHelper("std::set<int>::iterator", true,
                                    "std::set<int>::", "iterator");
  TestGetTypeScopeAndBasenameHelper(
      "std::set<int, std::less<int>>::iterator", true,
      "std::set<int, std::less<int>>::", "iterator");
  TestGetTypeScopeAndBasenameHelper(
      "std::set<int, std::less<int>>::iterator<bool>", true,
      "std::set<int, std::less<int>>::", "iterator<bool>");
}

TEST(Type, CompilerContextPattern) {
  std::vector<CompilerContext> mms = {
      {CompilerContextKind::Module, ConstString("A")},
      {CompilerContextKind::Module, ConstString("B")},
      {CompilerContextKind::Struct, ConstString("S")}};
  EXPECT_TRUE(contextMatches(mms, mms));
  std::vector<CompilerContext> mmc = {
      {CompilerContextKind::Module, ConstString("A")},
      {CompilerContextKind::Module, ConstString("B")},
      {CompilerContextKind::Class, ConstString("S")}};
  EXPECT_FALSE(contextMatches(mms, mmc));
  std::vector<CompilerContext> ms = {
      {CompilerContextKind::Module, ConstString("A")},
      {CompilerContextKind::Struct, ConstString("S")}};
  std::vector<CompilerContext> mas = {
      {CompilerContextKind::Module, ConstString("A")},
      {CompilerContextKind::AnyModule, ConstString("*")},
      {CompilerContextKind::Struct, ConstString("S")}};
  EXPECT_TRUE(contextMatches(mms, mas));
  EXPECT_TRUE(contextMatches(ms, mas));
  EXPECT_FALSE(contextMatches(mas, ms));
  std::vector<CompilerContext> mmms = {
      {CompilerContextKind::Module, ConstString("A")},
      {CompilerContextKind::Module, ConstString("B")},
      {CompilerContextKind::Module, ConstString("C")},
      {CompilerContextKind::Struct, ConstString("S")}};
  EXPECT_TRUE(contextMatches(mmms, mas));
  std::vector<CompilerContext> mme = {
      {CompilerContextKind::Module, ConstString("A")},
      {CompilerContextKind::Module, ConstString("B")},
      {CompilerContextKind::Enum, ConstString("S")}};
  std::vector<CompilerContext> mma = {
      {CompilerContextKind::Module, ConstString("A")},
      {CompilerContextKind::Module, ConstString("B")},
      {CompilerContextKind::AnyType, ConstString("S")}};
  EXPECT_TRUE(contextMatches(mme, mma));
  EXPECT_TRUE(contextMatches(mms, mma));
  std::vector<CompilerContext> mme2 = {
      {CompilerContextKind::Module, ConstString("A")},
      {CompilerContextKind::Module, ConstString("B")},
      {CompilerContextKind::Enum, ConstString("S2")}};
  EXPECT_FALSE(contextMatches(mme2, mma));
}
