//===-- TestType.cpp --------------------------------------------*- C++ -*-===//
//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
};

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
