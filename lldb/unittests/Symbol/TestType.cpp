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
#include "lldb/lldb-enumerations.h"

using namespace lldb;
using namespace lldb_private;

TEST(Type, GetTypeScopeAndBasename) {
  EXPECT_EQ(Type::GetTypeScopeAndBasename("int"),
            (Type::ParsedName{eTypeClassAny, {}, "int"}));
  EXPECT_EQ(Type::GetTypeScopeAndBasename("std::string"),
            (Type::ParsedName{eTypeClassAny, {"std"}, "string"}));
  EXPECT_EQ(Type::GetTypeScopeAndBasename("::std::string"),
            (Type::ParsedName{eTypeClassAny, {"::", "std"}, "string"}));
  EXPECT_EQ(Type::GetTypeScopeAndBasename("struct std::string"),
            (Type::ParsedName{eTypeClassStruct, {"std"}, "string"}));
  EXPECT_EQ(Type::GetTypeScopeAndBasename("std::set<int>"),
            (Type::ParsedName{eTypeClassAny, {"std"}, "set<int>"}));
  EXPECT_EQ(
      Type::GetTypeScopeAndBasename("std::set<int, std::less<int>>"),
      (Type::ParsedName{eTypeClassAny, {"std"}, "set<int, std::less<int>>"}));
  EXPECT_EQ(Type::GetTypeScopeAndBasename("std::string::iterator"),
            (Type::ParsedName{eTypeClassAny, {"std", "string"}, "iterator"}));
  EXPECT_EQ(Type::GetTypeScopeAndBasename("std::set<int>::iterator"),
            (Type::ParsedName{eTypeClassAny, {"std", "set<int>"}, "iterator"}));
  EXPECT_EQ(
      Type::GetTypeScopeAndBasename("std::set<int, std::less<int>>::iterator"),
      (Type::ParsedName{
          eTypeClassAny, {"std", "set<int, std::less<int>>"}, "iterator"}));
  EXPECT_EQ(Type::GetTypeScopeAndBasename(
                "std::set<int, std::less<int>>::iterator<bool>"),
            (Type::ParsedName{eTypeClassAny,
                              {"std", "set<int, std::less<int>>"},
                              "iterator<bool>"}));

  EXPECT_EQ(Type::GetTypeScopeAndBasename("std::"), std::nullopt);
  EXPECT_EQ(Type::GetTypeScopeAndBasename("foo<::bar"), std::nullopt);
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
