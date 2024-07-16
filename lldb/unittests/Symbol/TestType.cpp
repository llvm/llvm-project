//===-- TestType.cpp ------------------------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "lldb/Symbol/Type.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-enumerations.h"

using namespace lldb;
using namespace lldb_private;
using testing::Not;

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

namespace {
MATCHER_P(Matches, pattern, "") {
  TypeQuery query(pattern, TypeQueryOptions::e_none);
  return query.ContextMatches(arg);
}
MATCHER_P(MatchesIgnoringModules, pattern, "") {
  TypeQuery query(pattern, TypeQueryOptions::e_ignore_modules);
  return query.ContextMatches(arg);
}
} // namespace

TEST(Type, CompilerContextPattern) {
  auto make_module = [](llvm::StringRef name) {
    return CompilerContext(CompilerContextKind::Module, ConstString(name));
  };
  auto make_class = [](llvm::StringRef name) {
    return CompilerContext(CompilerContextKind::ClassOrStruct,
                           ConstString(name));
  };
  auto make_any_type = [](llvm::StringRef name) {
    return CompilerContext(CompilerContextKind::AnyType, ConstString(name));
  };
  auto make_enum = [](llvm::StringRef name) {
    return CompilerContext(CompilerContextKind::Enum, ConstString(name));
  };
  auto make_namespace = [](llvm::StringRef name) {
    return CompilerContext(CompilerContextKind::Namespace, ConstString(name));
  };

  EXPECT_THAT(
      (std::vector{make_module("A"), make_module("B"), make_class("C")}),
      Matches(
          std::vector{make_module("A"), make_module("B"), make_class("C")}));
  EXPECT_THAT(
      (std::vector{make_module("A"), make_module("B"), make_class("C")}),
      Not(Matches(std::vector{make_class("C")})));
  EXPECT_THAT(
      (std::vector{make_module("A"), make_module("B"), make_class("C")}),
      MatchesIgnoringModules(std::vector{make_class("C")}));
  EXPECT_THAT(
      (std::vector{make_module("A"), make_module("B"), make_class("C")}),
      MatchesIgnoringModules(std::vector{make_module("B"), make_class("C")}));
  EXPECT_THAT(
      (std::vector{make_module("A"), make_module("B"), make_class("C")}),
      Not(MatchesIgnoringModules(std::vector{make_module("A"), make_class("C")})));
  EXPECT_THAT((std::vector{make_module("A"), make_module("B"), make_enum("C")}),
              Matches(std::vector{make_module("A"), make_module("B"),
                                  make_any_type("C")}));
  EXPECT_THAT(
      (std::vector{make_module("A"), make_module("B"), make_class("C")}),
      Matches(
          std::vector{make_module("A"), make_module("B"), make_any_type("C")}));
  EXPECT_THAT(
      (std::vector{make_module("A"), make_module("B"), make_enum("C2")}),
      Not(Matches(std::vector{make_module("A"), make_module("B"),
                              make_any_type("C")})));
  EXPECT_THAT((std::vector{make_class("C")}),
              Matches(std::vector{make_class("C")}));
  EXPECT_THAT((std::vector{make_namespace("NS"), make_class("C")}),
              Not(Matches(std::vector{make_any_type("C")})));
}
