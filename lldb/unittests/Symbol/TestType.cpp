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

#include "lldb/Core/Declaration.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-enumerations.h"

using namespace lldb;
using namespace lldb_private;
using testing::ElementsAre;
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
MATCHER_P(MatchesWithStrictNamespaces, pattern, "") {
  TypeQuery query(pattern, TypeQueryOptions::e_strict_namespaces);
  return query.ContextMatches(arg);
}
} // namespace

TEST(Type, TypeQueryFlags) {
  TypeQuery q("foo", e_none);
  auto get = [](const TypeQuery &q) -> std::vector<bool> {
    return {q.GetFindOne(), q.GetExactMatch(), q.GetModuleSearch(),
            q.GetIgnoreModules(), q.GetStrictNamespaces()};
  };
  EXPECT_THAT(get(q), ElementsAre(false, false, false, false, false));

  q.SetFindOne(true);
  EXPECT_THAT(get(q), ElementsAre(true, false, false, false, false));

  q.SetIgnoreModules(true);
  EXPECT_THAT(get(q), ElementsAre(true, false, false, true, false));

  q.SetStrictNamespaces(true);
  EXPECT_THAT(get(q), ElementsAre(true, false, false, true, true));

  q.SetIgnoreModules(false);
  EXPECT_THAT(get(q), ElementsAre(true, false, false, false, true));
}

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
      Not(MatchesIgnoringModules(
          std::vector{make_module("A"), make_class("C")})));
  EXPECT_THAT((std::vector{make_module("A"), make_module("B"), make_enum("C")}),
              Matches(std::vector{make_module("A"), make_module("B"),
                                  make_any_type("C")}));
  EXPECT_THAT(
      (std::vector{make_module("A"), make_module("B"), make_class("C")}),
      Matches(
          std::vector{make_module("A"), make_module("B"), make_any_type("C")}));
  EXPECT_THAT((std::vector{make_module("A"), make_module("B"),
                           make_namespace(""), make_class("C")}),
              Matches(std::vector{make_module("A"), make_module("B"),
                                  make_any_type("C")}));
  EXPECT_THAT(
      (std::vector{make_module("A"), make_module("B"), make_enum("C2")}),
      Not(Matches(std::vector{make_module("A"), make_module("B"),
                              make_any_type("C")})));
  EXPECT_THAT((std::vector{make_class("C")}),
              Matches(std::vector{make_class("C")}));
  EXPECT_THAT((std::vector{make_namespace("NS"), make_class("C")}),
              Not(Matches(std::vector{make_any_type("C")})));

  EXPECT_THAT((std::vector{make_namespace(""), make_class("C")}),
              Matches(std::vector{make_class("C")}));
  EXPECT_THAT((std::vector{make_namespace(""), make_class("C")}),
              Not(MatchesWithStrictNamespaces(std::vector{make_class("C")})));
  EXPECT_THAT((std::vector{make_namespace(""), make_class("C")}),
              Matches(std::vector{make_namespace(""), make_class("C")}));
  EXPECT_THAT((std::vector{make_namespace(""), make_class("C")}),
              MatchesWithStrictNamespaces(
                  std::vector{make_namespace(""), make_class("C")}));
  EXPECT_THAT((std::vector{make_class("C")}),
              Not(Matches(std::vector{make_namespace(""), make_class("C")})));
  EXPECT_THAT((std::vector{make_class("C")}),
              Not(MatchesWithStrictNamespaces(
                  std::vector{make_namespace(""), make_class("C")})));
  EXPECT_THAT((std::vector{make_namespace(""), make_namespace("NS"),
                           make_namespace(""), make_class("C")}),
              Matches(std::vector{make_namespace("NS"), make_class("C")}));
  EXPECT_THAT(
      (std::vector{make_namespace(""), make_namespace(""), make_namespace("NS"),
                   make_namespace(""), make_namespace(""), make_class("C")}),
      Matches(std::vector{make_namespace("NS"), make_class("C")}));
  EXPECT_THAT((std::vector{make_module("A"), make_namespace("NS"),
                           make_namespace(""), make_class("C")}),
              MatchesIgnoringModules(
                  std::vector{make_namespace("NS"), make_class("C")}));
}

namespace {
/// Minimal SymbolFile mock that lets us call SymbolFileCommon::MakeType.
class CyclicTypeSymbolFile : public SymbolFileCommon {
  static char ID;

public:
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || SymbolFileCommon::isA(ClassID);
  }

  CyclicTypeSymbolFile() : SymbolFileCommon(/*objfile_sp=*/nullptr) {}

  llvm::StringRef GetPluginName() override { return "CyclicTypeSymbolFile"; }
  uint32_t CalculateAbilities() override { return 0; }
  lldb::LanguageType ParseLanguage(CompileUnit &) override {
    return lldb::eLanguageTypeC;
  }
  size_t ParseFunctions(CompileUnit &) override { return 0; }
  bool ParseLineTable(CompileUnit &) override { return false; }
  bool ParseDebugMacros(CompileUnit &) override { return false; }
  bool ParseSupportFiles(CompileUnit &, SupportFileList &) override {
    return false;
  }
  size_t ParseTypes(CompileUnit &) override { return 0; }
  bool ParseImportedModules(const SymbolContext &,
                            std::vector<SourceModule> &) override {
    return false;
  }
  size_t ParseBlocksRecursive(Function &) override { return 0; }
  size_t ParseVariablesForContext(const SymbolContext &) override { return 0; }
  Type *ResolveTypeUID(lldb::user_id_t) override { return nullptr; }
  std::optional<ArrayInfo>
  GetDynamicArrayInfoForUID(lldb::user_id_t,
                            const ExecutionContext *) override {
    return std::nullopt;
  }
  bool CompleteType(CompilerType &) override { return false; }
  uint32_t ResolveSymbolContext(const Address &, lldb::SymbolContextItem,
                                SymbolContext &) override {
    return 0;
  }
  void GetTypes(SymbolContextScope *, lldb::TypeClass, TypeList &) override {}

  uint32_t CalculateNumCompileUnits() override { return 0; }
  lldb::CompUnitSP ParseCompileUnitAtIndex(uint32_t) override { return {}; }
};

char CyclicTypeSymbolFile::ID;
} // namespace

// Two (malformed) types whose encoding chain forms a cycle (A's
// encoding is B, B's encoding is A). Resolving them should terminate.
TEST(Type, GetForwardCompilerTypeCycle) {
  CyclicTypeSymbolFile symbol_file;
  Declaration decl;

  // Create A and B with eEncodingIsConstUID and an unresolved (invalid)
  // CompilerType, so ResolveCompilerType enters the encoding-resolution path.
  lldb::user_id_t a_uid = 1;
  lldb::user_id_t b_uid = 2;
  std::optional<uint64_t> byte_size;
  SymbolContextScope *context = nullptr;
  TypeSP a =
      symbol_file.MakeType(a_uid, ConstString("A"), byte_size, context, b_uid,
                           Type::eEncodingIsConstUID, decl, CompilerType(),
                           Type::ResolveState::Unresolved);
  TypeSP b =
      symbol_file.MakeType(b_uid, ConstString("B"), byte_size, context, a_uid,
                           Type::eEncodingIsConstUID, decl, CompilerType(),
                           Type::ResolveState::Unresolved);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);

  // Pre-populate the encoding pointers so GetEncodingType bypasses
  // ResolveTypeUID and returns the cyclic peer directly.
  a->SetEncodingType(b.get());
  b->SetEncodingType(a.get());

  EXPECT_FALSE(a->GetForwardCompilerType().IsValid());
  EXPECT_FALSE(b->GetForwardCompilerType().IsValid());
}
