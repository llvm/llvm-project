//===-- CPlusPlusLanguageTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"
#include "Plugins/Language/CPlusPlus/CPlusPlusNameParser.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/lldb-enumerations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(CPlusPlusLanguage, MethodNameParsing) {
  struct TestCase {
    std::string input;
    std::string context, basename, arguments, qualifiers, scope_qualified_name;
  };

  TestCase test_cases[] = {
      {"main(int, char *[]) ", "", "main", "(int, char *[])", "", "main"},
      {"foo::bar(baz) const", "foo", "bar", "(baz)", "const", "foo::bar"},
      {"foo::~bar(baz)", "foo", "~bar", "(baz)", "", "foo::~bar"},
      {"a::b::c::d(e,f)", "a::b::c", "d", "(e,f)", "", "a::b::c::d"},
      {"void f(int)", "", "f", "(int)", "", "f"},

      // Operators
      {"std::basic_ostream<char, std::char_traits<char> >& "
       "std::operator<<<std::char_traits<char> >"
       "(std::basic_ostream<char, std::char_traits<char> >&, char const*)",
       "std", "operator<<<std::char_traits<char> >",
       "(std::basic_ostream<char, std::char_traits<char> >&, char const*)", "",
       "std::operator<<<std::char_traits<char> >"},
      {"operator delete[](void*, clang::ASTContext const&, unsigned long)", "",
       "operator delete[]", "(void*, clang::ASTContext const&, unsigned long)",
       "", "operator delete[]"},
      {"llvm::Optional<clang::PostInitializer>::operator bool() const",
       "llvm::Optional<clang::PostInitializer>", "operator bool", "()", "const",
       "llvm::Optional<clang::PostInitializer>::operator bool"},
      {"(anonymous namespace)::FactManager::operator[](unsigned short)",
       "(anonymous namespace)::FactManager", "operator[]", "(unsigned short)",
       "", "(anonymous namespace)::FactManager::operator[]"},
      {"const int& std::map<int, pair<short, int>>::operator[](short) const",
       "std::map<int, pair<short, int>>", "operator[]", "(short)", "const",
       "std::map<int, pair<short, int>>::operator[]"},
      {"CompareInsn::operator()(llvm::StringRef, InsnMatchEntry const&)",
       "CompareInsn", "operator()", "(llvm::StringRef, InsnMatchEntry const&)",
       "", "CompareInsn::operator()"},
      {"llvm::Optional<llvm::MCFixupKind>::operator*() const &",
       "llvm::Optional<llvm::MCFixupKind>", "operator*", "()", "const &",
       "llvm::Optional<llvm::MCFixupKind>::operator*"},
      // Internal classes
      {"operator<<(Cls, Cls)::Subclass::function()",
       "operator<<(Cls, Cls)::Subclass", "function", "()", "",
       "operator<<(Cls, Cls)::Subclass::function"},
      {"SAEC::checkFunction(context&) const::CallBack::CallBack(int)",
       "SAEC::checkFunction(context&) const::CallBack", "CallBack", "(int)", "",
       "SAEC::checkFunction(context&) const::CallBack::CallBack"},
      // Anonymous namespace
      {"XX::(anonymous namespace)::anon_class::anon_func() const",
       "XX::(anonymous namespace)::anon_class", "anon_func", "()", "const",
       "XX::(anonymous namespace)::anon_class::anon_func"},

      // Lambda
      {"main::{lambda()#1}::operator()() const::{lambda()#1}::operator()() "
       "const",
       "main::{lambda()#1}::operator()() const::{lambda()#1}", "operator()",
       "()", "const",
       "main::{lambda()#1}::operator()() const::{lambda()#1}::operator()"},

      // Function pointers
      {"string (*f(vector<int>&&))(float)", "", "f", "(vector<int>&&)", "",
       "f"},
      {"void (*&std::_Any_data::_M_access<void (*)()>())()", "std::_Any_data",
       "_M_access<void (*)()>", "()", "",
       "std::_Any_data::_M_access<void (*)()>"},
      {"void (*(*(*(*(*(*(*(* const&func1(int))())())())())())())())()", "",
       "func1", "(int)", "", "func1"},

      // Decltype
      {"decltype(nullptr)&& std::forward<decltype(nullptr)>"
       "(std::remove_reference<decltype(nullptr)>::type&)",
       "std", "forward<decltype(nullptr)>",
       "(std::remove_reference<decltype(nullptr)>::type&)", "",
       "std::forward<decltype(nullptr)>"},

      // Templates
      {"void llvm::PM<llvm::Module, llvm::AM<llvm::Module>>::"
       "addPass<llvm::VP>(llvm::VP)",
       "llvm::PM<llvm::Module, llvm::AM<llvm::Module>>", "addPass<llvm::VP>",
       "(llvm::VP)", "",
       "llvm::PM<llvm::Module, llvm::AM<llvm::Module>>::"
       "addPass<llvm::VP>"},
      {"void std::vector<Class, std::allocator<Class> >"
       "::_M_emplace_back_aux<Class const&>(Class const&)",
       "std::vector<Class, std::allocator<Class> >",
       "_M_emplace_back_aux<Class const&>", "(Class const&)", "",
       "std::vector<Class, std::allocator<Class> >::"
       "_M_emplace_back_aux<Class const&>"},
      {"unsigned long llvm::countTrailingOnes<unsigned int>"
       "(unsigned int, llvm::ZeroBehavior)",
       "llvm", "countTrailingOnes<unsigned int>",
       "(unsigned int, llvm::ZeroBehavior)", "",
       "llvm::countTrailingOnes<unsigned int>"},
      {"std::enable_if<(10u)<(64), bool>::type llvm::isUInt<10u>(unsigned "
       "long)",
       "llvm", "isUInt<10u>", "(unsigned long)", "", "llvm::isUInt<10u>"},
      {"f<A<operator<(X,Y)::Subclass>, sizeof(B)<sizeof(C)>()", "",
       "f<A<operator<(X,Y)::Subclass>, sizeof(B)<sizeof(C)>", "()", "",
       "f<A<operator<(X,Y)::Subclass>, sizeof(B)<sizeof(C)>"},
      {"llvm::Optional<llvm::MCFixupKind>::operator*() const volatile &&",
       "llvm::Optional<llvm::MCFixupKind>", "operator*", "()",
       "const volatile &&", "llvm::Optional<llvm::MCFixupKind>::operator*"},
      {"void foo<Dummy<char [10]>>()", "", "foo<Dummy<char [10]>>", "()", "",
       "foo<Dummy<char [10]>>"},

      // auto return type
      {"auto std::test_return_auto<int>() const", "std",
       "test_return_auto<int>", "()", "const", "std::test_return_auto<int>"},
      {"decltype(auto) std::test_return_auto<int>(int) const", "std",
       "test_return_auto<int>", "(int)", "const", "std::test_return_auto<int>"},

      // abi_tag on class method
      {"v1::v2::Dummy[abi:c1][abi:c2]<v1::v2::Dummy[abi:c1][abi:c2]<int>> "
       "v1::v2::Dummy[abi:c1][abi:c2]<v1::v2::Dummy[abi:c1][abi:c2]<int>>"
       "::method2<v1::v2::Dummy[abi:c1][abi:c2]<v1::v2::Dummy[abi:c1][abi:c2]<"
       "int>>>(int, v1::v2::Dummy<int>) const &&",
       // Context
       "v1::v2::Dummy[abi:c1][abi:c2]<v1::v2::Dummy[abi:c1][abi:c2]<int>>",
       // Basename
       "method2<v1::v2::Dummy[abi:c1][abi:c2]<v1::v2::Dummy[abi:c1][abi:c2]<"
       "int>>>",
       // Args, qualifiers
       "(int, v1::v2::Dummy<int>)", "const &&",
       // Full scope-qualified name without args
       "v1::v2::Dummy[abi:c1][abi:c2]<v1::v2::Dummy[abi:c1][abi:c2]<int>>"
       "::method2<v1::v2::Dummy[abi:c1][abi:c2]<v1::v2::Dummy[abi:c1][abi:c2]<"
       "int>>>"},

      // abi_tag on free function and template argument
      {"v1::v2::Dummy[abi:c1][abi:c2]<v1::v2::Dummy[abi:c1][abi:c2]<int>> "
       "v1::v2::with_tag_in_ns[abi:f1][abi:f2]<v1::v2::Dummy[abi:c1][abi:c2]"
       "<v1::v2::Dummy[abi:c1][abi:c2]<int>>>(int, v1::v2::Dummy<int>) const "
       "&&",
       // Context
       "v1::v2",
       // Basename
       "with_tag_in_ns[abi:f1][abi:f2]<v1::v2::Dummy[abi:c1][abi:c2]<v1::v2::"
       "Dummy[abi:c1][abi:c2]<int>>>",
       // Args, qualifiers
       "(int, v1::v2::Dummy<int>)", "const &&",
       // Full scope-qualified name without args
       "v1::v2::with_tag_in_ns[abi:f1][abi:f2]<v1::v2::Dummy[abi:c1][abi:c2]<"
       "v1::v2::Dummy[abi:c1][abi:c2]<int>>>"},

      // abi_tag with special characters
      {"auto ns::with_tag_in_ns[abi:special tag,0.0][abi:special "
       "tag,1.0]<Dummy<int>>"
       "(float) const &&",
       // Context
       "ns",
       // Basename
       "with_tag_in_ns[abi:special tag,0.0][abi:special tag,1.0]<Dummy<int>>",
       // Args, qualifiers
       "(float)", "const &&",
       // Full scope-qualified name without args
       "ns::with_tag_in_ns[abi:special tag,0.0][abi:special "
       "tag,1.0]<Dummy<int>>"},

      // abi_tag on operator overloads
      {"std::__1::error_code::operator bool[abi:v160000]() const",
       "std::__1::error_code", "operator bool[abi:v160000]", "()", "const",
       "std::__1::error_code::operator bool[abi:v160000]"},

      {"auto ns::foo::operator[][abi:v160000](size_t) const", "ns::foo",
       "operator[][abi:v160000]", "(size_t)", "const",
       "ns::foo::operator[][abi:v160000]"},

      {"auto Foo[abi:abc]<int>::operator<<<Foo[abi:abc]<int>>(int) &",
       "Foo[abi:abc]<int>", "operator<<<Foo[abi:abc]<int>>", "(int)", "&",
       "Foo[abi:abc]<int>::operator<<<Foo[abi:abc]<int>>"},

      {"auto A::operator<=>[abi:tag]<A::B>()", "A",
       "operator<=>[abi:tag]<A::B>", "()", "",
       "A::operator<=>[abi:tag]<A::B>"}};

  for (const auto &test : test_cases) {
    CPlusPlusLanguage::MethodName method(ConstString(test.input));
    EXPECT_TRUE(method.IsValid()) << test.input;
    if (method.IsValid()) {
      EXPECT_EQ(test.context, method.GetContext().str());
      EXPECT_EQ(test.basename, method.GetBasename().str());
      EXPECT_EQ(test.arguments, method.GetArguments().str());
      EXPECT_EQ(test.qualifiers, method.GetQualifiers().str());
      EXPECT_EQ(test.scope_qualified_name, method.GetScopeQualifiedName());
    }
  }
}

TEST(CPlusPlusLanguage, InvalidMethodNameParsing) {
  // Tests that we correctly reject malformed function names

  std::string test_cases[] = {
      "int Foo::operator[]<[10>()",
      "Foo::operator bool[10]()",
      "auto A::operator<<<(int)",
      "auto A::operator>>>(int)",
      "auto A::operator<<<Type[abi:tag]<>(int)",
      "auto A::operator<<<Type[abi:tag]<Type<int>>(int)",
      "auto A::foo[(int)",
      "auto A::foo[](int)",
      "auto A::foo[bar](int)",
      "auto A::foo[abi](int)",
      "auto A::foo[abi:(int)",
  };

  for (const auto &name : test_cases) {
    CPlusPlusLanguage::MethodName method{ConstString(name)};
    EXPECT_FALSE(method.IsValid()) << name;
  }
}

TEST(CPlusPlusLanguage, ContainsPath) {
  CPlusPlusLanguage::MethodName 
      reference_1(ConstString("int foo::bar::func01(int a, double b)"));
  CPlusPlusLanguage::MethodName
      reference_2(ConstString("int foofoo::bar::func01(std::string a, int b)"));
  CPlusPlusLanguage::MethodName reference_3(ConstString("int func01()"));
  CPlusPlusLanguage::MethodName 
      reference_4(ConstString("bar::baz::operator bool()"));
  CPlusPlusLanguage::MethodName reference_5(
      ConstString("bar::baz::operator bool<int, Type<double>>()"));
  CPlusPlusLanguage::MethodName reference_6(ConstString(
      "bar::baz::operator<<<Type<double>, Type<std::vector<double>>>()"));

  EXPECT_TRUE(reference_1.ContainsPath(""));
  EXPECT_TRUE(reference_1.ContainsPath("func01"));
  EXPECT_TRUE(reference_1.ContainsPath("bar::func01"));
  EXPECT_TRUE(reference_1.ContainsPath("foo::bar::func01"));
  EXPECT_FALSE(reference_1.ContainsPath("func"));
  EXPECT_FALSE(reference_1.ContainsPath("baz::func01"));
  EXPECT_FALSE(reference_1.ContainsPath("::bar::func01"));
  EXPECT_FALSE(reference_1.ContainsPath("::foo::baz::func01"));
  EXPECT_FALSE(reference_1.ContainsPath("foo::bar::baz::func01"));
  
  EXPECT_TRUE(reference_2.ContainsPath(""));
  EXPECT_TRUE(reference_2.ContainsPath("foofoo::bar::func01"));
  EXPECT_FALSE(reference_2.ContainsPath("foo::bar::func01"));
  
  EXPECT_TRUE(reference_3.ContainsPath(""));
  EXPECT_TRUE(reference_3.ContainsPath("func01"));
  EXPECT_FALSE(reference_3.ContainsPath("func"));
  EXPECT_FALSE(reference_3.ContainsPath("bar::func01"));

  EXPECT_TRUE(reference_4.ContainsPath(""));
  EXPECT_TRUE(reference_4.ContainsPath("operator"));
  EXPECT_TRUE(reference_4.ContainsPath("operator bool"));
  EXPECT_TRUE(reference_4.ContainsPath("baz::operator bool"));
  EXPECT_TRUE(reference_4.ContainsPath("bar::baz::operator bool"));
  EXPECT_FALSE(reference_4.ContainsPath("az::operator bool"));

  EXPECT_TRUE(reference_5.ContainsPath(""));
  EXPECT_TRUE(reference_5.ContainsPath("operator"));
  EXPECT_TRUE(reference_5.ContainsPath("operator bool"));
  EXPECT_TRUE(reference_5.ContainsPath("operator bool<int, Type<double>>"));
  EXPECT_FALSE(reference_5.ContainsPath("operator bool<int, double>"));
  EXPECT_FALSE(reference_5.ContainsPath("operator bool<int, Type<int>>"));

  EXPECT_TRUE(reference_6.ContainsPath(""));
  EXPECT_TRUE(reference_6.ContainsPath("operator"));
  EXPECT_TRUE(reference_6.ContainsPath("operator<<"));
  EXPECT_TRUE(reference_6.ContainsPath(
      "bar::baz::operator<<<Type<double>, Type<std::vector<double>>>()"));
  EXPECT_FALSE(reference_6.ContainsPath("operator<<<Type<double>>"));
}

TEST(CPlusPlusLanguage, ExtractContextAndIdentifier) {
  struct TestCase {
    std::string input;
    std::string context, basename;
  };

  TestCase test_cases[] = {
      {"main", "", "main"},
      {"main     ", "", "main"},
      {"foo01::bar", "foo01", "bar"},
      {"foo::~bar", "foo", "~bar"},
      {"std::vector<int>::push_back", "std::vector<int>", "push_back"},
      {"operator<<(Cls, Cls)::Subclass::function",
       "operator<<(Cls, Cls)::Subclass", "function"},
      {"std::vector<Class, std::allocator<Class>>"
       "::_M_emplace_back_aux<Class const&>",
       "std::vector<Class, std::allocator<Class>>",
       "_M_emplace_back_aux<Class const&>"},
      {"`anonymous namespace'::foo", "`anonymous namespace'", "foo"},
      {"`operator<<A>'::`2'::B<0>::operator>", "`operator<<A>'::`2'::B<0>",
       "operator>"},
      {"`anonymous namespace'::S::<<::__l2::Foo",
       "`anonymous namespace'::S::<<::__l2", "Foo"},
      // These cases are idiosyncratic in how clang generates debug info for
      // names when we have template parameters. They are not valid C++ names
      // but if we fix this we need to support them for older compilers.
      {"A::operator><A::B>", "A", "operator><A::B>"},
      {"operator><A::B>", "", "operator><A::B>"},
      {"A::operator<<A::B>", "A", "operator<<A::B>"},
      {"operator<<A::B>", "", "operator<<A::B>"},
      {"A::operator<<<A::B>", "A", "operator<<<A::B>"},
      {"operator<<<A::B>", "", "operator<<<A::B>"},
  };

  llvm::StringRef context, basename;
  for (const auto &test : test_cases) {
    EXPECT_TRUE(CPlusPlusLanguage::ExtractContextAndIdentifier(
        test.input.c_str(), context, basename));
    EXPECT_EQ(test.context, context.str());
    EXPECT_EQ(test.basename, basename.str());
  }

  EXPECT_FALSE(CPlusPlusLanguage::ExtractContextAndIdentifier("void", context,
                                                              basename));
  EXPECT_FALSE(
      CPlusPlusLanguage::ExtractContextAndIdentifier("321", context, basename));
  EXPECT_FALSE(
      CPlusPlusLanguage::ExtractContextAndIdentifier("", context, basename));
  EXPECT_FALSE(CPlusPlusLanguage::ExtractContextAndIdentifier(
      "selector:", context, basename));
  EXPECT_FALSE(CPlusPlusLanguage::ExtractContextAndIdentifier(
      "selector:otherField:", context, basename));
  EXPECT_FALSE(CPlusPlusLanguage::ExtractContextAndIdentifier(
      "abc::", context, basename));
  EXPECT_FALSE(CPlusPlusLanguage::ExtractContextAndIdentifier(
      "f<A<B><C>>", context, basename));

  EXPECT_TRUE(CPlusPlusLanguage::ExtractContextAndIdentifier(
      "A::operator<=><A::B>", context, basename));
  EXPECT_TRUE(CPlusPlusLanguage::ExtractContextAndIdentifier(
      "operator<=><A::B>", context, basename));
}

static std::vector<std::string> GenerateAlternate(llvm::StringRef Name) {
  std::vector<std::string> Strings;
  if (Language *CPlusPlusLang =
          Language::FindPlugin(lldb::eLanguageTypeC_plus_plus)) {
    std::vector<ConstString> Results =
        CPlusPlusLang->GenerateAlternateFunctionManglings(ConstString(Name));
    for (ConstString Str : Results)
      Strings.push_back(std::string(Str.GetStringRef()));
  }
  return Strings;
}

TEST(CPlusPlusLanguage, GenerateAlternateFunctionManglings) {
  using namespace testing;

  SubsystemRAII<CPlusPlusLanguage> lang;

  EXPECT_THAT(GenerateAlternate("_ZN1A1fEv"),
              UnorderedElementsAre("_ZNK1A1fEv", "_ZLN1A1fEv"));
  EXPECT_THAT(GenerateAlternate("_ZN1A1fEa"), Contains("_ZN1A1fEc"));
  EXPECT_THAT(GenerateAlternate("_ZN1A1fEx"), Contains("_ZN1A1fEl"));
  EXPECT_THAT(GenerateAlternate("_ZN1A1fEy"), Contains("_ZN1A1fEm"));
  EXPECT_THAT(GenerateAlternate("_ZN1A1fEai"), Contains("_ZN1A1fEci"));
  EXPECT_THAT(GenerateAlternate("_ZN1AC1Ev"), Contains("_ZN1AC2Ev"));
  EXPECT_THAT(GenerateAlternate("_ZN1AD1Ev"), Contains("_ZN1AD2Ev"));
  EXPECT_THAT(GenerateAlternate("_bogus"), IsEmpty());
}

TEST(CPlusPlusLanguage, CPlusPlusNameParser) {
  // Don't crash.
  CPlusPlusNameParser((const char *)nullptr);
}
