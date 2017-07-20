//===-- CPlusPlusLanguageTest.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "gtest/gtest.h"

#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"

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
      {"main::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const",
       "main::{lambda()#1}::operator()() const::{lambda()#1}", "operator()", "()", "const",
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
       "f<A<operator<(X,Y)::Subclass>, sizeof(B)<sizeof(C)>"}};

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
       "_M_emplace_back_aux<Class const&>"}};

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
}
