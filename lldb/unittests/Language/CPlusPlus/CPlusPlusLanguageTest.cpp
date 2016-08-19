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

TEST(CPlusPlusLanguage, MethodName)
{
    struct TestCase {
        std::string input;
        std::string context, basename, arguments, qualifiers, scope_qualified_name;
    };

    TestCase test_cases[] = {{"foo::bar(baz)", "foo", "bar", "(baz)", "", "foo::bar"},
                             {"std::basic_ostream<char, std::char_traits<char> >& "
                              "std::operator<<<std::char_traits<char> >"
                              "(std::basic_ostream<char, std::char_traits<char> >&, char const*)",
                              "std", "operator<<<std::char_traits<char> >",
                              "(std::basic_ostream<char, std::char_traits<char> >&, char const*)", "",
                              "std::operator<<<std::char_traits<char> >"}};

    for (const auto &test: test_cases)
    {
        CPlusPlusLanguage::MethodName method(ConstString(test.input));
        EXPECT_TRUE(method.IsValid());
        EXPECT_EQ(test.context, method.GetContext());
        EXPECT_EQ(test.basename, method.GetBasename());
        EXPECT_EQ(test.arguments, method.GetArguments());
        EXPECT_EQ(test.qualifiers, method.GetQualifiers());
        EXPECT_EQ(test.scope_qualified_name, method.GetScopeQualifiedName());
    }
}
