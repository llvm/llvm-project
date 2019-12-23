//===-- HighlighterTest.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Core/Highlighter.h"
#include "lldb/Host/FileSystem.h"

#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"
#include "Plugins/Language/ObjC/ObjCLanguage.h"
#include "Plugins/Language/ObjCPlusPlus/ObjCPlusPlusLanguage.h"
#include "TestingSupport/SubsystemRAII.h"

using namespace lldb_private;

namespace {
class HighlighterTest : public testing::Test {
  SubsystemRAII<FileSystem, CPlusPlusLanguage, ObjCLanguage,
                ObjCPlusPlusLanguage>
      subsystems;
};
} // namespace

static std::string getName(lldb::LanguageType type) {
  HighlighterManager m;
  return m.getHighlighterFor(type, "").GetName().str();
}

static std::string getName(llvm::StringRef path) {
  HighlighterManager m;
  return m.getHighlighterFor(lldb::eLanguageTypeUnknown, path).GetName().str();
}

TEST_F(HighlighterTest, HighlighterSelectionType) {
  EXPECT_EQ(getName(lldb::eLanguageTypeC_plus_plus), "clang");
  EXPECT_EQ(getName(lldb::eLanguageTypeC_plus_plus_03), "clang");
  EXPECT_EQ(getName(lldb::eLanguageTypeC_plus_plus_11), "clang");
  EXPECT_EQ(getName(lldb::eLanguageTypeC_plus_plus_14), "clang");
  EXPECT_EQ(getName(lldb::eLanguageTypeObjC), "clang");
  EXPECT_EQ(getName(lldb::eLanguageTypeObjC_plus_plus), "clang");

  EXPECT_EQ(getName(lldb::eLanguageTypeUnknown), "none");
  EXPECT_EQ(getName(lldb::eLanguageTypeJulia), "none");
  EXPECT_EQ(getName(lldb::eLanguageTypeHaskell), "none");
}

TEST_F(HighlighterTest, HighlighterSelectionPath) {
  EXPECT_EQ(getName("myfile.cc"), "clang");
  EXPECT_EQ(getName("moo.cpp"), "clang");
  EXPECT_EQ(getName("mar.cxx"), "clang");
  EXPECT_EQ(getName("foo.C"), "clang");
  EXPECT_EQ(getName("bar.CC"), "clang");
  EXPECT_EQ(getName("a/dir.CC"), "clang");
  EXPECT_EQ(getName("/a/dir.hpp"), "clang");
  EXPECT_EQ(getName("header.h"), "clang");

  EXPECT_EQ(getName(""), "none");
  EXPECT_EQ(getName("/dev/null"), "none");
  EXPECT_EQ(getName("Factory.java"), "none");
  EXPECT_EQ(getName("poll.py"), "none");
  EXPECT_EQ(getName("reducer.hs"), "none");
}

TEST_F(HighlighterTest, FallbackHighlighter) {
  HighlighterManager mgr;
  const Highlighter &h =
      mgr.getHighlighterFor(lldb::eLanguageTypePascal83, "foo.pas");

  HighlightStyle style;
  style.identifier.Set("[", "]");
  style.semicolons.Set("<", ">");

  const char *code = "program Hello;";
  std::string output = h.Highlight(style, code, llvm::Optional<size_t>());

  EXPECT_STREQ(output.c_str(), code);
}

static std::string
highlightDefault(llvm::StringRef code, HighlightStyle style,
                 llvm::Optional<size_t> cursor = llvm::Optional<size_t>()) {
  HighlighterManager mgr;
  return mgr.getDefaultHighlighter().Highlight(style, code, cursor);
}

TEST_F(HighlighterTest, DefaultHighlighter) {
  const char *code = "int my_main() { return 22; } \n";

  HighlightStyle style;
  EXPECT_EQ(code, highlightDefault(code, style));
}

TEST_F(HighlighterTest, DefaultHighlighterWithCursor) {
  HighlightStyle style;
  style.selected.Set("<c>", "</c>");
  EXPECT_EQ("<c>a</c> bc", highlightDefault("a bc", style, 0));
  EXPECT_EQ("a<c> </c>bc", highlightDefault("a bc", style, 1));
  EXPECT_EQ("a <c>b</c>c", highlightDefault("a bc", style, 2));
  EXPECT_EQ("a b<c>c</c>", highlightDefault("a bc", style, 3));
}

TEST_F(HighlighterTest, DefaultHighlighterWithCursorOutOfBounds) {
  HighlightStyle style;
  style.selected.Set("<c>", "</c>");
  EXPECT_EQ("a bc", highlightDefault("a bc", style, 4));
}
// Tests highlighting with the Clang highlighter.

static std::string
highlightC(llvm::StringRef code, HighlightStyle style,
           llvm::Optional<size_t> cursor = llvm::Optional<size_t>()) {
  HighlighterManager mgr;
  const Highlighter &h = mgr.getHighlighterFor(lldb::eLanguageTypeC, "main.c");
  return h.Highlight(style, code, cursor);
}

TEST_F(HighlighterTest, ClangEmptyInput) {
  HighlightStyle s;
  EXPECT_EQ("", highlightC("", s));
}

TEST_F(HighlighterTest, ClangScalarLiterals) {
  HighlightStyle s;
  s.scalar_literal.Set("<scalar>", "</scalar>");

  EXPECT_EQ(" int i = <scalar>22</scalar>;", highlightC(" int i = 22;", s));
}

TEST_F(HighlighterTest, ClangStringLiterals) {
  HighlightStyle s;
  s.string_literal.Set("<str>", "</str>");

  EXPECT_EQ("const char *f = 22 + <str>\"foo\"</str>;",
            highlightC("const char *f = 22 + \"foo\";", s));
}

TEST_F(HighlighterTest, ClangUnterminatedString) {
  HighlightStyle s;
  s.string_literal.Set("<str>", "</str>");

  EXPECT_EQ(" f = \"", highlightC(" f = \"", s));
}

TEST_F(HighlighterTest, Keywords) {
  HighlightStyle s;
  s.keyword.Set("<k>", "</k>");

  EXPECT_EQ(" <k>return</k> 1; ", highlightC(" return 1; ", s));
}

TEST_F(HighlighterTest, Colons) {
  HighlightStyle s;
  s.colon.Set("<c>", "</c>");

  EXPECT_EQ("foo<c>:</c><c>:</c>bar<c>:</c>", highlightC("foo::bar:", s));
}

TEST_F(HighlighterTest, ClangBraces) {
  HighlightStyle s;
  s.braces.Set("<b>", "</b>");

  EXPECT_EQ("a<b>{</b><b>}</b>", highlightC("a{}", s));
}

TEST_F(HighlighterTest, ClangSquareBrackets) {
  HighlightStyle s;
  s.square_brackets.Set("<sb>", "</sb>");

  EXPECT_EQ("a<sb>[</sb><sb>]</sb>", highlightC("a[]", s));
}

TEST_F(HighlighterTest, ClangCommas) {
  HighlightStyle s;
  s.comma.Set("<comma>", "</comma>");

  EXPECT_EQ(" bool f = foo()<comma>,</comma> 1;",
            highlightC(" bool f = foo(), 1;", s));
}

TEST_F(HighlighterTest, ClangPPDirectives) {
  HighlightStyle s;
  s.pp_directive.Set("<pp>", "</pp>");

  EXPECT_EQ("<pp>#</pp><pp>include</pp><pp> </pp><pp>\"foo\"</pp><pp> </pp>//c",
            highlightC("#include \"foo\" //c", s));
}

TEST_F(HighlighterTest, ClangPreserveNewLine) {
  HighlightStyle s;
  s.comment.Set("<cc>", "</cc>");

  EXPECT_EQ("<cc>//</cc>\n", highlightC("//\n", s));
}

TEST_F(HighlighterTest, ClangTrailingBackslashBeforeNewline) {
  HighlightStyle s;

  EXPECT_EQ("\\\n", highlightC("\\\n", s));
  EXPECT_EQ("\\\r\n", highlightC("\\\r\n", s));

  EXPECT_EQ("#define a \\\n", highlightC("#define a \\\n", s));
  EXPECT_EQ("#define a \\\r\n", highlightC("#define a \\\r\n", s));
  EXPECT_EQ("#define a \\\r", highlightC("#define a \\\r", s));
}

TEST_F(HighlighterTest, ClangTrailingBackslashWithWhitespace) {
  HighlightStyle s;

  EXPECT_EQ("\\  \n", highlightC("\\  \n", s));
  EXPECT_EQ("\\ \t\n", highlightC("\\ \t\n", s));
  EXPECT_EQ("\\ \n", highlightC("\\ \n", s));
  EXPECT_EQ("\\\t\n", highlightC("\\\t\n", s));

  EXPECT_EQ("#define a \\  \n", highlightC("#define a \\  \n", s));
  EXPECT_EQ("#define a \\ \t\n", highlightC("#define a \\ \t\n", s));
  EXPECT_EQ("#define a \\ \n", highlightC("#define a \\ \n", s));
  EXPECT_EQ("#define a \\\t\n", highlightC("#define a \\\t\n", s));
}

TEST_F(HighlighterTest, ClangTrailingBackslashMissingNewLine) {
  HighlightStyle s;
  EXPECT_EQ("\\", highlightC("\\", s));
  EXPECT_EQ("#define a\\", highlightC("#define a\\", s));
}

TEST_F(HighlighterTest, ClangComments) {
  HighlightStyle s;
  s.comment.Set("<cc>", "</cc>");

  EXPECT_EQ(" <cc>/*com */</cc> <cc>// com /*n*/</cc>",
            highlightC(" /*com */ // com /*n*/", s));
}

TEST_F(HighlighterTest, ClangOperators) {
  HighlightStyle s;
  s.operators.Set("[", "]");

  EXPECT_EQ(" 1[+]2[/]a[*]f[&]x[|][~]l", highlightC(" 1+2/a*f&x|~l", s));
}

TEST_F(HighlighterTest, ClangIdentifiers) {
  HighlightStyle s;
  s.identifier.Set("<id>", "</id>");

  EXPECT_EQ(" <id>foo</id> <id>c</id> = <id>bar</id>(); return 1;",
            highlightC(" foo c = bar(); return 1;", s));
}

TEST_F(HighlighterTest, ClangCursorPos) {
  HighlightStyle s;
  s.selected.Set("<c>", "</c>");

  EXPECT_EQ("<c> </c>foo c = bar(); return 1;",
            highlightC(" foo c = bar(); return 1;", s, 0));
  EXPECT_EQ(" <c>foo</c> c = bar(); return 1;",
            highlightC(" foo c = bar(); return 1;", s, 1));
  EXPECT_EQ(" <c>foo</c> c = bar(); return 1;",
            highlightC(" foo c = bar(); return 1;", s, 2));
  EXPECT_EQ(" <c>foo</c> c = bar(); return 1;",
            highlightC(" foo c = bar(); return 1;", s, 3));
  EXPECT_EQ(" foo<c> </c>c = bar(); return 1;",
            highlightC(" foo c = bar(); return 1;", s, 4));
  EXPECT_EQ(" foo <c>c</c> = bar(); return 1;",
            highlightC(" foo c = bar(); return 1;", s, 5));
}

TEST_F(HighlighterTest, ClangCursorPosEndOfLine) {
  HighlightStyle s;
  s.selected.Set("<c>", "</c>");

  EXPECT_EQ("f", highlightC("f", s, 1));
}

TEST_F(HighlighterTest, ClangCursorOutOfBounds) {
  HighlightStyle s;
  s.selected.Set("<c>", "</c>");
  EXPECT_EQ("f", highlightC("f", s, 2));
  EXPECT_EQ("f", highlightC("f", s, 3));
  EXPECT_EQ("f", highlightC("f", s, 4));
}

TEST_F(HighlighterTest, ClangCursorPosBeforeOtherToken) {
  HighlightStyle s;
  s.selected.Set("<c>", "</c>");
  s.identifier.Set("<id>", "</id>");

  EXPECT_EQ("<c> </c><id>foo</id> <id>c</id> = <id>bar</id>(); return 1;",
            highlightC(" foo c = bar(); return 1;", s, 0));
}

TEST_F(HighlighterTest, ClangCursorPosAfterOtherToken) {
  HighlightStyle s;
  s.selected.Set("<c>", "</c>");
  s.identifier.Set("<id>", "</id>");

  EXPECT_EQ(" <id>foo</id><c> </c><id>c</id> = <id>bar</id>(); return 1;",
            highlightC(" foo c = bar(); return 1;", s, 4));
}

TEST_F(HighlighterTest, ClangCursorPosInOtherToken) {
  HighlightStyle s;
  s.selected.Set("<c>", "</c>");
  s.identifier.Set("<id>", "</id>");

  EXPECT_EQ(" <id><c>foo</c></id> <id>c</id> = <id>bar</id>(); return 1;",
            highlightC(" foo c = bar(); return 1;", s, 1));
  EXPECT_EQ(" <id><c>foo</c></id> <id>c</id> = <id>bar</id>(); return 1;",
            highlightC(" foo c = bar(); return 1;", s, 2));
  EXPECT_EQ(" <id><c>foo</c></id> <id>c</id> = <id>bar</id>(); return 1;",
            highlightC(" foo c = bar(); return 1;", s, 3));
}
