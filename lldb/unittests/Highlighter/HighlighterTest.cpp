//===-- HighlighterTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/Highlighter/Clang/ClangHighlighter.h"
#include "Plugins/Highlighter/Default/DefaultHighlighter.h"
#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"
#include "Plugins/Language/ObjC/ObjCLanguage.h"
#include "Plugins/Language/ObjCPlusPlus/ObjCPlusPlusLanguage.h"
#include "lldb/Core/Highlighter.h"
#include "lldb/Host/FileSystem.h"

#if LLDB_ENABLE_TREESITTER
#include "Plugins/Highlighter/TreeSitter/Swift/SwiftTreeSitterHighlighter.h"
#endif

#include "TestingSupport/SubsystemRAII.h"
#include <optional>

using namespace lldb_private;

namespace {
class HighlighterTest : public testing::Test {
  // We need the language plugins for detecting the language based on the
  // filename.
  SubsystemRAII<FileSystem, ClangHighlighter,
#if LLDB_ENABLE_TREESITTER
                SwiftTreeSitterHighlighter,
#endif
                DefaultHighlighter, CPlusPlusLanguage, ObjCLanguage,
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

#if LLDB_ENABLE_TREESITTER
  EXPECT_EQ(getName(lldb::eLanguageTypeSwift), "tree-sitter-swift");
#endif

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
  EXPECT_EQ(getName("foo.m"), "clang");
  EXPECT_EQ(getName("foo.mm"), "clang");

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
  std::string output = h.Highlight(style, code, std::optional<size_t>());

  EXPECT_STREQ(output.c_str(), code);
}

static std::string
highlightDefault(llvm::StringRef code, HighlightStyle style,
                 std::optional<size_t> cursor = std::optional<size_t>()) {
  HighlighterManager mgr;
  return mgr.getHighlighterFor(lldb::LanguageType::eLanguageTypeUnknown, "")
      .Highlight(style, code, cursor);
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
           std::optional<size_t> cursor = std::optional<size_t>()) {
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

  EXPECT_EQ("foo<c>::</c>bar<c>:</c>", highlightC("foo::bar:", s));
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

#if LLDB_ENABLE_TREESITTER
static std::string
highlightSwift(llvm::StringRef code, HighlightStyle style,
               std::optional<size_t> cursor = std::optional<size_t>()) {
  HighlighterManager mgr;
  const Highlighter &h =
      mgr.getHighlighterFor(lldb::eLanguageTypeSwift, "main.swift");
  return h.Highlight(style, code, cursor);
}

TEST_F(HighlighterTest, SwiftComments) {
  HighlightStyle s;
  s.comment.Set("<cc>", "</cc>");

  EXPECT_EQ(" <cc>// I'm feeling lucky today</cc>",
            highlightSwift(" // I'm feeling lucky today", s));
  EXPECT_EQ(" <cc>/* This is a\nmultiline comment */</cc>",
            highlightSwift(" /* This is a\nmultiline comment */", s));
  EXPECT_EQ(" <cc>/* nested /* comment */ works */</cc>",
            highlightSwift(" /* nested /* comment */ works */", s));
}

TEST_F(HighlighterTest, SwiftKeywords) {
  HighlightStyle s;
  s.keyword.Set("<k>", "</k>");

  EXPECT_EQ(" <k>let</k> x = 5;", highlightSwift(" let x = 5;", s));
  EXPECT_EQ(" <k>var</k> y = 10;", highlightSwift(" var y = 10;", s));
  EXPECT_EQ(" func foo() { return 42; }",
            highlightSwift(" func foo() { return 42; }", s));
  EXPECT_EQ(" class <k>MyClass</k> {}", highlightSwift(" class MyClass {}", s));
  EXPECT_EQ(" struct <k>Point</k> {}", highlightSwift(" struct Point {}", s));
  EXPECT_EQ(" enum <k>Color</k> {}", highlightSwift(" enum Color {}", s));
  EXPECT_EQ(" if x { }", highlightSwift(" if x { }", s));
  EXPECT_EQ(" for i in 0...10 { }", highlightSwift(" for i in 0...10 { }", s));
  EXPECT_EQ(" guard <k>let</k> x = optional <k>else</k> { }",
            highlightSwift(" guard let x = optional else { }", s));
}

TEST_F(HighlighterTest, SwiftStringLiterals) {
  HighlightStyle s;
  s.string_literal.Set("<str>", "</str>");

  EXPECT_EQ(" let s = <str>\"</str><str>Hello, World!</str><str>\"</str>;",
            highlightSwift(" let s = \"Hello, World!\";", s));
  EXPECT_EQ(" let multi = <str>\"\"\"</str><str>\nLine 1\nLine "
            "2\n</str><str>\"\"\"</str>;",
            highlightSwift(" let multi = \"\"\"\nLine 1\nLine 2\n\"\"\";", s));
}

TEST_F(HighlighterTest, SwiftScalarLiterals) {
  HighlightStyle s;
  s.scalar_literal.Set("<scalar>", "</scalar>");

  EXPECT_EQ(" let i = <scalar>42</scalar>;", highlightSwift(" let i = 42;", s));
  EXPECT_EQ(" let hex = <scalar>0xFF</scalar>;",
            highlightSwift(" let hex = 0xFF;", s));
}

TEST_F(HighlighterTest, SwiftIdentifiers) {
  HighlightStyle s;
  s.identifier.Set("<id>", "</id>");

  EXPECT_EQ(" let <id>foo</id> = <id>bar</id>();",
            highlightSwift(" let foo = bar();", s));
  EXPECT_EQ(" <id>myVariable</id> = 10;",
            highlightSwift(" myVariable = 10;", s));
  EXPECT_EQ(" optional?.<id>property</id>",
            highlightSwift(" optional?.property", s));
  EXPECT_EQ(" @available(*, <id>deprecated</id>)",
            highlightSwift(" @available(*, deprecated)", s));
  EXPECT_EQ(" @objc func <id>foo</id>() { }",
            highlightSwift(" @objc func foo() { }", s));
  EXPECT_EQ(" let <id>x</id>: Int = 5", highlightSwift(" let x: Int = 5", s));
  EXPECT_EQ(" func <id>foo</id>() -> String { }",
            highlightSwift(" func foo() -> String { }", s));
}

TEST_F(HighlighterTest, SwiftOperators) {
  HighlightStyle s;
  s.operators.Set("[", "]");

  EXPECT_EQ(" 1[+]2[-]3[*]4[/]5", highlightSwift(" 1+2-3*4/5", s));
  EXPECT_EQ(" x [&&] y [||] z", highlightSwift(" x && y || z", s));
  EXPECT_EQ(" a [==] b [!=] c", highlightSwift(" a == b != c", s));
}

TEST_F(HighlighterTest, SwiftCursorPosition) {
  HighlightStyle s;
  s.selected.Set("<c>", "</c>");

  EXPECT_EQ("<c> </c>let x = 5;", highlightSwift(" let x = 5;", s, 0));
  EXPECT_EQ(" <c>l</c>et x = 5;", highlightSwift(" let x = 5;", s, 1));
  EXPECT_EQ(" l<c>e</c>t x = 5;", highlightSwift(" let x = 5;", s, 2));
  EXPECT_EQ(" le<c>t</c> x = 5;", highlightSwift(" let x = 5;", s, 3));
  EXPECT_EQ(" let<c> </c>x = 5;", highlightSwift(" let x = 5;", s, 4));
}

TEST_F(HighlighterTest, SwiftClosures) {
  HighlightStyle s;
  s.keyword.Set("<k>", "</k>");

  EXPECT_EQ(" <k>let</k> closure = { (x: <k>Int</k>) in return x * 2 }",
            highlightSwift(" let closure = { (x: Int) in return x * 2 }", s));
}
#endif
