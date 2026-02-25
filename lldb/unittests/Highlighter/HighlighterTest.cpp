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
#include "lldb/Host/Config.h"
#include "lldb/Host/FileSystem.h"

#if LLDB_ENABLE_TREESITTER
#include "Plugins/Highlighter/TreeSitter/Rust/RustTreeSitterHighlighter.h"
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
                SwiftTreeSitterHighlighter, RustTreeSitterHighlighter,
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
  EXPECT_EQ(getName(lldb::eLanguageTypeRust), "tree-sitter-rust");
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

static std::string
highlightRust(llvm::StringRef code, HighlightStyle style,
              std::optional<size_t> cursor = std::optional<size_t>()) {
  HighlighterManager mgr;
  const Highlighter &h =
      mgr.getHighlighterFor(lldb::eLanguageTypeRust, "main.rs");
  return h.Highlight(style, code, cursor);
}

TEST_F(HighlighterTest, RustComments) {
  HighlightStyle s;
  s.comment.Set("<cc>", "</cc>");

  EXPECT_EQ(" <cc>// I'm feeling lucky today</cc>",
            highlightRust(" // I'm feeling lucky today", s));
  EXPECT_EQ(" <cc>/* This is a\nmultiline comment */</cc>",
            highlightRust(" /* This is a\nmultiline comment */", s));
  EXPECT_EQ(" <cc>/* nested /* comment */ works */</cc>",
            highlightRust(" /* nested /* comment */ works */", s));
  EXPECT_EQ(" <cc>/// Documentation comment</cc>",
            highlightRust(" /// Documentation comment", s));
  EXPECT_EQ(" <cc>//! Inner doc comment</cc>",
            highlightRust(" //! Inner doc comment", s));
}

TEST_F(HighlighterTest, RustKeywords) {
  HighlightStyle s;
  s.keyword.Set("<k>", "</k>");

  EXPECT_EQ(" <k>let</k> x = 5;", highlightRust(" let x = 5;", s));
  EXPECT_EQ(" <k>let</k> <k>mut</k> y = 10;",
            highlightRust(" let mut y = 10;", s));
  EXPECT_EQ(" <k>fn</k> foo() { <k>return</k> 42; }",
            highlightRust(" fn foo() { return 42; }", s));
  EXPECT_EQ(" <k>struct</k> <k>Point</k> {}",
            highlightRust(" struct Point {}", s));
  EXPECT_EQ(" <k>enum</k> <k>Color</k> {}", highlightRust(" enum Color {}", s));
  EXPECT_EQ(" <k>impl</k> <k>MyStruct</k> {}",
            highlightRust(" impl MyStruct {}", s));
  EXPECT_EQ(" <k>trait</k> <k>MyTrait</k> {}",
            highlightRust(" trait MyTrait {}", s));
  EXPECT_EQ(" <k>if</k> x { }", highlightRust(" if x { }", s));
  EXPECT_EQ(" <k>for</k> i <k>in</k> 0..10 { }",
            highlightRust(" for i in 0..10 { }", s));
  EXPECT_EQ(" <k>while</k> x { }", highlightRust(" while x { }", s));
  EXPECT_EQ(" <k>match</k> x { _ => {} }",
            highlightRust(" match x { _ => {} }", s));
  EXPECT_EQ(" <k>pub</k> <k>fn</k> foo() {}",
            highlightRust(" pub fn foo() {}", s));
  EXPECT_EQ(" <k>const</k> MAX: u32 = 100;",
            highlightRust(" const MAX: u32 = 100;", s));
  EXPECT_EQ(" <k>static</k> GLOBAL: i32 = 0;",
            highlightRust(" static GLOBAL: i32 = 0;", s));
  EXPECT_EQ(" <k>if</k> <k>let</k> Some(foo) = foo_maybe {",
            highlightRust(" if let Some(foo) = foo_maybe {", s, 0));
}

TEST_F(HighlighterTest, RustStringLiterals) {
  HighlightStyle s;
  s.string_literal.Set("<str>", "</str>");

  EXPECT_EQ(" let s = <str>\"Hello, World!\"</str>;",
            highlightRust(" let s = \"Hello, World!\";", s));
  EXPECT_EQ(" let raw = <str>r\"C:\\\\path\"</str>;",
            highlightRust(" let raw = r\"C:\\\\path\";", s));
  EXPECT_EQ(" let raw2 = <str>r#\"He said \"hi\"\"#</str>;",
            highlightRust(" let raw2 = r#\"He said \"hi\"\"#;", s));
  EXPECT_EQ(" let byte_str = <str>b\"bytes\"</str>;",
            highlightRust(" let byte_str = b\"bytes\";", s));
}

TEST_F(HighlighterTest, RustScalarLiterals) {
  HighlightStyle s;
  s.scalar_literal.Set("<scalar>", "</scalar>");

  EXPECT_EQ(" let i = 42;", highlightRust(" let i = 42;", s));
  EXPECT_EQ(" let hex = 0xFF;", highlightRust(" let hex = 0xFF;", s));
  EXPECT_EQ(" let bin = 0b1010;", highlightRust(" let bin = 0b1010;", s));
  EXPECT_EQ(" let oct = 0o77;", highlightRust(" let oct = 0o77;", s));
  EXPECT_EQ(" let f = 3.14;", highlightRust(" let f = 3.14;", s));
  EXPECT_EQ(" let typed = 42u32;", highlightRust(" let typed = 42u32;", s));
  EXPECT_EQ(" let c = 'x';", highlightRust(" let c = 'x';", s));
}

TEST_F(HighlighterTest, RustIdentifiers) {
  HighlightStyle s;
  s.identifier.Set("<id>", "</id>");

  EXPECT_EQ(" let foo = <id>bar</id>();",
            highlightRust(" let foo = bar();", s));
  EXPECT_EQ(" my_variable = 10;", highlightRust(" my_variable = 10;", s));
  EXPECT_EQ(" let x: i32 = 5", highlightRust(" let x: i32 = 5", s));
  EXPECT_EQ(" fn <id>foo</id>() -> String { }",
            highlightRust(" fn foo() -> String { }", s));
  EXPECT_EQ(" fn <id>foo</id><'a>(x: &'a str) {}",
            highlightRust(" fn foo<'a>(x: &'a str) {}", s));
  EXPECT_EQ(" struct Foo<'a> { x: &'a i32 }",
            highlightRust(" struct Foo<'a> { x: &'a i32 }", s));
}

TEST_F(HighlighterTest, RustOperators) {
  HighlightStyle s;
  s.operators.Set("[", "]");

  EXPECT_EQ(" 1+2-3[*]4/5", highlightRust(" 1+2-3*4/5", s));
  EXPECT_EQ(" x && y || z", highlightRust(" x && y || z", s));
  EXPECT_EQ(" a == b != c", highlightRust(" a == b != c", s));
  EXPECT_EQ(" x [&]y", highlightRust(" x &y", s));
  EXPECT_EQ(" [*]ptr", highlightRust(" *ptr", s));
}

TEST_F(HighlighterTest, RustCursorPosition) {
  HighlightStyle s;
  s.selected.Set("<c>", "</c>");

  EXPECT_EQ("<c> </c>let x = 5;", highlightRust(" let x = 5;", s, 0));
  EXPECT_EQ(" <c>l</c>et x = 5;", highlightRust(" let x = 5;", s, 1));
  EXPECT_EQ(" l<c>e</c>t x = 5;", highlightRust(" let x = 5;", s, 2));
  EXPECT_EQ(" le<c>t</c> x = 5;", highlightRust(" let x = 5;", s, 3));
  EXPECT_EQ(" let<c> </c>x = 5;", highlightRust(" let x = 5;", s, 4));
}
#endif
