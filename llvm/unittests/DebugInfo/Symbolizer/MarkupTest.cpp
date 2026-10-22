
//===- unittest/DebugInfo/Symbolizer/MarkupTest.cpp - Markup parser tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/Symbolize/Markup.h"
#include "llvm/DebugInfo/Symbolize/MarkupFilter.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using namespace llvm;
using namespace llvm::symbolize;
using namespace testing;

Matcher<MarkupNode> isNode(StringRef Text, StringRef Tag = "",
                           Matcher<SmallVector<StringRef>> Fields = IsEmpty()) {
  return AllOf(Field("Text", &MarkupNode::Text, Text),
               Field("Tag", &MarkupNode::Tag, Tag),
               Field("Fields", &MarkupNode::Fields, Fields));
}

TEST(SymbolizerMarkup, NoLines) {
  EXPECT_EQ(MarkupParser{}.nextNode(), std::nullopt);
}

TEST(SymbolizerMarkup, LinesWithoutMarkup) {
  MarkupParser Parser;

  Parser.parseLine("text");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("text")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("discarded");
  Parser.parseLine("kept");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("kept")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("text\n");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("text\n")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("text\r\n");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("text\r\n")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{}}")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{}}}")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{}}}")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{:field}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{:field}}}")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{tag:");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{tag:")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{tag:field}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{tag:field}}")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("a\033[2mb");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("a\033[2mb")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("a\033[38mb");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("a\033[38mb")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("a\033[4mb");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("a\033[4mb")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);
}

TEST(SymbolizerMarkup, LinesWithMarkup) {
  MarkupParser Parser;

  Parser.parseLine("{{{tag}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{tag}}}", "tag")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{tag:f1:f2:f3}}}");
  EXPECT_THAT(Parser.nextNode(),
              testing::Optional(isNode("{{{tag:f1:f2:f3}}}", "tag",
                                       ElementsAre("f1", "f2", "f3"))));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{tag:}}}");
  EXPECT_THAT(Parser.nextNode(),
              testing::Optional(isNode("{{{tag:}}}", "tag", ElementsAre(""))));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{tag:}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{tag:}}")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{t2g}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{t2g}}}", "t2g")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{tAg}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{tAg}}}", "tAg")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("a{{{b}}}c{{{d}}}e");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("a")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{b}}}", "b")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("c")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{d}}}", "d")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("e")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{}}}{{{tag}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{}}}")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{tag}}}", "tag")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("\033[0mA\033[1mB\033[30mC\033[37m");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("\033[0m")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("A")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("\033[1m")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("B")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("\033[30m")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("C")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("\033[37m")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{tag:\033[0m}}}");
  EXPECT_THAT(Parser.nextNode(),
              testing::Optional(
                  isNode("{{{tag:\033[0m}}}", "tag", ElementsAre("\033[0m"))));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);
}

TEST(SymbolizerMarkup, MultilineElements) {
  MarkupParser Parser(/*MultilineTags=*/{"first", "second"});

  Parser.parseLine("{{{tag:");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{tag:")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{first:");
  EXPECT_THAT(Parser.nextNode(), std::nullopt);
  Parser.parseLine("}}}{{{second:");
  EXPECT_THAT(
      Parser.nextNode(),
      testing::Optional(isNode("{{{first:}}}", "first", ElementsAre(""))));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);
  Parser.parseLine("}}}");
  EXPECT_THAT(
      Parser.nextNode(),
      testing::Optional(isNode("{{{second:}}}", "second", ElementsAre(""))));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{before{{{first:");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{before")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);
  Parser.parseLine("line");
  EXPECT_THAT(Parser.nextNode(), std::nullopt);
  Parser.parseLine("}}}after");
  EXPECT_THAT(Parser.nextNode(),
              testing::Optional(
                  isNode("{{{first:line}}}", "first", ElementsAre("line"))));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("after")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{first:");
  EXPECT_THAT(Parser.nextNode(), std::nullopt);
  Parser.flush();
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{first:")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{first:\n");
  EXPECT_THAT(Parser.nextNode(), std::nullopt);
  Parser.parseLine("}}}\n");
  EXPECT_THAT(
      Parser.nextNode(),
      testing::Optional(isNode("{{{first:\n}}}", "first", ElementsAre("\n"))));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("\n")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{first:\r\n");
  EXPECT_THAT(Parser.nextNode(), std::nullopt);
  Parser.parseLine("}}}\r\n");
  EXPECT_THAT(Parser.nextNode(),
              testing::Optional(
                  isNode("{{{first:\r\n}}}", "first", ElementsAre("\r\n"))));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("\r\n")));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);

  Parser.parseLine("{{{first:");
  EXPECT_THAT(Parser.nextNode(), std::nullopt);
  Parser.parseLine("\033[0m}}}");
  EXPECT_THAT(Parser.nextNode(),
              testing::Optional(isNode("{{{first:\033[0m}}}", "first",
                                       ElementsAre("\033[0m"))));
  EXPECT_THAT(Parser.nextNode(), std::nullopt);
}


// Non-markup fragments are eagerly flushed before any newline arrives.
TEST(MarkupFilter, EagerFragmentFlush) {
  std::string Out;
  raw_string_ostream OS(Out);
  LLVMSymbolizer Symbolizer;
  MarkupFilter Filter(OS, Symbolizer, /*ColorsEnabled=*/false);

  Filter.filter("hello ");
  EXPECT_EQ(Out, "hello ");

  Filter.filter("world");
  EXPECT_EQ(Out, "hello world");

  // Newline triggers complete-line processing.
  Filter.filter("\n");
  EXPECT_EQ(Out, "hello world\n");
}

// A fragment ending with '{', '{{', or containing '{{{' may be the start of a
// markup element and is held for more input.
TEST(MarkupFilter, HoldMarkupFragment) {
  std::string Out;
  raw_string_ostream OS(Out);
  LLVMSymbolizer Symbolizer;
  MarkupFilter Filter(OS, Symbolizer, /*ColorsEnabled=*/false);

  // Trailing '{' held (could be start of '{{{').
  Filter.filter("text {");
  EXPECT_EQ(Out, "");

  // Trailing "{{"" still held.
  Filter.filter("{");
  EXPECT_EQ(Out, "");

  // "{{{" present but still no newline. Should still be held.
  Filter.filter("{unknown:");
  EXPECT_EQ(Out, "");

  // Completing to a non-special element on a full line flushes everything.
  Filter.filter("foo}}}\n");
  EXPECT_EQ(Out, "text {{{unknown:foo}}}\n");
}

// Multiple complete lines in a single filter() call are all processed.
TEST(MarkupFilter, MultipleCompleteLines) {
  std::string Out;
  raw_string_ostream OS(Out);
  LLVMSymbolizer Symbolizer;
  MarkupFilter Filter(OS, Symbolizer, /*ColorsEnabled=*/false);

  Filter.filter("line1\nline2\nline3\n");
  EXPECT_EQ(Out, "line1\nline2\nline3\n");
}

// A complete line followed by a non-markup fragment: the line is processed and
// the fragment is eagerly flushed in the same call.
TEST(MarkupFilter, CompleteLineThenFragment) {
  std::string Out;
  raw_string_ostream OS(Out);
  LLVMSymbolizer Symbolizer;
  MarkupFilter Filter(OS, Symbolizer, /*ColorsEnabled=*/false);

  Filter.filter("line1\nfragment");
  EXPECT_EQ(Out, "line1\nfragment");
}

// finish() processes a held markup fragment by appending a synthetic newline.
TEST(MarkupFilter, FinishProcessesHeldFragment) {
  std::string Out;
  raw_string_ostream OS(Out);
  LLVMSymbolizer Symbolizer;
  MarkupFilter Filter(OS, Symbolizer, /*ColorsEnabled=*/false);

  Filter.filter("pre {{{unknown:foo}}}");
  EXPECT_EQ(Out, "");

  Filter.finish();
  EXPECT_EQ(Out, "pre {{{unknown:foo}}}\n");
}

} // namespace
