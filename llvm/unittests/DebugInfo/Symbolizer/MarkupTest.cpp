
//===- unittest/DebugInfo/Symbolizer/MarkupTest.cpp - Markup parser tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/Symbolize/Markup.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"

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

TEST(SymbolizerMarkup, NoLines) { EXPECT_EQ(MarkupParser{}.nextNode(), None); }

TEST(SymbolizerMarkup, LinesWithoutMarkup) {
  MarkupParser Parser;

  Parser.parseLine("text");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("text")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("discarded");
  Parser.parseLine("kept");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("kept")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{{");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{{}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{}}")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{}}}")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{{}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{}}}")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{{:field}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{:field}}}")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{{tag:");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{tag:")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{{tag:field}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{tag:field}}")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("a\033[2mb");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("a\033[2mb")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("a\033[38mb");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("a\033[38mb")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("a\033[4mb");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("a\033[4mb")));
  EXPECT_THAT(Parser.nextNode(), None);
}

TEST(SymbolizerMarkup, LinesWithMarkup) {
  MarkupParser Parser;

  Parser.parseLine("{{{tag}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{tag}}}", "tag")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{{tag:f1:f2:f3}}}");
  EXPECT_THAT(Parser.nextNode(),
              testing::Optional(isNode("{{{tag:f1:f2:f3}}}", "tag",
                                       ElementsAre("f1", "f2", "f3"))));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{{tag:}}}");
  EXPECT_THAT(Parser.nextNode(),
              testing::Optional(isNode("{{{tag:}}}", "tag", ElementsAre(""))));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{{tag:}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{tag:}}")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{{t2g}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{t2g}}}", "t2g")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{{tAg}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{tAg}}}", "tAg")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("a{{{b}}}c{{{d}}}e");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("a")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{b}}}", "b")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("c")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{d}}}", "d")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("e")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{{}}}{{{tag}}}");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{}}}")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("{{{tag}}}", "tag")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("\033[0mA\033[1mB\033[30mC\033[37m");
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("\033[0m")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("A")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("\033[1m")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("B")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("\033[30m")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("C")));
  EXPECT_THAT(Parser.nextNode(), testing::Optional(isNode("\033[37m")));
  EXPECT_THAT(Parser.nextNode(), None);

  Parser.parseLine("{{{tag:\033[0m}}}");
  EXPECT_THAT(Parser.nextNode(),
              testing::Optional(
                  isNode("{{{tag:\033[0m}}}", "tag", ElementsAre("\033[0m"))));
  EXPECT_THAT(Parser.nextNode(), None);
}

} // namespace
