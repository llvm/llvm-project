//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/XMLParser.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::xml;

namespace {
class Recorder : public Handler {
public:
  std::string output;

  void StartElement(llvm::StringRef name,
                    llvm::ArrayRef<Attribute> attributes) override {
    output += "<" + name.str();
    for (const Attribute &attr : attributes)
      output += " " + attr.name.str() + "=\"" + attr.value.str() + "\"";
    output += ">";
  }
  void EndElement(llvm::StringRef name) override {
    output += "</" + name.str() + ">";
  }
  void Characters(llvm::StringRef text) override { output += text; }
};
} // namespace

static llvm::Expected<std::string> ParseToString(llvm::StringRef input) {
  Recorder recorder;
  if (llvm::Error error = Parse(input, recorder))
    return std::move(error);
  return recorder.output;
}

TEST(XMLParserTest, Content) {
  EXPECT_THAT_EXPECTED(
      ParseToString("<!-- comment --><?pi data?>\n"
                    "<root a='1' b=\"2\"><first/>text<second>value</second>"
                    "<?pi?><!---->\n</root >\n<!-- trailing -->\n"),
      llvm::HasValue("<root a=\"1\" b=\"2\"><first></first>text<second>value"
                     "</second>\n</root>"));
}

TEST(XMLParserTest, CharacterData) {
  EXPECT_THAT_EXPECTED(
      ParseToString("<r a=' &lt;&#x41;&#66;\t\r\nx' b='&quot;&apos;&gt;&amp;'>"
                    "1&lt;2\r\n3\r4<![CDATA[<&>]]>&#x10000;</r>"),
      llvm::HasValue(
          "<r a=\" <AB  x\" b=\"\"'>&\">1<2\n3\n4<&>\xF0\x90\x80\x80</r>"));
  EXPECT_THAT_EXPECTED(ParseToString("\xEF\xBB\xBF<r>\xC3\xA9</r>"),
                       llvm::HasValue("<r>\xC3\xA9</r>"));
}

TEST(XMLParserTest, CharactersAreConcatenated) {
  class Collector : public Handler {
  public:
    std::vector<std::string> texts;
    void Characters(llvm::StringRef text) override {
      texts.push_back(text.str());
    }
  } collector;
  EXPECT_THAT_ERROR(Parse("<r>a&amp;b<![CDATA[<c>]]><!-- x --><?pi?>d&#x41;"
                          "<e/>f</r>",
                          collector),
                    llvm::Succeeded());
  EXPECT_EQ(collector.texts, (std::vector<std::string>{"a&b<c>dA", "f"}));
}

TEST(XMLParserTest, DeepNesting) {
  const size_t depth = 1000000;
  std::string input;
  for (size_t i = 0; i < depth; ++i)
    input += "<a>";
  for (size_t i = 0; i < depth; ++i)
    input += "</a>";
  Handler handler;
  EXPECT_THAT_ERROR(Parse(input, handler), llvm::Succeeded());
}

TEST(XMLParserTest, LineNumbers) {
  Handler handler;
  EXPECT_THAT_ERROR(
      Parse("<r>\n<a>\n</b></r>", handler),
      llvm::FailedWithMessage("line 3: end tag 'b' does not match 'a'"));
  EXPECT_THAT_ERROR(
      Parse("<r>\r\n<a>\r</b></r>", handler),
      llvm::FailedWithMessage("line 3: end tag 'b' does not match 'a'"));
  EXPECT_THAT_ERROR(Parse("<r>\r\n\r\x01</r>", handler),
                    llvm::FailedWithMessage("line 3: invalid character U+1"));
}

TEST(XMLParserTest, NotWellFormed) {
  const char *inputs[] = {
      "",
      " ",
      "text",
      "<r>",
      "<r></s>",
      "<r/><r/>",
      "<r/>text",
      "<1r/>",
      "<r a='1' a='2'/>",
      "<r a='1'b='2'/>",
      "<r a=1/>",
      "<r a/>",
      "<r a='<'/>",
      "<r a='1>",
      "<r>&undeclared;</r>",
      "<r>&lt</r>",
      "<r>&#0;</r>",
      "<r>&#xD800;</r>",
      "<r>&#x;</r>",
      "<r>&#X41;</r>",
      "<r>&#+65;</r>",
      "<r>&#6a;</r>",
      "<r>&#x110000;</r>",
      "<r>&#99999999999;</r>",
      "<r>a]]>b</r>",
      "<r><!-- a -- b --></r>",
      "<r><!-- a ---></r>",
      "<r><![CDATA[</r>",
      "<r><?xml version='1.0'?></r>",
      "<r><?pi</r>",
      "<r>\x01</r>",
      "<r>\xFF</r>",
      "<r>\xC0\xAF</r>",
      "<r>\xEF\xBF\xBE</r>",
  };
  for (const char *input : inputs) {
    SCOPED_TRACE(input);
    EXPECT_THAT_EXPECTED(ParseToString(input), llvm::Failed());
  }
}
