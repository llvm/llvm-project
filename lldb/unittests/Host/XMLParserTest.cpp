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

TEST(XMLParserTest, Prolog) {
  EXPECT_THAT_EXPECTED(
      ParseToString("<?xml version='1.0' encoding='UTF-8' standalone='no'?>\n"
                    "<!-- comment -->\n"
                    "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
                    "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
                    "<plist>a&external;b</plist>"),
      llvm::HasValue("<plist>ab</plist>"));
  EXPECT_THAT_EXPECTED(ParseToString("<!DOCTYPE r SYSTEM 'r.dtd'><r/>"),
                       llvm::HasValue("<r></r>"));
}

TEST(XMLParserTest, UTF16) {
  std::string text = "<?xml version='1.0' encoding='UTF-16'?><r>\xE9</r>";
  std::string utf16le = "\xFF\xFE", utf16be = "\xFE\xFF";
  for (char c : text) {
    utf16le += {c, '\0'};
    utf16be += {'\0', c};
  }
  EXPECT_THAT_EXPECTED(ParseToString(utf16le),
                       llvm::HasValue("<r>\xC3\xA9</r>"));
  EXPECT_THAT_EXPECTED(ParseToString(utf16be),
                       llvm::HasValue("<r>\xC3\xA9</r>"));
  EXPECT_THAT_EXPECTED(ParseToString("\xFF\xFE\xFF\xFE" + utf16le.substr(2)),
                       llvm::Failed());
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
      "<r>a]]>b</r>",
      "<r><!-- a -- b --></r>",
      "<r><!-- a ---></r>",
      "<r><![CDATA[</r>",
      "<r><?xml version='1.0'?></r>",
      "<r/><!DOCTYPE r>",
      " <?xml version='1.0'?><r/>",
      "<?xml?><r/>",
      "<?xml version='2.0'?><r/>",
      "<?xml encoding='UTF-8'?><r/>",
      "<?xml version='1.0' encoding='ISO-8859-1'?><r/>",
      "<?xml version='1.0' encoding='UTF-16'?><r/>",
      "<?xml version='1.0' standalone='maybe'?><r/>",
      "<?xml version='1.0' standalone='yes' encoding='UTF-8'?><r/>",
      "<!DOCTYPE r [<!ENTITY e 'x'>]><r>&e;</r>",
      "<!DOCTYPE r><r>&e;</r>",
      "<?xml version='1.0' standalone='yes'?><!DOCTYPE r SYSTEM 'r'><r>&e;</r>",
      "<!DOCTYPE r SYSTEM><r/>",
      "<!DOCTYPE r PUBLIC '{}' 'r.dtd'><r/>",
      "<!DOCTYPE r><!DOCTYPE r><r/>",
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
