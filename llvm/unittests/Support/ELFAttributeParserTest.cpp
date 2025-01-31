//===----- unittests/ELFAttributeParserTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ELFAttributeParser.h"
#include "llvm/Support/ELFAttributes.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

static const TagNameMap emptyTagNameMap;

// This class is used to test the common part of the ELF attribute section.
class AttributeHeaderParser : public ELFAttributeParser {
  Error handler(uint64_t tag, bool &handled) override {
    // Treat all attributes as handled.
    handled = true;
    return Error::success();
  }

public:
  AttributeHeaderParser(ScopedPrinter *printer)
      : ELFAttributeParser(printer, emptyTagNameMap, "test") {}
  AttributeHeaderParser() : ELFAttributeParser(emptyTagNameMap, "test") {}
};

static void testParseError(ArrayRef<uint8_t> bytes, const char *msg) {
  AttributeHeaderParser parser;
  Error e = parser.parse(bytes, llvm::endianness::little);
  EXPECT_STREQ(toString(std::move(e)).c_str(), msg);
}

TEST(AttributeHeaderParser, UnrecognizedFormatVersion) {
  static const uint8_t bytes[] = {1};
  testParseError(bytes, "unrecognized format-version: 0x1");
}

TEST(AttributeHeaderParser, InvalidSectionLength) {
  static const uint8_t bytes[] = {'A', 3, 0, 0, 0};
  testParseError(bytes, "invalid section length 3 at offset 0x1");
}

TEST(AttributeHeaderParser, UnrecognizedTag) {
  static const uint8_t bytes[] = {'A', 14, 0, 0, 0, 't', 'e', 's',
                                  't', 0,  4, 5, 0, 0,   0};
  testParseError(bytes, "unrecognized tag 0x4 at offset 0xa");
}

TEST(AttributeHeaderParser, InvalidAttributeSize) {
  static const uint8_t bytes[] = {'A', 14, 0, 0, 0, 't', 'e', 's',
                                  't', 0,  1, 4, 0, 0,   0};
  testParseError(bytes, "invalid attribute size 4 at offset 0xa");
}

class AttributeParserJSONOutput : public testing::TestWithParam<bool> {
public:
  // Accepts the contents of an ELF attribute section and parses it to
  // JSON-formatted output. The error value from the attribute parse as
  // well as its formatted output are returned.
  std::pair<Error, std::string> parse(ArrayRef<uint8_t> Section,
                                      bool PrettyPrint) {
    std::string Output;
    raw_string_ostream OS{Output};
    JSONScopedPrinter Printer{OS, PrettyPrint};
    Parser P{&Printer, emptyTagNameMap, "vendor"};
    Error Err = P.parse(Section, endianness::little);
    return std::make_pair(std::move(Err), std::move(Output));
  }

private:
  class Parser : public ELFAttributeParser {
  public:
    using ELFAttributeParser::ELFAttributeParser;

  private:
    Error handler(uint64_t, bool &Handled) override {
      Handled = false; // No custom attributes are handled.
      return Error::success();
    }
  };
};

TEST_P(AttributeParserJSONOutput, Empty) {
  const uint8_t Section[] = {
      'A',                                // format magic number
      11,  0,   0,   0,                   // section length
      'v', 'e', 'n', 'd', 'o', 'r', '\0', // vendor name
  };
  // Parse and emit JSON. Pretty-printing is controlled by the
  // test parameter.
  auto [Err, Output] = this->parse(Section, GetParam());
  EXPECT_FALSE(bool{Err}) << Output << '\n' << toString(std::move(Err));
  // Check that 'Output' is valid JSON.
  Error JsonErr = json::parse(Output).takeError();
  EXPECT_FALSE(bool{JsonErr}) << Output << '\n' << toString(std::move(JsonErr));
}

TEST_P(AttributeParserJSONOutput, SingleSubsection) {
  const uint8_t Section[] = {
      'A',                                // format magic number
      18,  0,   0,   0,                   // section length
      'v', 'e', 'n', 'd', 'o', 'r', '\0', // vendor name
      1,                                  // tag (File=1, Section=2, Symbol=3)
      7,   0,   0,   0,                   // size
      32,                                 // tag (uleb128)
      16,                                 // value (16 uleb128)
  };
  auto [Err, Output] = this->parse(Section, GetParam());
  EXPECT_FALSE(bool{Err}) << Output << '\n' << toString(std::move(Err));
  // Check that 'Output' is valid JSON.
  Error JsonErr = json::parse(Output).takeError();
  EXPECT_FALSE(bool{JsonErr}) << Output << '\n' << toString(std::move(JsonErr));
}

INSTANTIATE_TEST_SUITE_P(AttributeParserJSONOutput, AttributeParserJSONOutput,
                         testing::Values(false, true));
