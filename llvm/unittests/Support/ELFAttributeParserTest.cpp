//===----- unittests/ELFAttributeParserTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ELFAttributeParser.h"
#include "llvm/Support/ELFAttributes.h"
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
  Error e = parser.parse(bytes, support::little);
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

TEST(AttributeHeaderParser, UnrecognizedVendorName) {
  static const uint8_t bytes[] = {'A', 7, 0, 0, 0, 'x', 'y', 0};
  testParseError(bytes, "unrecognized vendor-name: xy");
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
