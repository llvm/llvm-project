//===-- StringPrinterTests.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>

using namespace lldb;
using namespace lldb_private;
using lldb_private::formatters::StringPrinter;
using llvm::Optional;
using llvm::StringRef;

#define QUOTE(x) std::string("\"" x "\"")

/// Format \p input according to the specified string encoding and special char
/// escape style.
template <StringPrinter::StringElementType elem_ty>
static Optional<std::string> format(StringRef input,
                                    StringPrinter::EscapeStyle escape_style) {
  StreamString out;
  StringPrinter::ReadBufferAndDumpToStreamOptions opts;
  opts.SetStream(&out);
  opts.SetSourceSize(input.size());
  opts.SetNeedsZeroTermination(true);
  opts.SetEscapeNonPrintables(true);
  opts.SetIgnoreMaxLength(false);
  opts.SetEscapeStyle(escape_style);
  opts.SetData(DataExtractor(input.data(), input.size(),
                             endian::InlHostByteOrder(), sizeof(void *)));
  const bool success = StringPrinter::ReadBufferAndDumpToStream<elem_ty>(opts);
  if (!success)
    return std::nullopt;
  return out.GetString().str();
}

// Test ASCII formatting for C++. This behaves exactly like UTF8 formatting for
// C++, although that's questionable (see FIXME in StringPrinter.cpp).
TEST(StringPrinterTests, CxxASCII) {
  auto fmt = [](StringRef str) {
    return format<StringPrinter::StringElementType::ASCII>(
        str, StringPrinter::EscapeStyle::CXX);
  };

  // Special escapes.
  EXPECT_EQ(fmt({"\0", 1}), QUOTE(""));
  EXPECT_EQ(fmt("\a"), QUOTE(R"(\a)"));
  EXPECT_EQ(fmt("\b"), QUOTE(R"(\b)"));
  EXPECT_EQ(fmt("\f"), QUOTE(R"(\f)"));
  EXPECT_EQ(fmt("\n"), QUOTE(R"(\n)"));
  EXPECT_EQ(fmt("\r"), QUOTE(R"(\r)"));
  EXPECT_EQ(fmt("\t"), QUOTE(R"(\t)"));
  EXPECT_EQ(fmt("\v"), QUOTE(R"(\v)"));
  EXPECT_EQ(fmt("\""), QUOTE(R"(\")"));
  EXPECT_EQ(fmt("\'"), QUOTE(R"(')"));
  EXPECT_EQ(fmt("\\"), QUOTE(R"(\\)"));

  // Printable characters.
  EXPECT_EQ(fmt("'"), QUOTE("'"));
  EXPECT_EQ(fmt("a"), QUOTE("a"));
  EXPECT_EQ(fmt("Z"), QUOTE("Z"));
  EXPECT_EQ(fmt("🥑"), QUOTE("🥑"));

  // Octal (\nnn), hex (\xnn), extended octal (\unnnn or \Unnnnnnnn).
  EXPECT_EQ(fmt("\uD55C"), QUOTE("\uD55C"));
  EXPECT_EQ(fmt("\U00010348"), QUOTE("\U00010348"));

  EXPECT_EQ(fmt("\376"), QUOTE(R"(\xfe)")); // \376 is 254 in decimal.
  EXPECT_EQ(fmt("\xfe"), QUOTE(R"(\xfe)")); // \xfe is 254 in decimal.
}

// Test UTF8 formatting for C++.
TEST(StringPrinterTests, CxxUTF8) {
  auto fmt = [](StringRef str) {
    return format<StringPrinter::StringElementType::UTF8>(
        str, StringPrinter::EscapeStyle::CXX);
  };

  // Special escapes.
  EXPECT_EQ(fmt({"\0", 1}), QUOTE(""));
  EXPECT_EQ(fmt("\a"), QUOTE(R"(\a)"));
  EXPECT_EQ(fmt("\b"), QUOTE(R"(\b)"));
  EXPECT_EQ(fmt("\f"), QUOTE(R"(\f)"));
  EXPECT_EQ(fmt("\n"), QUOTE(R"(\n)"));
  EXPECT_EQ(fmt("\r"), QUOTE(R"(\r)"));
  EXPECT_EQ(fmt("\t"), QUOTE(R"(\t)"));
  EXPECT_EQ(fmt("\v"), QUOTE(R"(\v)"));
  EXPECT_EQ(fmt("\""), QUOTE(R"(\")"));
  EXPECT_EQ(fmt("\'"), QUOTE(R"(')"));
  EXPECT_EQ(fmt("\\"), QUOTE(R"(\\)"));

  // Printable characters.
  EXPECT_EQ(fmt("'"), QUOTE("'"));
  EXPECT_EQ(fmt("a"), QUOTE("a"));
  EXPECT_EQ(fmt("Z"), QUOTE("Z"));
  EXPECT_EQ(fmt("🥑"), QUOTE("🥑"));

  // Octal (\nnn), hex (\xnn), extended octal (\unnnn or \Unnnnnnnn).
  EXPECT_EQ(fmt("\uD55C"), QUOTE("\uD55C"));
  EXPECT_EQ(fmt("\U00010348"), QUOTE("\U00010348"));

  EXPECT_EQ(fmt("\376"), QUOTE(R"(\xfe)")); // \376 is 254 in decimal.
  EXPECT_EQ(fmt("\xfe"), QUOTE(R"(\xfe)")); // \xfe is 254 in decimal.
}

// Test UTF8 formatting for Swift.
TEST(StringPrinterTests, SwiftUTF8) {
  auto fmt = [](StringRef str) {
    return format<StringPrinter::StringElementType::UTF8>(
        str, StringPrinter::EscapeStyle::Swift);
  };

  // Special escapes.
  EXPECT_EQ(fmt({"\0", 1}), QUOTE(""));
  EXPECT_EQ(fmt("\a"), QUOTE(R"(\a)"));
  EXPECT_EQ(fmt("\b"), QUOTE(R"(\u{8})"));
  EXPECT_EQ(fmt("\f"), QUOTE(R"(\u{c})"));
  EXPECT_EQ(fmt("\n"), QUOTE(R"(\n)"));
  EXPECT_EQ(fmt("\r"), QUOTE(R"(\r)"));
  EXPECT_EQ(fmt("\t"), QUOTE(R"(\t)"));
  EXPECT_EQ(fmt("\v"), QUOTE(R"(\u{b})"));
  EXPECT_EQ(fmt("\""), QUOTE(R"(\")"));
  EXPECT_EQ(fmt("\'"), QUOTE(R"(\')"));
  EXPECT_EQ(fmt("\\"), QUOTE(R"(\\)"));

  // Printable characters.
  EXPECT_EQ(fmt("'"), QUOTE(R"(\')"));
  EXPECT_EQ(fmt("a"), QUOTE("a"));
  EXPECT_EQ(fmt("Z"), QUOTE("Z"));
  EXPECT_EQ(fmt("🥑"), QUOTE("🥑"));

  // Octal (\nnn), hex (\xnn), extended octal (\unnnn or \Unnnnnnnn).
  EXPECT_EQ(fmt("\uD55C"), QUOTE("\uD55C"));
  EXPECT_EQ(fmt("\U00010348"), QUOTE("\U00010348"));

  EXPECT_EQ(fmt("\376"), QUOTE(R"(\u{fe})")); // \376 is 254 in decimal.
  EXPECT_EQ(fmt("\xfe"), QUOTE(R"(\u{fe})")); // \xfe is 254 in decimal.
}
