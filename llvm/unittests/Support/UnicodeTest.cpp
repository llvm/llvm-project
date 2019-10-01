//===- unittests/Support/UnicodeTest.cpp - Unicode.h tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Unicode.h"
#include "gtest/gtest.h"

namespace llvm {
namespace sys {
namespace unicode {
namespace {

TEST(Unicode, columnWidthUTF8) {
  EXPECT_EQ(0, columnWidthUTF8(""));
  EXPECT_EQ(1, columnWidthUTF8(" "));
  EXPECT_EQ(1, columnWidthUTF8("a"));
  EXPECT_EQ(1, columnWidthUTF8("~"));

  EXPECT_EQ(6, columnWidthUTF8("abcdef"));

  EXPECT_EQ(-1, columnWidthUTF8("\x01"));
  EXPECT_EQ(-1, columnWidthUTF8("aaaaaaaaaa\x01"));
  EXPECT_EQ(-1, columnWidthUTF8("\342\200\213")); // 200B ZERO WIDTH SPACE

  // 00AD SOFT HYPHEN is displayed on most terminals as a space or a dash. Some
  // text editors display it only when a line is broken at it, some use it as a
  // line-break hint, but don't display. We choose terminal-oriented
  // interpretation.
  EXPECT_EQ(1, columnWidthUTF8("\302\255"));

  EXPECT_EQ(0, columnWidthUTF8("\314\200"));     // 0300 COMBINING GRAVE ACCENT
  EXPECT_EQ(1, columnWidthUTF8("\340\270\201")); // 0E01 THAI CHARACTER KO KAI
  EXPECT_EQ(2, columnWidthUTF8("\344\270\200")); // CJK UNIFIED IDEOGRAPH-4E00

  EXPECT_EQ(4, columnWidthUTF8("\344\270\200\344\270\200"));
  EXPECT_EQ(3, columnWidthUTF8("q\344\270\200"));
  EXPECT_EQ(3, columnWidthUTF8("\314\200\340\270\201\344\270\200"));

  // Invalid UTF-8 strings, columnWidthUTF8 should error out.
  EXPECT_EQ(-2, columnWidthUTF8("\344"));
  EXPECT_EQ(-2, columnWidthUTF8("\344\270"));
  EXPECT_EQ(-2, columnWidthUTF8("\344\270\033"));
  EXPECT_EQ(-2, columnWidthUTF8("\344\270\300"));
  EXPECT_EQ(-2, columnWidthUTF8("\377\366\355"));

  EXPECT_EQ(-2, columnWidthUTF8("qwer\344"));
  EXPECT_EQ(-2, columnWidthUTF8("qwer\344\270"));
  EXPECT_EQ(-2, columnWidthUTF8("qwer\344\270\033"));
  EXPECT_EQ(-2, columnWidthUTF8("qwer\344\270\300"));
  EXPECT_EQ(-2, columnWidthUTF8("qwer\377\366\355"));

  // UTF-8 sequences longer than 4 bytes correspond to unallocated Unicode
  // characters.
  EXPECT_EQ(-2, columnWidthUTF8("\370\200\200\200\200"));     // U+200000
  EXPECT_EQ(-2, columnWidthUTF8("\374\200\200\200\200\200")); // U+4000000
}

TEST(Unicode, isPrintable) {
  EXPECT_FALSE(isPrintable(0)); // <control-0000>-<control-001F>
  EXPECT_FALSE(isPrintable(0x01));
  EXPECT_FALSE(isPrintable(0x1F));
  EXPECT_TRUE(isPrintable(' '));
  EXPECT_TRUE(isPrintable('A'));
  EXPECT_TRUE(isPrintable('~'));
  EXPECT_FALSE(isPrintable(0x7F)); // <control-007F>..<control-009F>
  EXPECT_FALSE(isPrintable(0x90));
  EXPECT_FALSE(isPrintable(0x9F));

  EXPECT_TRUE(isPrintable(0xAC));
  EXPECT_TRUE(isPrintable(0xAD)); // SOFT HYPHEN is displayed on most terminals
                                  // as either a space or a dash.
  EXPECT_TRUE(isPrintable(0xAE));

  EXPECT_TRUE(isPrintable(0x0377));  // GREEK SMALL LETTER PAMPHYLIAN DIGAMMA
  EXPECT_FALSE(isPrintable(0x0378)); // <reserved-0378>..<reserved-0379>

  EXPECT_FALSE(isPrintable(0x0600)); // ARABIC NUMBER SIGN

  EXPECT_FALSE(isPrintable(0x1FFFF)); // <reserved-1F774>..<noncharacter-1FFFF>
  EXPECT_TRUE(isPrintable(0x20000));  // CJK UNIFIED IDEOGRAPH-20000

  EXPECT_FALSE(isPrintable(0x10FFFF)); // noncharacter
}

} // namespace
} // namespace unicode
} // namespace sys
} // namespace llvm
