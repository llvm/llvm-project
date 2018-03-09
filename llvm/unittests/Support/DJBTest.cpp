//===---------- llvm/unittest/Support/DJBTest.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DJB.h"
#include "llvm/ADT/Twine.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(DJBTest, caseFolding) {
  struct TestCase {
    StringLiteral One;
    StringLiteral Two;
  };

  static constexpr TestCase Tests[] = {
      {{"ASDF"}, {"asdf"}},
      {{"qWeR"}, {"QwEr"}},
      {{"qqqqqqqqqqqqqqqqqqqq"}, {"QQQQQQQQQQQQQQQQQQQQ"}},

      {{"I"}, {"i"}},
      // Latin Small Letter Dotless I
      {{u8"\u0130"}, {"i"}},
      // Latin Capital Letter I With Dot Above
      {{u8"\u0131"}, {"i"}},

      // Latin Capital Letter A With Grave
      {{u8"\u00c0"}, {u8"\u00e0"}},
      // Latin Capital Letter A With Macron
      {{u8"\u0100"}, {u8"\u0101"}},
      // Latin Capital Letter L With Acute
      {{u8"\u0139"}, {u8"\u013a"}},
      // Cyrillic Capital Letter Ie
      {{u8"\u0415"}, {u8"\u0435"}},
      // Latin Capital Letter A With Circumflex And Grave
      {{u8"\u1ea6"}, {u8"\u1ea7"}},
      // Kelvin Sign
      {{u8"\u212a"}, {u8"\u006b"}},
      // Glagolitic Capital Letter Chrivi
      {{u8"\u2c1d"}, {u8"\u2c4d"}},
      // Fullwidth Latin Capital Letter M
      {{u8"\uff2d"}, {u8"\uff4d"}},
      // Old Hungarian Capital Letter Ej
      {{u8"\U00010c92"}, {u8"\U00010cd2"}},
  };

  for (const TestCase &T : Tests) {
    SCOPED_TRACE("Comparing '" + T.One + "' and '" + T.Two + "'");
    EXPECT_EQ(caseFoldingDjbHash(T.One), caseFoldingDjbHash(T.Two));
  }
}

TEST(DJBTest, knownValuesLowerCase) {
  struct TestCase {
    StringLiteral Text;
    uint32_t Hash;
  };
  static constexpr TestCase Tests[] = {
      {{""}, 5381u},
      {{"f"}, 177675u},
      {{"fo"}, 5863386u},
      {{"foo"}, 193491849u},
      {{"foob"}, 2090263819u},
      {{"fooba"}, 259229388u},
      {{"foobar"}, 4259602622u},
      {{"pneumonoultramicroscopicsilicovolcanoconiosis"}, 3999417781u},
  };

  for (const TestCase &T : Tests) {
    SCOPED_TRACE("Text: '" + T.Text + "'");
    EXPECT_EQ(T.Hash, djbHash(T.Text));
    EXPECT_EQ(T.Hash, caseFoldingDjbHash(T.Text));
    EXPECT_EQ(T.Hash, caseFoldingDjbHash(T.Text.upper()));
  }
}

TEST(DJBTest, knownValuesUnicode) {
  EXPECT_EQ(5866553u, djbHash(u8"\u0130"));
  EXPECT_EQ(177678u, caseFoldingDjbHash(u8"\u0130"));
  EXPECT_EQ(
      1302161417u,
      djbHash(
          u8"\u0130\u0131\u00c0\u00e0\u0100\u0101\u0139\u013a\u0415\u0435\u1ea6"
          u8"\u1ea7\u212a\u006b\u2c1d\u2c4d\uff2d\uff4d\U00010c92\U00010cd2"));
  EXPECT_EQ(
      1145571043u,
      caseFoldingDjbHash(
          u8"\u0130\u0131\u00c0\u00e0\u0100\u0101\u0139\u013a\u0415\u0435\u1ea6"
          u8"\u1ea7\u212a\u006b\u2c1d\u2c4d\uff2d\uff4d\U00010c92\U00010cd2"));
}
