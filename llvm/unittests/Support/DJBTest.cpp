//===---------- llvm/unittest/Support/DJBTest.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
      {{/*U+130*/ "\xc4\xb0"}, {"i"}},
      // Latin Capital Letter I With Dot Above
      {{/*U+131*/ "\xc4\xb1"}, {"i"}},

      // Latin Capital Letter A With Grave
      {{/*U+c0*/ "\xc3\x80"}, {/*U+e0*/ "\xc3\xa0"}},
      // Latin Capital Letter A With Macron
      {{/*U+100*/ "\xc4\x80"}, {/*U+101*/ "\xc4\x81"}},
      // Latin Capital Letter L With Acute
      {{/*U+139*/ "\xc4\xb9"}, {/*U+13a*/ "\xc4\xba"}},
      // Cyrillic Capital Letter Ie
      {{/*U+415*/ "\xd0\x95"}, {/*U+435*/ "\xd0\xb5"}},
      // Latin Capital Letter A With Circumflex And Grave
      {{/*U+1ea6*/ "\xe1\xba\xa6"}, {/*U+1ea7*/ "\xe1\xba\xa7"}},
      // Kelvin Sign
      {{/*U+212a*/ "\xe2\x84\xaa"}, {"k"}},
      // Glagolitic Capital Letter Chrivi
      {{/*U+2c1d*/ "\xe2\xb0\x9d"}, {/*U+2c4d*/ "\xe2\xb1\x8d"}},
      // Fullwidth Latin Capital Letter M
      {{/*U+ff2d*/ "\xef\xbc\xad"}, {/*U+ff4d*/ "\xef\xbd\x8d"}},
      // Old Hungarian Capital Letter Ej
      {{/*U+10c92*/ "\xf0\x90\xb2\x92"}, {/*U+10cd2*/ "\xf0\x90\xb3\x92"}},
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
  EXPECT_EQ(5866553u, djbHash(/*U+130*/ "\xc4\xb0"));
  EXPECT_EQ(177678u, caseFoldingDjbHash(/*U+130*/ "\xc4\xb0"));
  EXPECT_EQ(
      1302161417u,
      djbHash("\xc4\xb0\xc4\xb1\xc3\x80\xc3\xa0\xc4\x80\xc4\x81\xc4\xb9\xc4\xba"
              "\xd0\x95\xd0\xb5\xe1\xba\xa6\xe1\xba\xa7\xe2\x84\xaa\x6b\xe2\xb0"
              "\x9d\xe2\xb1\x8d\xef\xbc\xad\xef\xbd\x8d\xf0\x90\xb2\x92\xf0\x90"
              "\xb3\x92"));
  EXPECT_EQ(
      1145571043u,
      caseFoldingDjbHash(
          "\xc4\xb0\xc4\xb1\xc3\x80\xc3\xa0\xc4\x80\xc4\x81\xc4\xb9\xc4\xba"
          "\xd0\x95\xd0\xb5\xe1\xba\xa6\xe1\xba\xa7\xe2\x84\xaa\x6b\xe2\xb0"
          "\x9d\xe2\xb1\x8d\xef\xbc\xad\xef\xbd\x8d\xf0\x90\xb2\x92\xf0\x90"
          "\xb3\x92"));
}
