//===- unittests/Support/UnicodeTest.cpp - Unicode.h tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Unicode.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/edit_distance.h"
#include "llvm/Support/ConvertUTF.h"
#include "gmock/gmock.h"
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
  EXPECT_EQ(-1, columnWidthUTF8("\t"));
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

  // test the validity of a fast path in columnWidthUTF8
  for (unsigned char c = 0; c < 128; ++c) {
    const UTF8 buf8[2] = {c, 0};
    const UTF8 *Target8 = &buf8[0];
    UTF32 buf32[1];
    UTF32 *Target32 = &buf32[0];
    auto status = ConvertUTF8toUTF32(&Target8, Target8 + 1, &Target32,
                                     Target32 + 1, strictConversion);
    EXPECT_EQ(status, conversionOK);
    EXPECT_EQ((columnWidthUTF8(reinterpret_cast<const char *>(buf8)) == 1),
              (bool)isPrintable(buf32[0]));
  }
}

TEST(Unicode, nameToCodepointStrict) {
  auto map = [](StringRef Str) {
    return nameToCodepointStrict(Str).value_or(0xFFFF'FFFF);
  };

  // generated codepoints
  EXPECT_EQ(0x03400u, map("CJK UNIFIED IDEOGRAPH-3400"));
  EXPECT_EQ(0x04DBFu, map("CJK UNIFIED IDEOGRAPH-4DBF"));
  EXPECT_EQ(0x04E00u, map("CJK UNIFIED IDEOGRAPH-4E00"));
  EXPECT_EQ(0x09FFCu, map("CJK UNIFIED IDEOGRAPH-9FFC"));
  EXPECT_EQ(0x20000u, map("CJK UNIFIED IDEOGRAPH-20000"));
  EXPECT_EQ(0x2A6DDu, map("CJK UNIFIED IDEOGRAPH-2A6DD"));
  EXPECT_EQ(0x2A700u, map("CJK UNIFIED IDEOGRAPH-2A700"));
  EXPECT_EQ(0x2B740u, map("CJK UNIFIED IDEOGRAPH-2B740"));
  EXPECT_EQ(0x2B81Du, map("CJK UNIFIED IDEOGRAPH-2B81D"));
  EXPECT_EQ(0x2B820u, map("CJK UNIFIED IDEOGRAPH-2B820"));
  EXPECT_EQ(0x2CEA1u, map("CJK UNIFIED IDEOGRAPH-2CEA1"));
  EXPECT_EQ(0x2CEB0u, map("CJK UNIFIED IDEOGRAPH-2CEB0"));
  EXPECT_EQ(0x2EBE0u, map("CJK UNIFIED IDEOGRAPH-2EBE0"));
  EXPECT_EQ(0x30000u, map("CJK UNIFIED IDEOGRAPH-30000"));
  EXPECT_EQ(0x3134Au, map("CJK UNIFIED IDEOGRAPH-3134A"));
  EXPECT_EQ(0x17000u, map("TANGUT IDEOGRAPH-17000"));
  EXPECT_EQ(0x187F7u, map("TANGUT IDEOGRAPH-187F7"));
  EXPECT_EQ(0x18D00u, map("TANGUT IDEOGRAPH-18D00"));
  EXPECT_EQ(0x18D08u, map("TANGUT IDEOGRAPH-18D08"));
  EXPECT_EQ(0x18B00u, map("KHITAN SMALL SCRIPT CHARACTER-18B00"));
  EXPECT_EQ(0x18CD5u, map("KHITAN SMALL SCRIPT CHARACTER-18CD5"));
  EXPECT_EQ(0x1B170u, map("NUSHU CHARACTER-1B170"));
  EXPECT_EQ(0x1B2FBu, map("NUSHU CHARACTER-1B2FB"));
  EXPECT_EQ(0x0F900u, map("CJK COMPATIBILITY IDEOGRAPH-F900"));
  EXPECT_EQ(0x0FA6Du, map("CJK COMPATIBILITY IDEOGRAPH-FA6D"));
  EXPECT_EQ(0x0FA70u, map("CJK COMPATIBILITY IDEOGRAPH-FA70"));
  EXPECT_EQ(0x0FAD9u, map("CJK COMPATIBILITY IDEOGRAPH-FAD9"));
  EXPECT_EQ(0x2F800u, map("CJK COMPATIBILITY IDEOGRAPH-2F800"));
  EXPECT_EQ(0x2FA1Du, map("CJK COMPATIBILITY IDEOGRAPH-2FA1D"));
  EXPECT_EQ(0x31350u, map("CJK UNIFIED IDEOGRAPH-31350")); // Unicode 15.0

  EXPECT_EQ(0xAC00u, map("HANGUL SYLLABLE GA"));
  EXPECT_EQ(0xAC14u, map("HANGUL SYLLABLE GASS"));
  EXPECT_EQ(0xAC2Bu, map("HANGUL SYLLABLE GAELH"));
  EXPECT_EQ(0xAC7Bu, map("HANGUL SYLLABLE GEOLB"));
  EXPECT_EQ(0xC640u, map("HANGUL SYLLABLE WA"));
  EXPECT_EQ(0xC544u, map("HANGUL SYLLABLE A"));
  EXPECT_EQ(0xC5D0u, map("HANGUL SYLLABLE E"));
  EXPECT_EQ(0xC774u, map("HANGUL SYLLABLE I"));

  EXPECT_EQ(0x1F984u, map("UNICORN FACE"));
  EXPECT_EQ(0x00640u, map("ARABIC TATWEEL"));
  EXPECT_EQ(0x02C05u, map("GLAGOLITIC CAPITAL LETTER YESTU"));
  EXPECT_EQ(0x13000u, map("EGYPTIAN HIEROGLYPH A001"));
  EXPECT_EQ(0x02235u, map("BECAUSE"));
  EXPECT_EQ(0x1F514u, map("BELL"));
  EXPECT_EQ(0x1F9A9u, map("FLAMINGO"));
  EXPECT_EQ(0x1F9A9u, map("FLAMINGO"));
  EXPECT_EQ(0x1F402u, map("OX")); // 2 characters
  EXPECT_EQ(0x0FBF9u, map("ARABIC LIGATURE UIGHUR KIRGHIZ YEH WITH HAMZA "
                          "ABOVE WITH ALEF MAKSURA ISOLATED FORM"));
  EXPECT_EQ(0x11F04u, map("KAWI LETTER A")); // Unicode 15.0
  EXPECT_EQ(0x1FA77u, map("PINK HEART")); // Unicode 15.0

  // Aliases
  EXPECT_EQ(0x0000u, map("NULL"));
  EXPECT_EQ(0x0007u, map("ALERT"));
  EXPECT_EQ(0x0009u, map("HORIZONTAL TABULATION"));
  EXPECT_EQ(0x0009u, map("CHARACTER TABULATION"));
  EXPECT_EQ(0x000Au, map("LINE FEED"));
  EXPECT_EQ(0x000Au, map("NEW LINE"));
  EXPECT_EQ(0x0089u, map("CHARACTER TABULATION WITH JUSTIFICATION"));
  EXPECT_EQ(0x0089u, map("HORIZONTAL TABULATION WITH JUSTIFICATION"));
  EXPECT_EQ(0x2118u,
            map("WEIERSTRASS ELLIPTIC FUNCTION"));      // correction
  EXPECT_EQ(0x2118u, map("SCRIPT CAPITAL P"));          // correction
  EXPECT_EQ(0xFEFFu, map("BYTE ORDER MARK"));           // alternate
  EXPECT_EQ(0xFEFFu, map("ZERO WIDTH NO-BREAK SPACE")); // alternate

  // Should perform exact case match
  EXPECT_EQ(0xFFFFFFFFu, map(""));
  EXPECT_EQ(0xFFFFFFFFu, map("NOT A UNICODE CHARACTER"));
  EXPECT_EQ(0xFFFFFFFFu, map("unicorn face"));
  EXPECT_EQ(0xFFFFFFFFu, map("UNICORN FaCE"));
  EXPECT_EQ(0xFFFFFFFFu, map("UNICORNFaCE"));
  EXPECT_EQ(0xFFFFFFFFu, map("UNICORN"));
  EXPECT_EQ(0xFFFFFFFFu, map("HANGUL SYLLABLE i"));
  EXPECT_EQ(0xFFFFFFFFu, map("hANGUL SYLLABLE i"));
  EXPECT_EQ(0xFFFFFFFFu, map("HANGULSYLLABLEI"));
  EXPECT_EQ(0xFFFFFFFFu, map("HANGUL SYLLABLE"));
  EXPECT_EQ(0xFFFFFFFFu, map("cJK COMPATIBILITY IDEOGRAPH-2FA1D"));
  EXPECT_EQ(0xFFFFFFFFu, map("CJK COMPATIBILITY IDEOGRAPH-2FA1d"));
  EXPECT_EQ(0xFFFFFFFFu, map("CJK COMPATIBILITY IDEOGRAPH 2FA1D"));
  EXPECT_EQ(0xFFFFFFFF, map("CJK COMPATIBILITY IDEOGRAPH-NOTANUMBER"));
  EXPECT_EQ(0xFFFFFFFFu, map("CJK COMPATIBILITY IDEOGRAPH-1"));
  EXPECT_EQ(0xFFFFFFFFu, map("ZERO WIDTH NO BREAK SPACE"));

  // Should not support abbreviations or figments
  EXPECT_EQ(0xFFFFFFFFu, map("FVS1"));
  EXPECT_EQ(0xFFFFFFFFu, map("HIGH OCTET PRESET"));
  EXPECT_EQ(0xFFFFFFFFu, map("BEL"));
}

TEST(Unicode, nameToCodepointLoose) {
  auto map = [](StringRef Str) {
    auto Opt = nameToCodepointLooseMatching(Str);
    if (!Opt)
      return char32_t(0xFFFF'FFFF);
    return Opt->CodePoint;
  };

  // generated codepoints
  EXPECT_EQ(0x04DBFu, map("CJK UNIFIED IDEOGRAPH-4DBF"));
  EXPECT_EQ(0x04E00u, map("CJK UNIFIED IDEOGRAPH-4E00"));
  EXPECT_EQ(0x09FFCu, map("CJK UNIFIED IDEOGRAPH-9FFC"));
  EXPECT_EQ(0x20000u, map("CJK UNIFIED IDEOGRAPH-20000"));
  EXPECT_EQ(0x2A6DDu, map("CJK UNIFIED IDEOGRAPH-2A6DD"));
  EXPECT_EQ(0x2A700u, map("CJK UNIFIED IDEOGRAPH-2A700"));
  EXPECT_EQ(0x2B740u, map("CJK UNIFIED IDEOGRAPH-2B740"));
  EXPECT_EQ(0x03400u, map("CJK UNIFIED IDEOGRAPH-3400"));
  EXPECT_EQ(0x2B81Du, map("CJK UNIFIED IDEOGRAPH-2B81D"));
  EXPECT_EQ(0x2B820u, map("CJK UNIFIED IDEOGRAPH-2B820"));
  EXPECT_EQ(0x2CEA1u, map("CJK UNIFIED IDEOGRAPH-2CEA1"));
  EXPECT_EQ(0x2CEB0u, map("CJK UNIFIED IDEOGRAPH-2CEB0"));
  EXPECT_EQ(0x2EBE0u, map("CJK UNIFIED IDEOGRAPH-2EBE0"));
  EXPECT_EQ(0x30000u, map("CJK UNIFIED IDEOGRAPH-30000"));
  EXPECT_EQ(0x3134Au, map("CJK UNIFIED IDEOGRAPH-3134A"));
  EXPECT_EQ(0x17000u, map("TANGUT IDEOGRAPH-17000"));
  EXPECT_EQ(0x187F7u, map("TANGUT IDEOGRAPH-187F7"));
  EXPECT_EQ(0x18D00u, map("TANGUT IDEOGRAPH-18D00"));
  EXPECT_EQ(0x18D08u, map("TANGUT IDEOGRAPH-18D08"));
  EXPECT_EQ(0x18B00u, map("KHITAN SMALL SCRIPT CHARACTER-18B00"));
  EXPECT_EQ(0x18CD5u, map("KHITAN SMALL SCRIPT CHARACTER-18CD5"));
  EXPECT_EQ(0x1B170u, map("NUSHU CHARACTER-1B170"));
  EXPECT_EQ(0x1B2FBu, map("NUSHU CHARACTER-1B2FB"));
  EXPECT_EQ(0x0F900u, map("CJK COMPATIBILITY IDEOGRAPH-F900"));
  EXPECT_EQ(0x0FA6Du, map("CJK COMPATIBILITY IDEOGRAPH-FA6D"));
  EXPECT_EQ(0x0FA70u, map("CJK COMPATIBILITY IDEOGRAPH-FA70"));
  EXPECT_EQ(0x0FAD9u, map("CJK COMPATIBILITY IDEOGRAPH-FAD9"));
  EXPECT_EQ(0x2F800u, map("CJK COMPATIBILITY IDEOGRAPH-2F800"));
  EXPECT_EQ(0x2FA1Du, map("CJK COMPATIBILITY IDEOGRAPH-2FA1D"));

  EXPECT_EQ(0xAC00u, map("HANGUL SYLLABLE GA"));
  EXPECT_EQ(0xAC14u, map("HANGUL SYLLABLE GASS"));
  EXPECT_EQ(0xAC2Bu, map("HANGUL SYLLABLE GAELH"));
  EXPECT_EQ(0xAC7Bu, map("HANGUL SYLLABLE GEOLB"));
  EXPECT_EQ(0xC640u, map("HANGUL SYLLABLE WA"));
  EXPECT_EQ(0xC544u, map("HANGUL SYLLABLE A"));
  EXPECT_EQ(0xC5D0u, map("HANGUL SYLLABLE E"));
  EXPECT_EQ(0xC774u, map("HANGUL SYLLABLE I"));

  EXPECT_EQ(0x1F984u, map("UNICORN FACE"));
  EXPECT_EQ(0x00640u, map("ARABIC TATWEEL"));
  EXPECT_EQ(0x02C05u, map("GLAGOLITIC CAPITAL LETTER YESTU"));
  EXPECT_EQ(0x13000u, map("EGYPTIAN HIEROGLYPH A001"));
  EXPECT_EQ(0x02235u, map("BECAUSE"));
  EXPECT_EQ(0x1F514u, map("BELL"));
  EXPECT_EQ(0x1F9A9u, map("FLAMINGO"));
  EXPECT_EQ(0x1F402u, map("OX")); // 2 characters
  EXPECT_EQ(0x0FBF9u, map("ARABIC LIGATURE UIGHUR KIRGHIZ YEH WITH HAMZA "
                          "ABOVE WITH ALEF MAKSURA ISOLATED FORM"));

  // Aliases
  EXPECT_EQ(0x0000u, map("NULL"));
  EXPECT_EQ(0x0007u, map("ALERT"));
  EXPECT_EQ(0x0009u, map("HORIZONTAL TABULATION"));
  EXPECT_EQ(0x0009u, map("CHARACTER TABULATION"));
  EXPECT_EQ(0x000Au, map("LINE FEED"));
  EXPECT_EQ(0x000Au, map("NEW LINE"));
  EXPECT_EQ(0x0089u, map("CHARACTER TABULATION WITH JUSTIFICATION"));
  EXPECT_EQ(0x0089u, map("HORIZONTAL TABULATION WITH JUSTIFICATION"));
  EXPECT_EQ(0x2118u,
            map("WEIERSTRASS ELLIPTIC FUNCTION"));      // correction
  EXPECT_EQ(0x2118u, map("SCRIPT CAPITAL P"));          // correction
  EXPECT_EQ(0xFEFFu, map("BYTE ORDER MARK"));           // alternate
  EXPECT_EQ(0xFEFFu, map("ZERO WIDTH NO-BREAK SPACE")); // alternate
  EXPECT_EQ(0xFEFFu, map("ZERO WIDTH NO BREAK SPACE")); // alternate

  // Should perform loose matching
  EXPECT_EQ(0xFFFFFFFFu, map(""));
  EXPECT_EQ(0xFFFFFFFFu, map("NOT A UNICODE CHARACTER"));
  EXPECT_EQ(0x0001F984u, map("unicorn face"));
  EXPECT_EQ(0x0001F984u, map("UNICORN FaCE"));
  EXPECT_EQ(0x0001F984u, map("UNICORNFaCE"));
  EXPECT_EQ(0xFFFFFFFFu, map("UNICORN"));
  EXPECT_EQ(0xC774u, map("HANGUL SYLLABLE i"));
  EXPECT_EQ(0xC774u, map("hANGUL SYLLABLE i"));
  EXPECT_EQ(0xC774u, map("HANGULSYLLABLEI"));
  EXPECT_EQ(0xFFFFFFFFu, map("HANGUL SYLLABLE"));

  EXPECT_EQ(0x2FA1Du, map("cJK COMPATIBILITY IDEOGRAPH-2FA1D"));
  EXPECT_EQ(0x2FA1Du, map("CJK COMPATIBILITY IDEOGRAPH-2FA1d"));
  EXPECT_EQ(0x2FA1Du, map("CJK COMPATIBILITY IDEOGRAPH 2FA1D"));

  EXPECT_EQ(0xFFFFFFFFu, map("CJK COMPATIBILITY IDEOGRAPH-NOTANUMBER"));
  EXPECT_EQ(0xFFFFFFFFu, map("CJK COMPATIBILITY IDEOGRAPH-1"));

  // https://unicode.org/reports/tr44/#Matching_Names
  // UAX44-LM2: Medial hypens are ignored, non medial hyphens are not
  EXPECT_EQ(0x1FBC5u, map("S-T-I-C-K-F-I-G-U-R-E"));
  EXPECT_EQ(0xFFFFFFFFu, map("-STICK FIGURE"));
  EXPECT_EQ(0xFFFFFFFFu, map("STICK FIGURE-"));
  EXPECT_EQ(0xFFFFFFFFu, map("STICK FIGURE -"));
  EXPECT_EQ(0xFFFFFFFFu, map("STICK FIGURE --"));
  EXPECT_EQ(0xFFFFFFFFu, map("STICK--FIGURE"));

  EXPECT_EQ(0x0F68u, map("TIBETAN LETTER A"));
  EXPECT_EQ(0x0F68u, map("TIBETAN LETTERA"));
  EXPECT_EQ(0x0F68u, map("TIBETAN LETTER-A"));
  EXPECT_EQ(0x0F60u, map("TIBETAN LETTER -A"));
  EXPECT_EQ(0x0F60u, map("TIBETAN LETTER  -A"));
  ;

  // special case
  EXPECT_EQ(0x1180u, map("HANGUL JUNGSEONG O-E"));
  EXPECT_EQ(0x116Cu, map("HANGUL JUNGSEONG OE"));

  // names that are prefix to existing characters should not match
  EXPECT_FALSE(nameToCodepointLooseMatching("B"));
  EXPECT_FALSE(nameToCodepointLooseMatching("BE"));
  EXPECT_FALSE(nameToCodepointLooseMatching("BEE"));
  EXPECT_FALSE(nameToCodepointLooseMatching("BEET"));
  EXPECT_FALSE(nameToCodepointLooseMatching("BEETL"));
  EXPECT_TRUE(nameToCodepointLooseMatching("BEETLE"));
}

} // namespace

bool operator==(MatchForCodepointName a, MatchForCodepointName b) {
  return a.Name == b.Name && a.Distance == b.Distance && a.Value == b.Value;
}

namespace {

TEST(Unicode, nearestMatchesForCodepointName) {
  auto Normalize = [](StringRef Name) {
    std::string Out;
    Out.reserve(Name.size());
    for (char C : Name) {
      if (isAlnum(C))
        Out.push_back(toUpper(C));
    }
    return Out;
  };

  auto L = [&](StringRef name) {
    auto v = nearestMatchesForCodepointName(name, 3);
    for (auto &r : v) {
      auto A = Normalize(r.Name);
      auto B = Normalize(name);
      EXPECT_EQ(StringRef(A).edit_distance(B, true), r.Distance);
    }
    return v;
  };
  using ::testing::ElementsAre;
  using M = MatchForCodepointName;

  ASSERT_THAT(L(""), ElementsAre(M{"OX", 2, 0x1F402}, M{"ANT", 3, 0x1F41C},
                                 M{"ARC", 3, 0x2312}));
  // shortest name
  ASSERT_THAT(L("OX"), ElementsAre(M{"OX", 0, 0x1F402}, M{"AXE", 2, 0x1FA93},
                                   M{"BOY", 2, 0x1F466}));

  // longest name
  ASSERT_THAT(L("ARABIC LIGATURE UIGHUR KIRGHIZ YEH WITH HAMZA ABOVE WITH ALEF "
                "MAKSURA INITIAL FORM"),
              ElementsAre(M{"ARABIC LIGATURE UIGHUR KIRGHIZ YEH WITH HAMZA "
                            "ABOVE WITH ALEF MAKSURA INITIAL FORM",
                            0, 0xFBFB},
                          M{"ARABIC LIGATURE UIGHUR KIRGHIZ YEH WITH HAMZA "
                            "ABOVE WITH ALEF MAKSURA FINAL FORM",
                            4, 0xFBFA},
                          M{"ARABIC LIGATURE UIGHUR KIRGHIZ YEH WITH HAMZA "
                            "ABOVE WITH ALEF MAKSURA ISOLATED FORM",
                            7, 0xFBF9}));

  // same result with underscore, spaces, etc
  ASSERT_THAT(L("______ARABICLIGATUREUIGHUR KIRGHIZ YEH with HAMZA ABOVE WITH "
                "ALEF MAKsURAINITIAL form_"),
              ElementsAre(M{"ARABIC LIGATURE UIGHUR KIRGHIZ YEH WITH HAMZA "
                            "ABOVE WITH ALEF MAKSURA INITIAL FORM",
                            0, 0xFBFB},
                          M{"ARABIC LIGATURE UIGHUR KIRGHIZ YEH WITH HAMZA "
                            "ABOVE WITH ALEF MAKSURA FINAL FORM",
                            4, 0xFBFA},
                          M{"ARABIC LIGATURE UIGHUR KIRGHIZ YEH WITH HAMZA "
                            "ABOVE WITH ALEF MAKSURA ISOLATED FORM",
                            7, 0xFBF9}));

  ASSERT_THAT(L("GREEK CAPITAL LETTER LAMBDA"),
              ElementsAre(M{"GREEK CAPITAL LETTER LAMDA", 1, 0x39B},
                          M{"GREEK CAPITAL LETTER GAMMA", 3, 0x0393},
                          M{"GREEK CAPITAL LETTER ALPHA", 4, 0x0391}));

  ASSERT_THAT(L("greekcapitalletter-lambda"),
              ElementsAre(M{"GREEK CAPITAL LETTER LAMDA", 1, 0x39B},
                          M{"GREEK CAPITAL LETTER GAMMA", 3, 0x0393},
                          M{"GREEK CAPITAL LETTER ALPHA", 4, 0x0391}));

  // typo http://www.unicode.org/notes/tn27/tn27-5.html
  ASSERT_THAT(
      L("PRESENTATION FORM FOR VERTICAL RIGHT WHITE LENTICULAR BRAKCET"),
      ElementsAre(
          M{"PRESENTATION FORM FOR VERTICAL RIGHT WHITE LENTICULAR BRAKCET", 0,
            0xFE18}, // typo
          M{"PRESENTATION FORM FOR VERTICAL RIGHT WHITE LENTICULAR BRACKET", 2,
            0xFE18}, // correction
          M{"PRESENTATION FORM FOR VERTICAL LEFT WHITE LENTICULAR BRACKET", 6,
            0xFE17}));

  // typo http://www.unicode.org/notes/tn27/tn27-5.html
  ASSERT_THAT(
      L("BYZANTINE MUSICAL SYMBOL FHTORA SKLIRON CHROMA VASIS"),
      ElementsAre(
          M{"BYZANTINE MUSICAL SYMBOL FHTORA SKLIRON CHROMA VASIS", 0, 0x1D0C5},
          M{"BYZANTINE MUSICAL SYMBOL FTHORA SKLIRON CHROMA VASIS", 2, 0x1D0C5},
          M{"BYZANTINE MUSICAL SYMBOL FTHORA SKLIRON CHROMA SYNAFI", 7,
            0x1D0C6}));
}

} // namespace
} // namespace unicode
} // namespace sys
} // namespace llvm
