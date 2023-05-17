//===- FormatVariadicTest.cpp - Unit tests for string formatting ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatAdapters.h"
#include "gtest/gtest.h"

using namespace llvm;

// Compile-time tests templates in the detail namespace.
namespace {
struct Format : public FormatAdapter<int> {
  Format(int N) : FormatAdapter<int>(std::move(N)) {}
  void format(raw_ostream &OS, StringRef Opt) override { OS << "Format"; }
};

using detail::uses_format_member;
using detail::uses_missing_provider;

static_assert(uses_format_member<Format>::value, "");
static_assert(uses_format_member<Format &>::value, "");
static_assert(uses_format_member<Format &&>::value, "");
static_assert(uses_format_member<const Format>::value, "");
static_assert(uses_format_member<const Format &>::value, "");
static_assert(uses_format_member<const volatile Format>::value, "");
static_assert(uses_format_member<const volatile Format &>::value, "");

struct NoFormat {};
static_assert(uses_missing_provider<NoFormat>::value, "");
}

TEST(FormatVariadicTest, EmptyFormatString) {
  auto Replacements = formatv_object_base::parseFormatString("");
  EXPECT_EQ(0U, Replacements.size());
}

TEST(FormatVariadicTest, NoReplacements) {
  const StringRef kFormatString = "This is a test";
  auto Replacements = formatv_object_base::parseFormatString(kFormatString);
  ASSERT_EQ(1U, Replacements.size());
  EXPECT_EQ(kFormatString, Replacements[0].Spec);
  EXPECT_EQ(ReplacementType::Literal, Replacements[0].Type);
}

TEST(FormatVariadicTest, EscapedBrace) {
  // {{ should be replaced with {
  auto Replacements = formatv_object_base::parseFormatString("{{");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ("{", Replacements[0].Spec);
  EXPECT_EQ(ReplacementType::Literal, Replacements[0].Type);

  // An even number N of braces should be replaced with N/2 braces.
  Replacements = formatv_object_base::parseFormatString("{{{{{{");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ("{{{", Replacements[0].Spec);
  EXPECT_EQ(ReplacementType::Literal, Replacements[0].Type);

  // } does not require doubling up.
  Replacements = formatv_object_base::parseFormatString("}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ("}", Replacements[0].Spec);
  EXPECT_EQ(ReplacementType::Literal, Replacements[0].Type);

  // } does not require doubling up.
  Replacements = formatv_object_base::parseFormatString("}}}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ("}}}", Replacements[0].Spec);
  EXPECT_EQ(ReplacementType::Literal, Replacements[0].Type);
}

TEST(FormatVariadicTest, ValidReplacementSequence) {
  // 1. Simple replacement - parameter index only
  auto Replacements = formatv_object_base::parseFormatString("{0}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(0u, Replacements[0].Align);
  EXPECT_EQ("", Replacements[0].Options);

  Replacements = formatv_object_base::parseFormatString("{1}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(1u, Replacements[0].Index);
  EXPECT_EQ(0u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Right, Replacements[0].Where);
  EXPECT_EQ("", Replacements[0].Options);

  // 2. Parameter index with right alignment
  Replacements = formatv_object_base::parseFormatString("{0,3}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(3u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Right, Replacements[0].Where);
  EXPECT_EQ("", Replacements[0].Options);

  // 3. And left alignment
  Replacements = formatv_object_base::parseFormatString("{0,-3}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(3u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Left, Replacements[0].Where);
  EXPECT_EQ("", Replacements[0].Options);

  // 4. And center alignment
  Replacements = formatv_object_base::parseFormatString("{0,=3}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(3u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Center, Replacements[0].Where);
  EXPECT_EQ("", Replacements[0].Options);

  // 4. Parameter index with option string
  Replacements = formatv_object_base::parseFormatString("{0:foo}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(0u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Right, Replacements[0].Where);
  EXPECT_EQ("foo", Replacements[0].Options);

  // 5. Parameter index with alignment before option string
  Replacements = formatv_object_base::parseFormatString("{0,-3:foo}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(3u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Left, Replacements[0].Where);
  EXPECT_EQ("foo", Replacements[0].Options);

  // 7. Parameter indices, options, and alignment can all have whitespace.
  Replacements = formatv_object_base::parseFormatString("{ 0, -3 : foo }");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(3u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Left, Replacements[0].Where);
  EXPECT_EQ("foo", Replacements[0].Options);

  // 8. Everything after the first option specifier is part of the style, even
  // if it contains another option specifier.
  Replacements = formatv_object_base::parseFormatString("{0:0:1}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ("0:0:1", Replacements[0].Spec);
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(0u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Right, Replacements[0].Where);
  EXPECT_EQ("0:1", Replacements[0].Options);

  // 9. Custom padding character
  Replacements = formatv_object_base::parseFormatString("{0,p+4:foo}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ("0,p+4:foo", Replacements[0].Spec);
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(4u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Right, Replacements[0].Where);
  EXPECT_EQ('p', Replacements[0].Pad);
  EXPECT_EQ("foo", Replacements[0].Options);

  // Format string special characters are allowed as padding character
  Replacements = formatv_object_base::parseFormatString("{0,-+4:foo}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ("0,-+4:foo", Replacements[0].Spec);
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(4u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Right, Replacements[0].Where);
  EXPECT_EQ('-', Replacements[0].Pad);
  EXPECT_EQ("foo", Replacements[0].Options);

  Replacements = formatv_object_base::parseFormatString("{0,+-4:foo}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ("0,+-4:foo", Replacements[0].Spec);
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(4u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Left, Replacements[0].Where);
  EXPECT_EQ('+', Replacements[0].Pad);
  EXPECT_EQ("foo", Replacements[0].Options);

  Replacements = formatv_object_base::parseFormatString("{0,==4:foo}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ("0,==4:foo", Replacements[0].Spec);
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(4u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Center, Replacements[0].Where);
  EXPECT_EQ('=', Replacements[0].Pad);
  EXPECT_EQ("foo", Replacements[0].Options);

  Replacements = formatv_object_base::parseFormatString("{0,:=4:foo}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ("0,:=4:foo", Replacements[0].Spec);
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(4u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Center, Replacements[0].Where);
  EXPECT_EQ(':', Replacements[0].Pad);
  EXPECT_EQ("foo", Replacements[0].Options);
}

TEST(FormatVariadicTest, DefaultReplacementValues) {
  // 2. If options string is missing, it defaults to empty.
  auto Replacements = formatv_object_base::parseFormatString("{0,3}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(3u, Replacements[0].Align);
  EXPECT_EQ("", Replacements[0].Options);

  // Including if the colon is present but contains no text.
  Replacements = formatv_object_base::parseFormatString("{0,3:}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(3u, Replacements[0].Align);
  EXPECT_EQ("", Replacements[0].Options);

  // 3. If alignment is missing, it defaults to 0, right, space
  Replacements = formatv_object_base::parseFormatString("{0:foo}");
  ASSERT_EQ(1u, Replacements.size());
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(AlignStyle::Right, Replacements[0].Where);
  EXPECT_EQ(' ', Replacements[0].Pad);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(0u, Replacements[0].Align);
  EXPECT_EQ("foo", Replacements[0].Options);
}

TEST(FormatVariadicTest, MultipleReplacements) {
  auto Replacements =
      formatv_object_base::parseFormatString("{0} {1:foo}-{2,-3:bar}");
  ASSERT_EQ(5u, Replacements.size());
  // {0}
  EXPECT_EQ(ReplacementType::Format, Replacements[0].Type);
  EXPECT_EQ(0u, Replacements[0].Index);
  EXPECT_EQ(0u, Replacements[0].Align);
  EXPECT_EQ(AlignStyle::Right, Replacements[0].Where);
  EXPECT_EQ("", Replacements[0].Options);

  // " "
  EXPECT_EQ(ReplacementType::Literal, Replacements[1].Type);
  EXPECT_EQ(" ", Replacements[1].Spec);

  // {1:foo} - Options=foo
  EXPECT_EQ(ReplacementType::Format, Replacements[2].Type);
  EXPECT_EQ(1u, Replacements[2].Index);
  EXPECT_EQ(0u, Replacements[2].Align);
  EXPECT_EQ(AlignStyle::Right, Replacements[2].Where);
  EXPECT_EQ("foo", Replacements[2].Options);

  // "-"
  EXPECT_EQ(ReplacementType::Literal, Replacements[3].Type);
  EXPECT_EQ("-", Replacements[3].Spec);

  // {2:bar,-3} - Options=bar, Align=-3
  EXPECT_EQ(ReplacementType::Format, Replacements[4].Type);
  EXPECT_EQ(2u, Replacements[4].Index);
  EXPECT_EQ(3u, Replacements[4].Align);
  EXPECT_EQ(AlignStyle::Left, Replacements[4].Where);
  EXPECT_EQ("bar", Replacements[4].Options);
}

TEST(FormatVariadicTest, FormatNoReplacements) {
  EXPECT_EQ("", formatv("").str());
  EXPECT_EQ("Test", formatv("Test").str());
}

TEST(FormatVariadicTest, FormatBasicTypesOneReplacement) {
  EXPECT_EQ("1", formatv("{0}", 1).str());
  EXPECT_EQ("c", formatv("{0}", 'c').str());
  EXPECT_EQ("-3", formatv("{0}", -3).str());
  EXPECT_EQ("Test", formatv("{0}", "Test").str());
  EXPECT_EQ("Test2", formatv("{0}", StringRef("Test2")).str());
  EXPECT_EQ("Test3", formatv("{0}", std::string("Test3")).str());
}

TEST(FormatVariadicTest, IntegralHexFormatting) {
  // 1. Trivial cases.  Make sure hex is not the default.
  EXPECT_EQ("0", formatv("{0}", 0).str());
  EXPECT_EQ("2748", formatv("{0}", 0xABC).str());
  EXPECT_EQ("-2748", formatv("{0}", -0xABC).str());

  // 3. various hex prefixes.
  EXPECT_EQ("0xFF", formatv("{0:X}", 255).str());
  EXPECT_EQ("0xFF", formatv("{0:X+}", 255).str());
  EXPECT_EQ("0xff", formatv("{0:x}", 255).str());
  EXPECT_EQ("0xff", formatv("{0:x+}", 255).str());
  EXPECT_EQ("FF", formatv("{0:X-}", 255).str());
  EXPECT_EQ("ff", formatv("{0:x-}", 255).str());

  // 5. Precision pads left of the most significant digit but right of the
  // prefix (if one exists).
  EXPECT_EQ("0xFF", formatv("{0:X2}", 255).str());
  EXPECT_EQ("0xFF", formatv("{0:X+2}", 255).str());
  EXPECT_EQ("0x0ff", formatv("{0:x3}", 255).str());
  EXPECT_EQ("0x0ff", formatv("{0:x+3}", 255).str());
  EXPECT_EQ("00FF", formatv("{0:X-4}", 255).str());
  EXPECT_EQ("00ff", formatv("{0:x-4}", 255).str());

  // 6. Try some larger types.
  EXPECT_EQ("0xDEADBEEFDEADBEEF",
            formatv("{0:X16}", -2401053088876216593LL).str());
  EXPECT_EQ("0xFEEBDAEDFEEBDAED",
            formatv("{0:X16}", 0xFEEBDAEDFEEBDAEDULL).str());
  EXPECT_EQ("0x00000000DEADBEEF", formatv("{0:X16}", 0xDEADBEEF).str());

  // 7. Padding should take into account the prefix
  EXPECT_EQ("0xff", formatv("{0,4:x}", 255).str());
  EXPECT_EQ(" 0xff", formatv("{0,5:x+}", 255).str());
  EXPECT_EQ("  FF", formatv("{0,4:X-}", 255).str());
  EXPECT_EQ("   ff", formatv("{0,5:x-}", 255).str());

  // 8. Including when it's been zero-padded
  EXPECT_EQ("  0x0ff", formatv("{0,7:x3}", 255).str());
  EXPECT_EQ(" 0x00ff", formatv("{0,7:x+4}", 255).str());
  EXPECT_EQ("  000FF", formatv("{0,7:X-5}", 255).str());
  EXPECT_EQ(" 0000ff", formatv("{0,7:x-6}", 255).str());

  // 9. Precision with default format specifier should work too
  EXPECT_EQ("    255", formatv("{0,7:3}", 255).str());
  EXPECT_EQ("   0255", formatv("{0,7:4}", 255).str());
  EXPECT_EQ("  00255", formatv("{0,7:5}", 255).str());
  EXPECT_EQ(" 000255", formatv("{0,7:6}", 255).str());
}

TEST(FormatVariadicTest, PointerFormatting) {
  // 1. Trivial cases.  Hex is default.  Default Precision is pointer width.
  if (sizeof(void *) == 4) {
    EXPECT_EQ("0x00000000", formatv("{0}", (void *)0).str());
    EXPECT_EQ("0x00000ABC", formatv("{0}", (void *)0xABC).str());
  } else {
    EXPECT_EQ("0x0000000000000000", formatv("{0}", (void *)0).str());
    EXPECT_EQ("0x0000000000000ABC", formatv("{0}", (void *)0xABC).str());
  }

  // 2. But we can reduce the precision explicitly.
  EXPECT_EQ("0x0", formatv("{0:0}", (void *)0).str());
  EXPECT_EQ("0xABC", formatv("{0:0}", (void *)0xABC).str());
  EXPECT_EQ("0x0000", formatv("{0:4}", (void *)0).str());
  EXPECT_EQ("0x0ABC", formatv("{0:4}", (void *)0xABC).str());

  // 3. various hex prefixes.
  EXPECT_EQ("0x0ABC", formatv("{0:X4}", (void *)0xABC).str());
  EXPECT_EQ("0x0abc", formatv("{0:x4}", (void *)0xABC).str());
  EXPECT_EQ("0ABC", formatv("{0:X-4}", (void *)0xABC).str());
  EXPECT_EQ("0abc", formatv("{0:x-4}", (void *)0xABC).str());
}

TEST(FormatVariadicTest, IntegralNumberFormatting) {
  // 1. Test comma grouping with default widths and precisions.
  EXPECT_EQ("0", formatv("{0:N}", 0).str());
  EXPECT_EQ("10", formatv("{0:N}", 10).str());
  EXPECT_EQ("100", formatv("{0:N}", 100).str());
  EXPECT_EQ("1,000", formatv("{0:N}", 1000).str());
  EXPECT_EQ("1,234,567,890", formatv("{0:N}", 1234567890).str());
  EXPECT_EQ("-10", formatv("{0:N}", -10).str());
  EXPECT_EQ("-100", formatv("{0:N}", -100).str());
  EXPECT_EQ("-1,000", formatv("{0:N}", -1000).str());
  EXPECT_EQ("-1,234,567,890", formatv("{0:N}", -1234567890).str());

  // 2. If there is no comma, width and precision pad to the same absolute
  // size.
  EXPECT_EQ(" 1", formatv("{0,2:N}", 1).str());

  // 3. But if there is a comma or negative sign, width factors them in but
  // precision doesn't.
  EXPECT_EQ(" 1,000", formatv("{0,6:N}", 1000).str());
  EXPECT_EQ(" -1,000", formatv("{0,7:N}", -1000).str());

  // 4. Large widths all line up.
  EXPECT_EQ("      1,000", formatv("{0,11:N}", 1000).str());
  EXPECT_EQ("     -1,000", formatv("{0,11:N}", -1000).str());
  EXPECT_EQ("   -100,000", formatv("{0,11:N}", -100000).str());
}

TEST(FormatVariadicTest, StringFormatting) {
  const char FooArray[] = "FooArray";
  const char *FooPtr = "FooPtr";
  llvm::StringRef FooRef("FooRef");
  constexpr StringLiteral FooLiteral("FooLiteral");
  std::string FooString("FooString");
  // 1. Test that we can print various types of strings.
  EXPECT_EQ(FooArray, formatv("{0}", FooArray).str());
  EXPECT_EQ(FooPtr, formatv("{0}", FooPtr).str());
  EXPECT_EQ(FooRef, formatv("{0}", FooRef).str());
  EXPECT_EQ(FooLiteral, formatv("{0}", FooLiteral).str());
  EXPECT_EQ(FooString, formatv("{0}", FooString).str());

  // 2. Test that the precision specifier prints the correct number of
  // characters.
  EXPECT_EQ("FooA", formatv("{0:4}", FooArray).str());
  EXPECT_EQ("FooP", formatv("{0:4}", FooPtr).str());
  EXPECT_EQ("FooR", formatv("{0:4}", FooRef).str());
  EXPECT_EQ("FooS", formatv("{0:4}", FooString).str());

  // 3. And that padding works.
  EXPECT_EQ("  FooA", formatv("{0,6:4}", FooArray).str());
  EXPECT_EQ("  FooP", formatv("{0,6:4}", FooPtr).str());
  EXPECT_EQ("  FooR", formatv("{0,6:4}", FooRef).str());
  EXPECT_EQ("  FooS", formatv("{0,6:4}", FooString).str());
}

TEST(FormatVariadicTest, CharFormatting) {
  // 1. Not much to see here.  Just print a char with and without padding.
  EXPECT_EQ("C", formatv("{0}", 'C').str());
  EXPECT_EQ("  C", formatv("{0,3}", 'C').str());

  // 2. char is really an integral type though, where the only difference is
  // that the "default" is to print the ASCII.  So if a non-default presentation
  // specifier exists, it should print as an integer.
  EXPECT_EQ("37", formatv("{0:D}", (char)37).str());
  EXPECT_EQ("  037", formatv("{0,5:D3}", (char)37).str());
}

TEST(FormatVariadicTest, BoolTest) {
  // 1. Default style is lowercase text (same as 't')
  EXPECT_EQ("true", formatv("{0}", true).str());
  EXPECT_EQ("false", formatv("{0}", false).str());
  EXPECT_EQ("true", formatv("{0:t}", true).str());
  EXPECT_EQ("false", formatv("{0:t}", false).str());

  // 2. T - uppercase text
  EXPECT_EQ("TRUE", formatv("{0:T}", true).str());
  EXPECT_EQ("FALSE", formatv("{0:T}", false).str());

  // 3. D / d - integral
  EXPECT_EQ("1", formatv("{0:D}", true).str());
  EXPECT_EQ("0", formatv("{0:D}", false).str());
  EXPECT_EQ("1", formatv("{0:d}", true).str());
  EXPECT_EQ("0", formatv("{0:d}", false).str());

  // 4. Y - uppercase yes/no
  EXPECT_EQ("YES", formatv("{0:Y}", true).str());
  EXPECT_EQ("NO", formatv("{0:Y}", false).str());

  // 5. y - lowercase yes/no
  EXPECT_EQ("yes", formatv("{0:y}", true).str());
  EXPECT_EQ("no", formatv("{0:y}", false).str());
}

TEST(FormatVariadicTest, DoubleFormatting) {
  // Test exponents, fixed point, and percent formatting.

  // 1. Signed, unsigned, and zero exponent format.
  EXPECT_EQ("0.000000E+00", formatv("{0:E}", 0.0).str());
  EXPECT_EQ("-0.000000E+00", formatv("{0:E}", -0.0).str());
  EXPECT_EQ("1.100000E+00", formatv("{0:E}", 1.1).str());
  EXPECT_EQ("-1.100000E+00", formatv("{0:E}", -1.1).str());
  EXPECT_EQ("1.234568E+03", formatv("{0:E}", 1234.5678).str());
  EXPECT_EQ("-1.234568E+03", formatv("{0:E}", -1234.5678).str());
  EXPECT_EQ("1.234568E-03", formatv("{0:E}", .0012345678).str());
  EXPECT_EQ("-1.234568E-03", formatv("{0:E}", -.0012345678).str());

  // 2. With padding and precision.
  EXPECT_EQ("  0.000E+00", formatv("{0,11:E3}", 0.0).str());
  EXPECT_EQ(" -1.100E+00", formatv("{0,11:E3}", -1.1).str());
  EXPECT_EQ("  1.235E+03", formatv("{0,11:E3}", 1234.5678).str());
  EXPECT_EQ(" -1.235E-03", formatv("{0,11:E3}", -.0012345678).str());

  // 3. Signed, unsigned, and zero fixed point format.
  EXPECT_EQ("0.00", formatv("{0:F}", 0.0).str());
  EXPECT_EQ("-0.00", formatv("{0:F}", -0.0).str());
  EXPECT_EQ("1.10", formatv("{0:F}", 1.1).str());
  EXPECT_EQ("-1.10", formatv("{0:F}", -1.1).str());
  EXPECT_EQ("1234.57", formatv("{0:F}", 1234.5678).str());
  EXPECT_EQ("-1234.57", formatv("{0:F}", -1234.5678).str());
  EXPECT_EQ("0.00", formatv("{0:F}", .0012345678).str());
  EXPECT_EQ("-0.00", formatv("{0:F}", -.0012345678).str());

  // 2. With padding and precision.
  EXPECT_EQ("   0.000", formatv("{0,8:F3}", 0.0).str());
  EXPECT_EQ("  -1.100", formatv("{0,8:F3}", -1.1).str());
  EXPECT_EQ("1234.568", formatv("{0,8:F3}", 1234.5678).str());
  EXPECT_EQ("  -0.001", formatv("{0,8:F3}", -.0012345678).str());
}

TEST(FormatVariadicTest, CustomPaddingCharacter) {
  // 1. Padding with custom character
  EXPECT_EQ("==123", formatv("{0,=+5}", 123).str());
  EXPECT_EQ("=123=", formatv("{0,==5}", 123).str());
  EXPECT_EQ("123==", formatv("{0,=-5}", 123).str());

  // 2. Combined with zero padding
  EXPECT_EQ("=00123=", formatv("{0,==7:5}", 123).str());
}

struct format_tuple {
  const char *Fmt;
  explicit format_tuple(const char *Fmt) : Fmt(Fmt) {}

  template <typename... Ts> auto operator()(Ts &&... Values) const {
    return formatv(Fmt, std::forward<Ts>(Values)...);
  }
};

TEST(FormatVariadicTest, BigTest) {
  using Tuple =
      std::tuple<char, int, const char *, StringRef, std::string, double, float,
                 void *, int, double, int64_t, uint64_t, double, uint8_t>;
  Tuple Ts[] = {
      Tuple('a', 1, "Str", StringRef(), std::string(), 3.14159, -.17532f,
            (void *)nullptr, 123456, 6.02E23, -908234908423, 908234908422234,
            std::numeric_limits<double>::quiet_NaN(), 0xAB),
      Tuple('x', 0xDDB5B, "LongerStr", "StringRef", "std::string", -2.7,
            .08215f, (void *)nullptr, 0, 6.62E-34, -908234908423,
            908234908422234, std::numeric_limits<double>::infinity(), 0x0)};
  // Test long string formatting with many edge cases combined.
  const char *Intro =
      "There are {{{0}} items in the tuple, and {{{1}} tuple(s) in the array.";
  const char *Header =
      "{0,6}|{1,8}|{2,=10}|{3,=10}|{4,=13}|{5,7}|{6,7}|{7,10}|{8,"
      "-7}|{9,10}|{10,16}|{11,17}|{12,6}|{13,4}";
  const char *Line =
      "{0,6}|{1,8:X}|{2,=10}|{3,=10:5}|{4,=13}|{5,7:3}|{6,7:P2}|{7,"
      "10:X8}|{8,-7:N}|{9,10:E4}|{10,16:N}|{11,17:D}|{12,6}|{13,"
      "4:X}";

  std::string S;
  llvm::raw_string_ostream Stream(S);
  Stream << formatv(Intro, std::tuple_size<Tuple>::value, std::size(Ts))
         << "\n";
  Stream << formatv(Header, "Char", "HexInt", "Str", "Ref", "std::str",
                    "double", "float", "pointer", "comma", "exp", "bigint",
                    "bigint2", "limit", "byte")
         << "\n";
  for (auto &Item : Ts) {
    Stream << std::apply(format_tuple(Line), Item) << "\n";
  }
  Stream.flush();
  const char *Expected =
      R"foo(There are {14} items in the tuple, and {2} tuple(s) in the array.
  Char|  HexInt|   Str    |   Ref    |  std::str   | double|  float|   pointer|comma  |       exp|          bigint|          bigint2| limit|byte
     a|     0x1|   Str    |          |             |  3.142|-17.53%|0x00000000|123,456|6.0200E+23|-908,234,908,423|  908234908422234|   nan|0xAB
     x| 0xDDB5B|LongerStr |  Strin   | std::string | -2.700|  8.21%|0x00000000|0      |6.6200E-34|-908,234,908,423|  908234908422234|   INF| 0x0
)foo";

  EXPECT_EQ(Expected, S);
}

TEST(FormatVariadicTest, Range) {
  std::vector<int> IntRange = {1, 1, 2, 3, 5, 8, 13};

  // 1. Simple range with default separator and element style.
  EXPECT_EQ("1, 1, 2, 3, 5, 8, 13",
            formatv("{0}", make_range(IntRange.begin(), IntRange.end())).str());
  EXPECT_EQ("1, 2, 3, 5, 8",
            formatv("{0}", make_range(IntRange.begin() + 1, IntRange.end() - 1))
                .str());

  // 2. Non-default separator
  EXPECT_EQ(
      "1/1/2/3/5/8/13",
      formatv("{0:$[/]}", make_range(IntRange.begin(), IntRange.end())).str());

  // 3. Default separator, non-default element style.
  EXPECT_EQ(
      "0x1, 0x1, 0x2, 0x3, 0x5, 0x8, 0xd",
      formatv("{0:@[x]}", make_range(IntRange.begin(), IntRange.end())).str());

  // 4. Non-default separator and element style.
  EXPECT_EQ(
      "0x1 + 0x1 + 0x2 + 0x3 + 0x5 + 0x8 + 0xd",
      formatv("{0:$[ + ]@[x]}", make_range(IntRange.begin(), IntRange.end()))
          .str());

  // 5. Element style and/or separator using alternate delimeters to allow using
  // delimeter characters as part of the separator.
  EXPECT_EQ(
      "<0x1><0x1><0x2><0x3><0x5><0x8><0xd>",
      formatv("<{0:$[><]@(x)}>", make_range(IntRange.begin(), IntRange.end()))
          .str());
  EXPECT_EQ(
      "[0x1][0x1][0x2][0x3][0x5][0x8][0xd]",
      formatv("[{0:$(][)@[x]}]", make_range(IntRange.begin(), IntRange.end()))
          .str());
  EXPECT_EQ(
      "(0x1)(0x1)(0x2)(0x3)(0x5)(0x8)(0xd)",
      formatv("({0:$<)(>@<x>})", make_range(IntRange.begin(), IntRange.end()))
          .str());

  // 5. Empty range.
  EXPECT_EQ("", formatv("{0:$[+]@[x]}",
                        make_range(IntRange.begin(), IntRange.begin()))
                    .str());

  // 6. Empty separator and style.
  EXPECT_EQ("11235813",
            formatv("{0:$[]@<>}", make_range(IntRange.begin(), IntRange.end()))
                .str());
}

TEST(FormatVariadicTest, Adapter) {
  class Negative : public FormatAdapter<int> {
  public:
    explicit Negative(int N) : FormatAdapter<int>(std::move(N)) {}
    void format(raw_ostream &S, StringRef Options) override { S << -Item; }
  };

  EXPECT_EQ("-7", formatv("{0}", Negative(7)).str());

  int N = 171;

  EXPECT_EQ("  171  ",
            formatv("{0}", fmt_align(N, AlignStyle::Center, 7)).str());
  EXPECT_EQ("--171--",
            formatv("{0}", fmt_align(N, AlignStyle::Center, 7, '-')).str());
  EXPECT_EQ(" 171   ", formatv("{0}", fmt_pad(N, 1, 3)).str());
  EXPECT_EQ("171171171171171", formatv("{0}", fmt_repeat(N, 5)).str());

  EXPECT_EQ(" ABABABABAB   ",
            formatv("{0:X-}", fmt_pad(fmt_repeat(N, 5), 1, 3)).str());
  EXPECT_EQ("   AB    AB    AB    AB    AB     ",
            formatv("{0,=34:X-}", fmt_repeat(fmt_pad(N, 1, 3), 5)).str());
}

TEST(FormatVariadicTest, MoveConstructor) {
  auto fmt = formatv("{0} {1}", 1, 2);
  auto fmt2 = std::move(fmt);
  std::string S = std::string(fmt2);
  EXPECT_EQ("1 2", S);
}
TEST(FormatVariadicTest, ImplicitConversions) {
  std::string S = std::string(formatv("{0} {1}", 1, 2));
  EXPECT_EQ("1 2", S);

  SmallString<4> S2 = formatv("{0} {1}", 1, 2);
  EXPECT_EQ("1 2", S2);
}

TEST(FormatVariadicTest, FormatAdapter) {
  EXPECT_EQ("Format", formatv("{0}", Format(1)).str());

  Format var(1);
  EXPECT_EQ("Format", formatv("{0}", var).str());
  EXPECT_EQ("Format", formatv("{0}", std::move(var)).str());

  // Not supposed to compile
  // const Format cvar(1);
  // EXPECT_EQ("Format", formatv("{0}", cvar).str());
}

TEST(FormatVariadicTest, FormatFormatvObject) {
  EXPECT_EQ("Format", formatv("F{0}t", formatv("o{0}a", "rm")).str());
  EXPECT_EQ("[   ! ]", formatv("[{0,+5}]", formatv("{0,-2}", "!")).str());
}

namespace {
struct Recorder {
  int Copied = 0, Moved = 0;
  Recorder() = default;
  Recorder(const Recorder &Copy) : Copied(1 + Copy.Copied), Moved(Copy.Moved) {}
  Recorder(const Recorder &&Move)
      : Copied(Move.Copied), Moved(1 + Move.Moved) {}
};
} // namespace
namespace llvm {
template <> struct format_provider<Recorder> {
  static void format(const Recorder &R, raw_ostream &OS, StringRef style) {
    OS << R.Copied << "C " << R.Moved << "M";
  }
};
} // namespace

TEST(FormatVariadicTest, CopiesAndMoves) {
  Recorder R;
  EXPECT_EQ("0C 0M", formatv("{0}", R).str());
  EXPECT_EQ("0C 3M", formatv("{0}", std::move(R)).str());
  EXPECT_EQ("0C 3M", formatv("{0}", Recorder()).str());
  EXPECT_EQ(0, R.Copied);
  EXPECT_EQ(0, R.Moved);
}

namespace adl {
struct X {};
raw_ostream &operator<<(raw_ostream &OS, const X &) { return OS << "X"; }
} // namespace adl
TEST(FormatVariadicTest, FormatStreamable) {
  adl::X X;
  EXPECT_EQ("X", formatv("{0}", X).str());
}

TEST(FormatVariadicTest, FormatError) {
  auto E1 = make_error<StringError>("X", inconvertibleErrorCode());
  EXPECT_EQ("X", formatv("{0}", E1).str());
  EXPECT_TRUE(E1.isA<StringError>()); // not consumed
  EXPECT_EQ("X", formatv("{0}", fmt_consume(std::move(E1))).str());
  EXPECT_FALSE(E1.isA<StringError>()); // consumed
}

TEST(FormatVariadicTest, FormatFilterRange) {
  std::vector<int> Vec{0, 1, 2};
  auto Range = map_range(Vec, [](int V) { return V + 1; });
  EXPECT_EQ("1, 2, 3", formatv("{0}", Range).str());
}

namespace {

enum class Base { First };

class IntegerValuesRange final
    : public indexed_accessor_range<IntegerValuesRange, Base, int, int *, int> {
public:
  using indexed_accessor_range<IntegerValuesRange, Base, int, int *,
                               int>::indexed_accessor_range;

  static int dereference(const Base &, ptrdiff_t Index) {
    return static_cast<int>(Index);
  }
};

TEST(FormatVariadicTest, FormatRangeNonRef) {
  IntegerValuesRange Range(Base(), 0, 3);
  EXPECT_EQ("0, 1, 2",
            formatv("{0}", make_range(Range.begin(), Range.end())).str());
}

} // namespace
