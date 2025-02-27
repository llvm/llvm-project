//===- unittest/Support/OptionParsingTest.cpp - OptTable tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::opt;

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#define OPTTABLE_STR_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_STR_TABLE_CODE

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
  LastOption
#undef OPTION
};

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

#define OPTTABLE_PREFIXES_UNION_CODE
#include "Opts.inc"
#undef OPTTABLE_PREFIXES_UNION_CODE

enum OptionFlags {
  OptFlag1 = (1 << 4),
  OptFlag2 = (1 << 5),
  OptFlag3 = (1 << 6)
};

enum OptionVisibility {
  SubtoolVis = (1 << 2),
  MultiLineVis = (1 << 3),
};

static constexpr OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

namespace {
class TestOptTable : public GenericOptTable {
public:
  TestOptTable(bool IgnoreCase = false)
      : GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable,
                        IgnoreCase) {}
};

class TestPrecomputedOptTable : public PrecomputedOptTable {
public:
  TestPrecomputedOptTable(bool IgnoreCase = false)
      : PrecomputedOptTable(OptionStrTable, OptionPrefixesTable, InfoTable,
                            OptionPrefixesUnion, IgnoreCase) {}
};
}

const char *Args[] = {
  "-A",
  "-Bhi",
  "--C=desu",
  "-C", "bye",
  "-D,adena",
  "-E", "apple", "bloom",
  "-Fblarg",
  "-F", "42",
  "-Gchuu", "2"
  };

// Test fixture
template <typename T> class OptTableTest : public ::testing::Test {};

template <typename T> class DISABLED_OptTableTest : public ::testing::Test {};

// Test both precomputed and computed OptTables with the same suite of tests.
using OptTableTestTypes =
    ::testing::Types<TestOptTable, TestPrecomputedOptTable>;

TYPED_TEST_SUITE(OptTableTest, OptTableTestTypes, );
TYPED_TEST_SUITE(DISABLED_OptTableTest, OptTableTestTypes, );

TYPED_TEST(OptTableTest, OptionParsing) {
  TypeParam T;
  unsigned MAI, MAC;
  InputArgList AL = T.ParseArgs(Args, MAI, MAC);

  // Check they all exist.
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_B));
  EXPECT_TRUE(AL.hasArg(OPT_C));
  EXPECT_TRUE(AL.hasArg(OPT_D));
  EXPECT_TRUE(AL.hasArg(OPT_E));
  EXPECT_TRUE(AL.hasArg(OPT_F));
  EXPECT_TRUE(AL.hasArg(OPT_G));

  // Check the values.
  EXPECT_EQ("hi", AL.getLastArgValue(OPT_B));
  EXPECT_EQ("bye", AL.getLastArgValue(OPT_C));
  EXPECT_EQ("adena", AL.getLastArgValue(OPT_D));
  std::vector<std::string> Es = AL.getAllArgValues(OPT_E);
  EXPECT_EQ("apple", Es[0]);
  EXPECT_EQ("bloom", Es[1]);
  EXPECT_EQ("42", AL.getLastArgValue(OPT_F));
  std::vector<std::string> Gs = AL.getAllArgValues(OPT_G);
  EXPECT_EQ("chuu", Gs[0]);
  EXPECT_EQ("2", Gs[1]);

  // Check the help text.
  std::string Help;
  raw_string_ostream RSO(Help);
  T.printHelp(RSO, "test", "title!");
  EXPECT_NE(std::string::npos, Help.find("-A"));

  // Check usage line.
  T.printHelp(RSO, "name [options] file...", "title!");
  EXPECT_NE(std::string::npos, Help.find("USAGE: name [options] file...\n"));

  // Test aliases.
  auto Cs = AL.filtered(OPT_C);
  ASSERT_NE(Cs.begin(), Cs.end());
  EXPECT_EQ("desu", StringRef((*Cs.begin())->getValue()));
  ArgStringList ASL;
  (*Cs.begin())->render(AL, ASL);
  ASSERT_EQ(2u, ASL.size());
  EXPECT_EQ("-C", StringRef(ASL[0]));
  EXPECT_EQ("desu", StringRef(ASL[1]));
}

TYPED_TEST(OptTableTest, ParseWithFlagExclusions) {
  TypeParam T;
  unsigned MAI, MAC;

  // Exclude flag3 to avoid parsing as OPT_SLASH_C.
  InputArgList AL = T.ParseArgs(Args, MAI, MAC,
                                /*FlagsToInclude=*/0,
                                /*FlagsToExclude=*/OptFlag3);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_C));
  EXPECT_FALSE(AL.hasArg(OPT_SLASH_C));

  // Exclude flag1 to avoid parsing as OPT_C.
  AL = T.ParseArgs(Args, MAI, MAC,
                   /*FlagsToInclude=*/0,
                   /*FlagsToExclude=*/OptFlag1);
  EXPECT_TRUE(AL.hasArg(OPT_B));
  EXPECT_FALSE(AL.hasArg(OPT_C));
  EXPECT_TRUE(AL.hasArg(OPT_SLASH_C));

  const char *NewArgs[] = { "/C", "foo", "--C=bar" };
  AL = T.ParseArgs(NewArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_SLASH_C));
  EXPECT_TRUE(AL.hasArg(OPT_C));
  EXPECT_EQ("foo", AL.getLastArgValue(OPT_SLASH_C));
  EXPECT_EQ("bar", AL.getLastArgValue(OPT_C));
}

TYPED_TEST(OptTableTest, ParseWithVisibility) {
  TypeParam T;
  unsigned MAI, MAC;

  const char *STArgs[] = {"-A", "-Q", "-R"};

  // With no visibility specified, we find all of the arguments.
  InputArgList AL = T.ParseArgs(STArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_Q));
  EXPECT_TRUE(AL.hasArg(OPT_R));

  // Default visibility omits SubtoolVis.
  AL = T.ParseArgs(STArgs, MAI, MAC, Visibility(DefaultVis));
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_FALSE(AL.hasArg(OPT_Q));
  EXPECT_TRUE(AL.hasArg(OPT_R));

  // ~SubtoolVis still finds arguments that are visible in Default.
  AL = T.ParseArgs(STArgs, MAI, MAC, Visibility(~SubtoolVis));
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_FALSE(AL.hasArg(OPT_Q));
  EXPECT_TRUE(AL.hasArg(OPT_R));

  // Only SubtoolVis.
  AL = T.ParseArgs(STArgs, MAI, MAC, Visibility(SubtoolVis));
  EXPECT_FALSE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_Q));
  EXPECT_TRUE(AL.hasArg(OPT_R));

  // Both Default and SubtoolVis are found.
  AL = T.ParseArgs(STArgs, MAI, MAC, Visibility(DefaultVis | SubtoolVis));
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_Q));
  EXPECT_TRUE(AL.hasArg(OPT_R));
}

TYPED_TEST(OptTableTest, ParseAliasInGroup) {
  TypeParam T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-I" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_H));
}

TYPED_TEST(OptTableTest, AliasArgs) {
  TypeParam T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-J", "-Joo" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_B));
  EXPECT_EQ("foo", AL.getAllArgValues(OPT_B)[0]);
  EXPECT_EQ("bar", AL.getAllArgValues(OPT_B)[1]);
}

TYPED_TEST(OptTableTest, IgnoreCase) {
  TypeParam T(true);
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-a", "-joo" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_B));
}

#if defined(__clang__)
// Disable the warning that triggers on exactly what is being tested.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-move"
#endif

TYPED_TEST(OptTableTest, InputArgListSelfAssign) {
  TypeParam T;
  unsigned MAI, MAC;
  InputArgList AL = T.ParseArgs(Args, MAI, MAC,
                                /*FlagsToInclude=*/0,
                                /*FlagsToExclude=*/OptFlag3);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_C));
  EXPECT_FALSE(AL.hasArg(OPT_SLASH_C));

  AL = std::move(AL);

  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_C));
  EXPECT_FALSE(AL.hasArg(OPT_SLASH_C));
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

TYPED_TEST(OptTableTest, DoNotIgnoreCase) {
  TypeParam T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-a", "-joo" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_FALSE(AL.hasArg(OPT_A));
  EXPECT_FALSE(AL.hasArg(OPT_B));
}

TYPED_TEST(OptTableTest, SlurpEmpty) {
  TypeParam T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-A", "-slurp" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_Slurp));
  EXPECT_EQ(0U, AL.getAllArgValues(OPT_Slurp).size());
}

TYPED_TEST(OptTableTest, Slurp) {
  TypeParam T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-A", "-slurp", "-B", "--", "foo" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_EQ(AL.size(), 2U);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_FALSE(AL.hasArg(OPT_B));
  EXPECT_TRUE(AL.hasArg(OPT_Slurp));
  EXPECT_EQ(3U, AL.getAllArgValues(OPT_Slurp).size());
  EXPECT_EQ("-B", AL.getAllArgValues(OPT_Slurp)[0]);
  EXPECT_EQ("--", AL.getAllArgValues(OPT_Slurp)[1]);
  EXPECT_EQ("foo", AL.getAllArgValues(OPT_Slurp)[2]);
}

TYPED_TEST(OptTableTest, SlurpJoinedEmpty) {
  TypeParam T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-A", "-slurpjoined" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_SlurpJoined));
  EXPECT_EQ(AL.getAllArgValues(OPT_SlurpJoined).size(), 0U);
}

TYPED_TEST(OptTableTest, SlurpJoinedOneJoined) {
  TypeParam T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-A", "-slurpjoinedfoo" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_SlurpJoined));
  EXPECT_EQ(AL.getAllArgValues(OPT_SlurpJoined).size(), 1U);
  EXPECT_EQ(AL.getAllArgValues(OPT_SlurpJoined)[0], "foo");
}

TYPED_TEST(OptTableTest, SlurpJoinedAndSeparate) {
  TypeParam T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-A", "-slurpjoinedfoo", "bar", "baz" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_SlurpJoined));
  EXPECT_EQ(3U, AL.getAllArgValues(OPT_SlurpJoined).size());
  EXPECT_EQ("foo", AL.getAllArgValues(OPT_SlurpJoined)[0]);
  EXPECT_EQ("bar", AL.getAllArgValues(OPT_SlurpJoined)[1]);
  EXPECT_EQ("baz", AL.getAllArgValues(OPT_SlurpJoined)[2]);
}

TYPED_TEST(OptTableTest, SlurpJoinedButSeparate) {
  TypeParam T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-A", "-slurpjoined", "foo", "bar", "baz" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_SlurpJoined));
  EXPECT_EQ(3U, AL.getAllArgValues(OPT_SlurpJoined).size());
  EXPECT_EQ("foo", AL.getAllArgValues(OPT_SlurpJoined)[0]);
  EXPECT_EQ("bar", AL.getAllArgValues(OPT_SlurpJoined)[1]);
  EXPECT_EQ("baz", AL.getAllArgValues(OPT_SlurpJoined)[2]);
}

TYPED_TEST(OptTableTest, FlagAliasToJoined) {
  TypeParam T;
  unsigned MAI, MAC;

  // Check that a flag alias provides an empty argument to a joined option.
  const char *MyArgs[] = { "-K" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_EQ(AL.size(), 1U);
  EXPECT_TRUE(AL.hasArg(OPT_B));
  EXPECT_EQ(1U, AL.getAllArgValues(OPT_B).size());
  EXPECT_EQ("", AL.getAllArgValues(OPT_B)[0]);
}

TYPED_TEST(OptTableTest, FindNearest) {
  TypeParam T;
  std::string Nearest;

  // Options that are too short should not be considered
  // "near" other short options.
  EXPECT_GT(T.findNearest("-A", Nearest), 4U);
  EXPECT_GT(T.findNearest("/C", Nearest), 4U);
  EXPECT_GT(T.findNearest("--C=foo", Nearest), 4U);

  // The nearest candidate should mirror the amount of prefix
  // characters used in the original string.
  EXPECT_EQ(1U, T.findNearest("-blorb", Nearest));
  EXPECT_EQ(Nearest, "-blorp");
  EXPECT_EQ(1U, T.findNearest("--blorm", Nearest));
  EXPECT_EQ(Nearest, "--blorp");
  EXPECT_EQ(1U, T.findNearest("-blarg", Nearest));
  EXPECT_EQ(Nearest, "-blarn");
  EXPECT_EQ(1U, T.findNearest("--blarm", Nearest));
  EXPECT_EQ(Nearest, "--blarn");
  EXPECT_EQ(1U, T.findNearest("-fjormp", Nearest));
  EXPECT_EQ(Nearest, "--fjormp");

  // The nearest candidate respects the prefix and value delimiter
  // of the original string.
  EXPECT_EQ(1U, T.findNearest("/framb:foo", Nearest));
  EXPECT_EQ(Nearest, "/cramb:foo");

  // `--glormp` should have an editing distance > 0 from `--glormp=`.
  EXPECT_GT(T.findNearest("--glorrmp", Nearest), 0U);
  EXPECT_EQ(Nearest, "--glorrmp=");
  EXPECT_EQ(0U, T.findNearest("--glorrmp=foo", Nearest));

  // `--blurmps` should correct to `--blurmp`, not `--blurmp=`, even though
  // both naively have an editing distance of 1.
  EXPECT_EQ(1U, T.findNearest("--blurmps", Nearest));
  EXPECT_EQ(Nearest, "--blurmp");

  // ...but `--blurmps=foo` should correct to `--blurmp=foo`.
  EXPECT_EQ(1U, T.findNearest("--blurmps=foo", Nearest));
  EXPECT_EQ(Nearest, "--blurmp=foo");

  // Flags should be included and excluded as specified.
  EXPECT_EQ(1U, T.findNearest("-doopf", Nearest,
                              /*FlagsToInclude=*/OptFlag2,
                              /*FlagsToExclude=*/0));
  EXPECT_EQ(Nearest, "-doopf2");
  EXPECT_EQ(1U, T.findNearest("-doopf", Nearest,
                              /*FlagsToInclude=*/0,
                              /*FlagsToExclude=*/OptFlag2));
  EXPECT_EQ(Nearest, "-doopf1");

  // Spelling should respect visibility.
  EXPECT_EQ(1U, T.findNearest("-xyzzy", Nearest, Visibility(DefaultVis)));
  EXPECT_EQ(Nearest, "-xyzzy2");
  EXPECT_EQ(1U, T.findNearest("-xyzzy", Nearest, Visibility(SubtoolVis)));
  EXPECT_EQ(Nearest, "-xyzzy1");
}

TYPED_TEST(DISABLED_OptTableTest, FindNearestFIXME) {
  TypeParam T;
  std::string Nearest;

  // FIXME: Options with joined values should not have those values considered
  // when calculating distance. The test below would fail if run, but it should
  // succeed.
  EXPECT_EQ(1U, T.findNearest("--erbghFoo", Nearest));
  EXPECT_EQ(Nearest, "--ermghFoo");
}

TYPED_TEST(OptTableTest, ParseGroupedShortOptions) {
  TypeParam T;
  T.setGroupedShortOptions(true);
  unsigned MAI, MAC;

  // Grouped short options can be followed by a long Flag (-Joo), or a non-Flag
  // option (-C=1).
  const char *Args1[] = {"-AIJ", "-AIJoo", "-AC=1"};
  InputArgList AL = T.ParseArgs(Args1, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_H));
  ASSERT_EQ((size_t)2, AL.getAllArgValues(OPT_B).size());
  EXPECT_EQ("foo", AL.getAllArgValues(OPT_B)[0]);
  EXPECT_EQ("bar", AL.getAllArgValues(OPT_B)[1]);
  ASSERT_TRUE(AL.hasArg(OPT_C));
  EXPECT_EQ("1", AL.getAllArgValues(OPT_C)[0]);

  // Prefer a long option to a short option.
  const char *Args2[] = {"-AB"};
  InputArgList AL2 = T.ParseArgs(Args2, MAI, MAC);
  EXPECT_TRUE(!AL2.hasArg(OPT_A));
  EXPECT_TRUE(AL2.hasArg(OPT_AB));

  // Short options followed by a long option. We probably should disallow this.
  const char *Args3[] = {"-AIblorp"};
  InputArgList AL3 = T.ParseArgs(Args3, MAI, MAC);
  EXPECT_TRUE(AL3.hasArg(OPT_A));
  EXPECT_TRUE(AL3.hasArg(OPT_Blorp));
}

TYPED_TEST(OptTableTest, ParseDashDash) {
  TypeParam T;
  T.setDashDashParsing(true);
  unsigned MAI, MAC;

  const char *Args1[] = {"-A", "--"};
  InputArgList AL = T.ParseArgs(Args1, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_EQ(size_t(0), AL.getAllArgValues(OPT_INPUT).size());
  EXPECT_EQ(size_t(0), AL.getAllArgValues(OPT_UNKNOWN).size());

  const char *Args2[] = {"-A", "--", "-A", "--", "-B"};
  AL = T.ParseArgs(Args2, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_FALSE(AL.hasArg(OPT_B));
  const std::vector<std::string> Input = AL.getAllArgValues(OPT_INPUT);
  ASSERT_EQ(size_t(3), Input.size());
  EXPECT_EQ("-A", Input[0]);
  EXPECT_EQ("--", Input[1]);
  EXPECT_EQ("-B", Input[2]);
  EXPECT_EQ(size_t(0), AL.getAllArgValues(OPT_UNKNOWN).size());

  T.setDashDashParsing(false);
  AL = T.ParseArgs(Args2, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_B));
  EXPECT_EQ(size_t(0), AL.getAllArgValues(OPT_INPUT).size());
  const std::vector<std::string> Unknown = AL.getAllArgValues(OPT_UNKNOWN);
  ASSERT_EQ(size_t(2), Unknown.size());
  EXPECT_EQ("--", Unknown[0]);
  EXPECT_EQ("--", Unknown[1]);
}

TYPED_TEST(OptTableTest, UnknownOptions) {
  TypeParam T;
  unsigned MAI, MAC;
  const char *Args[] = {"-u", "--long", "0"};
  for (int I = 0; I < 2; ++I) {
    T.setGroupedShortOptions(I != 0);
    InputArgList AL = T.ParseArgs(Args, MAI, MAC);
    const std::vector<std::string> Unknown = AL.getAllArgValues(OPT_UNKNOWN);
    ASSERT_EQ((size_t)2, Unknown.size());
    EXPECT_EQ("-u", Unknown[0]);
    EXPECT_EQ("--long", Unknown[1]);
  }
}

TYPED_TEST(OptTableTest, FlagsWithoutValues) {
  TypeParam T;
  T.setGroupedShortOptions(true);
  unsigned MAI, MAC;
  const char *Args[] = {"-A=1", "-A="};
  InputArgList AL = T.ParseArgs(Args, MAI, MAC);
  const std::vector<std::string> Unknown = AL.getAllArgValues(OPT_UNKNOWN);
  ASSERT_EQ((size_t)2, Unknown.size());
  EXPECT_EQ("-A=1", Unknown[0]);
  EXPECT_EQ("-A=", Unknown[1]);
}

TYPED_TEST(OptTableTest, UnknownGroupedShortOptions) {
  TypeParam T;
  T.setGroupedShortOptions(true);
  unsigned MAI, MAC;
  const char *Args[] = {"-AuzK", "-AuzK"};
  InputArgList AL = T.ParseArgs(Args, MAI, MAC);
  const std::vector<std::string> Unknown = AL.getAllArgValues(OPT_UNKNOWN);
  ASSERT_EQ((size_t)4, Unknown.size());
  EXPECT_EQ("-u", Unknown[0]);
  EXPECT_EQ("-z", Unknown[1]);
  EXPECT_EQ("-u", Unknown[2]);
  EXPECT_EQ("-z", Unknown[3]);
}

TYPED_TEST(OptTableTest, PrintMultilineHelpText) {
  TypeParam T;
  std::string Help;
  raw_string_ostream RSO(Help);
  T.printHelp(RSO, "usage", "title", /*ShowHidden=*/false,
              /*ShowAllAliases=*/false, Visibility(MultiLineVis));
  EXPECT_STREQ(Help.c_str(), R"(OVERVIEW: title

USAGE: usage

OPTIONS:
  -multiline-help-with-long-name
                  This a help text that has
                  multiple lines in it
                  and a long name
  -multiline-help This a help text that has
                  multiple lines in it
)");
}
