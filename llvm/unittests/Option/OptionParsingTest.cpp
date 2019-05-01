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
#include "llvm/Option/Option.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::opt;

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  OPT_##ID,
#include "Opts.inc"
  LastOption
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Opts.inc"
#undef PREFIX

enum OptionFlags {
  OptFlag1 = (1 << 4),
  OptFlag2 = (1 << 5),
  OptFlag3 = (1 << 6)
};

static const OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {PREFIX, NAME,  HELPTEXT,    METAVAR,     OPT_##ID,  Option::KIND##Class,    \
   PARAM,  FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS, VALUES},
#include "Opts.inc"
#undef OPTION
};

namespace {
class TestOptTable : public OptTable {
public:
  TestOptTable(bool IgnoreCase = false)
    : OptTable(InfoTable, IgnoreCase) {}
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

TEST(Option, OptionParsing) {
  TestOptTable T;
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
  T.PrintHelp(RSO, "test", "title!");
  EXPECT_NE(std::string::npos, Help.find("-A"));

  // Check usage line.
  T.PrintHelp(RSO, "name [options] file...", "title!");
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

TEST(Option, ParseWithFlagExclusions) {
  TestOptTable T;
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

TEST(Option, ParseAliasInGroup) {
  TestOptTable T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-I" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_H));
}

TEST(Option, AliasArgs) {
  TestOptTable T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-J", "-Joo" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_B));
  EXPECT_EQ("foo", AL.getAllArgValues(OPT_B)[0]);
  EXPECT_EQ("bar", AL.getAllArgValues(OPT_B)[1]);
}

TEST(Option, IgnoreCase) {
  TestOptTable T(true);
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-a", "-joo" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_B));
}

TEST(Option, DoNotIgnoreCase) {
  TestOptTable T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-a", "-joo" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_FALSE(AL.hasArg(OPT_A));
  EXPECT_FALSE(AL.hasArg(OPT_B));
}

TEST(Option, SlurpEmpty) {
  TestOptTable T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-A", "-slurp" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_Slurp));
  EXPECT_EQ(0U, AL.getAllArgValues(OPT_Slurp).size());
}

TEST(Option, Slurp) {
  TestOptTable T;
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

TEST(Option, SlurpJoinedEmpty) {
  TestOptTable T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-A", "-slurpjoined" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_SlurpJoined));
  EXPECT_EQ(AL.getAllArgValues(OPT_SlurpJoined).size(), 0U);
}

TEST(Option, SlurpJoinedOneJoined) {
  TestOptTable T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-A", "-slurpjoinedfoo" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_TRUE(AL.hasArg(OPT_A));
  EXPECT_TRUE(AL.hasArg(OPT_SlurpJoined));
  EXPECT_EQ(AL.getAllArgValues(OPT_SlurpJoined).size(), 1U);
  EXPECT_EQ(AL.getAllArgValues(OPT_SlurpJoined)[0], "foo");
}

TEST(Option, SlurpJoinedAndSeparate) {
  TestOptTable T;
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

TEST(Option, SlurpJoinedButSeparate) {
  TestOptTable T;
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

TEST(Option, FlagAliasToJoined) {
  TestOptTable T;
  unsigned MAI, MAC;

  // Check that a flag alias provides an empty argument to a joined option.
  const char *MyArgs[] = { "-K" };
  InputArgList AL = T.ParseArgs(MyArgs, MAI, MAC);
  EXPECT_EQ(AL.size(), 1U);
  EXPECT_TRUE(AL.hasArg(OPT_B));
  EXPECT_EQ(1U, AL.getAllArgValues(OPT_B).size());
  EXPECT_EQ("", AL.getAllArgValues(OPT_B)[0]);
}

TEST(Option, FindNearest) {
  TestOptTable T;
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

  // Flags should be included and excluded as specified.
  EXPECT_EQ(1U, T.findNearest("-doopf", Nearest, /*FlagsToInclude=*/OptFlag2));
  EXPECT_EQ(Nearest, "-doopf2");
  EXPECT_EQ(1U, T.findNearest("-doopf", Nearest,
                              /*FlagsToInclude=*/0,
                              /*FlagsToExclude=*/OptFlag2));
  EXPECT_EQ(Nearest, "-doopf1");
}

TEST(DISABLED_Option, FindNearestFIXME) {
  TestOptTable T;
  std::string Nearest;

  // FIXME: Options with joined values should not have those values considered
  // when calculating distance. The test below would fail if run, but it should
  // succeed.
  EXPECT_EQ(1U, T.findNearest("--erbghFoo", Nearest));
  EXPECT_EQ(Nearest, "--ermghFoo");

}
