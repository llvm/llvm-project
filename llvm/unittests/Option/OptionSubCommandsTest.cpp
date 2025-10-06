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
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::opt;

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace {
enum ID {
  OPT_INVALID = 0,
#define OPTION(PREFIXES, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS,       \
               VISIBILITY, PARAM, HELPTEXT, HELPTEXTSFORVARIANTS, METAVAR,     \
               VALUES, SUBCOMMANDIDS_OFFSET)                                   \
  OPT_##ID,
#include "SubCommandOpts.inc"
#undef OPTION
};
#define OPTTABLE_STR_TABLE_CODE
#include "SubCommandOpts.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "SubCommandOpts.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

#define OPTTABLE_SUBCOMMAND_IDS_TABLE_CODE
#include "SubCommandOpts.inc"
#undef OPTTABLE_SUBCOMMAND_IDS_TABLE_CODE

#define OPTTABLE_SUBCOMMANDS_CODE
#include "SubCommandOpts.inc"
#undef OPTTABLE_SUBCOMMANDS_CODE

static constexpr OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "SubCommandOpts.inc"
#undef OPTION
};

class TestOptSubCommandTable : public GenericOptTable {
public:
  TestOptSubCommandTable(bool IgnoreCase = false)
      : GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable,
                        /*IgnoreCase=*/false, OptionSubCommands,
                        OptionSubCommandIDsTable) {}
};

// Test fixture
template <typename T> class OptSubCommandTableTest : public ::testing::Test {};

// Test both precomputed and computed OptTables with the same suite of tests.
using OptSubCommandTableTestTypes = ::testing::Types<TestOptSubCommandTable>;

TYPED_TEST_SUITE(OptSubCommandTableTest, OptSubCommandTableTestTypes, );

TYPED_TEST(OptSubCommandTableTest, SubCommandParsing) {
  TypeParam T;
  unsigned MAI, MAC;

  std::string ErrMsg;
  raw_string_ostream RSO1(ErrMsg);

  auto HandleMultipleSubcommands = [&](ArrayRef<StringRef> SubCommands) {
    ErrMsg.clear();
    RSO1 << "Multiple subcommands passed\n";
    for (auto SC : SubCommands)
      RSO1 << "\n" << SC;
  };

  auto HandleOtherPositionals = [&](ArrayRef<StringRef> Positionals) {
    ErrMsg.clear();
    RSO1 << "Unregistered positionals passed\n";
    for (auto SC : Positionals)
      RSO1 << "\n" << SC;
  };

  {
    // Test case 1: Toplevel option, no subcommand
    const char *Args[] = {"-version"};
    InputArgList AL = T.ParseArgs(Args, MAI, MAC);
    EXPECT_TRUE(AL.hasArg(OPT_version));
    StringRef SC = AL.getSubCommand(
        T.getSubCommands(), HandleMultipleSubcommands, HandleOtherPositionals);
    EXPECT_TRUE(SC.empty());
    EXPECT_FALSE(AL.hasArg(OPT_uppercase));
    EXPECT_FALSE(AL.hasArg(OPT_lowercase));
  }

  {
    // Test case 2: Subcommand 'foo' with its valid options
    const char *Args[] = {"foo", "-uppercase"};
    InputArgList AL = T.ParseArgs(Args, MAI, MAC);
    StringRef SC = AL.getSubCommand(
        T.getSubCommands(), HandleMultipleSubcommands, HandleOtherPositionals);
    EXPECT_EQ(SC, "foo");
    EXPECT_TRUE(AL.hasArg(OPT_uppercase));
    EXPECT_FALSE(AL.hasArg(OPT_lowercase));
    EXPECT_FALSE(AL.hasArg(OPT_version));
    EXPECT_EQ(std::string::npos, ErrMsg.find("Multiple subcommands passed"))
        << "Did not expect error message as this is a valid use case.";
    EXPECT_EQ(std::string::npos, ErrMsg.find("Unregistered positionals passed"))
        << "Did not expect error message as this is a valid use case.";
  }

  {
    // Test case 3: Check valid use of subcommand which follows a valid
    // subcommand option.
    const char *Args[] = {"-uppercase", "foo"};
    InputArgList AL = T.ParseArgs(Args, MAI, MAC);
    StringRef SC = AL.getSubCommand(
        T.getSubCommands(), HandleMultipleSubcommands, HandleOtherPositionals);
    EXPECT_EQ(SC, "foo");
    EXPECT_TRUE(AL.hasArg(OPT_uppercase));
    EXPECT_FALSE(AL.hasArg(OPT_lowercase));
    EXPECT_FALSE(AL.hasArg(OPT_version));
    EXPECT_EQ(std::string::npos, ErrMsg.find("Multiple subcommands passed"))
        << "Did not expect error message as this is a valid use case.";
    EXPECT_EQ(std::string::npos, ErrMsg.find("Unregistered positionals passed"))
        << "Did not expect error message as this is a valid use case.";
  }

  {
    // Test case 4: Check invalid use of passing multiple subcommands.
    const char *Args[] = {"-uppercase", "foo", "bar"};
    InputArgList AL = T.ParseArgs(Args, MAI, MAC);
    StringRef SC = AL.getSubCommand(
        T.getSubCommands(), HandleMultipleSubcommands, HandleOtherPositionals);
    // No valid subcommand should be returned as this is an invalid invocation.
    EXPECT_TRUE(SC.empty());
    // Expect the multiple subcommands error message.
    EXPECT_NE(std::string::npos, ErrMsg.find("Multiple subcommands passed"));
    EXPECT_NE(std::string::npos, ErrMsg.find("foo"));
    EXPECT_NE(std::string::npos, ErrMsg.find("bar"));
    EXPECT_EQ(std::string::npos, ErrMsg.find("Unregistered positionals passed"))
        << "Did not expect error message as this is a valid use case.";
  }

  {
    // Test case 5: Check invalid use of passing unregistered subcommands.
    const char *Args[] = {"foobar"};
    InputArgList AL = T.ParseArgs(Args, MAI, MAC);
    StringRef SC = AL.getSubCommand(
        T.getSubCommands(), HandleMultipleSubcommands, HandleOtherPositionals);
    // No valid subcommand should be returned as this is an invalid invocation.
    EXPECT_TRUE(SC.empty());
    // Expect the unregistered subcommands error message.
    EXPECT_NE(std::string::npos,
              ErrMsg.find("Unregistered positionals passed"));
    EXPECT_NE(std::string::npos, ErrMsg.find("foobar"));
  }

  {
    // Test case 6: Check invalid use of a valid subcommand which follows a
    // valid subcommand option but the option is not registered with the given
    // subcommand.
    const char *Args[] = {"-lowercase", "bar"};
    InputArgList AL = T.ParseArgs(Args, MAI, MAC);
    StringRef SC = AL.getSubCommand(
        T.getSubCommands(), HandleMultipleSubcommands, HandleOtherPositionals);
    auto HandleSubCommandArg = [&](ID OptionType) {
      if (!AL.hasArg(OptionType))
        return false;
      auto O = T.getOption(OptionType);
      if (!O.isRegisteredSC(SC)) {
        ErrMsg.clear();
        RSO1 << "Option [" << O.getName() << "] is not valid for SubCommand ["
             << SC << "]\n";
        return false;
      }
      return true;
    };
    EXPECT_EQ(SC, "bar");                  // valid subcommand
    EXPECT_TRUE(AL.hasArg(OPT_lowercase)); // valid option
    EXPECT_FALSE(HandleSubCommandArg(OPT_lowercase));
    EXPECT_NE(
        std::string::npos,
        ErrMsg.find("Option [lowercase] is not valid for SubCommand [bar]"));
  }
}

TYPED_TEST(OptSubCommandTableTest, SubCommandHelp) {
  TypeParam T;
  std::string Help;
  raw_string_ostream RSO(Help);

  // Toplevel help
  T.printHelp(RSO, "Test Usage String", "OverviewString");
  EXPECT_NE(std::string::npos, Help.find("OVERVIEW:"));
  EXPECT_NE(std::string::npos, Help.find("OverviewString"));
  EXPECT_NE(std::string::npos, Help.find("USAGE:"));
  EXPECT_NE(std::string::npos, Help.find("Test Usage String"));
  EXPECT_NE(std::string::npos, Help.find("SUBCOMMANDS:"));
  EXPECT_NE(std::string::npos, Help.find("foo"));
  EXPECT_NE(std::string::npos, Help.find("bar"));
  EXPECT_NE(std::string::npos, Help.find("HelpText for SubCommand foo."));
  EXPECT_NE(std::string::npos, Help.find("HelpText for SubCommand bar."));
  EXPECT_NE(std::string::npos, Help.find("OPTIONS:"));
  EXPECT_NE(std::string::npos, Help.find("--help"));
  EXPECT_NE(std::string::npos, Help.find("-version"));
  // uppercase is not a global option and should not be shown.
  EXPECT_EQ(std::string::npos, Help.find("-uppercase"));

  // Help for subcommand foo
  Help.clear();
  StringRef SC1 = "foo";
  T.printHelp(RSO, "Test Usage String", "OverviewString", false, false,
              Visibility(), SC1);
  EXPECT_NE(std::string::npos, Help.find("OVERVIEW:"));
  EXPECT_NE(std::string::npos, Help.find("OverviewString"));
  // SubCommand "foo" definition for tablegen has NO dedicated usage string so
  // not expected to see USAGE.
  EXPECT_EQ(std::string::npos, Help.find("USAGE:"));
  EXPECT_NE(std::string::npos, Help.find("HelpText for SubCommand foo."));
  EXPECT_NE(std::string::npos, Help.find("-uppercase"));
  EXPECT_NE(std::string::npos, Help.find("-lowercase"));
  EXPECT_EQ(std::string::npos, Help.find("-version"));
  EXPECT_EQ(std::string::npos, Help.find("SUBCOMMANDS:"));

  // Help for subcommand bar
  Help.clear();
  StringRef SC2 = "bar";
  T.printHelp(RSO, "Test Usage String", "OverviewString", false, false,
              Visibility(), SC2);
  EXPECT_NE(std::string::npos, Help.find("OVERVIEW:"));
  EXPECT_NE(std::string::npos, Help.find("OverviewString"));
  // SubCommand "bar" definition for tablegen has a dedicated usage string.
  EXPECT_NE(std::string::npos, Help.find("USAGE:"));
  EXPECT_NE(std::string::npos, Help.find("Subcommand bar <options>"));
  EXPECT_NE(std::string::npos, Help.find("HelpText for SubCommand bar."));
  EXPECT_NE(std::string::npos, Help.find("-uppercase"));
  // lowercase is not an option for bar and should not be shown.
  EXPECT_EQ(std::string::npos, Help.find("-lowercase"));
  // version is a global option and should not be shown.
  EXPECT_EQ(std::string::npos, Help.find("-version"));
}
} // end anonymous namespace
