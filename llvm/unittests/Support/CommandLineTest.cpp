//===- llvm/unittest/Support/CommandLineTest.cpp - CommandLine tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Config/config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/StringSaver.h"
#include "gtest/gtest.h"
#include <fstream>
#include <stdlib.h>
#include <string>

using namespace llvm;

namespace {

class TempEnvVar {
 public:
  TempEnvVar(const char *name, const char *value)
      : name(name) {
    const char *old_value = getenv(name);
    EXPECT_EQ(nullptr, old_value) << old_value;
#if HAVE_SETENV
    setenv(name, value, true);
#else
#   define SKIP_ENVIRONMENT_TESTS
#endif
  }

  ~TempEnvVar() {
#if HAVE_SETENV
    // Assume setenv and unsetenv come together.
    unsetenv(name);
#else
    (void)name; // Suppress -Wunused-private-field.
#endif
  }

 private:
  const char *const name;
};

template <typename T, typename Base = cl::opt<T>>
class StackOption : public Base {
public:
  template <class... Ts>
  explicit StackOption(Ts &&... Ms) : Base(std::forward<Ts>(Ms)...) {}

  ~StackOption() override { this->removeArgument(); }

  template <class DT> StackOption<T> &operator=(const DT &V) {
    this->setValue(V);
    return *this;
  }
};

class StackSubCommand : public cl::SubCommand {
public:
  StackSubCommand(StringRef Name,
                  StringRef Description = StringRef())
      : SubCommand(Name, Description) {}

  StackSubCommand() : SubCommand() {}

  ~StackSubCommand() { unregisterSubCommand(); }
};


cl::OptionCategory TestCategory("Test Options", "Description");
TEST(CommandLineTest, ModifyExisitingOption) {
  StackOption<int> TestOption("test-option", cl::desc("old description"));

  static const char Description[] = "New description";
  static const char ArgString[] = "new-test-option";
  static const char ValueString[] = "Integer";

  StringMap<cl::Option *> &Map =
      cl::getRegisteredOptions(*cl::TopLevelSubCommand);

  ASSERT_TRUE(Map.count("test-option") == 1) <<
    "Could not find option in map.";

  cl::Option *Retrieved = Map["test-option"];
  ASSERT_EQ(&TestOption, Retrieved) << "Retrieved wrong option.";

  ASSERT_EQ(&cl::GeneralCategory,Retrieved->Category) <<
    "Incorrect default option category.";

  Retrieved->setCategory(TestCategory);
  ASSERT_EQ(&TestCategory,Retrieved->Category) <<
    "Failed to modify option's option category.";

  Retrieved->setDescription(Description);
  ASSERT_STREQ(Retrieved->HelpStr.data(), Description)
      << "Changing option description failed.";

  Retrieved->setArgStr(ArgString);
  ASSERT_STREQ(ArgString, Retrieved->ArgStr.data())
      << "Failed to modify option's Argument string.";

  Retrieved->setValueStr(ValueString);
  ASSERT_STREQ(Retrieved->ValueStr.data(), ValueString)
      << "Failed to modify option's Value string.";

  Retrieved->setHiddenFlag(cl::Hidden);
  ASSERT_EQ(cl::Hidden, TestOption.getOptionHiddenFlag()) <<
    "Failed to modify option's hidden flag.";
}
#ifndef SKIP_ENVIRONMENT_TESTS

const char test_env_var[] = "LLVM_TEST_COMMAND_LINE_FLAGS";

cl::opt<std::string> EnvironmentTestOption("env-test-opt");
TEST(CommandLineTest, ParseEnvironment) {
  TempEnvVar TEV(test_env_var, "-env-test-opt=hello");
  EXPECT_EQ("", EnvironmentTestOption);
  cl::ParseEnvironmentOptions("CommandLineTest", test_env_var);
  EXPECT_EQ("hello", EnvironmentTestOption);
}

// This test used to make valgrind complain
// ("Conditional jump or move depends on uninitialised value(s)")
//
// Warning: Do not run any tests after this one that try to gain access to
// registered command line options because this will likely result in a
// SEGFAULT. This can occur because the cl::opt in the test below is declared
// on the stack which will be destroyed after the test completes but the
// command line system will still hold a pointer to a deallocated cl::Option.
TEST(CommandLineTest, ParseEnvironmentToLocalVar) {
  // Put cl::opt on stack to check for proper initialization of fields.
  StackOption<std::string> EnvironmentTestOptionLocal("env-test-opt-local");
  TempEnvVar TEV(test_env_var, "-env-test-opt-local=hello-local");
  EXPECT_EQ("", EnvironmentTestOptionLocal);
  cl::ParseEnvironmentOptions("CommandLineTest", test_env_var);
  EXPECT_EQ("hello-local", EnvironmentTestOptionLocal);
}

#endif  // SKIP_ENVIRONMENT_TESTS

TEST(CommandLineTest, UseOptionCategory) {
  StackOption<int> TestOption2("test-option", cl::cat(TestCategory));

  ASSERT_EQ(&TestCategory,TestOption2.Category) << "Failed to assign Option "
                                                  "Category.";
}

typedef void ParserFunction(StringRef Source, StringSaver &Saver,
                            SmallVectorImpl<const char *> &NewArgv,
                            bool MarkEOLs);

void testCommandLineTokenizer(ParserFunction *parse, StringRef Input,
                              const char *const Output[], size_t OutputSize) {
  SmallVector<const char *, 0> Actual;
  BumpPtrAllocator A;
  StringSaver Saver(A);
  parse(Input, Saver, Actual, /*MarkEOLs=*/false);
  EXPECT_EQ(OutputSize, Actual.size());
  for (unsigned I = 0, E = Actual.size(); I != E; ++I) {
    if (I < OutputSize) {
      EXPECT_STREQ(Output[I], Actual[I]);
    }
  }
}

TEST(CommandLineTest, TokenizeGNUCommandLine) {
  const char Input[] =
      "foo\\ bar \"foo bar\" \'foo bar\' 'foo\\\\bar' -DFOO=bar\\(\\) "
      "foo\"bar\"baz C:\\\\src\\\\foo.cpp \"C:\\src\\foo.cpp\"";
  const char *const Output[] = {
      "foo bar",     "foo bar",   "foo bar",          "foo\\bar",
      "-DFOO=bar()", "foobarbaz", "C:\\src\\foo.cpp", "C:srcfoo.cpp"};
  testCommandLineTokenizer(cl::TokenizeGNUCommandLine, Input, Output,
                           array_lengthof(Output));
}

TEST(CommandLineTest, TokenizeWindowsCommandLine) {
  const char Input[] = "a\\b c\\\\d e\\\\\"f g\" h\\\"i j\\\\\\\"k \"lmn\" o pqr "
                      "\"st \\\"u\" \\v";
  const char *const Output[] = { "a\\b", "c\\\\d", "e\\f g", "h\"i", "j\\\"k",
                                 "lmn", "o", "pqr", "st \"u", "\\v" };
  testCommandLineTokenizer(cl::TokenizeWindowsCommandLine, Input, Output,
                           array_lengthof(Output));
}

TEST(CommandLineTest, TokenizeConfigFile1) {
  const char *Input = "\\";
  const char *const Output[] = { "\\" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output,
                           array_lengthof(Output));
}

TEST(CommandLineTest, TokenizeConfigFile2) {
  const char *Input = "\\abc";
  const char *const Output[] = { "abc" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output,
                           array_lengthof(Output));
}

TEST(CommandLineTest, TokenizeConfigFile3) {
  const char *Input = "abc\\";
  const char *const Output[] = { "abc\\" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output,
                           array_lengthof(Output));
}

TEST(CommandLineTest, TokenizeConfigFile4) {
  const char *Input = "abc\\\n123";
  const char *const Output[] = { "abc123" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output,
                           array_lengthof(Output));
}

TEST(CommandLineTest, TokenizeConfigFile5) {
  const char *Input = "abc\\\r\n123";
  const char *const Output[] = { "abc123" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output,
                           array_lengthof(Output));
}

TEST(CommandLineTest, TokenizeConfigFile6) {
  const char *Input = "abc\\\n";
  const char *const Output[] = { "abc" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output,
                           array_lengthof(Output));
}

TEST(CommandLineTest, TokenizeConfigFile7) {
  const char *Input = "abc\\\r\n";
  const char *const Output[] = { "abc" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output,
                           array_lengthof(Output));
}

TEST(CommandLineTest, TokenizeConfigFile8) {
  SmallVector<const char *, 0> Actual;
  BumpPtrAllocator A;
  StringSaver Saver(A);
  cl::tokenizeConfigFile("\\\n", Saver, Actual, /*MarkEOLs=*/false);
  EXPECT_TRUE(Actual.empty());
}

TEST(CommandLineTest, TokenizeConfigFile9) {
  SmallVector<const char *, 0> Actual;
  BumpPtrAllocator A;
  StringSaver Saver(A);
  cl::tokenizeConfigFile("\\\r\n", Saver, Actual, /*MarkEOLs=*/false);
  EXPECT_TRUE(Actual.empty());
}

TEST(CommandLineTest, TokenizeConfigFile10) {
  const char *Input = "\\\nabc";
  const char *const Output[] = { "abc" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output,
                           array_lengthof(Output));
}

TEST(CommandLineTest, TokenizeConfigFile11) {
  const char *Input = "\\\r\nabc";
  const char *const Output[] = { "abc" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output,
                           array_lengthof(Output));
}

TEST(CommandLineTest, AliasesWithArguments) {
  static const size_t ARGC = 3;
  const char *const Inputs[][ARGC] = {
    { "-tool", "-actual=x", "-extra" },
    { "-tool", "-actual", "x" },
    { "-tool", "-alias=x", "-extra" },
    { "-tool", "-alias", "x" }
  };

  for (size_t i = 0, e = array_lengthof(Inputs); i < e; ++i) {
    StackOption<std::string> Actual("actual");
    StackOption<bool> Extra("extra");
    StackOption<std::string> Input(cl::Positional);

    cl::alias Alias("alias", llvm::cl::aliasopt(Actual));

    cl::ParseCommandLineOptions(ARGC, Inputs[i]);
    EXPECT_EQ("x", Actual);
    EXPECT_EQ(0, Input.getNumOccurrences());

    Alias.removeArgument();
  }
}

void testAliasRequired(int argc, const char *const *argv) {
  StackOption<std::string> Option("option", cl::Required);
  cl::alias Alias("o", llvm::cl::aliasopt(Option));

  cl::ParseCommandLineOptions(argc, argv);
  EXPECT_EQ("x", Option);
  EXPECT_EQ(1, Option.getNumOccurrences());

  Alias.removeArgument();
}

TEST(CommandLineTest, AliasRequired) {
  const char *opts1[] = { "-tool", "-option=x" };
  const char *opts2[] = { "-tool", "-o", "x" };
  testAliasRequired(array_lengthof(opts1), opts1);
  testAliasRequired(array_lengthof(opts2), opts2);
}

TEST(CommandLineTest, HideUnrelatedOptions) {
  StackOption<int> TestOption1("hide-option-1");
  StackOption<int> TestOption2("hide-option-2", cl::cat(TestCategory));

  cl::HideUnrelatedOptions(TestCategory);

  ASSERT_EQ(cl::ReallyHidden, TestOption1.getOptionHiddenFlag())
      << "Failed to hide extra option.";
  ASSERT_EQ(cl::NotHidden, TestOption2.getOptionHiddenFlag())
      << "Hid extra option that should be visable.";

  StringMap<cl::Option *> &Map =
      cl::getRegisteredOptions(*cl::TopLevelSubCommand);
  ASSERT_EQ(cl::NotHidden, Map["help"]->getOptionHiddenFlag())
      << "Hid default option that should be visable.";
}

cl::OptionCategory TestCategory2("Test Options set 2", "Description");

TEST(CommandLineTest, HideUnrelatedOptionsMulti) {
  StackOption<int> TestOption1("multi-hide-option-1");
  StackOption<int> TestOption2("multi-hide-option-2", cl::cat(TestCategory));
  StackOption<int> TestOption3("multi-hide-option-3", cl::cat(TestCategory2));

  const cl::OptionCategory *VisibleCategories[] = {&TestCategory,
                                                   &TestCategory2};

  cl::HideUnrelatedOptions(makeArrayRef(VisibleCategories));

  ASSERT_EQ(cl::ReallyHidden, TestOption1.getOptionHiddenFlag())
      << "Failed to hide extra option.";
  ASSERT_EQ(cl::NotHidden, TestOption2.getOptionHiddenFlag())
      << "Hid extra option that should be visable.";
  ASSERT_EQ(cl::NotHidden, TestOption3.getOptionHiddenFlag())
      << "Hid extra option that should be visable.";

  StringMap<cl::Option *> &Map =
      cl::getRegisteredOptions(*cl::TopLevelSubCommand);
  ASSERT_EQ(cl::NotHidden, Map["help"]->getOptionHiddenFlag())
      << "Hid default option that should be visable.";
}

TEST(CommandLineTest, SetValueInSubcategories) {
  cl::ResetCommandLineParser();

  StackSubCommand SC1("sc1", "First subcommand");
  StackSubCommand SC2("sc2", "Second subcommand");

  StackOption<bool> TopLevelOpt("top-level", cl::init(false));
  StackOption<bool> SC1Opt("sc1", cl::sub(SC1), cl::init(false));
  StackOption<bool> SC2Opt("sc2", cl::sub(SC2), cl::init(false));

  EXPECT_FALSE(TopLevelOpt);
  EXPECT_FALSE(SC1Opt);
  EXPECT_FALSE(SC2Opt);
  const char *args[] = {"prog", "-top-level"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(TopLevelOpt);
  EXPECT_FALSE(SC1Opt);
  EXPECT_FALSE(SC2Opt);

  TopLevelOpt = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(TopLevelOpt);
  EXPECT_FALSE(SC1Opt);
  EXPECT_FALSE(SC2Opt);
  const char *args2[] = {"prog", "sc1", "-sc1"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(3, args2, StringRef(), &llvm::nulls()));
  EXPECT_FALSE(TopLevelOpt);
  EXPECT_TRUE(SC1Opt);
  EXPECT_FALSE(SC2Opt);

  SC1Opt = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(TopLevelOpt);
  EXPECT_FALSE(SC1Opt);
  EXPECT_FALSE(SC2Opt);
  const char *args3[] = {"prog", "sc2", "-sc2"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(3, args3, StringRef(), &llvm::nulls()));
  EXPECT_FALSE(TopLevelOpt);
  EXPECT_FALSE(SC1Opt);
  EXPECT_TRUE(SC2Opt);
}

TEST(CommandLineTest, LookupFailsInWrongSubCommand) {
  cl::ResetCommandLineParser();

  StackSubCommand SC1("sc1", "First subcommand");
  StackSubCommand SC2("sc2", "Second subcommand");

  StackOption<bool> SC1Opt("sc1", cl::sub(SC1), cl::init(false));
  StackOption<bool> SC2Opt("sc2", cl::sub(SC2), cl::init(false));

  std::string Errs;
  raw_string_ostream OS(Errs);

  const char *args[] = {"prog", "sc1", "-sc2"};
  EXPECT_FALSE(cl::ParseCommandLineOptions(3, args, StringRef(), &OS));
  OS.flush();
  EXPECT_FALSE(Errs.empty());
}

TEST(CommandLineTest, AddToAllSubCommands) {
  cl::ResetCommandLineParser();

  StackSubCommand SC1("sc1", "First subcommand");
  StackOption<bool> AllOpt("everywhere", cl::sub(*cl::AllSubCommands),
                           cl::init(false));
  StackSubCommand SC2("sc2", "Second subcommand");

  const char *args[] = {"prog", "-everywhere"};
  const char *args2[] = {"prog", "sc1", "-everywhere"};
  const char *args3[] = {"prog", "sc2", "-everywhere"};

  std::string Errs;
  raw_string_ostream OS(Errs);

  EXPECT_FALSE(AllOpt);
  EXPECT_TRUE(cl::ParseCommandLineOptions(2, args, StringRef(), &OS));
  EXPECT_TRUE(AllOpt);

  AllOpt = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(AllOpt);
  EXPECT_TRUE(cl::ParseCommandLineOptions(3, args2, StringRef(), &OS));
  EXPECT_TRUE(AllOpt);

  AllOpt = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(AllOpt);
  EXPECT_TRUE(cl::ParseCommandLineOptions(3, args3, StringRef(), &OS));
  EXPECT_TRUE(AllOpt);

  // Since all parsing succeeded, the error message should be empty.
  OS.flush();
  EXPECT_TRUE(Errs.empty());
}

TEST(CommandLineTest, ReparseCommandLineOptions) {
  cl::ResetCommandLineParser();

  StackOption<bool> TopLevelOpt("top-level", cl::sub(*cl::TopLevelSubCommand),
                                cl::init(false));

  const char *args[] = {"prog", "-top-level"};

  EXPECT_FALSE(TopLevelOpt);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(TopLevelOpt);

  TopLevelOpt = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(TopLevelOpt);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(TopLevelOpt);
}

TEST(CommandLineTest, RemoveFromRegularSubCommand) {
  cl::ResetCommandLineParser();

  StackSubCommand SC("sc", "Subcommand");
  StackOption<bool> RemoveOption("remove-option", cl::sub(SC), cl::init(false));
  StackOption<bool> KeepOption("keep-option", cl::sub(SC), cl::init(false));

  const char *args[] = {"prog", "sc", "-remove-option"};

  std::string Errs;
  raw_string_ostream OS(Errs);

  EXPECT_FALSE(RemoveOption);
  EXPECT_TRUE(cl::ParseCommandLineOptions(3, args, StringRef(), &OS));
  EXPECT_TRUE(RemoveOption);
  OS.flush();
  EXPECT_TRUE(Errs.empty());

  RemoveOption.removeArgument();

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(cl::ParseCommandLineOptions(3, args, StringRef(), &OS));
  OS.flush();
  EXPECT_FALSE(Errs.empty());
}

TEST(CommandLineTest, RemoveFromTopLevelSubCommand) {
  cl::ResetCommandLineParser();

  StackOption<bool> TopLevelRemove(
      "top-level-remove", cl::sub(*cl::TopLevelSubCommand), cl::init(false));
  StackOption<bool> TopLevelKeep(
      "top-level-keep", cl::sub(*cl::TopLevelSubCommand), cl::init(false));

  const char *args[] = {"prog", "-top-level-remove"};

  EXPECT_FALSE(TopLevelRemove);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(TopLevelRemove);

  TopLevelRemove.removeArgument();

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
}

TEST(CommandLineTest, RemoveFromAllSubCommands) {
  cl::ResetCommandLineParser();

  StackSubCommand SC1("sc1", "First Subcommand");
  StackSubCommand SC2("sc2", "Second Subcommand");
  StackOption<bool> RemoveOption("remove-option", cl::sub(*cl::AllSubCommands),
                                 cl::init(false));
  StackOption<bool> KeepOption("keep-option", cl::sub(*cl::AllSubCommands),
                               cl::init(false));

  const char *args0[] = {"prog", "-remove-option"};
  const char *args1[] = {"prog", "sc1", "-remove-option"};
  const char *args2[] = {"prog", "sc2", "-remove-option"};

  // It should work for all subcommands including the top-level.
  EXPECT_FALSE(RemoveOption);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args0, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(RemoveOption);

  RemoveOption = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(RemoveOption);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(3, args1, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(RemoveOption);

  RemoveOption = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(RemoveOption);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(3, args2, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(RemoveOption);

  RemoveOption.removeArgument();

  // It should not work for any subcommands including the top-level.
  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(2, args0, StringRef(), &llvm::nulls()));
  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(3, args1, StringRef(), &llvm::nulls()));
  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(3, args2, StringRef(), &llvm::nulls()));
}

TEST(CommandLineTest, GetRegisteredSubcommands) {
  cl::ResetCommandLineParser();

  StackSubCommand SC1("sc1", "First Subcommand");
  StackOption<bool> Opt1("opt1", cl::sub(SC1), cl::init(false));
  StackSubCommand SC2("sc2", "Second subcommand");
  StackOption<bool> Opt2("opt2", cl::sub(SC2), cl::init(false));

  const char *args0[] = {"prog", "sc1"};
  const char *args1[] = {"prog", "sc2"};

  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args0, StringRef(), &llvm::nulls()));
  EXPECT_FALSE(Opt1);
  EXPECT_FALSE(Opt2);
  for (auto *S : cl::getRegisteredSubcommands()) {
    if (*S) {
      EXPECT_EQ("sc1", S->getName());
    }
  }

  cl::ResetAllOptionOccurrences();
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args1, StringRef(), &llvm::nulls()));
  EXPECT_FALSE(Opt1);
  EXPECT_FALSE(Opt2);
  for (auto *S : cl::getRegisteredSubcommands()) {
    if (*S) {
      EXPECT_EQ("sc2", S->getName());
    }
  }
}

TEST(CommandLineTest, ArgumentLimit) {
  std::string args(32 * 4096, 'a');
  EXPECT_FALSE(llvm::sys::commandLineFitsWithinSystemLimits("cl", args.data()));
}

TEST(CommandLineTest, ResponseFileWindows) {
  if (!Triple(sys::getProcessTriple()).isOSWindows())
    return;

  StackOption<std::string, cl::list<std::string>> InputFilenames(
      cl::Positional, cl::desc("<input files>"), cl::ZeroOrMore);
  StackOption<bool> TopLevelOpt("top-level", cl::init(false));

  // Create response file.
  int FileDescriptor;
  SmallString<64> TempPath;
  std::error_code EC =
      llvm::sys::fs::createTemporaryFile("resp-", ".txt", FileDescriptor, TempPath);
  EXPECT_TRUE(!EC);

  std::ofstream RspFile(TempPath.c_str());
  EXPECT_TRUE(RspFile.is_open());
  RspFile << "-top-level\npath\\dir\\file1\npath/dir/file2";
  RspFile.close();

  llvm::SmallString<128> RspOpt;
  RspOpt.append(1, '@');
  RspOpt.append(TempPath.c_str());
  const char *args[] = {"prog", RspOpt.c_str()};
  EXPECT_FALSE(TopLevelOpt);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(TopLevelOpt);
  EXPECT_TRUE(InputFilenames[0] == "path\\dir\\file1");
  EXPECT_TRUE(InputFilenames[1] == "path/dir/file2");

  llvm::sys::fs::remove(TempPath.c_str());
}

TEST(CommandLineTest, ResponseFiles) {
  llvm::SmallString<128> TestDir;
  std::error_code EC =
    llvm::sys::fs::createUniqueDirectory("unittest", TestDir);
  EXPECT_TRUE(!EC);

  // Create included response file of first level.
  llvm::SmallString<128> IncludedFileName;
  llvm::sys::path::append(IncludedFileName, TestDir, "resp1");
  std::ofstream IncludedFile(IncludedFileName.c_str());
  EXPECT_TRUE(IncludedFile.is_open());
  IncludedFile << "-option_1 -option_2\n"
                  "@incdir/resp2\n"
                  "-option_3=abcd\n";
  IncludedFile.close();

  // Directory for included file.
  llvm::SmallString<128> IncDir;
  llvm::sys::path::append(IncDir, TestDir, "incdir");
  EC = llvm::sys::fs::create_directory(IncDir);
  EXPECT_TRUE(!EC);

  // Create included response file of second level.
  llvm::SmallString<128> IncludedFileName2;
  llvm::sys::path::append(IncludedFileName2, IncDir, "resp2");
  std::ofstream IncludedFile2(IncludedFileName2.c_str());
  EXPECT_TRUE(IncludedFile2.is_open());
  IncludedFile2 << "-option_21 -option_22\n";
  IncludedFile2 << "-option_23=abcd\n";
  IncludedFile2.close();

  // Prepare 'file' with reference to response file.
  SmallString<128> IncRef;
  IncRef.append(1, '@');
  IncRef.append(IncludedFileName.c_str());
  llvm::SmallVector<const char *, 4> Argv =
                          { "test/test", "-flag_1", IncRef.c_str(), "-flag_2" };

  // Expand response files.
  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver(A);
  bool Res = llvm::cl::ExpandResponseFiles(
                    Saver, llvm::cl::TokenizeGNUCommandLine, Argv, false, true);
  EXPECT_TRUE(Res);
  EXPECT_EQ(Argv.size(), 9U);
  EXPECT_STREQ(Argv[0], "test/test");
  EXPECT_STREQ(Argv[1], "-flag_1");
  EXPECT_STREQ(Argv[2], "-option_1");
  EXPECT_STREQ(Argv[3], "-option_2");
  EXPECT_STREQ(Argv[4], "-option_21");
  EXPECT_STREQ(Argv[5], "-option_22");
  EXPECT_STREQ(Argv[6], "-option_23=abcd");
  EXPECT_STREQ(Argv[7], "-option_3=abcd");
  EXPECT_STREQ(Argv[8], "-flag_2");

  llvm::sys::fs::remove(IncludedFileName2);
  llvm::sys::fs::remove(IncDir);
  llvm::sys::fs::remove(IncludedFileName);
  llvm::sys::fs::remove(TestDir);
}

TEST(CommandLineTest, SetDefautValue) {
  cl::ResetCommandLineParser();

  StackOption<std::string> Opt1("opt1", cl::init("true"));
  StackOption<bool> Opt2("opt2", cl::init(true));
  cl::alias Alias("alias", llvm::cl::aliasopt(Opt2));
  StackOption<int> Opt3("opt3", cl::init(3));

  const char *args[] = {"prog", "-opt1=false", "-opt2", "-opt3"};

  EXPECT_TRUE(
    cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));

  EXPECT_TRUE(Opt1 == "false");
  EXPECT_TRUE(Opt2);
  EXPECT_TRUE(Opt3 == 3);

  Opt2 = false;
  Opt3 = 1;

  cl::ResetAllOptionOccurrences();

  for (auto &OM : cl::getRegisteredOptions(*cl::TopLevelSubCommand)) {
    cl::Option *O = OM.second;
    if (O->ArgStr == "opt2") {
      continue;
    }
    O->setDefault();
  }

  EXPECT_TRUE(Opt1 == "true");
  EXPECT_TRUE(Opt2);
  EXPECT_TRUE(Opt3 == 3);
  Alias.removeArgument();
}

TEST(CommandLineTest, ReadConfigFile) {
  llvm::SmallVector<const char *, 1> Argv;

  llvm::SmallString<128> TestDir;
  std::error_code EC =
      llvm::sys::fs::createUniqueDirectory("unittest", TestDir);
  EXPECT_TRUE(!EC);

  llvm::SmallString<128> TestCfg;
  llvm::sys::path::append(TestCfg, TestDir, "foo");
  std::ofstream ConfigFile(TestCfg.c_str());
  EXPECT_TRUE(ConfigFile.is_open());
  ConfigFile << "# Comment\n"
                "-option_1\n"
                "@subconfig\n"
                "-option_3=abcd\n"
                "-option_4=\\\n"
                "cdef\n";
  ConfigFile.close();

  llvm::SmallString<128> TestCfg2;
  llvm::sys::path::append(TestCfg2, TestDir, "subconfig");
  std::ofstream ConfigFile2(TestCfg2.c_str());
  EXPECT_TRUE(ConfigFile2.is_open());
  ConfigFile2 << "-option_2\n"
                 "\n"
                 "   # comment\n";
  ConfigFile2.close();

  // Make sure the current directory is not the directory where config files
  // resides. In this case the code that expands response files will not find
  // 'subconfig' unless it resolves nested inclusions relative to the including
  // file.
  llvm::SmallString<128> CurrDir;
  EC = llvm::sys::fs::current_path(CurrDir);
  EXPECT_TRUE(!EC);
  EXPECT_TRUE(StringRef(CurrDir) != StringRef(TestDir));

  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver(A);
  bool Result = llvm::cl::readConfigFile(TestCfg, Saver, Argv);

  EXPECT_TRUE(Result);
  EXPECT_EQ(Argv.size(), 4U);
  EXPECT_STREQ(Argv[0], "-option_1");
  EXPECT_STREQ(Argv[1], "-option_2");
  EXPECT_STREQ(Argv[2], "-option_3=abcd");
  EXPECT_STREQ(Argv[3], "-option_4=cdef");

  llvm::sys::fs::remove(TestCfg2);
  llvm::sys::fs::remove(TestCfg);
  llvm::sys::fs::remove(TestDir);
}

TEST(CommandLineTest, PositionalEatArgsError) {
  StackOption<std::string, cl::list<std::string>> PosEatArgs(
      "positional-eat-args", cl::Positional, cl::desc("<arguments>..."),
      cl::ZeroOrMore, cl::PositionalEatsArgs);

  const char *args[] = {"prog", "-positional-eat-args=XXXX"};
  const char *args2[] = {"prog", "-positional-eat-args=XXXX", "-foo"};
  const char *args3[] = {"prog", "-positional-eat-args", "-foo"};

  std::string Errs;
  raw_string_ostream OS(Errs);
  EXPECT_FALSE(cl::ParseCommandLineOptions(2, args, StringRef(), &OS)); OS.flush();
  EXPECT_FALSE(Errs.empty()); Errs.clear();
  EXPECT_FALSE(cl::ParseCommandLineOptions(3, args2, StringRef(), &OS)); OS.flush();
  EXPECT_FALSE(Errs.empty()); Errs.clear();
  EXPECT_TRUE(cl::ParseCommandLineOptions(3, args3, StringRef(), &OS)); OS.flush();
  EXPECT_TRUE(Errs.empty());
}

#ifdef _WIN32
TEST(CommandLineTest, GetCommandLineArguments) {
  int argc = __argc;
  char **argv = __argv;

  // GetCommandLineArguments is called in InitLLVM.
  llvm::InitLLVM X(argc, argv);

  EXPECT_EQ(llvm::sys::path::is_absolute(argv[0]),
            llvm::sys::path::is_absolute(__argv[0]));

  EXPECT_TRUE(llvm::sys::path::filename(argv[0])
              .equals_lower("supporttests.exe"))
      << "Filename of test executable is "
      << llvm::sys::path::filename(argv[0]);
}
#endif

TEST(CommandLineTest, PrefixOptions) {
  cl::ResetCommandLineParser();

  StackOption<std::string, cl::list<std::string>> IncludeDirs(
      "I", cl::Prefix, cl::desc("Declare an include directory"));

  // Test non-prefixed variant works with cl::Prefix options.
  EXPECT_TRUE(IncludeDirs.empty());
  const char *args[] = {"prog", "-I=/usr/include"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(IncludeDirs.size() == 1);
  EXPECT_TRUE(IncludeDirs.front().compare("/usr/include") == 0);

  IncludeDirs.erase(IncludeDirs.begin());
  cl::ResetAllOptionOccurrences();

  // Test non-prefixed variant works with cl::Prefix options when value is
  // passed in following argument.
  EXPECT_TRUE(IncludeDirs.empty());
  const char *args2[] = {"prog", "-I", "/usr/include"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(3, args2, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(IncludeDirs.size() == 1);
  EXPECT_TRUE(IncludeDirs.front().compare("/usr/include") == 0);

  IncludeDirs.erase(IncludeDirs.begin());
  cl::ResetAllOptionOccurrences();

  // Test prefixed variant works with cl::Prefix options.
  EXPECT_TRUE(IncludeDirs.empty());
  const char *args3[] = {"prog", "-I/usr/include"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args3, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(IncludeDirs.size() == 1);
  EXPECT_TRUE(IncludeDirs.front().compare("/usr/include") == 0);

  StackOption<std::string, cl::list<std::string>> MacroDefs(
      "D", cl::AlwaysPrefix, cl::desc("Define a macro"),
      cl::value_desc("MACRO[=VALUE]"));

  cl::ResetAllOptionOccurrences();

  // Test non-prefixed variant does not work with cl::AlwaysPrefix options:
  // equal sign is part of the value.
  EXPECT_TRUE(MacroDefs.empty());
  const char *args4[] = {"prog", "-D=HAVE_FOO"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args4, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(MacroDefs.size() == 1);
  EXPECT_TRUE(MacroDefs.front().compare("=HAVE_FOO") == 0);

  MacroDefs.erase(MacroDefs.begin());
  cl::ResetAllOptionOccurrences();

  // Test non-prefixed variant does not allow value to be passed in following
  // argument with cl::AlwaysPrefix options.
  EXPECT_TRUE(MacroDefs.empty());
  const char *args5[] = {"prog", "-D", "HAVE_FOO"};
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(3, args5, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(MacroDefs.empty());

  cl::ResetAllOptionOccurrences();

  // Test prefixed variant works with cl::AlwaysPrefix options.
  EXPECT_TRUE(MacroDefs.empty());
  const char *args6[] = {"prog", "-DHAVE_FOO"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args6, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(MacroDefs.size() == 1);
  EXPECT_TRUE(MacroDefs.front().compare("HAVE_FOO") == 0);
}

} // anonymous namespace
