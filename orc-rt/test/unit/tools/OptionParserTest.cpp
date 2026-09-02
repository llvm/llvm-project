//===- OptionParserTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/tools/OptionParser.h"
#include "orc-rt/support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace orc_rt;

class OptionParserTest : public ::testing::Test {
protected:
  std::string Host;
  int Port = 0;
  bool Verbose = false;
  bool Help = false;
  OptionParser Parser;

  void SetUp() override {
    Parser.addValue("host", "Hostname", std::string("localhost"), Host,
                    OptionParser::OptionKind::Value, 'h');
    Parser.addFlag("help", "Display this help message", false, Help, '?');
    Parser.addValue("port", "Port number", 8080, Port,
                    OptionParser::OptionKind::Value, 'p');
    Parser.addFlag("verbose", "Enable verbose logging", false, Verbose, 'v');
  }
};

TEST_F(OptionParserTest, NoopTest) {
  OptionParser Parser;
  const char *Argv[] = {""};
  auto Err = Parser.parse(std::begin(Argv), std::end(Argv));
  EXPECT_FALSE(!!Err);
}

TEST_F(OptionParserTest, ValueRequired) {
  const char *Argv[] = {"--host"};
  auto Err = Parser.parse(std::begin(Argv), std::end(Argv));
  if (!Err) {
    ADD_FAILURE() << "--host requires a value, shouldn't succeed.";
  } else {
    orc_rt::consumeError(std::move(Err));
  }
}

TEST_F(OptionParserTest, UnknownOption) {
  const char *Argv[] = {"--unknown=foo"};
  auto Err = Parser.parse(std::begin(Argv), std::end(Argv));
  if (!Err) {
    ADD_FAILURE() << "unknown option, shouldn't succeed.";
  } else {
    orc_rt::consumeError(std::move(Err));
  }
}

TEST_F(OptionParserTest, InvalidInteger) {
  const char *Argv[] = {"--port=not_a_number"};
  auto Err = Parser.parse(std::begin(Argv), std::end(Argv));
  if (!Err) {
    ADD_FAILURE() << "Invalid integer, shouldn't succeed.";
  } else {
    orc_rt::consumeError(std::move(Err));
  }
}

TEST_F(OptionParserTest, ParseFullConfiguration) {
  const char *Argv[] = {"--host=example.com", "--port=8080", "--verbose=true"};
  cantFail(Parser.parse(std::begin(Argv), std::end(Argv)));
  EXPECT_EQ(Host, "example.com");
  EXPECT_EQ(Port, 8080);
  EXPECT_EQ(Verbose, true);
}

TEST_F(OptionParserTest, ShortFlagClustering) {
  const char *Argv[] = {"-v?"};
  cantFail(Parser.parse(std::begin(Argv), std::end(Argv)));
  EXPECT_TRUE(Verbose);
  EXPECT_TRUE(Help);
}

TEST_F(OptionParserTest, ShortFlagWithValue) {
  const char *Argv[] = {"-p", "1234", "-hlocalhost"};
  cantFail(Parser.parse(std::begin(Argv), std::end(Argv)));
  EXPECT_EQ(Port, 1234);
  EXPECT_EQ(Host, "localhost");
}

TEST_F(OptionParserTest, ClusterWithValueAtEnd) {
  const char *Argv[] = {"-vp9999"};
  cantFail(Parser.parse(std::begin(Argv), std::end(Argv)));
  EXPECT_TRUE(Verbose);
  EXPECT_EQ(Port, 9999);
}

TEST_F(OptionParserTest, DoubleDashTerminatesOptionParsing) {
  const char *Argv[] = {"-v", "--", "-p", "1234"};
  cantFail(Parser.parse(std::begin(Argv), std::end(Argv)));

  EXPECT_TRUE(Verbose);
  EXPECT_EQ(Port, 8080); // Should remain default
  ASSERT_EQ(Parser.positionals().size(), 2u);
  EXPECT_EQ(Parser.positionals()[0], "-p");
  EXPECT_EQ(Parser.positionals()[1], "1234");
}

TEST_F(OptionParserTest, ParseAsMainWithEmptyArgsSucceeds) {
  const char *Argv[] = {"appname"};
  auto Err = Parser.parseAsMainArgs(std::size(Argv), const_cast<char **>(Argv));
  EXPECT_FALSE(!!Err);
}

TEST_F(OptionParserTest, ParseAsMainWithRegularArgsSucceeds) {
  const char *Argv[] = {"appname", "-v", "--", "-p", "1234"};
  cantFail(Parser.parseAsMainArgs(std::size(Argv), const_cast<char **>(Argv)));

  EXPECT_TRUE(Verbose);
  EXPECT_EQ(Port, 8080); // Should remain default
  ASSERT_EQ(Parser.positionals().size(), 2u);
  EXPECT_EQ(Parser.positionals()[0], "-p");
  EXPECT_EQ(Parser.positionals()[1], "1234");
}

TEST_F(OptionParserTest, ParseAsMainWithEmplyListFails) {
  const char *Argv[] = {};
  auto Err = Parser.parseAsMainArgs(0, const_cast<char **>(Argv));

  EXPECT_TRUE(!!Err);
  consumeError(std::move(Err));
}

TEST_F(OptionParserTest, PrintHelpAlignmentWithShortFlags) {
  std::string LogFile;
  Parser.addValue("log-file", "Path to log", std::string("out.log"), LogFile);

  std::string Result = Parser.formatHelp("appname");

  auto GetColumn = [&](std::string_view SearchTerm) -> size_t {
    size_t Pos = Result.find(SearchTerm);
    if (Pos == std::string::npos)
      return 0;
    size_t LineStart = Result.rfind('\n', Pos);
    return (LineStart == std::string::npos) ? Pos : (Pos - LineStart - 1);
  };

  size_t PortDescCol = GetColumn("Port number");
  size_t LogDescCol = GetColumn("Path to log");
  size_t VerbDescCol = GetColumn("Enable verbose");

  ASSERT_NE(PortDescCol, 0u);
  EXPECT_EQ(PortDescCol, LogDescCol)
      << "Descriptions should align even if short flag is missing";
  EXPECT_EQ(LogDescCol, VerbDescCol);

  size_t LogFlagPos = Result.find("--log-file");
  size_t PortFlagPos = Result.find("--port");

  size_t LogFlagCol = (LogFlagPos - Result.rfind('\n', LogFlagPos) - 1);
  size_t PortFlagCol = (PortFlagPos - Result.rfind('\n', PortFlagPos) - 1);

  EXPECT_EQ(LogFlagCol, PortFlagCol);
}
