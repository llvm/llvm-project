#include "orc-rt-utils/CommandLine.h"
#include "orc-rt/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace orc_rt;

class CommandLineParserTest : public ::testing::Test {
protected:
  std::string Host;
  int Port = 0;
  bool Verbose = false;
  bool Help = false;
  CommandLineParser Parser;

  void SetUp() override {
    // Host: -h, Help: -?, Port: -p, Verbose: -v
    Parser.addValue("host", "Hostname", std::string("localhost"), Host, CommandLineParser::OptionKind::Value, 'h');
    Parser.addFlag("help", "Display this help message", false, Help, '?');
    Parser.addValue("port", "Port number", 8080, Port, CommandLineParser::OptionKind::Value, 'p');
    Parser.addFlag("verbose", "Enable verbose logging", false, Verbose, 'v');
  }
};

TEST_F(CommandLineParserTest, NoopTest) {
  CommandLineParser Parser;
  const char *Argv[] = {"appname"};
  auto Err = Parser.parse(1, const_cast<char **>(Argv));
  EXPECT_FALSE(!!Err);
}

TEST_F(CommandLineParserTest, ValueRequired) {
  const char *Argv[] = {"appname", "--host"};
  auto Err = Parser.parse(std::begin(Argv), std::end(Argv));
  if (!Err) {
    ADD_FAILURE() << "--host requires a value, shouldn't succeed.";
  } else {
    orc_rt::consumeError(std::move(Err));
  }
}

TEST_F(CommandLineParserTest, UnknownOption) {
  const char *Argv[] = {"appname", "--unknown=foo"};
  auto Err = Parser.parse(std::begin(Argv), std::end(Argv));
  if (!Err) {
    ADD_FAILURE() << "unknown option, shouldn't succeed.";
  } else {
    orc_rt::consumeError(std::move(Err));
  }
}

TEST_F(CommandLineParserTest, InvalidInteger) {
  const char *Argv[] = {"appname", "--port=not_a_number"};
  auto Err = Parser.parse(std::begin(Argv), std::end(Argv));
  if (!Err) {
    ADD_FAILURE() << "Invalid integer, shouldn't succeed.";
  } else {
    orc_rt::consumeError(std::move(Err));
  }
}

TEST_F(CommandLineParserTest, ParseFullConfiguration) {
  const char *Argv[] = {"appname", "--host=example.com", "--port=8080", "--verbose=true"};
  cantFail(Parser.parse(std::begin(Argv), std::end(Argv)));
  EXPECT_EQ(Host, "example.com");
  EXPECT_EQ(Port, 8080);
  EXPECT_EQ(Verbose, true);
}

TEST_F(CommandLineParserTest, ShortFlagClustering) {
  // Tests combining -v and -?
  const char *Argv[] = {"appname", "-v?"};
  cantFail(Parser.parse(std::begin(Argv), std::end(Argv)));
  EXPECT_TRUE(Verbose);
  EXPECT_TRUE(Help);
}

TEST_F(CommandLineParserTest, ShortFlagWithValue) {
  // Tests -p 1234 (separate) and -hlocalhost (joined)
  const char *Argv[] = {"appname", "-p", "1234", "-hlocalhost"};
  cantFail(Parser.parse(std::begin(Argv), std::end(Argv)));
  EXPECT_EQ(Port, 1234);
  EXPECT_EQ(Host, "localhost");
}

TEST_F(CommandLineParserTest, ClusterWithValueAtEnd) {
  // Tests flags followed immediately by a value: -vp9999
  const char *Argv[] = {"appname", "-vp9999"};
  cantFail(Parser.parse(std::begin(Argv), std::end(Argv)));
  EXPECT_TRUE(Verbose);
  EXPECT_EQ(Port, 9999);
}

TEST_F(CommandLineParserTest, DoubleDashTerminatesOptionParsing) {
  const char *Argv[] = {"appname", "-v", "--", "-p", "1234"};
  cantFail(Parser.parse(static_cast<int>(std::size(Argv)), const_cast<char **>(Argv)));

  EXPECT_TRUE(Verbose);
  EXPECT_EQ(Port, 8080); // Should remain default
  ASSERT_EQ(Parser.positionals().size(), 2u);
  EXPECT_EQ(Parser.positionals()[0], "-p");
  EXPECT_EQ(Parser.positionals()[1], "1234");
}

TEST_F(CommandLineParserTest, PrintHelpAlignmentWithShortFlags) {
  // Add a long-only flag to test alignment padding
  std::string LogFile;
  Parser.addValue("log-file", "Path to log", std::string("out.log"), LogFile);

  std::stringstream SS;
  Parser.printHelp(SS, "appname");
  std::string Result = SS.str();

  auto GetColumn = [&](std::string_view SearchTerm) -> size_t {
    size_t Pos = Result.find(SearchTerm);
    if (Pos == std::string::npos) return 0;
    size_t LineStart = Result.rfind('\n', Pos);
    return (LineStart == std::string::npos) ? Pos : (Pos - LineStart - 1);
  };

  size_t PortDescCol = GetColumn("Port number");
  size_t LogDescCol  = GetColumn("Path to log");
  size_t VerbDescCol = GetColumn("Enable verbose");

  ASSERT_NE(PortDescCol, 0u);
  EXPECT_EQ(PortDescCol, LogDescCol) << "Descriptions should align even if short flag is missing";
  EXPECT_EQ(LogDescCol, VerbDescCol);

  // Ensure the long-only flag is indented to match the "-x, --long" format
  size_t LogFlagPos = Result.find("--log-file");
  size_t PortFlagPos = Result.find("--port");

  size_t LogFlagCol = (LogFlagPos - Result.rfind('\n', LogFlagPos) - 1);
  size_t PortFlagCol = (PortFlagPos - Result.rfind('\n', PortFlagPos) - 1);

  // --port is preceded by "-p, " (4 chars). --log-file should be preceded by 4 spaces.
  EXPECT_EQ(LogFlagCol, PortFlagCol);
}
