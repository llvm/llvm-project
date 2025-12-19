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
    // Configure the parser with the specific options requested
    Parser.addValue("host", "Hostname", std::string("localhost"), Host);
    Parser.addFlag("help", "Display this help message", false, Help);
    Parser.addValue("port", "Port number", 8080, Port);
    Parser.addFlag("verbose", "Enable verbose logging", false, Verbose);
  }
};

TEST_F(CommandLineParserTest, NoopTest) {
  CommandLineParser Parser;
  const char *Argv[] = {"appname"};
  auto Err = Parser.parse(1, const_cast<char **>(Argv));
  EXPECT_TRUE(!Err);
}

// TODO(wyles): improve expected failure tests
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
  const char *Argv[] = {"appname", "--host=example.com", "--port=8080",
                        "--verbose=true"};

  cantFail(Parser.parse(std::begin(Argv), std::end(Argv)));

  EXPECT_EQ(Host, "example.com");
  EXPECT_EQ(Port, 8080);
  EXPECT_EQ(Verbose, true);
}

TEST_F(CommandLineParserTest, ValueWithoutEqualSign) {
  std::vector<const char *> Argv = {"appname", "--port", "12345", "--host",
                                    "example.com"};

  cantFail(Parser.parse(std::begin(Argv), std::end(Argv)));
  EXPECT_EQ(Port, 12345);
  EXPECT_EQ(Host, "example.com");
}

TEST_F(CommandLineParserTest, DoubleDashTerminatesOptionParsingWithArgvArgc) {
  bool Verbose = false;
  int Port = 0;

  CommandLineParser Parser;
  Parser.addFlag("verbose", "enable verbose mode", false, Verbose)
      .addValue("port", "port number", 8080, Port);

  const char *Argv[] = {"appname", "--verbose=true", "--", "--not-an-option",
                        "file.txt"};
  cantFail(Parser.parse(static_cast<int>(std::size(Argv)),
                        const_cast<char **>(Argv)));

  EXPECT_TRUE(Verbose);
  EXPECT_EQ(Port, 8080);

  const auto &Pos = Parser.positionals();
  ASSERT_EQ(Pos.size(), 2u);
  EXPECT_EQ(Pos[0], "--not-an-option");
  EXPECT_EQ(Pos[1], "file.txt");
}

TEST_F(CommandLineParserTest, PrintHelpFunctionalAlignment) {
  std::string OutputDir = "";
  Parser.addValue("output-dir", "Directory for output files",
                  std::string("/tmp"), OutputDir);

  Parser.addFlag("help", "Display this help message", false, Help);

  std::stringstream SS;
  Parser.printHelp(SS, "appname");
  std::string Result = SS.str();

  auto GetDescriptionColumn = [&](std::string_view SearchTerm) -> size_t {
    size_t Pos = Result.find(SearchTerm);
    if (Pos == std::string::npos)
      return 0;
    size_t LineStart = Result.rfind('\n', Pos);
    return (LineStart == std::string::npos) ? Pos : (Pos - LineStart - 1);
  };

  size_t HelpCol = GetDescriptionColumn("Display this help");
  size_t PortCol = GetDescriptionColumn("Port number");
  size_t DirCol = GetDescriptionColumn("Directory for output files");

  ASSERT_NE(HelpCol, 0);
  EXPECT_EQ(HelpCol, PortCol) << "Help and Port descriptions are not aligned!";
  EXPECT_EQ(PortCol, DirCol)
      << "Port and OutputDir descriptions are not aligned!";

  EXPECT_TRUE(Result.find("--output-dir=<value>") != std::string::npos);
}
