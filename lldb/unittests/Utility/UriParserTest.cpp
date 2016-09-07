#include "Utility/UriParser.h"
#include "gtest/gtest.h"

namespace {
class UriParserTest : public ::testing::Test {};
}

// result strings (scheme/hostname/port/path) passed into UriParser::Parse
// are initialized to kAsdf so we can verify that they are unmodified if the
// URI is invalid
static const char *kAsdf = "asdf";

class UriTestCase {
public:
  UriTestCase(const char *uri, const char *scheme, const char *hostname,
              int port, const char *path)
      : m_uri(uri), m_result(true), m_scheme(scheme), m_hostname(hostname),
        m_port(port), m_path(path) {}

  UriTestCase(const char *uri)
      : m_uri(uri), m_result(false), m_scheme(kAsdf), m_hostname(kAsdf),
        m_port(1138), m_path(kAsdf) {}

  const char *m_uri;
  bool m_result;
  const char *m_scheme;
  const char *m_hostname;
  int m_port;
  const char *m_path;
};

#define VALIDATE                                                               \
  std::string scheme(kAsdf);                                                   \
  std::string hostname(kAsdf);                                                 \
  int port(1138);                                                              \
  std::string path(kAsdf);                                                     \
  EXPECT_EQ(testCase.m_result,                                                 \
            UriParser::Parse(testCase.m_uri, scheme, hostname, port, path));   \
  EXPECT_STREQ(testCase.m_scheme, scheme.c_str());                             \
  EXPECT_STREQ(testCase.m_hostname, hostname.c_str());                         \
  EXPECT_EQ(testCase.m_port, port);                                            \
  EXPECT_STREQ(testCase.m_path, path.c_str());

TEST_F(UriParserTest, Minimal) {
  const UriTestCase testCase("x://y", "x", "y", -1, "/");
  VALIDATE
}

TEST_F(UriParserTest, MinimalPort) {
  const UriTestCase testCase("x://y:1", "x", "y", 1, "/");
  VALIDATE
}

TEST_F(UriParserTest, MinimalPath) {
  const UriTestCase testCase("x://y/", "x", "y", -1, "/");
  VALIDATE
}

TEST_F(UriParserTest, MinimalPortPath) {
  const UriTestCase testCase("x://y:1/", "x", "y", 1, "/");
  VALIDATE
}

TEST_F(UriParserTest, LongPath) {
  const UriTestCase testCase("x://y/abc/def/xyz", "x", "y", -1, "/abc/def/xyz");
  VALIDATE
}

TEST_F(UriParserTest, TypicalPortPath) {
  const UriTestCase testCase("connect://192.168.100.132:5432/", "connect",
                             "192.168.100.132", 5432, "/");
  VALIDATE
}

TEST_F(UriParserTest, BracketedHostnamePort) {
  const UriTestCase testCase("connect://[192.168.100.132]:5432/", "connect",
                             "192.168.100.132", 5432, "/");
  VALIDATE
}

TEST_F(UriParserTest, BracketedHostname) {
  const UriTestCase testCase("connect://[192.168.100.132]", "connect",
                             "192.168.100.132", -1, "/");
  VALIDATE
}

TEST_F(UriParserTest, BracketedHostnameWithColon) {
  const UriTestCase testCase("connect://[192.168.100.132:5555]:1234", "connect",
                             "192.168.100.132:5555", 1234, "/");
  VALIDATE
}

TEST_F(UriParserTest, SchemeHostSeparator) {
  const UriTestCase testCase("x:/y");
  VALIDATE
}

TEST_F(UriParserTest, SchemeHostSeparator2) {
  const UriTestCase testCase("x:y");
  VALIDATE
}

TEST_F(UriParserTest, SchemeHostSeparator3) {
  const UriTestCase testCase("x//y");
  VALIDATE
}

TEST_F(UriParserTest, SchemeHostSeparator4) {
  const UriTestCase testCase("x/y");
  VALIDATE
}

TEST_F(UriParserTest, BadPort) {
  const UriTestCase testCase("x://y:a/");
  VALIDATE
}

TEST_F(UriParserTest, BadPort2) {
  const UriTestCase testCase("x://y:5432a/");
  VALIDATE
}

TEST_F(UriParserTest, Empty) {
  const UriTestCase testCase("");
  VALIDATE
}

TEST_F(UriParserTest, PortOverflow) {
  const UriTestCase testCase("x://"
                             "y:"
                             "0123456789012345678901234567890123456789012345678"
                             "9012345678901234567890123456789012345678901234567"
                             "89/");
  VALIDATE
}
