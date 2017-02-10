#include "lldb/Utility/UriParser.h"
#include "gtest/gtest.h"

using namespace lldb_private;

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
  llvm::StringRef scheme(kAsdf);                                               \
  llvm::StringRef hostname(kAsdf);                                             \
  int port(1138);                                                              \
  llvm::StringRef path(kAsdf);                                                 \
  EXPECT_EQ(testCase.m_result,                                                 \
            UriParser::Parse(testCase.m_uri, scheme, hostname, port, path));   \
  EXPECT_STREQ(testCase.m_scheme, scheme.str().c_str());                       \
  EXPECT_STREQ(testCase.m_hostname, hostname.str().c_str());                   \
  EXPECT_EQ(testCase.m_port, port);                                            \
  EXPECT_STREQ(testCase.m_path, path.str().c_str());

TEST(UriParserTest, Minimal) {
  const UriTestCase testCase("x://y", "x", "y", -1, "/");
  VALIDATE
}

TEST(UriParserTest, MinimalPort) {
  const UriTestCase testCase("x://y:1", "x", "y", 1, "/");
  llvm::StringRef scheme(kAsdf);
  llvm::StringRef hostname(kAsdf);
  int port(1138);
  llvm::StringRef path(kAsdf);
  bool result = UriParser::Parse(testCase.m_uri, scheme, hostname, port, path);
  EXPECT_EQ(testCase.m_result, result);

  EXPECT_STREQ(testCase.m_scheme, scheme.str().c_str());
  EXPECT_STREQ(testCase.m_hostname, hostname.str().c_str());
  EXPECT_EQ(testCase.m_port, port);
  EXPECT_STREQ(testCase.m_path, path.str().c_str());
}

TEST(UriParserTest, MinimalPath) {
  const UriTestCase testCase("x://y/", "x", "y", -1, "/");
  VALIDATE
}

TEST(UriParserTest, MinimalPortPath) {
  const UriTestCase testCase("x://y:1/", "x", "y", 1, "/");
  VALIDATE
}

TEST(UriParserTest, LongPath) {
  const UriTestCase testCase("x://y/abc/def/xyz", "x", "y", -1, "/abc/def/xyz");
  VALIDATE
}

TEST(UriParserTest, TypicalPortPath) {
  const UriTestCase testCase("connect://192.168.100.132:5432/", "connect",
                             "192.168.100.132", 5432, "/");
  VALIDATE;
}

TEST(UriParserTest, BracketedHostnamePort) {
  const UriTestCase testCase("connect://[192.168.100.132]:5432/", "connect",
                             "192.168.100.132", 5432, "/");
  llvm::StringRef scheme(kAsdf);
  llvm::StringRef hostname(kAsdf);
  int port(1138);
  llvm::StringRef path(kAsdf);
  bool result = UriParser::Parse(testCase.m_uri, scheme, hostname, port, path);
  EXPECT_EQ(testCase.m_result, result);

  EXPECT_STREQ(testCase.m_scheme, scheme.str().c_str());
  EXPECT_STREQ(testCase.m_hostname, hostname.str().c_str());
  EXPECT_EQ(testCase.m_port, port);
  EXPECT_STREQ(testCase.m_path, path.str().c_str());
}

TEST(UriParserTest, BracketedHostname) {
  const UriTestCase testCase("connect://[192.168.100.132]", "connect",
                             "192.168.100.132", -1, "/");
  VALIDATE
}

TEST(UriParserTest, BracketedHostnameWithColon) {
  const UriTestCase testCase("connect://[192.168.100.132:5555]:1234", "connect",
                             "192.168.100.132:5555", 1234, "/");
  VALIDATE
}

TEST(UriParserTest, SchemeHostSeparator) {
  const UriTestCase testCase("x:/y");
  VALIDATE
}

TEST(UriParserTest, SchemeHostSeparator2) {
  const UriTestCase testCase("x:y");
  VALIDATE
}

TEST(UriParserTest, SchemeHostSeparator3) {
  const UriTestCase testCase("x//y");
  VALIDATE
}

TEST(UriParserTest, SchemeHostSeparator4) {
  const UriTestCase testCase("x/y");
  VALIDATE
}

TEST(UriParserTest, BadPort) {
  const UriTestCase testCase("x://y:a/");
  VALIDATE
}

TEST(UriParserTest, BadPort2) {
  const UriTestCase testCase("x://y:5432a/");
  VALIDATE
}

TEST(UriParserTest, Empty) {
  const UriTestCase testCase("");
  VALIDATE
}

TEST(UriParserTest, PortOverflow) {
  const UriTestCase testCase("x://"
                             "y:"
                             "0123456789012345678901234567890123456789012345678"
                             "9012345678901234567890123456789012345678901234567"
                             "89/");
  VALIDATE
}
