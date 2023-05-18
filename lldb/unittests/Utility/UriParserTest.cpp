#include "lldb/Utility/UriParser.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(UriParserTest, Minimal) {
  EXPECT_EQ((URI{"x", "y", std::nullopt, "/"}), URI::Parse("x://y"));
}

TEST(UriParserTest, MinimalPort) {
  EXPECT_EQ((URI{"x", "y", 1, "/"}), URI::Parse("x://y:1"));
}

TEST(UriParserTest, MinimalPath) {
  EXPECT_EQ((URI{"x", "y", std::nullopt, "/"}), URI::Parse("x://y/"));
}

TEST(UriParserTest, MinimalPortPath) {
  EXPECT_EQ((URI{"x", "y", 1, "/"}), URI::Parse("x://y:1/"));
}

TEST(UriParserTest, LongPath) {
  EXPECT_EQ((URI{"x", "y", std::nullopt, "/abc/def/xyz"}),
            URI::Parse("x://y/abc/def/xyz"));
}

TEST(UriParserTest, TypicalPortPathIPv4) {
  EXPECT_EQ((URI{"connect", "192.168.100.132", 5432, "/"}),
            URI::Parse("connect://192.168.100.132:5432/"));
}

TEST(UriParserTest, TypicalPortPathIPv6) {
  EXPECT_EQ(
      (URI{"connect", "2601:600:107f:db64:a42b:4faa:284:3082", 5432, "/"}),
      URI::Parse("connect://[2601:600:107f:db64:a42b:4faa:284:3082]:5432/"));
}

TEST(UriParserTest, BracketedHostnamePort) {
  EXPECT_EQ((URI{"connect", "192.168.100.132", 5432, "/"}),
            URI::Parse("connect://[192.168.100.132]:5432/"));
}

TEST(UriParserTest, BracketedHostname) {
  EXPECT_EQ((URI{"connect", "192.168.100.132", std::nullopt, "/"}),
            URI::Parse("connect://[192.168.100.132]"));
}

TEST(UriParserTest, BracketedHostnameWithPortIPv4) {
  // Android device over IPv4: port is a part of the hostname.
  EXPECT_EQ((URI{"connect", "192.168.100.132:1234", std::nullopt, "/"}),
            URI::Parse("connect://[192.168.100.132:1234]"));
}

TEST(UriParserTest, BracketedHostnameWithPortIPv6) {
  // Android device over IPv6: port is a part of the hostname.
  EXPECT_EQ((URI{"connect", "[2601:600:107f:db64:a42b:4faa:284]:1234",
                 std::nullopt, "/"}),
            URI::Parse("connect://[[2601:600:107f:db64:a42b:4faa:284]:1234]"));
}

TEST(UriParserTest, BracketedHostnameWithColon) {
  EXPECT_EQ((URI{"connect", "192.168.100.132:5555", 1234, "/"}),
            URI::Parse("connect://[192.168.100.132:5555]:1234"));
}

TEST(UriParserTest, SchemeHostSeparator) {
  EXPECT_EQ(std::nullopt, URI::Parse("x:/y"));
}

TEST(UriParserTest, SchemeHostSeparator2) {
  EXPECT_EQ(std::nullopt, URI::Parse("x:y"));
}

TEST(UriParserTest, SchemeHostSeparator3) {
  EXPECT_EQ(std::nullopt, URI::Parse("x//y"));
}

TEST(UriParserTest, SchemeHostSeparator4) {
  EXPECT_EQ(std::nullopt, URI::Parse("x/y"));
}

TEST(UriParserTest, BadPort) {
  EXPECT_EQ(std::nullopt, URI::Parse("x://y:a/"));
}

TEST(UriParserTest, BadPort2) {
  EXPECT_EQ(std::nullopt, URI::Parse("x://y:5432a/"));
}

TEST(UriParserTest, Empty) { EXPECT_EQ(std::nullopt, URI::Parse("")); }

TEST(UriParserTest, PortOverflow) {
  EXPECT_EQ(std::nullopt,
            URI::Parse("x://"
                       "y:"
                       "0123456789012345678901234567890123456789012345678"
                       "9012345678901234567890123456789012345678901234567"
                       "89/"));
}
