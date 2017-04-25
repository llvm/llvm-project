//===-- SocketAddressTest.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Host/SocketAddress.h"

namespace {
class SocketAddressTest : public testing::Test {
public:
  static void SetUpTestCase() {
#ifdef _MSC_VER
    WSADATA data;
    ASSERT_EQ(0, WSAStartup(MAKEWORD(2, 2), &data));
#endif
  }
  static void TearDownTestCase() {
#ifdef _MSC_VER
    ASSERT_EQ(0, WSACleanup());
#endif
  }
};
} // namespace

using namespace lldb_private;

TEST_F(SocketAddressTest, Set) {
  SocketAddress sa;
  ASSERT_TRUE(sa.SetToLocalhost(AF_INET, 1138));
  ASSERT_STREQ("127.0.0.1", sa.GetIPAddress().c_str());
  ASSERT_EQ(1138, sa.GetPort());

  ASSERT_TRUE(sa.SetToAnyAddress(AF_INET, 0));
  ASSERT_STREQ("0.0.0.0", sa.GetIPAddress().c_str());
  ASSERT_EQ(0, sa.GetPort());

  ASSERT_TRUE(sa.SetToLocalhost(AF_INET6, 1139));
  ASSERT_TRUE(sa.GetIPAddress() == "::1" ||
              sa.GetIPAddress() == "0:0:0:0:0:0:0:1")
      << "Address was: " << sa.GetIPAddress();
  ASSERT_EQ(1139, sa.GetPort());
}

TEST_F(SocketAddressTest, GetAddressInfo) {
  auto addr = SocketAddress::GetAddressInfo("127.0.0.1", nullptr, AF_UNSPEC,
                                            SOCK_STREAM, IPPROTO_TCP);
  ASSERT_EQ(1u, addr.size());
  EXPECT_EQ(AF_INET, addr[0].GetFamily());
  EXPECT_EQ("127.0.0.1", addr[0].GetIPAddress());
}

#ifdef _WIN32

// we need to test our inet_ntop implementation for Windows XP
const char *inet_ntop(int af, const void *src, char *dst, socklen_t size);

TEST_F(SocketAddressTest, inet_ntop) {
  const uint8_t address4[4] = {255, 0, 1, 100};
  const uint8_t address6[16] = {0, 1, 2,  3,  4,  5,  6,   7,
                                8, 9, 10, 11, 12, 13, 255, 0};

  char buffer[INET6_ADDRSTRLEN];
  memset(buffer, 'x', sizeof(buffer));
  EXPECT_STREQ("1:203:405:607:809:a0b:c0d:ff00",
               inet_ntop(AF_INET6, address6, buffer, sizeof(buffer)));
  memset(buffer, 'x', sizeof(buffer));
  EXPECT_STREQ("1:203:405:607:809:a0b:c0d:ff00",
               inet_ntop(AF_INET6, address6, buffer, 31));
  memset(buffer, 'x', sizeof(buffer));
  EXPECT_STREQ(nullptr, inet_ntop(AF_INET6, address6, buffer, 0));
  memset(buffer, 'x', sizeof(buffer));
  EXPECT_STREQ(nullptr, inet_ntop(AF_INET6, address6, buffer, 30));

  memset(buffer, 'x', sizeof(buffer));
  EXPECT_STREQ("255.0.1.100",
               inet_ntop(AF_INET, address4, buffer, sizeof(buffer)));
  memset(buffer, 'x', sizeof(buffer));
  EXPECT_STREQ("255.0.1.100", inet_ntop(AF_INET, address4, buffer, 12));
  memset(buffer, 'x', sizeof(buffer));
  EXPECT_STREQ(nullptr, inet_ntop(AF_INET, address4, buffer, 0));
  memset(buffer, 'x', sizeof(buffer));
  EXPECT_STREQ(nullptr, inet_ntop(AF_INET, address4, buffer, 11));
}

#endif
