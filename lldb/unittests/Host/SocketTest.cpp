//===-- SocketTest.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SocketTestUtilities.h"
#include "lldb/Utility/UriParser.h"
#include "gtest/gtest.h"

using namespace lldb_private;

class SocketTest : public testing::Test {
public:
  void SetUp() override {
    ASSERT_THAT_ERROR(Socket::Initialize(), llvm::Succeeded());
  }

  void TearDown() override { Socket::Terminate(); }
};

TEST_F(SocketTest, DecodeHostAndPort) {
  std::string host_str;
  std::string port_str;
  int32_t port;
  Status error;
  EXPECT_TRUE(Socket::DecodeHostAndPort("localhost:1138", host_str, port_str,
                                        port, &error));
  EXPECT_STREQ("localhost", host_str.c_str());
  EXPECT_STREQ("1138", port_str.c_str());
  EXPECT_EQ(1138, port);
  EXPECT_TRUE(error.Success());

  EXPECT_FALSE(Socket::DecodeHostAndPort("google.com:65536", host_str, port_str,
                                         port, &error));
  EXPECT_TRUE(error.Fail());
  EXPECT_STREQ("invalid host:port specification: 'google.com:65536'",
               error.AsCString());

  EXPECT_FALSE(Socket::DecodeHostAndPort("google.com:-1138", host_str, port_str,
                                         port, &error));
  EXPECT_TRUE(error.Fail());
  EXPECT_STREQ("invalid host:port specification: 'google.com:-1138'",
               error.AsCString());

  EXPECT_FALSE(Socket::DecodeHostAndPort("google.com:65536", host_str, port_str,
                                         port, &error));
  EXPECT_TRUE(error.Fail());
  EXPECT_STREQ("invalid host:port specification: 'google.com:65536'",
               error.AsCString());

  EXPECT_TRUE(
      Socket::DecodeHostAndPort("12345", host_str, port_str, port, &error));
  EXPECT_STREQ("", host_str.c_str());
  EXPECT_STREQ("12345", port_str.c_str());
  EXPECT_EQ(12345, port);
  EXPECT_TRUE(error.Success());

  EXPECT_TRUE(
      Socket::DecodeHostAndPort("*:0", host_str, port_str, port, &error));
  EXPECT_STREQ("*", host_str.c_str());
  EXPECT_STREQ("0", port_str.c_str());
  EXPECT_EQ(0, port);
  EXPECT_TRUE(error.Success());

  EXPECT_TRUE(
      Socket::DecodeHostAndPort("*:65535", host_str, port_str, port, &error));
  EXPECT_STREQ("*", host_str.c_str());
  EXPECT_STREQ("65535", port_str.c_str());
  EXPECT_EQ(65535, port);
  EXPECT_TRUE(error.Success());

  EXPECT_TRUE(
      Socket::DecodeHostAndPort("[::1]:12345", host_str, port_str, port, &error));
  EXPECT_STREQ("::1", host_str.c_str());
  EXPECT_STREQ("12345", port_str.c_str());
  EXPECT_EQ(12345, port);
  EXPECT_TRUE(error.Success());

  EXPECT_TRUE(
      Socket::DecodeHostAndPort("[abcd:12fg:AF58::1]:12345", host_str, port_str, port, &error));
  EXPECT_STREQ("abcd:12fg:AF58::1", host_str.c_str());
  EXPECT_STREQ("12345", port_str.c_str());
  EXPECT_EQ(12345, port);
  EXPECT_TRUE(error.Success());
}

#ifndef LLDB_DISABLE_POSIX
TEST_F(SocketTest, DomainListenConnectAccept) {
  llvm::SmallString<64> Path;
  std::error_code EC = llvm::sys::fs::createUniqueDirectory("DomainListenConnectAccept", Path);
  ASSERT_FALSE(EC);
  llvm::sys::path::append(Path, "test");

  std::unique_ptr<DomainSocket> socket_a_up;
  std::unique_ptr<DomainSocket> socket_b_up;
  CreateDomainConnectedSockets(Path, &socket_a_up, &socket_b_up);
}
#endif

TEST_F(SocketTest, TCPListen0ConnectAccept) {
  std::unique_ptr<TCPSocket> socket_a_up;
  std::unique_ptr<TCPSocket> socket_b_up;
  CreateTCPConnectedSockets("127.0.0.1", &socket_a_up, &socket_b_up);
}

TEST_F(SocketTest, TCPGetAddress) {
  std::unique_ptr<TCPSocket> socket_a_up;
  std::unique_ptr<TCPSocket> socket_b_up;
  if (!IsAddressFamilySupported("127.0.0.1")) {
    GTEST_LOG_(WARNING) << "Skipping test due to missing IPv4 support.";
    return;
  }
  CreateTCPConnectedSockets("127.0.0.1", &socket_a_up, &socket_b_up);

  EXPECT_EQ(socket_a_up->GetLocalPortNumber(),
            socket_b_up->GetRemotePortNumber());
  EXPECT_EQ(socket_b_up->GetLocalPortNumber(),
            socket_a_up->GetRemotePortNumber());
  EXPECT_NE(socket_a_up->GetLocalPortNumber(),
            socket_b_up->GetLocalPortNumber());
  EXPECT_STREQ("127.0.0.1", socket_a_up->GetRemoteIPAddress().c_str());
  EXPECT_STREQ("127.0.0.1", socket_b_up->GetRemoteIPAddress().c_str());
}

TEST_F(SocketTest, UDPConnect) {
  Socket *socket;

  bool child_processes_inherit = false;
  auto error = UDPSocket::Connect("127.0.0.1:0", child_processes_inherit,
                                  socket);

  std::unique_ptr<Socket> socket_up(socket);

  EXPECT_TRUE(error.Success());
  EXPECT_TRUE(socket_up->IsValid());
}

TEST_F(SocketTest, TCPListen0GetPort) {
  Socket *server_socket;
  Predicate<uint16_t> port_predicate;
  port_predicate.SetValue(0, eBroadcastNever);
  Status err =
      Socket::TcpListen("10.10.12.3:0", false, server_socket, &port_predicate);
  std::unique_ptr<TCPSocket> socket_up((TCPSocket*)server_socket);
  EXPECT_TRUE(socket_up->IsValid());
  EXPECT_NE(socket_up->GetLocalPortNumber(), 0);
}

TEST_F(SocketTest, TCPGetConnectURI) {
  std::unique_ptr<TCPSocket> socket_a_up;
  std::unique_ptr<TCPSocket> socket_b_up;
  if (!IsAddressFamilySupported("127.0.0.1")) {
    GTEST_LOG_(WARNING) << "Skipping test due to missing IPv4 support.";
    return;
  }
  CreateTCPConnectedSockets("127.0.0.1", &socket_a_up, &socket_b_up);

  llvm::StringRef scheme;
  llvm::StringRef hostname;
  int port;
  llvm::StringRef path;
  std::string uri(socket_a_up->GetRemoteConnectionURI());
  EXPECT_TRUE(UriParser::Parse(uri, scheme, hostname, port, path));
  EXPECT_EQ(scheme, "connect");
  EXPECT_EQ(port, socket_a_up->GetRemotePortNumber());
}

TEST_F(SocketTest, UDPGetConnectURI) {
  if (!IsAddressFamilySupported("127.0.0.1")) {
    GTEST_LOG_(WARNING) << "Skipping test due to missing IPv4 support.";
    return;
  }
  Socket *socket;
  bool child_processes_inherit = false;
  auto error =
      UDPSocket::Connect("127.0.0.1:0", child_processes_inherit, socket);

  llvm::StringRef scheme;
  llvm::StringRef hostname;
  int port;
  llvm::StringRef path;
  std::string uri(socket->GetRemoteConnectionURI());
  EXPECT_TRUE(UriParser::Parse(uri, scheme, hostname, port, path));
  EXPECT_EQ(scheme, "udp");
}

#ifndef LLDB_DISABLE_POSIX
TEST_F(SocketTest, DomainGetConnectURI) {
  llvm::SmallString<64> domain_path;
  std::error_code EC =
      llvm::sys::fs::createUniqueDirectory("DomainListenConnectAccept", domain_path);
  ASSERT_FALSE(EC);
  llvm::sys::path::append(domain_path, "test");

  std::unique_ptr<DomainSocket> socket_a_up;
  std::unique_ptr<DomainSocket> socket_b_up;
  CreateDomainConnectedSockets(domain_path, &socket_a_up, &socket_b_up);

  llvm::StringRef scheme;
  llvm::StringRef hostname;
  int port;
  llvm::StringRef path;
  std::string uri(socket_a_up->GetRemoteConnectionURI());
  EXPECT_TRUE(UriParser::Parse(uri, scheme, hostname, port, path));
  EXPECT_EQ(scheme, "unix-connect");
  EXPECT_EQ(path, domain_path);
}
#endif