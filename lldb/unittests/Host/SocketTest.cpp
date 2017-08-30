//===-- SocketTest.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <functional>
#include <thread>

#include "gtest/gtest.h"

#include "lldb/Host/Config.h"
#include "lldb/Host/Socket.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/common/UDPSocket.h"

#ifndef LLDB_DISABLE_POSIX
#include "lldb/Host/posix/DomainSocket.h"
#endif

using namespace lldb_private;

class SocketTest : public testing::Test {
public:
  void SetUp() override {
#if defined(_MSC_VER)
    WSADATA data;
    ::WSAStartup(MAKEWORD(2, 2), &data);
#endif
  }

  void TearDown() override {
#if defined(_MSC_VER)
    ::WSACleanup();
#endif
  }

protected:
  static void AcceptThread(Socket *listen_socket,
                           const char *listen_remote_address,
                           bool child_processes_inherit, Socket **accept_socket,
                           Status *error) {
    *error = listen_socket->Accept(*accept_socket);
  }

  template <typename SocketType>
  void CreateConnectedSockets(
      const char *listen_remote_address,
      const std::function<std::string(const SocketType &)> &get_connect_addr,
      std::unique_ptr<SocketType> *a_up, std::unique_ptr<SocketType> *b_up) {
    bool child_processes_inherit = false;
    Status error;
    std::unique_ptr<SocketType> listen_socket_up(
        new SocketType(true, child_processes_inherit));
    EXPECT_FALSE(error.Fail());
    error = listen_socket_up->Listen(listen_remote_address, 5);
    EXPECT_FALSE(error.Fail());
    EXPECT_TRUE(listen_socket_up->IsValid());

    Status accept_error;
    Socket *accept_socket;
    std::thread accept_thread(AcceptThread, listen_socket_up.get(),
                              listen_remote_address, child_processes_inherit,
                              &accept_socket, &accept_error);

    std::string connect_remote_address = get_connect_addr(*listen_socket_up);
    std::unique_ptr<SocketType> connect_socket_up(
        new SocketType(true, child_processes_inherit));
    EXPECT_FALSE(error.Fail());
    error = connect_socket_up->Connect(connect_remote_address);
    EXPECT_FALSE(error.Fail());
    EXPECT_TRUE(connect_socket_up->IsValid());

    a_up->swap(connect_socket_up);
    EXPECT_TRUE(error.Success());
    EXPECT_NE(nullptr, a_up->get());
    EXPECT_TRUE((*a_up)->IsValid());

    accept_thread.join();
    b_up->reset(static_cast<SocketType *>(accept_socket));
    EXPECT_TRUE(accept_error.Success());
    EXPECT_NE(nullptr, b_up->get());
    EXPECT_TRUE((*b_up)->IsValid());

    listen_socket_up.reset();
  }
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
  char *file_name_str = tempnam(nullptr, nullptr);
  EXPECT_NE(nullptr, file_name_str);
  const std::string file_name(file_name_str);
  free(file_name_str);

  std::unique_ptr<DomainSocket> socket_a_up;
  std::unique_ptr<DomainSocket> socket_b_up;
  CreateConnectedSockets<DomainSocket>(
      file_name.c_str(), [=](const DomainSocket &) { return file_name; },
      &socket_a_up, &socket_b_up);
}
#endif

TEST_F(SocketTest, TCPListen0ConnectAccept) {
  std::unique_ptr<TCPSocket> socket_a_up;
  std::unique_ptr<TCPSocket> socket_b_up;
  CreateConnectedSockets<TCPSocket>(
      "127.0.0.1:0",
      [=](const TCPSocket &s) {
        char connect_remote_address[64];
        snprintf(connect_remote_address, sizeof(connect_remote_address),
                 "localhost:%u", s.GetLocalPortNumber());
        return std::string(connect_remote_address);
      },
      &socket_a_up, &socket_b_up);
}

TEST_F(SocketTest, TCPGetAddress) {
  std::unique_ptr<TCPSocket> socket_a_up;
  std::unique_ptr<TCPSocket> socket_b_up;
  CreateConnectedSockets<TCPSocket>(
      "127.0.0.1:0",
      [=](const TCPSocket &s) {
        char connect_remote_address[64];
        snprintf(connect_remote_address, sizeof(connect_remote_address),
                 "localhost:%u", s.GetLocalPortNumber());
        return std::string(connect_remote_address);
      },
      &socket_a_up, &socket_b_up);

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
