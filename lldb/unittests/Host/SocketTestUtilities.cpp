//===----------------- SocketTestUtilities.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SocketTestUtilities.h"
#include "lldb/Utility/StreamString.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#endif

using namespace lldb_private;

static void AcceptThread(Socket *listen_socket, bool child_processes_inherit,
                         Socket **accept_socket, Status *error) {
  *error = listen_socket->Accept(*accept_socket);
}

template <typename SocketType>
void lldb_private::CreateConnectedSockets(
    llvm::StringRef listen_remote_address,
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
                            child_processes_inherit, &accept_socket,
                            &accept_error);

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

bool lldb_private::CreateTCPConnectedSockets(
    std::string listen_remote_ip, std::unique_ptr<TCPSocket> *socket_a_up,
    std::unique_ptr<TCPSocket> *socket_b_up) {
  StreamString strm;
  strm.Printf("[%s]:0", listen_remote_ip.c_str());
  CreateConnectedSockets<TCPSocket>(
      strm.GetString(),
      [=](const TCPSocket &s) {
        char connect_remote_address[64];
        snprintf(connect_remote_address, sizeof(connect_remote_address),
                 "[%s]:%u", listen_remote_ip.c_str(), s.GetLocalPortNumber());
        return std::string(connect_remote_address);
      },
      socket_a_up, socket_b_up);
  return true;
}

#ifndef LLDB_DISABLE_POSIX
void lldb_private::CreateDomainConnectedSockets(
    llvm::StringRef path, std::unique_ptr<DomainSocket> *socket_a_up,
    std::unique_ptr<DomainSocket> *socket_b_up) {
  return CreateConnectedSockets<DomainSocket>(
      path, [=](const DomainSocket &) { return path.str(); }, socket_a_up,
      socket_b_up);
}
#endif

bool lldb_private::IsAddressFamilySupported(std::string ip) {
  auto addresses = lldb_private::SocketAddress::GetAddressInfo(
      ip.c_str(), NULL, AF_UNSPEC, SOCK_STREAM, IPPROTO_TCP);
  return addresses.size() > 0;
}

bool lldb_private::IsIPv4(std::string ip) {
  struct sockaddr_in sock_addr;
  return inet_pton(AF_INET, ip.c_str(), &(sock_addr.sin_addr)) != 0;
}
