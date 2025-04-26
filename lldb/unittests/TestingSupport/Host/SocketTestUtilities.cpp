//===-- SocketTestUtilities.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport/Host/SocketTestUtilities.h"
#include "lldb/Host/Config.h"
#include "lldb/Utility/StreamString.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#endif

using namespace lldb_private;

template <typename SocketType>
void lldb_private::CreateConnectedSockets(
    llvm::StringRef listen_remote_address,
    const std::function<std::string(const SocketType &)> &get_connect_addr,
    std::unique_ptr<SocketType> *a_up, std::unique_ptr<SocketType> *b_up) {
  Status error;
  auto listen_socket_up = std::make_unique<SocketType>(true);
  ASSERT_THAT_ERROR(error.ToError(), llvm::Succeeded());
  error = listen_socket_up->Listen(listen_remote_address, 5);
  ASSERT_THAT_ERROR(error.ToError(), llvm::Succeeded());
  ASSERT_TRUE(listen_socket_up->IsValid());

  std::string connect_remote_address = get_connect_addr(*listen_socket_up);
  auto connect_socket_up = std::make_unique<SocketType>(true);
  ASSERT_THAT_ERROR(error.ToError(), llvm::Succeeded());
  error = connect_socket_up->Connect(connect_remote_address);
  ASSERT_THAT_ERROR(error.ToError(), llvm::Succeeded());
  ASSERT_TRUE(connect_socket_up->IsValid());

  a_up->swap(connect_socket_up);
  ASSERT_TRUE((*a_up)->IsValid());

  Socket *accept_socket;
  ASSERT_THAT_ERROR(
      listen_socket_up->Accept(std::chrono::seconds(1), accept_socket)
          .takeError(),
      llvm::Succeeded());

  b_up->reset(static_cast<SocketType *>(accept_socket));
  ASSERT_NE(nullptr, b_up->get());
  ASSERT_TRUE((*b_up)->IsValid());

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

#if LLDB_ENABLE_POSIX
void lldb_private::CreateDomainConnectedSockets(
    llvm::StringRef path, std::unique_ptr<DomainSocket> *socket_a_up,
    std::unique_ptr<DomainSocket> *socket_b_up) {
  return CreateConnectedSockets<DomainSocket>(
      path, [=](const DomainSocket &) { return path.str(); }, socket_a_up,
      socket_b_up);
}
#endif

static bool CheckIPSupport(llvm::StringRef Proto, llvm::StringRef Addr) {
  llvm::Expected<std::unique_ptr<TCPSocket>> Sock = Socket::TcpListen(Addr);
  if (Sock)
    return true;
  llvm::Error Err = Sock.takeError();
  GTEST_LOG_(WARNING) << llvm::formatv(
                             "Creating a canary {0} TCP socket failed: {1}.",
                             Proto, Err)
                             .str();
  bool HasProtocolError = false;
  handleAllErrors(std::move(Err), [&](std::unique_ptr<llvm::ECError> ECErr) {
    std::error_code ec = ECErr->convertToErrorCode();
    if (ec == std::make_error_code(std::errc::address_family_not_supported) ||
        ec == std::make_error_code(std::errc::address_not_available))
      HasProtocolError = true;
  });
  if (HasProtocolError) {
    GTEST_LOG_(WARNING)
        << llvm::formatv(
               "Assuming the host does not support {0}. Skipping test.", Proto)
               .str();
    return false;
  }
  GTEST_LOG_(WARNING) << "Continuing anyway. The test will probably fail.";
  return true;
}

bool lldb_private::HostSupportsIPv4() {
  return CheckIPSupport("IPv4", "127.0.0.1:0");
}

bool lldb_private::HostSupportsIPv6() {
  return CheckIPSupport("IPv6", "[::1]:0");
}

bool lldb_private::HostSupportsLocalhostToIPv4() {
  if (!HostSupportsIPv4())
    return false;

  auto addresses = SocketAddress::GetAddressInfo(
      "localhost", nullptr, AF_UNSPEC, SOCK_STREAM, IPPROTO_TCP);
  return llvm::any_of(addresses, [](const SocketAddress &addr) {
    return addr.GetFamily() == AF_INET;
  });
}

bool lldb_private::HostSupportsLocalhostToIPv6() {
  if (!HostSupportsIPv6())
    return false;

  auto addresses = SocketAddress::GetAddressInfo(
      "localhost", nullptr, AF_UNSPEC, SOCK_STREAM, IPPROTO_TCP);
  return llvm::any_of(addresses, [](const SocketAddress &addr) {
    return addr.GetFamily() == AF_INET6;
  });
}

llvm::Expected<std::string> lldb_private::GetLocalhostIP() {
  if (HostSupportsIPv4())
    return "127.0.0.1";
  if (HostSupportsIPv6())
    return "[::1]";
  return llvm::make_error<llvm::StringError>(
      "Neither IPv4 nor IPv6 appear to be supported",
      llvm::inconvertibleErrorCode());
}
