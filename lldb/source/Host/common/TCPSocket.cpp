//===-- TCPSocket.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(_MSC_VER)
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#endif

#include "lldb/Host/common/TCPSocket.h"

#include "lldb/Host/Config.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/WindowsError.h"
#include "llvm/Support/raw_ostream.h"

#if LLDB_ENABLE_POSIX
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#endif

#if defined(_WIN32)
#include <winsock2.h>
#endif

using namespace lldb;
using namespace lldb_private;

static const int kType = SOCK_STREAM;

TCPSocket::TCPSocket(bool should_close) : Socket(ProtocolTcp, should_close) {}

TCPSocket::TCPSocket(NativeSocket socket, const TCPSocket &listen_socket)
    : Socket(ProtocolTcp, listen_socket.m_should_close_fd) {
  m_socket = socket;
}

TCPSocket::TCPSocket(NativeSocket socket, bool should_close)
    : Socket(ProtocolTcp, should_close) {
  m_socket = socket;
}

TCPSocket::~TCPSocket() { CloseListenSockets(); }

bool TCPSocket::IsValid() const {
  return m_socket != kInvalidSocketValue || m_listen_sockets.size() != 0;
}

// Return the port number that is being used by the socket.
uint16_t TCPSocket::GetLocalPortNumber() const {
  if (m_socket != kInvalidSocketValue) {
    SocketAddress sock_addr;
    socklen_t sock_addr_len = sock_addr.GetMaxLength();
    if (::getsockname(m_socket, sock_addr, &sock_addr_len) == 0)
      return sock_addr.GetPort();
  } else if (!m_listen_sockets.empty()) {
    SocketAddress sock_addr;
    socklen_t sock_addr_len = sock_addr.GetMaxLength();
    if (::getsockname(m_listen_sockets.begin()->first, sock_addr,
                      &sock_addr_len) == 0)
      return sock_addr.GetPort();
  }
  return 0;
}

std::string TCPSocket::GetLocalIPAddress() const {
  // We bound to port zero, so we need to figure out which port we actually
  // bound to
  if (m_socket != kInvalidSocketValue) {
    SocketAddress sock_addr;
    socklen_t sock_addr_len = sock_addr.GetMaxLength();
    if (::getsockname(m_socket, sock_addr, &sock_addr_len) == 0)
      return sock_addr.GetIPAddress();
  }
  return "";
}

uint16_t TCPSocket::GetRemotePortNumber() const {
  if (m_socket != kInvalidSocketValue) {
    SocketAddress sock_addr;
    socklen_t sock_addr_len = sock_addr.GetMaxLength();
    if (::getpeername(m_socket, sock_addr, &sock_addr_len) == 0)
      return sock_addr.GetPort();
  }
  return 0;
}

std::string TCPSocket::GetRemoteIPAddress() const {
  // We bound to port zero, so we need to figure out which port we actually
  // bound to
  if (m_socket != kInvalidSocketValue) {
    SocketAddress sock_addr;
    socklen_t sock_addr_len = sock_addr.GetMaxLength();
    if (::getpeername(m_socket, sock_addr, &sock_addr_len) == 0)
      return sock_addr.GetIPAddress();
  }
  return "";
}

std::string TCPSocket::GetRemoteConnectionURI() const {
  if (m_socket != kInvalidSocketValue) {
    return std::string(llvm::formatv(
        "connect://[{0}]:{1}", GetRemoteIPAddress(), GetRemotePortNumber()));
  }
  return "";
}

std::vector<std::string> TCPSocket::GetListeningConnectionURI() const {
  std::vector<std::string> URIs;
  for (const auto &[fd, addr] : m_listen_sockets)
    URIs.emplace_back(llvm::formatv("connection://[{0}]:{1}",
                                    addr.GetIPAddress(), addr.GetPort()));
  return URIs;
}

Status TCPSocket::CreateSocket(int domain) {
  Status error;
  if (IsValid())
    error = Close();
  if (error.Fail())
    return error;
  m_socket = Socket::CreateSocket(domain, kType, IPPROTO_TCP, error);
  return error;
}

Status TCPSocket::Connect(llvm::StringRef name) {

  Log *log = GetLog(LLDBLog::Communication);
  LLDB_LOG(log, "Connect to host/port {0}", name);

  Status error;
  llvm::Expected<HostAndPort> host_port = DecodeHostAndPort(name);
  if (!host_port)
    return Status::FromError(host_port.takeError());

  std::vector<SocketAddress> addresses =
      SocketAddress::GetAddressInfo(host_port->hostname.c_str(), nullptr,
                                    AF_UNSPEC, SOCK_STREAM, IPPROTO_TCP);
  for (SocketAddress &address : addresses) {
    error = CreateSocket(address.GetFamily());
    if (error.Fail())
      continue;

    address.SetPort(host_port->port);

    if (llvm::sys::RetryAfterSignal(-1, ::connect, GetNativeSocket(),
                                    &address.sockaddr(),
                                    address.GetLength()) == -1) {
      Close();
      continue;
    }

    if (SetOptionNoDelay() == -1) {
      Close();
      continue;
    }

    error.Clear();
    return error;
  }

  error = Status::FromErrorString("Failed to connect port");
  return error;
}

Status TCPSocket::Listen(llvm::StringRef name, int backlog) {
  Log *log = GetLog(LLDBLog::Connection);
  LLDB_LOG(log, "Listen to {0}", name);

  Status error;
  llvm::Expected<HostAndPort> host_port = DecodeHostAndPort(name);
  if (!host_port)
    return Status::FromError(host_port.takeError());

  if (host_port->hostname == "*")
    host_port->hostname = "0.0.0.0";
  std::vector<SocketAddress> addresses = SocketAddress::GetAddressInfo(
      host_port->hostname.c_str(), nullptr, AF_UNSPEC, SOCK_STREAM, IPPROTO_TCP);
  for (SocketAddress &address : addresses) {
    int fd =
        Socket::CreateSocket(address.GetFamily(), kType, IPPROTO_TCP, error);
    if (error.Fail() || fd < 0)
      continue;

    // enable local address reuse
    if (SetOption(fd, SOL_SOCKET, SO_REUSEADDR, 1) == -1) {
      CloseSocket(fd);
      continue;
    }

    SocketAddress listen_address = address;
    if(!listen_address.IsLocalhost())
      listen_address.SetToAnyAddress(address.GetFamily(), host_port->port);
    else
      listen_address.SetPort(host_port->port);

    int err =
        ::bind(fd, &listen_address.sockaddr(), listen_address.GetLength());
    if (err != -1)
      err = ::listen(fd, backlog);

    if (err == -1) {
      error = GetLastError();
      CloseSocket(fd);
      continue;
    }

    if (host_port->port == 0) {
      socklen_t sa_len = listen_address.GetLength();
      if (getsockname(fd, &listen_address.sockaddr(), &sa_len) == 0)
        host_port->port = listen_address.GetPort();
    }
    m_listen_sockets[fd] = listen_address;
  }

  if (m_listen_sockets.empty()) {
    assert(error.Fail());
    return error;
  }
  return Status();
}

void TCPSocket::CloseListenSockets() {
  for (auto socket : m_listen_sockets)
    CloseSocket(socket.first);
  m_listen_sockets.clear();
}

llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>>
TCPSocket::Accept(MainLoopBase &loop,
                  std::function<void(std::unique_ptr<Socket> socket)> sock_cb) {
  if (m_listen_sockets.size() == 0)
    return llvm::createStringError("No open listening sockets!");

  std::vector<MainLoopBase::ReadHandleUP> handles;
  for (auto socket : m_listen_sockets) {
    auto fd = socket.first;
    auto io_sp = std::make_shared<TCPSocket>(fd, false);
    auto cb = [this, fd, sock_cb](MainLoopBase &loop) {
      lldb_private::SocketAddress AcceptAddr;
      socklen_t sa_len = AcceptAddr.GetMaxLength();
      Status error;
      NativeSocket sock =
          AcceptSocket(fd, &AcceptAddr.sockaddr(), &sa_len, error);
      Log *log = GetLog(LLDBLog::Host);
      if (error.Fail()) {
        LLDB_LOG(log, "AcceptSocket({0}): {1}", fd, error);
        return;
      }

      const lldb_private::SocketAddress &AddrIn = m_listen_sockets[fd];
      if (!AddrIn.IsAnyAddr() && AcceptAddr != AddrIn) {
        CloseSocket(sock);
        LLDB_LOG(log, "rejecting incoming connection from {0} (expecting {1})",
                 AcceptAddr.GetIPAddress(), AddrIn.GetIPAddress());
        return;
      }
      std::unique_ptr<TCPSocket> sock_up(new TCPSocket(sock, *this));

      // Keep our TCP packets coming without any delays.
      sock_up->SetOptionNoDelay();

      sock_cb(std::move(sock_up));
    };
    Status error;
    handles.emplace_back(loop.RegisterReadObject(io_sp, cb, error));
    if (error.Fail())
      return error.ToError();
  }

  return handles;
}

int TCPSocket::SetOptionNoDelay() {
  return SetOption(IPPROTO_TCP, TCP_NODELAY, 1);
}

int TCPSocket::SetOptionReuseAddress() {
  return SetOption(SOL_SOCKET, SO_REUSEADDR, 1);
}
