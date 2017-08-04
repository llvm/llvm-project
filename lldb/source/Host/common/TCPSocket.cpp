//===-- TCPSocket.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(_MSC_VER)
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#endif

#include "lldb/Host/common/TCPSocket.h"

#include "lldb/Core/Log.h"
#include "lldb/Host/Config.h"

#ifndef LLDB_DISABLE_POSIX
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#endif

using namespace lldb;
using namespace lldb_private;

namespace {

const int kDomain = AF_INET;
const int kType = SOCK_STREAM;
}

TCPSocket::TCPSocket(NativeSocket socket, bool should_close)
    : Socket(socket, ProtocolTcp, should_close) {}

TCPSocket::TCPSocket(bool child_processes_inherit, Error &error)
    : TCPSocket(CreateSocket(kDomain, kType, IPPROTO_TCP,
                             child_processes_inherit, error),
                true) {}

// Return the port number that is being used by the socket.
uint16_t TCPSocket::GetLocalPortNumber() const {
  if (m_socket != kInvalidSocketValue) {
    SocketAddress sock_addr;
    socklen_t sock_addr_len = sock_addr.GetMaxLength();
    if (::getsockname(m_socket, sock_addr, &sock_addr_len) == 0)
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

Error TCPSocket::Connect(llvm::StringRef name) {
  if (m_socket == kInvalidSocketValue)
    return Error("Invalid socket");

  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_COMMUNICATION));
  if (log)
    log->Printf("TCPSocket::%s (host/port = %s)", __FUNCTION__, name.data());

  Error error;
  std::string host_str;
  std::string port_str;
  int32_t port = INT32_MIN;
  if (!DecodeHostAndPort(name, host_str, port_str, port, &error))
    return error;

  struct sockaddr_in sa;
  ::memset(&sa, 0, sizeof(sa));
  sa.sin_family = kDomain;
  sa.sin_port = htons(port);

  int inet_pton_result = ::inet_pton(kDomain, host_str.c_str(), &sa.sin_addr);

  if (inet_pton_result <= 0) {
    struct hostent *host_entry = gethostbyname(host_str.c_str());
    if (host_entry)
      host_str = ::inet_ntoa(*(struct in_addr *)*host_entry->h_addr_list);
    inet_pton_result = ::inet_pton(kDomain, host_str.c_str(), &sa.sin_addr);
    if (inet_pton_result <= 0) {
      if (inet_pton_result == -1)
        SetLastError(error);
      else
        error.SetErrorStringWithFormat("invalid host string: '%s'",
                                       host_str.c_str());

      return error;
    }
  }

  if (-1 ==
      ::connect(GetNativeSocket(), (const struct sockaddr *)&sa, sizeof(sa))) {
    SetLastError(error);
    return error;
  }

  // Keep our TCP packets coming without any delays.
  SetOptionNoDelay();
  error.Clear();
  return error;
}

Error TCPSocket::Listen(llvm::StringRef name, int backlog) {
  Error error;

  // enable local address reuse
  SetOptionReuseAddress();

  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_CONNECTION));
  if (log)
    log->Printf("TCPSocket::%s (%s)", __FUNCTION__, name.data());

  std::string host_str;
  std::string port_str;
  int32_t port = INT32_MIN;
  if (!DecodeHostAndPort(name, host_str, port_str, port, &error))
    return error;

  SocketAddress bind_addr;

  // Only bind to the loopback address if we are expecting a connection from
  // localhost to avoid any firewall issues.
  const bool bind_addr_success = (host_str == "127.0.0.1")
                                     ? bind_addr.SetToLocalhost(kDomain, port)
                                     : bind_addr.SetToAnyAddress(kDomain, port);

  if (!bind_addr_success) {
    error.SetErrorString("Failed to bind port");
    return error;
  }

  int err = ::bind(GetNativeSocket(), bind_addr, bind_addr.GetLength());
  if (err != -1)
    err = ::listen(GetNativeSocket(), backlog);

  if (err == -1)
    SetLastError(error);

  return error;
}

Error TCPSocket::Accept(llvm::StringRef name, bool child_processes_inherit,
                        Socket *&conn_socket) {
  Error error;
  std::string host_str;
  std::string port_str;
  int32_t port;
  if (!DecodeHostAndPort(name, host_str, port_str, port, &error))
    return error;

  const sa_family_t family = kDomain;
  const int socktype = kType;
  const int protocol = IPPROTO_TCP;
  SocketAddress listen_addr;
  if (host_str.empty())
    listen_addr.SetToLocalhost(family, port);
  else if (host_str.compare("*") == 0)
    listen_addr.SetToAnyAddress(family, port);
  else {
    if (!listen_addr.getaddrinfo(host_str.c_str(), port_str.c_str(), family,
                                 socktype, protocol)) {
      error.SetErrorStringWithFormat("unable to resolve hostname '%s'",
                                     host_str.c_str());
      return error;
    }
  }

  bool accept_connection = false;
  std::unique_ptr<TCPSocket> accepted_socket;

  // Loop until we are happy with our connection
  while (!accept_connection) {
    struct sockaddr_in accept_addr;
    ::memset(&accept_addr, 0, sizeof accept_addr);
#if !(defined(__linux__) || defined(_WIN32))
    accept_addr.sin_len = sizeof accept_addr;
#endif
    socklen_t accept_addr_len = sizeof accept_addr;

    int sock = AcceptSocket(GetNativeSocket(), (struct sockaddr *)&accept_addr,
                            &accept_addr_len, child_processes_inherit, error);

    if (error.Fail())
      break;

    bool is_same_addr = true;
#if !(defined(__linux__) || (defined(_WIN32)))
    is_same_addr = (accept_addr_len == listen_addr.sockaddr_in().sin_len);
#endif
    if (is_same_addr)
      is_same_addr = (accept_addr.sin_addr.s_addr ==
                      listen_addr.sockaddr_in().sin_addr.s_addr);

    if (is_same_addr ||
        (listen_addr.sockaddr_in().sin_addr.s_addr == INADDR_ANY)) {
      accept_connection = true;
      accepted_socket.reset(new TCPSocket(sock, true));
    } else {
      const uint8_t *accept_ip = (const uint8_t *)&accept_addr.sin_addr.s_addr;
      const uint8_t *listen_ip =
          (const uint8_t *)&listen_addr.sockaddr_in().sin_addr.s_addr;
      ::fprintf(stderr, "error: rejecting incoming connection from %u.%u.%u.%u "
                        "(expecting %u.%u.%u.%u)\n",
                accept_ip[0], accept_ip[1], accept_ip[2], accept_ip[3],
                listen_ip[0], listen_ip[1], listen_ip[2], listen_ip[3]);
      accepted_socket.reset();
    }
  }

  if (!accepted_socket)
    return error;

  // Keep our TCP packets coming without any delays.
  accepted_socket->SetOptionNoDelay();
  error.Clear();
  conn_socket = accepted_socket.release();
  return error;
}

int TCPSocket::SetOptionNoDelay() {
  return SetOption(IPPROTO_TCP, TCP_NODELAY, 1);
}

int TCPSocket::SetOptionReuseAddress() {
  return SetOption(SOL_SOCKET, SO_REUSEADDR, 1);
}
