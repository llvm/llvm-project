//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/common/DomainSocket.h"
#include "lldb/Utility/LLDBLog.h"
#ifdef __linux__
#include <lldb/Host/linux/AbstractSocket.h>
#endif

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <algorithm>
#include <cstddef>
#include <memory>

#ifdef _WIN32
#include <afunix.h>
#else
#include <sys/socket.h>
#include <sys/un.h>
#endif

using namespace lldb;
using namespace lldb_private;

static const int kDomain = AF_UNIX;
static const int kType = SOCK_STREAM;

std::string DomainSocket::NativePathToURIPath(llvm::StringRef path) {
  // A drive-letter path (e.g. "C:\\dir\\sock") is not a valid URI authority, so
  // it is carried in the RFC 8089 URI form "/C:/dir/sock".
  if (path.size() >= 2 && llvm::isAlpha(path[0]) && path[1] == ':') {
    std::string uri_path = "/" + path.str();
    std::replace(uri_path.begin(), uri_path.end(), '\\', '/');
    return uri_path;
  }
  // A UNC path (e.g. "\\\\server\\share") is already anchored by its leading
  // slashes.
  if (path.starts_with("\\\\")) {
    std::string uri_path = path.str();
    std::replace(uri_path.begin(), uri_path.end(), '\\', '/');
    return uri_path;
  }
  return path.str();
}

std::string DomainSocket::URIPathToNativePath(llvm::StringRef path) {
  // Reverse of the drive-letter mapping: "/C:/dir/sock" -> "C:\\dir\\sock".
  if (path.size() >= 3 && path[0] == '/' && llvm::isAlpha(path[1]) &&
      path[2] == ':') {
    std::string native = path.drop_front().str();
    std::replace(native.begin(), native.end(), '/', '\\');
    return native;
  }
  // Reverse of the UNC mapping: "//server/share" -> "\\\\server\\share".
  if (path.starts_with("//")) {
    std::string native = path.str();
    std::replace(native.begin(), native.end(), '/', '\\');
    return native;
  }
  return path.str();
}

static bool SetSockAddr(llvm::StringRef name, const size_t name_offset,
                        sockaddr_un *saddr_un, socklen_t &saddr_un_len) {
  if (name.size() + name_offset > sizeof(saddr_un->sun_path))
    return false;

  memset(saddr_un, 0, sizeof(*saddr_un));
  saddr_un->sun_family = kDomain;

  memcpy(saddr_un->sun_path + name_offset, name.data(), name.size());

  // Compute the address length explicitly rather than via SUN_LEN: that macro
  // is not available on Windows.
  saddr_un_len =
      offsetof(struct sockaddr_un, sun_path) + name_offset + name.size();

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) ||       \
    defined(__OpenBSD__)
  saddr_un->sun_len = saddr_un_len;
#endif

  return true;
}

DomainSocket::DomainSocket(bool should_close)
    : DomainSocket(kInvalidSocketValue, should_close) {}

DomainSocket::DomainSocket(NativeSocket socket, bool should_close)
    : Socket(ProtocolUnixDomain, should_close) {
  m_socket = socket;
}

DomainSocket::DomainSocket(SocketProtocol protocol)
    : Socket(protocol, /*should_close=*/true) {}

DomainSocket::DomainSocket(NativeSocket socket,
                           const DomainSocket &listen_socket)
    : Socket(ProtocolUnixDomain, listen_socket.m_should_close_fd) {
  m_socket = socket;
}

DomainSocket::DomainSocket(SocketProtocol protocol, NativeSocket socket,
                           bool should_close)
    : Socket(protocol, should_close) {
  m_socket = socket;
}

Status DomainSocket::Connect(llvm::StringRef name) {
  std::string native_name = URIPathToNativePath(name);
  sockaddr_un saddr_un;
  socklen_t saddr_un_len;
  if (!SetSockAddr(native_name, GetNameOffset(), &saddr_un, saddr_un_len))
    return Status::FromErrorString("Failed to set socket address");

  Status error;
  m_socket = CreateSocket(kDomain, kType, 0, error);
  if (error.Fail())
    return error;
  if (llvm::sys::RetryAfterSignal(-1, ::connect, GetNativeSocket(),
                                  (struct sockaddr *)&saddr_un,
                                  saddr_un_len) < 0)
    SetLastError(error);

  return error;
}

Status DomainSocket::Listen(llvm::StringRef name, int backlog) {
  std::string native_name = URIPathToNativePath(name);
  sockaddr_un saddr_un;
  socklen_t saddr_un_len;
  if (!SetSockAddr(native_name, GetNameOffset(), &saddr_un, saddr_un_len))
    return Status::FromErrorString("Failed to set socket address");

  DeleteSocketFile(native_name);

  Status error;
  m_socket = CreateSocket(kDomain, kType, 0, error);
  if (error.Fail())
    return error;
  if (::bind(GetNativeSocket(), (struct sockaddr *)&saddr_un, saddr_un_len) ==
      0)
    if (::listen(GetNativeSocket(), backlog) == 0)
      return error;

  SetLastError(error);
  return error;
}

llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>> DomainSocket::Accept(
    MainLoopBase &loop,
    std::function<void(std::unique_ptr<Socket> socket)> sock_cb) {
  // TODO: Refactor MainLoop to avoid the shared_ptr requirement.
  auto io_sp = std::make_shared<DomainSocket>(GetNativeSocket(), false);
  auto cb = [this, sock_cb](MainLoopBase &loop) {
    Log *log = GetLog(LLDBLog::Host);
    Status error;
    auto conn_fd = AcceptSocket(GetNativeSocket(), nullptr, nullptr, error);
    if (error.Fail()) {
      LLDB_LOG(log, "AcceptSocket({0}): {1}", GetNativeSocket(), error);
      return;
    }
    std::unique_ptr<DomainSocket> sock_up(new DomainSocket(conn_fd, *this));
    sock_cb(std::move(sock_up));
  };

  Status error;
  std::vector<MainLoopBase::ReadHandleUP> handles;
  handles.emplace_back(loop.RegisterReadObject(io_sp, cb, error));
  if (error.Fail())
    return error.ToError();
  return handles;
}

size_t DomainSocket::GetNameOffset() const { return 0; }

void DomainSocket::DeleteSocketFile(llvm::StringRef name) {
  llvm::sys::fs::remove(name);
}

std::string DomainSocket::GetSocketName() const {
  if (m_socket == kInvalidSocketValue)
    return "";

  struct sockaddr_un saddr_un;
  saddr_un.sun_family = AF_UNIX;
  socklen_t sock_addr_len = sizeof(struct sockaddr_un);
  if (::getpeername(m_socket, (struct sockaddr *)&saddr_un, &sock_addr_len) !=
      0)
    return "";

  if (sock_addr_len <= offsetof(struct sockaddr_un, sun_path))
    return ""; // Unnamed domain socket

  llvm::StringRef name(saddr_un.sun_path + GetNameOffset(),
                       sock_addr_len - offsetof(struct sockaddr_un, sun_path) -
                           GetNameOffset());
  name = name.rtrim('\0');

  return name.str();
}

std::string DomainSocket::GetRemoteConnectionURI() const {
  std::string name = GetSocketName();
  if (name.empty())
    return name;

  if (GetNameOffset() == 0)
    return llvm::formatv("unix-connect://{0}", NativePathToURIPath(name));
  return llvm::formatv("unix-abstract-connect://{0}", name);
}

std::vector<std::string> DomainSocket::GetListeningConnectionURI() const {
  if (m_socket == kInvalidSocketValue)
    return {};

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  socklen_t addr_len = sizeof(struct sockaddr_un);
  if (::getsockname(m_socket, (struct sockaddr *)&addr, &addr_len) != 0)
    return {};

  return {
      llvm::formatv("unix-connect://{0}", NativePathToURIPath(addr.sun_path))};
}

llvm::Expected<std::unique_ptr<DomainSocket>>
DomainSocket::FromBoundNativeSocket(NativeSocket sockfd, bool should_close) {
  // Check if fd represents domain socket or abstract socket.
  struct sockaddr_un addr;
  socklen_t addr_len = sizeof(addr);
  if (getsockname(sockfd, (struct sockaddr *)&addr, &addr_len) == -1)
    return llvm::createStringError("not a socket or error occurred");
  if (addr.sun_family != AF_UNIX)
    return llvm::createStringError("bad socket type");
#ifdef __linux__
  if (addr_len > offsetof(struct sockaddr_un, sun_path) &&
      addr.sun_path[0] == '\0')
    return std::make_unique<AbstractSocket>(sockfd, should_close);
#endif
  return std::make_unique<DomainSocket>(sockfd, should_close);
}
